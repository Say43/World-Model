"""Plücker raymap pose conditioning for nanoWM.

Consumes exactly the pose interface `data-engineer` produces (see
`src/data/trajectories.py`):
  - camera-to-world poses: `(..., 4, 4)` float32.
  - OpenGL camera axes: +X right, +Y up, -Z forward (camera looks down its
    own -Z axis).
  - World space: right-handed, Z-up.
  - Intrinsics: separate pinhole `(fx, fy, cx, cy)`, resolution-independent.

The raymap spatial resolution (`height`, `width` here) is a caller-supplied
parameter, never hardcoded -- it must match whatever `tokens_per_frame`
grid the DiT is configured with (see `dit.DiTConfig.raymap_resolution`),
which is itself open until the M0 autoencoder gate. `intrinsics` passed to
`plucker_raymap` must already be expressed in that raymap grid's pixel
units, not the original render resolution -- use `scale_intrinsics` below
to rescale from the source render resolution.

Plücker coordinates: a ray is represented as `(d, m)` where `d` is the
(unit) direction and `m = o x d` is the moment (o = ray origin, the camera
center). This 6-vector is invariant to reparametrization of the ray, is
smooth as a function of the pose, and is the representation used by
RTFM-style pose-conditioned world models.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def scale_intrinsics(
    intrinsics: torch.Tensor, src_size: Tuple[int, int], dst_size: Tuple[int, int]
) -> torch.Tensor:
    """Rescale `(fx, fy, cx, cy)` (..., 4) from `src_size=(H,W)` pixel units
    to `dst_size=(H,W)` pixel units (e.g. render resolution -> raymap/patch
    grid resolution). Pure elementwise scaling, no assumptions about batch
    shape beyond the trailing size-4 axis.
    """
    src_h, src_w = src_size
    dst_h, dst_w = dst_size
    scale_x = dst_w / src_w
    scale_y = dst_h / src_h
    fx, fy, cx, cy = intrinsics.unbind(-1)
    return torch.stack([fx * scale_x, fy * scale_y, cx * scale_x, cy * scale_y], dim=-1)


def _pixel_grid(height: int, width: int, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pixel-center grid coordinates, `(height, width)` each, `u` = column
    (x, right), `v` = row (y, down) -- matches the convention used by
    `src/data/render.py:project_to_screen` so a raymap built here and a
    frame rendered by that module refer to the same pixel centers."""
    vs = torch.arange(height, device=device, dtype=dtype) + 0.5
    us = torch.arange(width, device=device, dtype=dtype) + 0.5
    grid_v, grid_u = torch.meshgrid(vs, us, indexing="ij")
    return grid_u, grid_v  # each (H, W)


def plucker_raymap(
    poses: torch.Tensor,
    intrinsics: torch.Tensor,
    height: int,
    width: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Build per-pixel Plücker raymaps.

    Args:
        poses: `(..., 4, 4)` float, camera-to-world (OpenGL axes).
        intrinsics: `(..., 4)` float, `(fx, fy, cx, cy)` in `height`/`width`
            pixel units (see `scale_intrinsics` if they aren't already).
        height, width: raymap grid resolution. A free parameter -- must
            match `tokens_per_frame` == `height * width` at the call site.

    Returns:
        `(..., height * width, 6)` float, flattened row-major (v-major,
        matching `_pixel_grid`/`project_to_screen`), last dim `[d (3), m (3)]`
        with `d` unit-norm and `m = o x d`.

    Inverts exactly the pinhole model `src/data/render.py:project_to_screen`
    uses for rendering: `u = fx * x_cam/depth + cx`, `v = cy - fy *
    y_cam/depth`, `depth = -z_cam` (positive in front, OpenGL -Z-forward).
    Solving for the camera-space ray direction at `depth=1` gives
    `((u-cx)/fx, (cy-v)/fy, -1)`.
    """
    batch_shape = poses.shape[:-2]
    device, dtype = poses.device, poses.dtype

    fx, fy, cx, cy = intrinsics.unbind(-1)  # each (...,)
    grid_u, grid_v = _pixel_grid(height, width, device=device, dtype=dtype)  # (H, W)

    # Broadcast intrinsics (..., 1, 1) against the pixel grid (H, W).
    fx = fx[..., None, None]
    fy = fy[..., None, None]
    cx = cx[..., None, None]
    cy = cy[..., None, None]

    x_cam = (grid_u - cx) / fx
    y_cam = (cy - grid_v) / fy
    z_cam = -torch.ones_like(x_cam)
    dirs_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # (..., H, W, 3)
    dirs_cam = dirs_cam.reshape(*batch_shape, height * width, 3)

    rot = poses[..., :3, :3]  # (..., 3, 3)
    origin = poses[..., :3, 3]  # (..., 3)

    # world direction = R @ dir_cam, batched over the flattened pixel axis.
    dirs_world = torch.einsum("...ij,...nj->...ni", rot, dirs_cam)
    dirs_world = dirs_world / (dirs_world.norm(dim=-1, keepdim=True) + eps)

    origin_b = origin[..., None, :].expand_as(dirs_world)  # (..., N, 3)
    moment = torch.cross(origin_b, dirs_world, dim=-1)

    return torch.cat([dirs_world, moment], dim=-1)  # (..., N, 6)


class RaymapEncoder(nn.Module):
    """Projects per-pixel Plücker 6-vectors into the transformer's channel
    dimension. A small 2-layer MLP rather than a bare linear layer: the 6
    raw Plücker components are a low-dimensional, highly structured input
    (unit direction + moment) and a single linear layer forces the model to
    do all pose reasoning downstream in the (shared, adaLN-modulated)
    transformer blocks. The extra hidden layer is deliberately kept narrow
    (`hidden_mult` config, default 2x) since pose conditioning is not meant
    to dominate the parameter budget.
    """

    def __init__(self, dim: int, hidden_mult: int = 2):
        super().__init__()
        hidden = max(dim // hidden_mult, 6)
        self.net = nn.Sequential(
            nn.Linear(6, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, raymap: torch.Tensor) -> torch.Tensor:
        """`raymap`: (..., N, 6) -> (..., N, dim)."""
        return self.net(raymap)
