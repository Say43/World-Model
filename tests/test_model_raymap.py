"""Tests for src/model/raymap.py Plücker raymap construction."""
import numpy as np
import torch

from src.data.trajectories import generate_closed_circuit, make_intrinsics
from src.data.render import project_to_screen
from src.model.raymap import RaymapEncoder, plucker_raymap, scale_intrinsics


def test_plucker_raymap_shape():
    poses = torch.eye(4).view(1, 1, 4, 4).repeat(2, 3, 1, 1)
    intr = torch.tensor([100.0, 100.0, 32.0, 32.0]).view(1, 1, 4).repeat(2, 3, 1)
    out = plucker_raymap(poses, intr, height=8, width=8)
    assert out.shape == (2, 3, 64, 6)


def test_plucker_direction_is_unit_norm():
    poses = torch.eye(4).view(1, 4, 4)
    intr = torch.tensor([[50.0, 50.0, 16.0, 16.0]])
    out = plucker_raymap(poses, intr, height=4, width=4)
    d = out[..., :3]
    norms = d.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_plucker_moment_orthogonal_to_direction_from_origin_camera():
    """At the identity pose (camera at world origin), the moment m = o x d
    with o=0 must be exactly zero."""
    poses = torch.eye(4).view(1, 4, 4)
    intr = torch.tensor([[50.0, 50.0, 16.0, 16.0]])
    out = plucker_raymap(poses, intr, height=4, width=4)
    m = out[..., 3:]
    assert torch.allclose(m, torch.zeros_like(m), atol=1e-6)


def test_plucker_center_pixel_looks_down_minus_z_at_identity():
    """The principal-point ray at the identity pose should be exactly
    (0, 0, -1): camera looks down its own -Z axis (world == camera axes
    at the identity pose)."""
    poses = torch.eye(4).view(1, 4, 4)
    # Odd resolution so a pixel center lands exactly on the principal point.
    intr = torch.tensor([[50.0, 50.0, 2.5, 2.5]])
    out = plucker_raymap(poses, intr, height=5, width=5)
    center_idx = 2 * 5 + 2  # row=2, col=2 -> pixel center (2.5, 2.5)
    d = out[0, center_idx, :3]
    assert torch.allclose(d, torch.tensor([0.0, 0.0, -1.0]), atol=1e-5)


def test_plucker_consistent_with_render_projection():
    """Round-trip check against src/data/render.py's projection convention:
    a world point lying along a raymap ray at some positive depth should
    project back to (approximately) the pixel center that generated it."""
    traj = generate_closed_circuit(seed=1, length=32)
    pose_np = traj.poses[0]
    intr = traj.intrinsics
    width, height = 32, 24
    scaled = make_intrinsics(width, height)  # independent intrinsics at this resolution

    poses_t = torch.from_numpy(pose_np).view(1, 4, 4)
    intr_t = torch.tensor([scaled.as_tuple()], dtype=torch.float32)
    raymap = plucker_raymap(poses_t, intr_t, height=height, width=width)

    row, col = 10, 15
    idx = row * width + col
    d = raymap[0, idx, :3].numpy().astype(np.float64)
    origin = pose_np[:3, 3].astype(np.float64)

    depth = 3.0
    world_point = origin + depth * d
    screen_xy, proj_depth = project_to_screen(
        world_point[None, :], pose_np, scaled.as_tuple(), width, height
    )
    assert proj_depth[0] > 0
    assert abs(screen_xy[0, 0] - (col + 0.5)) < 1e-3
    assert abs(screen_xy[0, 1] - (row + 0.5)) < 1e-3


def test_scale_intrinsics():
    intr = torch.tensor([[100.0, 100.0, 128.0, 128.0]])
    scaled = scale_intrinsics(intr, src_size=(256, 256), dst_size=(8, 8))
    expected = torch.tensor([[100.0 / 32, 100.0 / 32, 4.0, 4.0]])
    assert torch.allclose(scaled, expected)


def test_raymap_encoder_shape():
    enc = RaymapEncoder(dim=48)
    raymap = torch.randn(2, 4, 16, 6)
    out = enc(raymap)
    assert out.shape == (2, 4, 16, 48)
