"""Camera trajectory generation for nanoWM.

Two loop-closing path families, both required by the persistence metric
(revisit-PSNR, see .claude/agents/data-engineer.md and CLAUDE.md):

  - `closed_circuit`: a closed centripetal-ish Catmull-Rom spline through a
    ring of randomized control points. The camera travels `num_loops`
    times around the ring, so the tail of the trajectory revisits its head.
  - `out_and_back`: the camera travels forward along a randomized spline
    path and then re-traverses the *same* path back to the start, so the
    return leg revisits nearly every pose from the outbound leg.

Pose convention (must match what `model-architect` consumes for Plücker
raymaps):
  - Poses are camera-to-world 4x4 matrices (`numpy.float32`, shape (4, 4)).
  - Camera axes follow OpenGL convention: +X right, +Y up, -Z forward
    (i.e. the camera looks down its own -Z axis).
  - World space is right-handed, Z-up.
  - Intrinsics are a separate pinhole tuple `(fx, fy, cx, cy)` in pixels,
    resolution-independent of this module (resolution/AE choice is open
    until M0 per the CLAUDE.md decision log — do not hardcode it here).

Each frame carries `revisit_of: Optional[int]`: the index of the earliest
earlier frame in the *same* trajectory whose pose lies within
`(pos_eps, rot_theta_eps)` of this frame's pose, or `None` if no such frame
exists. Tolerances are configurable, not hardcoded.

Everything here is deterministic from an integer seed.
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional, Tuple

import numpy as np

DEFAULT_POS_EPS = 0.35
DEFAULT_ROT_THETA_EPS = np.deg2rad(15.0)
VALID_LENGTHS = (32, 64, 128)


@dataclasses.dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.fx, self.fy, self.cx, self.cy)


@dataclasses.dataclass
class Trajectory:
    seed: int
    kind: str  # "closed_circuit" | "out_and_back"
    poses: np.ndarray  # (N, 4, 4) float32, camera-to-world
    intrinsics: Intrinsics
    revisit_of: List[Optional[int]]
    pos_eps: float
    rot_theta_eps: float

    def __post_init__(self) -> None:
        assert self.poses.ndim == 3 and self.poses.shape[1:] == (4, 4)
        self.poses = self.poses.astype(np.float32, copy=False)
        assert len(self.revisit_of) == self.poses.shape[0]

    @property
    def length(self) -> int:
        return int(self.poses.shape[0])

    @property
    def num_revisits(self) -> int:
        return sum(1 for r in self.revisit_of if r is not None)


def make_intrinsics(width: int, height: int, fov_y_deg: float = 60.0) -> Intrinsics:
    """Pinhole intrinsics from an image size and vertical FOV.

    Resolution (`width`/`height`) is a caller-supplied parameter, never
    hardcoded here — it is set by whatever consumes this trajectory
    (renderer / AE choice), which is still open per the M0 decision log.
    """
    fov_y = np.deg2rad(fov_y_deg)
    fy = height / (2.0 * np.tan(fov_y / 2.0))
    fx = fy  # square pixels
    cx = width / 2.0
    cy = height / 2.0
    return Intrinsics(fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy))


def _look_at_c2w(position: np.ndarray, forward: np.ndarray, world_up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Build an OpenGL-convention camera-to-world matrix.

    `forward` is the world-space direction the camera looks along
    (need not be normalized). Camera looks down its own -Z, so the
    camera-space +Z basis vector is -forward.
    """
    f = np.asarray(forward, dtype=np.float64)
    f = f / (np.linalg.norm(f) + 1e-12)
    up = np.asarray(world_up, dtype=np.float64)

    right = np.cross(f, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-8:
        # forward nearly parallel to world_up: pick an arbitrary stable up.
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(f, up)
        right_norm = np.linalg.norm(right)
    right = right / right_norm
    cam_up = np.cross(right, f)
    cam_up = cam_up / (np.linalg.norm(cam_up) + 1e-12)

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right  # +X right
    c2w[:3, 1] = cam_up  # +Y up
    c2w[:3, 2] = -f  # +Z is behind the camera (looks down -Z)
    c2w[:3, 3] = np.asarray(position, dtype=np.float64)
    return c2w.astype(np.float32)


def _catmull_rom_point(p0, p1, p2, p3, t: float) -> np.ndarray:
    """Uniform Catmull-Rom interpolation between p1 and p2 at t in [0, 1]."""
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2 * p1)
        + (-p0 + p2) * t
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
        + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
    )


def _sample_closed_spline(control_points: np.ndarray, u: float) -> np.ndarray:
    """Sample a closed Catmull-Rom spline at global parameter u in [0, K)
    where K = number of control points. Wraps around."""
    k = control_points.shape[0]
    seg = int(np.floor(u)) % k
    t = u - np.floor(u)
    p0 = control_points[(seg - 1) % k]
    p1 = control_points[seg % k]
    p2 = control_points[(seg + 1) % k]
    p3 = control_points[(seg + 2) % k]
    return _catmull_rom_point(p0, p1, p2, p3, t)


def _random_ring_control_points(
    rng: np.random.Generator,
    room_extent: Tuple[float, float, float],
    num_points: int,
    margin: float = 0.8,
    height_range: Tuple[float, float] = (1.2, 1.8),
) -> np.ndarray:
    size_x, size_y, height_z = room_extent
    max_r_x = size_x / 2.0 - margin
    max_r_y = size_y / 2.0 - margin
    angles = np.sort(rng.uniform(0, 2 * np.pi, size=num_points))
    radii_x = rng.uniform(0.4, 1.0, size=num_points) * max_r_x
    radii_y = rng.uniform(0.4, 1.0, size=num_points) * max_r_y
    zs = rng.uniform(*height_range, size=num_points)
    zs = np.clip(zs, 0.1, max(0.1, height_z - 0.2))
    xs = radii_x * np.cos(angles)
    ys = radii_y * np.sin(angles)
    return np.stack([xs, ys, zs], axis=1)


def rotation_geodesic_angle(r1: np.ndarray, r2: np.ndarray) -> float:
    """Angle (radians) of the relative rotation between two 3x3 matrices."""
    rel = r1.T @ r2
    cos_theta = (np.trace(rel) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.arccos(cos_theta))


def label_revisits(
    poses: np.ndarray, pos_eps: float, rot_theta_eps: float
) -> List[Optional[int]]:
    """For each frame, find the earliest earlier frame within tolerance.

    Position distance uses Euclidean distance between camera centers
    (poses[:, :3, 3]); rotation distance uses the geodesic angle between
    the 3x3 rotation blocks. A frame revisits an earlier one only if both
    are within tolerance.
    """
    n = poses.shape[0]
    revisit_of: List[Optional[int]] = [None] * n
    positions = poses[:, :3, 3]
    for i in range(n):
        for j in range(i):
            dpos = float(np.linalg.norm(positions[i] - positions[j]))
            if dpos > pos_eps:
                continue
            drot = rotation_geodesic_angle(poses[j, :3, :3], poses[i, :3, :3])
            if drot > rot_theta_eps:
                continue
            revisit_of[i] = j
            break
    return revisit_of


def generate_closed_circuit(
    seed: int,
    length: int = 64,
    room_extent: Tuple[float, float, float] = (6.0, 6.0, 3.0),
    num_control_points: int = 8,
    num_loops: float = 1.0,
    intrinsics: Optional[Intrinsics] = None,
    pos_eps: float = DEFAULT_POS_EPS,
    rot_theta_eps: float = DEFAULT_ROT_THETA_EPS,
) -> Trajectory:
    """Closed Catmull-Rom circuit: camera loops `num_loops` times around a
    randomized ring inside the room, looking along its direction of travel.

    With `num_loops >= 1` the trajectory is guaranteed to revisit its own
    early poses near the end (loop closure), satisfying the hard
    persistence-metric requirement in .claude/agents/data-engineer.md.
    """
    if length not in VALID_LENGTHS:
        raise ValueError(f"length must be one of {VALID_LENGTHS}, got {length}")
    rng = np.random.default_rng(seed)
    control_points = _random_ring_control_points(rng, room_extent, num_control_points)
    k = control_points.shape[0]

    poses = np.zeros((length, 4, 4), dtype=np.float32)
    du = 1e-3
    for i in range(length):
        frac = i / length  # in [0, 1)
        u = frac * num_loops * k
        pos = _sample_closed_spline(control_points, u)
        pos_ahead = _sample_closed_spline(control_points, u + du)
        forward = pos_ahead - pos
        poses[i] = _look_at_c2w(pos, forward)

    intr = intrinsics or make_intrinsics(width=256, height=256)
    revisit_of = label_revisits(poses, pos_eps, rot_theta_eps)
    return Trajectory(
        seed=seed,
        kind="closed_circuit",
        poses=poses,
        intrinsics=intr,
        revisit_of=revisit_of,
        pos_eps=pos_eps,
        rot_theta_eps=rot_theta_eps,
    )


def generate_out_and_back(
    seed: int,
    length: int = 64,
    room_extent: Tuple[float, float, float] = (6.0, 6.0, 3.0),
    num_control_points: int = 5,
    intrinsics: Optional[Intrinsics] = None,
    pos_eps: float = DEFAULT_POS_EPS,
    rot_theta_eps: float = DEFAULT_ROT_THETA_EPS,
) -> Trajectory:
    """Out-and-back path: an open Catmull-Rom-ish path traversed forward
    then backward along the same physical points.

    Uses a clamped (non-closed) Catmull-Rom path by duplicating the first
    and last control points, sampled forward for the first half of frames
    and backward for the second half. The return leg re-visits essentially
    every outbound pose (up to eps), guaranteeing many revisit_of labels.
    """
    if length not in VALID_LENGTHS:
        raise ValueError(f"length must be one of {VALID_LENGTHS}, got {length}")
    if length % 2 != 0:
        raise ValueError("out_and_back requires an even length")
    rng = np.random.default_rng(seed)

    size_x, size_y, height_z = room_extent
    margin = 0.8
    max_r_x = size_x / 2.0 - margin
    max_r_y = size_y / 2.0 - margin
    t_line = np.linspace(0.0, 1.0, num_control_points)
    xs = (t_line * 2 - 1) * max_r_x * rng.uniform(0.6, 1.0)
    ys = rng.uniform(-max_r_y, max_r_y, size=num_control_points) * 0.5
    zs = np.clip(rng.uniform(1.2, 1.8, size=num_control_points), 0.1, max(0.1, height_z - 0.2))
    control_points = np.stack([xs, ys, zs], axis=1)
    # clamp endpoints by duplication so the open spline starts/ends exactly
    # at the first/last control point.
    padded = np.concatenate([control_points[:1], control_points, control_points[-1:]], axis=0)
    k_segments = num_control_points - 1  # number of interior segments

    half = length // 2

    def sample_open(u: float) -> np.ndarray:
        u = float(np.clip(u, 0.0, k_segments - 1e-6))
        seg = int(np.floor(u))
        t = u - seg
        p0, p1, p2, p3 = padded[seg], padded[seg + 1], padded[seg + 2], padded[seg + 3]
        return _catmull_rom_point(p0, p1, p2, p3, t)

    du = 1e-3
    outbound_positions = []
    outbound_forwards = []
    for i in range(half):
        frac = i / (half - 1) if half > 1 else 0.0
        u = frac * k_segments
        pos = sample_open(u)
        pos_ahead = sample_open(min(u + du, k_segments - 1e-6))
        forward = pos_ahead - pos
        if np.linalg.norm(forward) < 1e-9:
            forward = pos - sample_open(max(u - du, 0.0))
        outbound_positions.append(pos)
        outbound_forwards.append(forward)

    poses = np.zeros((length, 4, 4), dtype=np.float32)
    for i in range(half):
        poses[i] = _look_at_c2w(outbound_positions[i], outbound_forwards[i])
    # Return leg: same physical points in reverse order, camera still faces
    # its direction of travel (now the reverse direction).
    for i in range(half):
        src = half - 1 - i
        poses[half + i] = _look_at_c2w(outbound_positions[src], -outbound_forwards[src])

    intr = intrinsics or make_intrinsics(width=256, height=256)
    revisit_of = label_revisits(poses, pos_eps, rot_theta_eps)
    return Trajectory(
        seed=seed,
        kind="out_and_back",
        poses=poses,
        intrinsics=intr,
        revisit_of=revisit_of,
        pos_eps=pos_eps,
        rot_theta_eps=rot_theta_eps,
    )
