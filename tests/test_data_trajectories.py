"""Tests for src/data/trajectories.py: determinism, loop closure, and pose
matrix shape/convention checks."""
import numpy as np
import pytest

from src.data.trajectories import (
    DEFAULT_MIN_REVISIT_GAP,
    VALID_LENGTHS,
    generate_closed_circuit,
    generate_out_and_back,
    rotation_geodesic_angle,
)


@pytest.mark.parametrize("length", VALID_LENGTHS)
def test_closed_circuit_determinism(length):
    a = generate_closed_circuit(seed=7, length=length)
    b = generate_closed_circuit(seed=7, length=length)
    np.testing.assert_array_equal(a.poses, b.poses)
    assert a.revisit_of == b.revisit_of


@pytest.mark.parametrize("length", VALID_LENGTHS)
def test_out_and_back_determinism(length):
    a = generate_out_and_back(seed=11, length=length)
    b = generate_out_and_back(seed=11, length=length)
    np.testing.assert_array_equal(a.poses, b.poses)
    assert a.revisit_of == b.revisit_of


def test_different_seed_differs():
    a = generate_closed_circuit(seed=1, length=64)
    b = generate_closed_circuit(seed=2, length=64)
    assert not np.array_equal(a.poses, b.poses)


def _revisit_pairs(traj):
    return [(i, r) for i, r in enumerate(traj.revisit_of) if r is not None]


@pytest.mark.parametrize("length", VALID_LENGTHS)
def test_closed_circuit_produces_revisit_labels(length):
    traj = generate_closed_circuit(seed=3, length=length)
    pairs = _revisit_pairs(traj)
    assert pairs, "a closed circuit must revisit its own head near the end"


@pytest.mark.parametrize("length", VALID_LENGTHS)
def test_out_and_back_produces_multiple_revisits(length):
    traj = generate_out_and_back(seed=4, length=length)
    assert len(_revisit_pairs(traj)) > 1, "out-and-back return leg must revisit many outbound poses"


@pytest.mark.parametrize("length", VALID_LENGTHS)
@pytest.mark.parametrize("generate", [generate_closed_circuit, generate_out_and_back])
def test_revisits_lie_outside_the_model_context_window(generate, length):
    """A revisit inside the context window is not a persistence test.

    Guards the metric itself, not just the label count: if adjacent frames
    could be labeled as revisits, revisit-PSNR would measure "reproduce the
    previous frame" -- answerable straight from the attention window -- and
    would look healthy while testing nothing. Every labeled pair must be
    further apart than DEFAULT_MIN_REVISIT_GAP, which exceeds the 16-frame
    DiT context (see CLAUDE.md decision log).
    """
    traj = generate(seed=5, length=length)
    pairs = _revisit_pairs(traj)
    assert pairs, "no revisit labels at all -- loop closure is broken"
    smallest = min(i - r for i, r in pairs)
    assert smallest >= DEFAULT_MIN_REVISIT_GAP, (
        f"revisit pair only {smallest} frames apart; inside the context window "
        f"the model can copy the answer instead of remembering it"
    )


@pytest.mark.parametrize("length", VALID_LENGTHS)
def test_out_and_back_return_leg_keeps_outbound_heading(length):
    """Return-leg poses must share the outbound viewing direction.

    Facing the direction of travel would rotate each return pose 180 degrees
    from its outbound twin: same camera position, opposite view, no shared
    image content for revisit-PSNR to compare.
    """
    traj = generate_out_and_back(seed=6, length=length)
    half = length // 2
    for i in (1, half // 2, half - 2):
        src = half - 1 - i
        drot = rotation_geodesic_angle(traj.poses[src, :3, :3], traj.poses[half + i, :3, :3])
        assert np.degrees(drot) < 90.0, (
            f"return frame {half + i} faces {np.degrees(drot):.1f} deg away from "
            f"outbound frame {src} -- it is looking at a different part of the scene"
        )


def test_invalid_length_rejected():
    with pytest.raises(ValueError):
        generate_closed_circuit(seed=0, length=50)
    with pytest.raises(ValueError):
        generate_out_and_back(seed=0, length=50)


def test_pose_shape_and_dtype():
    traj = generate_closed_circuit(seed=0, length=32)
    assert traj.poses.shape == (32, 4, 4)
    assert traj.poses.dtype == np.float32


def test_pose_is_rigid_transform():
    """Bottom row is [0,0,0,1] and rotation block is orthonormal (det=+1)."""
    traj = generate_closed_circuit(seed=0, length=32)
    for pose in traj.poses:
        np.testing.assert_allclose(pose[3], [0, 0, 0, 1], atol=1e-5)
        r = pose[:3, :3].astype(np.float64)
        should_be_identity = r.T @ r
        np.testing.assert_allclose(should_be_identity, np.eye(3), atol=1e-3)
        det = np.linalg.det(r)
        assert det > 0, "rotation block must be a proper rotation (det > 0), not a reflection"


def test_opengl_camera_axis_convention_forward_is_minus_z():
    """The camera's -Z axis (column 2, negated) should point roughly toward
    the scene interior for a camera placed inside the room looking around,
    i.e. it should not be degenerate and should be a unit vector."""
    traj = generate_closed_circuit(seed=2, length=32)
    for pose in traj.poses:
        right = pose[:3, 0]
        up = pose[:3, 1]
        back = pose[:3, 2]  # +Z camera axis = -forward
        np.testing.assert_allclose(np.linalg.norm(right), 1.0, atol=1e-3)
        np.testing.assert_allclose(np.linalg.norm(up), 1.0, atol=1e-3)
        np.testing.assert_allclose(np.linalg.norm(back), 1.0, atol=1e-3)
        # OpenGL convention is right-handed: right x up = +Z (= back,
        # i.e. the negative of the forward/-Z look direction).
        np.testing.assert_allclose(np.cross(right, up), back, atol=1e-3)


def test_revisit_of_indices_point_backward_in_time():
    traj = generate_out_and_back(seed=6, length=64)
    for i, r in enumerate(traj.revisit_of):
        if r is not None:
            assert 0 <= r < i


def test_revisit_labels_respect_tolerance():
    pos_eps = 0.1
    rot_eps = np.deg2rad(5.0)
    traj = generate_closed_circuit(seed=8, length=64, pos_eps=pos_eps, rot_theta_eps=rot_eps)
    positions = traj.poses[:, :3, 3]
    for i, r in enumerate(traj.revisit_of):
        if r is None:
            continue
        dpos = np.linalg.norm(positions[i] - positions[r])
        assert dpos <= pos_eps + 1e-6
        drot = rotation_geodesic_angle(traj.poses[r, :3, :3], traj.poses[i, :3, :3])
        assert drot <= rot_eps + 1e-6


def test_intrinsics_are_positive_pinhole():
    traj = generate_closed_circuit(seed=0, length=32)
    fx, fy, cx, cy = traj.intrinsics.as_tuple()
    assert fx > 0 and fy > 0 and cx > 0 and cy > 0
