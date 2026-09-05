"""Tests for scripts/preprocess/precompute_dataset.py's non-AE parts.

Encoding through the real AE needs torch/diffusers + network and is not
exercised here (mirrors tests/test_eval_ae_ceiling.py's approach of testing
the surrounding logic with a fake reconstructor).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "preprocess"))

from precompute_dataset import build_trajectory, render_trajectory_frames  # noqa: E402
from src.data.render import Renderer  # noqa: E402
from src.data.scenes import generate_scene  # noqa: E402


def test_build_trajectory_deterministic():
    a = build_trajectory(scene_seed=0, traj_idx=1, length=32)
    b = build_trajectory(scene_seed=0, traj_idx=1, length=32)
    np.testing.assert_array_equal(a.poses, b.poses)
    assert a.revisit_of == b.revisit_of


def test_build_trajectory_alternates_kind():
    even = build_trajectory(scene_seed=0, traj_idx=0, length=32)
    odd = build_trajectory(scene_seed=0, traj_idx=1, length=32)
    assert even.kind == "closed_circuit"
    assert odd.kind == "out_and_back"


def test_render_trajectory_frames_shape():
    scene = generate_scene(seed=0)
    mesh = scene.merged()
    traj = build_trajectory(scene_seed=0, traj_idx=0, length=32)
    renderer = Renderer()
    frames = render_trajectory_frames(renderer, mesh, traj, resolution=64)
    renderer.release()
    assert frames.shape == (32, 64, 64, 3)
    assert frames.dtype == np.float32
    assert 0.0 <= frames.min() and frames.max() <= 1.0


def test_trajectory_arrays_omits_dinov2_key_when_not_extracted():
    from precompute_dataset import trajectory_arrays

    traj = build_trajectory(scene_seed=0, traj_idx=0, length=32)
    arrays = trajectory_arrays(traj, latents=np.zeros((32, 16, 128), dtype=np.float32))
    assert set(arrays) == {"latents", "poses", "intrinsics", "revisit_of"}


def test_trajectory_arrays_stores_dinov2_as_float32():
    from precompute_dataset import trajectory_arrays

    traj = build_trajectory(scene_seed=0, traj_idx=0, length=32)
    features = np.zeros((32, 16, 384), dtype=np.float64)
    arrays = trajectory_arrays(traj, np.zeros((32, 16, 128), dtype=np.float32), features)
    assert arrays["dinov2"].shape == (32, 16, 384)
    assert arrays["dinov2"].dtype == np.float32


def test_trajectory_arrays_rejects_frame_count_mismatch():
    """A silent mismatch here would be stored and only surface as a shape
    error deep inside a paid training run."""
    import pytest

    from precompute_dataset import trajectory_arrays

    traj = build_trajectory(scene_seed=0, traj_idx=0, length=32)
    with pytest.raises(ValueError, match="frames"):
        trajectory_arrays(traj, np.zeros((32, 16, 128), dtype=np.float32),
                          np.zeros((31, 16, 384), dtype=np.float32))


def test_revisit_of_encodes_none_as_minus_one():
    from precompute_dataset import trajectory_arrays

    traj = build_trajectory(scene_seed=0, traj_idx=1, length=64)
    arrays = trajectory_arrays(traj, np.zeros((64, 16, 128), dtype=np.float32))
    assert arrays["revisit_of"].dtype == np.int32
    assert set(np.unique(arrays["revisit_of"])) <= set(range(-1, 64))
