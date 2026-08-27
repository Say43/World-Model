"""Tests for src/data/render.py: backend fallback and basic raster sanity.

These run in an environment without moderngl/numba installed, so they
exercise (and pin down) the automatic-fallback-to-numpy path. That is
itself the scenario the M0 decision log calls out as needing verification
before relying on moderngl on Kaggle.
"""
import numpy as np

from src.data.render import BACKEND_NUMPY, Renderer, project_to_screen
from src.data.scenes import generate_scene
from src.data.trajectories import generate_closed_circuit


def test_renderer_falls_back_to_numpy_without_moderngl():
    renderer = Renderer()
    # moderngl is not installed in the test environment, so this pins the
    # automatic-fallback behavior explicitly.
    assert renderer.backend == BACKEND_NUMPY


def test_render_produces_correct_shape_and_range():
    scene = generate_scene(seed=1)
    mesh = scene.merged()
    traj = generate_closed_circuit(seed=1, length=32, room_extent=scene.room_extent)
    renderer = Renderer()

    width, height = 48, 32
    frame = renderer.render(mesh, traj.poses[0], traj.intrinsics.as_tuple(), width, height)

    assert frame.shape == (height, width, 3)
    assert frame.dtype == np.float32
    assert np.isfinite(frame).all()
    assert frame.min() >= 0.0 - 1e-5
    assert frame.max() <= 1.0 + 1e-5


def test_render_is_deterministic():
    scene = generate_scene(seed=2)
    mesh = scene.merged()
    traj = generate_closed_circuit(seed=2, length=32, room_extent=scene.room_extent)
    renderer = Renderer()

    frame_a = renderer.render(mesh, traj.poses[5], traj.intrinsics.as_tuple(), 40, 30)
    frame_b = renderer.render(mesh, traj.poses[5], traj.intrinsics.as_tuple(), 40, 30)
    np.testing.assert_array_equal(frame_a, frame_b)


def test_render_shows_more_than_background_from_inside_room():
    """A camera placed inside a generated room should see walls/floor, not
    just background — a basic smoke test that projection + rasterization
    aren't degenerate."""
    scene = generate_scene(seed=4)
    mesh = scene.merged()
    traj = generate_closed_circuit(seed=4, length=32, room_extent=scene.room_extent)
    renderer = Renderer()

    frame = renderer.render(mesh, traj.poses[0], traj.intrinsics.as_tuple(), 64, 64)
    background = np.array([0.05, 0.05, 0.05], dtype=np.float32)
    differs = np.any(np.abs(frame - background) > 1e-3, axis=-1)
    assert differs.mean() > 0.3


def test_project_to_screen_shapes():
    scene = generate_scene(seed=0)
    mesh = scene.merged()
    traj = generate_closed_circuit(seed=0, length=32, room_extent=scene.room_extent)
    xy, depth = project_to_screen(mesh.vertices, traj.poses[0], traj.intrinsics.as_tuple(), 64, 64)
    assert xy.shape == (mesh.vertices.shape[0], 2)
    assert depth.shape == (mesh.vertices.shape[0],)
