"""Tests for src/data/scenes.py: determinism and structural sanity."""
import numpy as np
import pytest

from src.data.scenes import (
    MAX_PRIMITIVES,
    MIN_PRIMITIVES,
    generate_scene,
)


def test_determinism_same_seed_identical_scene():
    a = generate_scene(seed=42)
    b = generate_scene(seed=42)

    assert a.room_extent == b.room_extent
    assert len(a.meshes) == len(b.meshes)

    ma, mb = a.merged(), b.merged()
    np.testing.assert_array_equal(ma.vertices, mb.vertices)
    np.testing.assert_array_equal(ma.normals, mb.normals)
    np.testing.assert_array_equal(ma.colors, mb.colors)
    np.testing.assert_array_equal(ma.indices, mb.indices)


def test_different_seeds_differ():
    a = generate_scene(seed=1)
    b = generate_scene(seed=2)
    ma, mb = a.merged(), b.merged()
    # Extremely unlikely to collide across independent seeds.
    same_shape = ma.vertices.shape == mb.vertices.shape
    if same_shape:
        assert not np.array_equal(ma.vertices, mb.vertices)


@pytest.mark.parametrize("seed", [0, 1, 7, 123, 999])
def test_primitive_count_in_range(seed):
    scene = generate_scene(seed=seed)
    n_primitives = len(scene.meshes) - 1  # first mesh is the room shell
    assert MIN_PRIMITIVES <= n_primitives <= MAX_PRIMITIVES


def test_room_shell_is_first_mesh_and_closed():
    scene = generate_scene(seed=5)
    room = scene.meshes[0]
    # 6 quads * 2 triangles each = 12 triangles, 6*4=24 vertices (unshared,
    # flat-shaded convention).
    assert room.num_triangles == 12
    assert room.vertices.shape[0] == 24


def test_merged_mesh_index_bounds_valid():
    scene = generate_scene(seed=17)
    merged = scene.merged()
    assert merged.indices.max() < merged.vertices.shape[0]
    assert merged.indices.min() >= 0


def test_geometry_format_is_renderer_agnostic_arrays():
    scene = generate_scene(seed=3)
    merged = scene.merged()
    assert merged.vertices.dtype == np.float32
    assert merged.normals.dtype == np.float32
    assert merged.colors.dtype == np.float32
    assert merged.indices.dtype == np.int32
    assert merged.vertices.shape[1] == 3
    assert merged.colors.shape[1] == 3
    assert merged.indices.shape[1] == 3


def test_primitives_rest_on_or_above_floor():
    scene = generate_scene(seed=9)
    for mesh in scene.meshes[1:]:
        assert mesh.vertices[:, 2].min() >= -1e-4
