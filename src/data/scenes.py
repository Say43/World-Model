"""Procedural scene generation for nanoWM.

Generates a room shell (floor/walls/ceiling, textured via per-face solid
colors — a placeholder UV/color scheme that either rasterizer backend can
consume) plus 5-15 randomized primitives (box, sphere, cylinder). Everything
is deterministic given an integer seed: same seed -> byte-identical geometry.

Geometry is returned in a renderer-agnostic format (`Scene`) so both the
moderngl backend and the numpy/numba software rasterizer in
`src/data/render.py` can consume it without caring how it was built. No
resolution, autoencoder, or rendering-backend choice is encoded here —
those are open decisions (see CLAUDE.md decision log) and this module must
not anticipate them.

Coordinate convention: world space is right-handed, Z-up (matches
`src/data/trajectories.py`). Units are arbitrary "meters".
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional, Tuple

import numpy as np

# Primitive kind tags, kept as plain ints so the geometry format has no
# enum/dataclass dependency a renderer would need to import.
PRIM_BOX = 0
PRIM_SPHERE = 1
PRIM_CYLINDER = 2

MIN_PRIMITIVES = 5
MAX_PRIMITIVES = 15


@dataclasses.dataclass
class Mesh:
    """A single triangle mesh, renderer-agnostic.

    vertices: (N, 3) float32 world-space positions.
    normals:  (N, 3) float32 unit normals, per-vertex.
    colors:   (N, 3) float32 RGB in [0, 1], per-vertex (flat-shaded via
              duplicated vertices per face, so no UV/texture sampling is
              required by a consuming rasterizer — this keeps the numpy
              fallback trivial while still giving moderngl vertex colors
              it can optionally light).
    indices:  (M, 3) int32 triangle indices into `vertices`.
    """

    vertices: np.ndarray
    normals: np.ndarray
    colors: np.ndarray
    indices: np.ndarray

    def __post_init__(self) -> None:
        assert self.vertices.ndim == 2 and self.vertices.shape[1] == 3
        assert self.normals.shape == self.vertices.shape
        assert self.colors.shape == self.vertices.shape
        assert self.indices.ndim == 2 and self.indices.shape[1] == 3
        self.vertices = self.vertices.astype(np.float32, copy=False)
        self.normals = self.normals.astype(np.float32, copy=False)
        self.colors = self.colors.astype(np.float32, copy=False)
        self.indices = self.indices.astype(np.int32, copy=False)

    @property
    def num_triangles(self) -> int:
        return int(self.indices.shape[0])


@dataclasses.dataclass
class Scene:
    """A full scene: room shell + primitives, mergeable into one draw call."""

    seed: int
    room_extent: Tuple[float, float, float]  # (size_x, size_y, height_z)
    meshes: List[Mesh]

    def merged(self) -> Mesh:
        """Concatenate all meshes into a single Mesh (one index space)."""
        if not self.meshes:
            return Mesh(
                vertices=np.zeros((0, 3), np.float32),
                normals=np.zeros((0, 3), np.float32),
                colors=np.zeros((0, 3), np.float32),
                indices=np.zeros((0, 3), np.int32),
            )
        verts, norms, cols, idxs = [], [], [], []
        offset = 0
        for m in self.meshes:
            verts.append(m.vertices)
            norms.append(m.normals)
            cols.append(m.colors)
            idxs.append(m.indices + offset)
            offset += m.vertices.shape[0]
        return Mesh(
            vertices=np.concatenate(verts, axis=0),
            normals=np.concatenate(norms, axis=0),
            colors=np.concatenate(cols, axis=0),
            indices=np.concatenate(idxs, axis=0),
        )


def _face(v0, v1, v2, v3, color, flip=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build one quad (as 2 triangles) with a flat per-face normal/color.

    Vertices given in CCW winding as seen from the front face; `flip`
    reverses winding (for interior-facing room walls).
    """
    verts = np.array([v0, v1, v2, v3], dtype=np.float32)
    if flip:
        verts = verts[[0, 3, 2, 1]]
    edge1 = verts[1] - verts[0]
    edge2 = verts[2] - verts[0]
    normal = np.cross(edge1, edge2)
    norm_len = np.linalg.norm(normal)
    normal = normal / norm_len if norm_len > 1e-12 else np.array([0.0, 0.0, 1.0], np.float32)
    normals = np.tile(normal, (4, 1)).astype(np.float32)
    colors = np.tile(np.asarray(color, dtype=np.float32), (4, 1))
    indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return verts, normals, colors, indices


def _quads_to_mesh(quads) -> Mesh:
    verts, norms, cols, idxs = [], [], [], []
    offset = 0
    for v, n, c, i in quads:
        verts.append(v)
        norms.append(n)
        cols.append(c)
        idxs.append(i + offset)
        offset += v.shape[0]
    return Mesh(
        vertices=np.concatenate(verts, axis=0),
        normals=np.concatenate(norms, axis=0),
        colors=np.concatenate(cols, axis=0),
        indices=np.concatenate(idxs, axis=0),
    )


def make_room_shell(
    size_x: float,
    size_y: float,
    height_z: float,
    floor_color=(0.55, 0.52, 0.48),
    wall_color=(0.75, 0.73, 0.70),
    ceiling_color=(0.9, 0.9, 0.9),
) -> Mesh:
    """Axis-aligned room shell centered at the origin in X/Y, floor at z=0.

    Faces point inward (camera lives inside the room).
    """
    hx, hy = size_x / 2.0, size_y / 2.0
    quads = []
    # Floor (z=0), normal should point +Z (up, into room).
    quads.append(_face((-hx, -hy, 0), (hx, -hy, 0), (hx, hy, 0), (-hx, hy, 0), floor_color, flip=False))
    # Ceiling (z=height), normal should point -Z (down, into room).
    quads.append(_face((-hx, -hy, height_z), (hx, -hy, height_z), (hx, hy, height_z), (-hx, hy, height_z), ceiling_color, flip=True))
    # Wall -Y (normal +Y into room)
    quads.append(_face((-hx, -hy, 0), (hx, -hy, 0), (hx, -hy, height_z), (-hx, -hy, height_z), wall_color, flip=True))
    # Wall +Y (normal -Y into room)
    quads.append(_face((-hx, hy, 0), (hx, hy, 0), (hx, hy, height_z), (-hx, hy, height_z), wall_color, flip=False))
    # Wall -X (normal +X into room)
    quads.append(_face((-hx, -hy, 0), (-hx, hy, 0), (-hx, hy, height_z), (-hx, -hy, height_z), wall_color, flip=False))
    # Wall +X (normal -X into room)
    quads.append(_face((hx, -hy, 0), (hx, hy, 0), (hx, hy, height_z), (hx, -hy, height_z), wall_color, flip=True))
    return _quads_to_mesh(quads)


def make_box(center, half_extents, color) -> Mesh:
    cx, cy, cz = center
    hx, hy, hz = half_extents
    quads = [
        _face((cx - hx, cy - hy, cz - hz), (cx + hx, cy - hy, cz - hz), (cx + hx, cy + hy, cz - hz), (cx - hx, cy + hy, cz - hz), color, flip=True),
        _face((cx - hx, cy - hy, cz + hz), (cx + hx, cy - hy, cz + hz), (cx + hx, cy + hy, cz + hz), (cx - hx, cy + hy, cz + hz), color, flip=False),
        _face((cx - hx, cy - hy, cz - hz), (cx + hx, cy - hy, cz - hz), (cx + hx, cy - hy, cz + hz), (cx - hx, cy - hy, cz + hz), color, flip=False),
        _face((cx - hx, cy + hy, cz - hz), (cx + hx, cy + hy, cz - hz), (cx + hx, cy + hy, cz + hz), (cx - hx, cy + hy, cz + hz), color, flip=True),
        _face((cx - hx, cy - hy, cz - hz), (cx - hx, cy + hy, cz - hz), (cx - hx, cy + hy, cz + hz), (cx - hx, cy - hy, cz + hz), color, flip=True),
        _face((cx + hx, cy - hy, cz - hz), (cx + hx, cy + hy, cz - hz), (cx + hx, cy + hy, cz + hz), (cx + hx, cy - hy, cz + hz), color, flip=False),
    ]
    return _quads_to_mesh(quads)


def make_sphere(center, radius, color, u_segments=10, v_segments=8) -> Mesh:
    """UV sphere, per-vertex color and outward normal (no shared vertices,
    so it composes with the flat-shaded convention used elsewhere)."""
    cx, cy, cz = center
    verts, norms, cols, idxs = [], [], [], []
    for vi in range(v_segments):
        theta0 = np.pi * vi / v_segments
        theta1 = np.pi * (vi + 1) / v_segments
        for ui in range(u_segments):
            phi0 = 2 * np.pi * ui / u_segments
            phi1 = 2 * np.pi * (ui + 1) / u_segments

            def pt(theta, phi):
                x = np.sin(theta) * np.cos(phi)
                y = np.sin(theta) * np.sin(phi)
                z = np.cos(theta)
                return np.array([x, y, z], dtype=np.float32)

            p00, p01 = pt(theta0, phi0), pt(theta0, phi1)
            p10, p11 = pt(theta1, phi0), pt(theta1, phi1)
            quad_normals = [p00, p01, p11, p10]
            quad_pts = [center + radius * n for n in quad_normals]
            base = len(verts) * 1  # placeholder, recomputed below
            offset = sum(v.shape[0] for v in verts)
            verts.append(np.stack(quad_pts).astype(np.float32))
            norms.append(np.stack(quad_normals).astype(np.float32))
            cols.append(np.tile(np.asarray(color, np.float32), (4, 1)))
            idxs.append(np.array([[0, 1, 2], [0, 2, 3]], np.int32) + offset)
    return Mesh(
        vertices=np.concatenate(verts, axis=0),
        normals=np.concatenate(norms, axis=0),
        colors=np.concatenate(cols, axis=0),
        indices=np.concatenate(idxs, axis=0),
    )


def make_cylinder(center, radius, half_height, color, segments=12) -> Mesh:
    """Capped cylinder, axis aligned with world Z, per-face flat shading."""
    cx, cy, cz = center
    quads = []
    for i in range(segments):
        a0 = 2 * np.pi * i / segments
        a1 = 2 * np.pi * (i + 1) / segments
        x0, y0 = radius * np.cos(a0), radius * np.sin(a0)
        x1, y1 = radius * np.cos(a1), radius * np.sin(a1)
        bot = (cx, cy, cz - half_height)
        top = (cx, cy, cz + half_height)
        p0b = (cx + x0, cy + y0, cz - half_height)
        p1b = (cx + x1, cy + y1, cz - half_height)
        p0t = (cx + x0, cy + y0, cz + half_height)
        p1t = (cx + x1, cy + y1, cz + half_height)
        # side wall
        quads.append(_face(p0b, p1b, p1t, p0t, color, flip=False))
        # bottom cap triangle (as degenerate quad through center)
        quads.append(_face(bot, p1b, p0b, bot, color, flip=True))
        # top cap
        quads.append(_face(top, p0t, p1t, top, color, flip=False))
    return _quads_to_mesh(quads)


def _random_color(rng: np.random.Generator) -> Tuple[float, float, float]:
    return tuple(float(x) for x in rng.uniform(0.15, 0.95, size=3))


def generate_scene(
    seed: int,
    room_size_range: Tuple[float, float] = (4.0, 9.0),
    room_height_range: Tuple[float, float] = (2.5, 4.0),
    min_primitives: int = MIN_PRIMITIVES,
    max_primitives: int = MAX_PRIMITIVES,
) -> Scene:
    """Deterministically generate one scene from an integer seed.

    Same `seed` (and same *_range/min/max arguments) always yields
    bit-identical geometry: a single `np.random.Generator(PCG64(seed))` is
    used for every random decision, in a fixed order.
    """
    rng = np.random.default_rng(seed)

    size_x = float(rng.uniform(*room_size_range))
    size_y = float(rng.uniform(*room_size_range))
    height_z = float(rng.uniform(*room_height_range))

    meshes = [make_room_shell(size_x, size_y, height_z)]

    n_prims = int(rng.integers(min_primitives, max_primitives + 1))
    margin = 0.5  # keep primitives off the walls
    for _ in range(n_prims):
        kind = int(rng.integers(0, 3))
        color = _random_color(rng)
        cx = float(rng.uniform(-size_x / 2 + margin, size_x / 2 - margin))
        cy = float(rng.uniform(-size_y / 2 + margin, size_y / 2 - margin))

        if kind == PRIM_BOX:
            hx = float(rng.uniform(0.15, 0.6))
            hy = float(rng.uniform(0.15, 0.6))
            hz = float(rng.uniform(0.15, 0.8))
            cz = hz  # resting on the floor
            meshes.append(make_box((cx, cy, cz), (hx, hy, hz), color))
        elif kind == PRIM_SPHERE:
            radius = float(rng.uniform(0.15, 0.5))
            cz = radius
            meshes.append(make_sphere((cx, cy, cz), radius, color))
        else:
            radius = float(rng.uniform(0.1, 0.4))
            half_h = float(rng.uniform(0.2, 0.7))
            cz = half_h
            meshes.append(make_cylinder((cx, cy, cz), radius, half_h, color))

    return Scene(seed=seed, room_extent=(size_x, size_y, height_z), meshes=meshes)
