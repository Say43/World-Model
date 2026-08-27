"""Renderer abstraction for nanoWM: moderngl primary, numpy/numba fallback.

Backend selection is automatic: an attempt is made to create a standalone
moderngl (EGL/headless) GL context; on any failure (missing package, no
EGL device, driver issue) the code falls back to a pure-numpy software
rasterizer (optionally numba-jitted if numba is importable). Which backend
ended up active is always logged explicitly via the `logging` module, per
the M0 decision log requirement to verify moderngl on Kaggle before relying
on it (see scripts/preprocess/check_kaggle_gl.py for the standalone probe).

Both backends consume the exact same renderer-agnostic `Mesh`/`Scene`
format from `src/data/scenes.py` and the same camera-to-world pose /
pinhole-intrinsics convention from `src/data/trajectories.py`. Neither
backend hardcodes output resolution — `width`/`height` are call
parameters, since tokens/frame (and therefore the resolution the model
will actually train at) is open until M0.

`moderngl` and `numba` are optional imports: this module, and everything
importing only the numpy path, must work in an environment with neither
installed (tests run this way).
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from src.data.scenes import Mesh

logger = logging.getLogger("nanowm.data.render")

BACKEND_MODERNGL = "moderngl"
BACKEND_NUMPY = "numpy"

try:
    import moderngl  # type: ignore

    _HAS_MODERNGL = True
except ImportError:
    moderngl = None  # type: ignore
    _HAS_MODERNGL = False

try:
    import numba  # type: ignore

    _HAS_NUMBA = True
except ImportError:
    numba = None  # type: ignore
    _HAS_NUMBA = False


AMBIENT = 0.35
DIFFUSE = 0.65
LIGHT_DIR = np.array([0.4, 0.5, 0.9])
LIGHT_DIR = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)


def shade_vertices(mesh: Mesh) -> np.ndarray:
    """Simple fixed-directional-light Lambertian shading, computed once on
    the CPU so both backends produce visually consistent output without
    needing lighting logic duplicated in GLSL.

    Returns (N, 3) float32 colors in [0, 1].
    """
    ndotl = np.clip(mesh.normals @ LIGHT_DIR, 0.0, 1.0).astype(np.float32)
    shade = AMBIENT + DIFFUSE * ndotl
    return np.clip(mesh.colors * shade[:, None], 0.0, 1.0).astype(np.float32)


def _invert_c2w(c2w: np.ndarray) -> np.ndarray:
    r = c2w[:3, :3]
    t = c2w[:3, 3]
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = r.T
    w2c[:3, 3] = -r.T @ t
    return w2c


def project_to_screen(
    vertices_world: np.ndarray,
    pose_c2w: np.ndarray,
    intrinsics: Tuple[float, float, float, float],
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project world-space vertices into screen pixel coordinates.

    OpenGL camera convention: the camera looks down its own -Z axis, so a
    point in front of the camera has negative camera-space Z. `depth`
    returned is positive-in-front (`-z_cam`).

    Returns (screen_xy (N,2) float64, depth (N,) float64). Points behind
    the camera have depth <= 0 and must be discarded by the caller.
    """
    fx, fy, cx, cy = intrinsics
    w2c = _invert_c2w(pose_c2w.astype(np.float64))
    n = vertices_world.shape[0]
    homo = np.concatenate([vertices_world.astype(np.float64), np.ones((n, 1))], axis=1)
    cam = (w2c @ homo.T).T  # (N, 4)
    x_cam, y_cam, z_cam = cam[:, 0], cam[:, 1], cam[:, 2]
    depth = -z_cam
    safe_depth = np.where(np.abs(depth) < 1e-9, 1e-9, depth)
    u = fx * (x_cam / safe_depth) + cx
    v = cy - fy * (y_cam / safe_depth)
    return np.stack([u, v], axis=1), depth


def _rasterize_numpy(
    mesh: Mesh,
    pose_c2w: np.ndarray,
    intrinsics: Tuple[float, float, float, float],
    width: int,
    height: int,
    background: Tuple[float, float, float] = (0.05, 0.05, 0.05),
) -> Tuple[np.ndarray, np.ndarray]:
    """Pure-numpy triangle rasterizer with a z-buffer.

    Loops over triangles (typically a few hundred for these scenes) and
    vectorizes the per-pixel work within each triangle's screen-space
    bounding box. If numba is available its presence is logged, but the
    reference implementation here stays pure numpy for correctness and to
    guarantee it runs with numba absent (as in the test environment).
    """
    screen_xy, depth = project_to_screen(mesh.vertices, pose_c2w, intrinsics, width, height)
    shaded = shade_vertices(mesh)

    color_buf = np.tile(np.asarray(background, dtype=np.float32), (height, width, 1))
    depth_buf = np.full((height, width), np.inf, dtype=np.float64)

    for tri in mesh.indices:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        d0, d1, d2 = depth[i0], depth[i1], depth[i2]
        if d0 <= 1e-6 or d1 <= 1e-6 or d2 <= 1e-6:
            continue  # skip triangles with any vertex behind/at the camera
        p0, p1, p2 = screen_xy[i0], screen_xy[i1], screen_xy[i2]

        min_x = max(int(np.floor(min(p0[0], p1[0], p2[0]))), 0)
        max_x = min(int(np.ceil(max(p0[0], p1[0], p2[0]))), width - 1)
        min_y = max(int(np.floor(min(p0[1], p1[1], p2[1]))), 0)
        max_y = min(int(np.ceil(max(p0[1], p1[1], p2[1]))), height - 1)
        if min_x > max_x or min_y > max_y:
            continue

        area = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
        if abs(area) < 1e-9:
            continue

        xs, ys = np.meshgrid(
            np.arange(min_x, max_x + 1) + 0.5, np.arange(min_y, max_y + 1) + 0.5
        )
        w0 = (p1[0] - xs) * (p2[1] - ys) - (p1[1] - ys) * (p2[0] - xs)
        w1 = (p2[0] - xs) * (p0[1] - ys) - (p2[1] - ys) * (p0[0] - xs)
        w2 = (p0[0] - xs) * (p1[1] - ys) - (p0[1] - ys) * (p1[0] - xs)

        if area < 0:
            inside = (w0 <= 0) & (w1 <= 0) & (w2 <= 0)
        else:
            inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not np.any(inside):
            continue

        b0, b1, b2 = w0 / area, w1 / area, w2 / area
        pixel_depth = b0 * d0 + b1 * d1 + b2 * d2

        sub_y, sub_x = np.nonzero(inside)
        gy = sub_y + min_y
        gx = sub_x + min_x
        pd = pixel_depth[sub_y, sub_x]

        closer = pd < depth_buf[gy, gx]
        if not np.any(closer):
            continue
        gy, gx, pd = gy[closer], gx[closer], pd[closer]
        bb0 = b0[sub_y, sub_x][closer]
        bb1 = b1[sub_y, sub_x][closer]
        bb2 = b2[sub_y, sub_x][closer]

        pixel_color = (
            bb0[:, None] * shaded[i0]
            + bb1[:, None] * shaded[i1]
            + bb2[:, None] * shaded[i2]
        )
        depth_buf[gy, gx] = pd
        color_buf[gy, gx] = pixel_color

    return color_buf, depth_buf


_MODERNGL_VERTEX_SHADER = """
#version 330
uniform mat4 mvp;
in vec3 in_position;
in vec3 in_color;
out vec3 v_color;
void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    v_color = in_color;
}
"""

_MODERNGL_FRAGMENT_SHADER = """
#version 330
in vec3 v_color;
out vec4 f_color;
void main() {
    f_color = vec4(v_color, 1.0);
}
"""


def _perspective_matrix(fx, fy, cx, cy, width, height, near=0.05, far=100.0) -> np.ndarray:
    """OpenGL-style perspective projection matrix from pinhole intrinsics."""
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = 2.0 * fx / width
    m[1, 1] = 2.0 * fy / height
    m[0, 2] = 1.0 - 2.0 * cx / width
    m[1, 2] = 2.0 * cy / height - 1.0
    m[2, 2] = -(far + near) / (far - near)
    m[2, 3] = -2.0 * far * near / (far - near)
    m[3, 2] = -1.0
    return m


def try_create_moderngl_context() -> Optional["moderngl.Context"]:
    """Attempt to create a standalone (headless/EGL) moderngl context.

    Returns None on any failure, logging the reason. Never raises.
    """
    if not _HAS_MODERNGL:
        logger.info("moderngl not installed; will use numpy fallback rasterizer.")
        return None
    try:
        ctx = moderngl.create_context(standalone=True)
        return ctx
    except Exception as exc:  # noqa: BLE001 - any GL/driver failure must fall back
        logger.warning("moderngl standalone context creation failed (%s); falling back to numpy rasterizer.", exc)
        return None


class Renderer:
    """Renders a `Mesh` from a camera-to-world pose + pinhole intrinsics.

    Automatically selects moderngl if a standalone GL context can be
    created, else falls back to the numpy software rasterizer. The active
    backend is exposed as `self.backend` and always logged at INFO level.
    """

    def __init__(self, prefer_backend: str = BACKEND_MODERNGL):
        self._ctx = None
        if prefer_backend == BACKEND_MODERNGL:
            self._ctx = try_create_moderngl_context()
        if self._ctx is not None:
            self.backend = BACKEND_MODERNGL
        else:
            self.backend = BACKEND_NUMPY
            if _HAS_NUMBA:
                logger.info("numba available; software rasterizer may use numba-jitted kernels.")
            else:
                logger.info("numba not installed; software rasterizer runs in pure numpy/Python.")
        logger.info("Renderer active backend: %s", self.backend)

    def release(self) -> None:
        if self._ctx is not None:
            self._ctx.release()
            self._ctx = None

    def render(
        self,
        mesh: Mesh,
        pose_c2w: np.ndarray,
        intrinsics: Tuple[float, float, float, float],
        width: int,
        height: int,
    ) -> np.ndarray:
        """Render one frame. Returns (H, W, 3) float32 RGB in [0, 1]."""
        if self.backend == BACKEND_MODERNGL:
            return self._render_moderngl(mesh, pose_c2w, intrinsics, width, height)
        color_buf, _depth = _rasterize_numpy(mesh, pose_c2w, intrinsics, width, height)
        return color_buf

    def _render_moderngl(self, mesh, pose_c2w, intrinsics, width, height) -> np.ndarray:
        assert self._ctx is not None
        ctx = self._ctx
        fx, fy, cx, cy = intrinsics
        proj = _perspective_matrix(fx, fy, cx, cy, width, height)
        w2c = _invert_c2w(pose_c2w.astype(np.float64))
        mvp = (proj @ w2c).astype("f4")

        shaded = shade_vertices(mesh)
        vbo_data = np.concatenate([mesh.vertices, shaded], axis=1).astype("f4")
        vbo = ctx.buffer(vbo_data.tobytes())
        ibo = ctx.buffer(mesh.indices.astype("i4").tobytes())

        prog = ctx.program(
            vertex_shader=_MODERNGL_VERTEX_SHADER, fragment_shader=_MODERNGL_FRAGMENT_SHADER
        )
        prog["mvp"].write(mvp.T.tobytes())  # column-major for GLSL

        vao = ctx.vertex_array(
            prog, [(vbo, "3f 3f", "in_position", "in_color")], ibo
        )
        color_rbo = ctx.renderbuffer((width, height))
        depth_rbo = ctx.depth_renderbuffer((width, height))
        fbo = ctx.framebuffer(color_attachments=[color_rbo], depth_attachment=depth_rbo)
        fbo.use()
        ctx.enable(ctx.DEPTH_TEST)
        fbo.clear(0.05, 0.05, 0.05, 1.0)
        vao.render()

        raw = fbo.read(components=3, dtype="f4")
        frame = np.frombuffer(raw, dtype=np.float32).reshape(height, width, 3)
        frame = np.flipud(frame)  # GL reads bottom-up

        vao.release()
        ibo.release()
        vbo.release()
        prog.release()
        fbo.release()
        color_rbo.release()
        depth_rbo.release()
        return frame
