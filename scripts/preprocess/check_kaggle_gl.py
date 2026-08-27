#!/usr/bin/env python3
"""Standalone Kaggle EGL/headless-GL probe for nanoWM.

This is the "throwaway notebook check" approved in the CLAUDE.md decision
log ("Renderer: verify moderngl EGL context on Kaggle first") — it costs
Kaggle weekly quota, not training budget, and its whole point is to catch a
dead-end GL context failure *before* M0 rather than during it.

Usage (on Kaggle, or any target machine):
    python scripts/preprocess/check_kaggle_gl.py [--width 64] [--height 64]

Prints a clear PASS/FAIL line plus diagnosis:
  - PASS: moderngl standalone context created, a triangle was rendered and
    read back, pixel values are sane (not all-zero, not NaN).
  - FAIL: import error, context-creation error, or a render/readback that
    produced garbage — with the specific exception/reason printed so it is
    actionable without re-running.

Exit code 0 on PASS, 1 on FAIL. This script does not depend on the rest of
`src/data/` on purpose: it must run even before those modules are trusted,
as the smallest possible reproduction of the moderngl-on-Kaggle question.
"""
import argparse
import sys
import traceback


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--backend", default=None, help="Force a moderngl backend string (e.g. 'egl'). Default: let moderngl choose.")
    args = parser.parse_args()

    print("nanoWM Kaggle GL probe")
    print("=" * 40)

    try:
        import moderngl
    except ImportError as exc:
        print(f"FAIL: could not import moderngl ({exc}).")
        print("Diagnosis: package not installed in this environment. Run `pip install moderngl` "
              "in the Kaggle notebook, or accept the numpy fallback renderer for this run.")
        return 1

    print(f"moderngl version: {getattr(moderngl, '__version__', 'unknown')}")

    try:
        if args.backend:
            ctx = moderngl.create_context(standalone=True, backend=args.backend)
        else:
            ctx = moderngl.create_context(standalone=True)
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: moderngl.create_context(standalone=True) raised: {exc!r}")
        print("Diagnosis: no usable headless GL context (commonly: no EGL device visible to the")
        print("container, missing libGL/libEGL, or a driver mismatch). This confirms the numpy")
        print("fallback rasterizer in src/data/render.py must be the primary backend for this run.")
        traceback.print_exc()
        return 1

    print(f"GL context created. vendor={ctx.info.get('GL_VENDOR')} renderer={ctx.info.get('GL_RENDERER')} version={ctx.info.get('GL_VERSION')}")

    try:
        prog = ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_pos;
                void main() { gl_Position = vec4(in_pos, 0.0, 1.0); }
            """,
            fragment_shader="""
                #version 330
                out vec4 f_color;
                void main() { f_color = vec4(1.0, 0.5, 0.25, 1.0); }
            """,
        )
        vertices = [
            -0.8, -0.8,
            0.8, -0.8,
            0.0, 0.8,
        ]
        import struct

        vbo = ctx.buffer(struct.pack(f"{len(vertices)}f", *vertices))
        vao = ctx.vertex_array(prog, [(vbo, "2f", "in_pos")])

        color_rbo = ctx.renderbuffer((args.width, args.height))
        fbo = ctx.framebuffer(color_attachments=[color_rbo])
        fbo.use()
        fbo.clear(0.0, 0.0, 0.0, 1.0)
        vao.render()

        raw = fbo.read(components=3, dtype="f1")
        import numpy as np

        frame = np.frombuffer(raw, dtype=np.uint8).reshape(args.height, args.width, 3)
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: context created but render/readback raised: {exc!r}")
        print("Diagnosis: context creation succeeded but the render pipeline is broken (shader "
              "compile failure, framebuffer incomplete, or readback mismatch). Treat as a GL FAIL "
              "for planning purposes; fall back to numpy rasterizer.")
        traceback.print_exc()
        return 1

    nonzero_fraction = float((frame.sum(axis=-1) > 0).mean())
    if nonzero_fraction < 0.01:
        print(f"FAIL: rendered frame is essentially blank (nonzero pixel fraction={nonzero_fraction:.4f}).")
        print("Diagnosis: context and pipeline ran without exceptions but produced no visible")
        print("triangle — likely a coordinate/viewport mismatch, not a Kaggle-environment issue, ")
        print("but still means moderngl is not verified working end-to-end here.")
        return 1

    print(f"Rendered triangle nonzero-pixel fraction: {nonzero_fraction:.4f}")
    print("PASS: moderngl standalone context works end-to-end on this machine.")
    print("=> moderngl may be used as the primary renderer per the M0 decision log; "
          "numpy fallback remains available in src/data/render.py regardless.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
