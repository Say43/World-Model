"""Fixed test-frame set for M0's AE comparison.

Every candidate must be measured on the *same* frames at each resolution,
otherwise a PSNR difference could come from scene sampling rather than the
AE. Deterministic from a fixed seed list, drawn across multiple scenes and
both loop trajectory kinds so the set is not biased toward one geometry.
"""
from __future__ import annotations

from typing import List

import numpy as np

from src.data.render import Renderer
from src.data.scenes import generate_scene
from src.data.trajectories import generate_closed_circuit, generate_out_and_back, make_intrinsics

SCENE_SEEDS = [0, 1, 2, 3]
FRAMES_PER_SCENE = 50  # 4 scenes x 50 = 200 frames total, per CLAUDE.md's M0 plan


def render_test_frames(resolution: int, scene_seeds: List[int] = SCENE_SEEDS,
                        frames_per_scene: int = FRAMES_PER_SCENE) -> np.ndarray:
    """Returns (N, resolution, resolution, 3) float32 RGB in [0, 1]."""
    renderer = Renderer()
    all_frames = []
    intr = make_intrinsics(width=resolution, height=resolution).as_tuple()
    for i, seed in enumerate(scene_seeds):
        scene = generate_scene(seed=seed)
        mesh = scene.merged()
        gen_fn = generate_closed_circuit if i % 2 == 0 else generate_out_and_back
        traj = gen_fn(seed=seed, length=64, intrinsics=make_intrinsics(width=resolution, height=resolution))
        step = max(1, traj.length // frames_per_scene)
        indices = list(range(0, traj.length, step))[:frames_per_scene]
        for idx in indices:
            frame = renderer.render(mesh, traj.poses[idx], intr, resolution, resolution)
            all_frames.append(frame)
    renderer.release()
    return np.stack(all_frames, axis=0).astype(np.float32)
