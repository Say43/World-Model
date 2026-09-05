#!/usr/bin/env python3
"""Render scenes/trajectories, encode through the M0-chosen AE, package for training.

Inference-side: costs Kaggle weekly quota, not the 10-hour training budget.
Run once per data-volume tier and cache the result as a Kaggle dataset
(CLAUDE.md: "Latents as Kaggle-Dataset ablegen, nicht neu berechnen").

Output is one .npz per trajectory plus a manifest.json, under --out-dir:
  latents        (T, 16, 128) float32   -- T frames, chosen AE's token grid
  poses          (T, 4, 4)    float32   -- camera-to-world
  intrinsics     (4,)         float32   -- (fx, fy, cx, cy)
  revisit_of     (T,)         int32     -- -1 where None
  t_values       (T,)         float32   -- placeholder noise level, filled by
                                            the training loop, not here

With --with-dinov2, one more array is stored per trajectory:
  dinov2         (T, 16, 384) float32   -- frozen DINOv2-small patch features
                                            pooled onto the same token grid

REPA's alignment target has to be computed here rather than in the training
loop: running a second frozen encoder every step would make the REPA arm of
M3's ablation slower for a reason that has nothing to do with the method,
and the comparison M3 exists to make is per-wall-clock-hour.

Usage:
    python scripts/preprocess/precompute_dataset.py --tier m1 --out-dir /kaggle/working/nanowm_data
    python scripts/preprocess/precompute_dataset.py --tier m3 --out-dir /kaggle/working/nanowm_data --with-dinov2
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.render import Renderer  # noqa: E402
from src.data.scenes import generate_scene  # noqa: E402
from src.data.trajectories import (  # noqa: E402
    generate_closed_circuit,
    generate_out_and_back,
    make_intrinsics,
)
from src.data.dinov2 import FEATURE_DIM as DINOV2_FEATURE_DIM  # noqa: E402
from src.data.dinov2 import MODEL_ID as DINOV2_MODEL_ID  # noqa: E402
from src.eval.chosen_ae import LATENT_GRID, RESOLUTION, encode_frames, load_chosen_ae  # noqa: E402

# Data-volume tiers per CLAUDE.md's M0 plan: 1 scene/1 trajectory for the M1
# overfit check; ~30-50 scenes x 3-4 trajectories for M3/M4.
TIERS = {
    "m1": {"scene_seeds": [0], "trajectories_per_scene": 1, "lengths": [128]},
    "m3": {"scene_seeds": list(range(40)), "trajectories_per_scene": 4, "lengths": [32, 64, 128]},
}


def render_trajectory_frames(renderer, mesh, traj, resolution: int) -> np.ndarray:
    intr = traj.intrinsics.as_tuple()
    frames = [renderer.render(mesh, traj.poses[i], intr, resolution, resolution) for i in range(traj.length)]
    return np.stack(frames, axis=0).astype(np.float32)


def build_trajectory(scene_seed: int, traj_idx: int, length: int):
    """Deterministic from (scene_seed, traj_idx, length): alternates loop
    kinds so a data volume tier is not biased toward one geometry."""
    seed = scene_seed * 1000 + traj_idx * 10 + length
    intr = make_intrinsics(width=RESOLUTION, height=RESOLUTION)
    if traj_idx % 2 == 0:
        return generate_closed_circuit(seed=seed, length=length, intrinsics=intr)
    return generate_out_and_back(seed=seed, length=length, intrinsics=intr)


def trajectory_arrays(traj, latents: np.ndarray, dinov2_features=None) -> dict:
    """The arrays that go into one trajectory's .npz.

    Separated from main() so the npz contract can be tested without loading
    either frozen encoder. `dinov2_features` is omitted rather than stored as
    an empty array when absent, so a dataset built without --with-dinov2 is
    distinguishable from one where extraction silently produced nothing.
    """
    arrays = {
        "latents": latents,
        "poses": traj.poses,
        "intrinsics": np.array(traj.intrinsics.as_tuple(), dtype=np.float32),
        "revisit_of": np.array([r if r is not None else -1 for r in traj.revisit_of], dtype=np.int32),
    }
    if dinov2_features is not None:
        if dinov2_features.shape[0] != latents.shape[0]:
            raise ValueError(
                f"DINOv2 features cover {dinov2_features.shape[0]} frames but "
                f"latents cover {latents.shape[0]}"
            )
        arrays["dinov2"] = dinov2_features.astype(np.float32)
    return arrays


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tier", choices=list(TIERS), required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--with-dinov2", action="store_true",
                   help="also store frozen DINOv2-small features (REPA's alignment target)")
    args = p.parse_args(argv)

    tier = TIERS[args.tier]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading chosen AE ({args.device})...")
    ae = load_chosen_ae(device=args.device)

    dinov2 = None
    if args.with_dinov2:
        from src.data.dinov2 import encode_frames as encode_dinov2_frames
        from src.data.dinov2 import load_dinov2
        print(f"Loading {DINOV2_MODEL_ID} ({args.device})...")
        dinov2 = load_dinov2(device=args.device)

    renderer = Renderer()

    manifest = []
    total_frames = 0
    start = time.time()
    for scene_seed in tier["scene_seeds"]:
        scene = generate_scene(seed=scene_seed)
        mesh = scene.merged()
        for traj_idx in range(tier["trajectories_per_scene"]):
            length = tier["lengths"][traj_idx % len(tier["lengths"])]
            traj = build_trajectory(scene_seed, traj_idx, length)
            frames = render_trajectory_frames(renderer, mesh, traj, RESOLUTION)
            latents = encode_frames(ae, frames, device=args.device)
            features = None
            if dinov2 is not None:
                features = encode_dinov2_frames(dinov2, frames, latent_grid=LATENT_GRID,
                                                device=args.device)

            name = f"scene{scene_seed:03d}_traj{traj_idx:02d}_{traj.kind}_len{length}"
            np.savez_compressed(
                out_dir / f"{name}.npz",
                **trajectory_arrays(traj, latents, features),
            )
            manifest.append({"name": name, "scene_seed": scene_seed, "kind": traj.kind,
                              "length": length, "num_revisits": traj.num_revisits})
            total_frames += length
            print(f"  {name}: {length} frames, {traj.num_revisits} revisits")

    renderer.release()
    elapsed = time.time() - start

    (out_dir / "manifest.json").write_text(json.dumps({
        "tier": args.tier,
        "ae_model_id": ae.config._name_or_path if hasattr(ae.config, "_name_or_path") else None,
        "resolution": RESOLUTION,
        "tokens_per_frame": 16,
        "dinov2_model_id": DINOV2_MODEL_ID if args.with_dinov2 else None,
        "dinov2_feature_dim": DINOV2_FEATURE_DIM if args.with_dinov2 else None,
        "num_trajectories": len(manifest),
        "total_frames": total_frames,
        "trajectories": manifest,
    }, indent=2), encoding="utf-8")

    npz_bytes = sum(f.stat().st_size for f in out_dir.glob("*.npz"))
    print(f"\nWrote {len(manifest)} trajectories, {total_frames} frames, "
          f"{npz_bytes / 1024**2:.1f} MB in {elapsed:.1f}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
