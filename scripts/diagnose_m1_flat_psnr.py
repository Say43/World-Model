#!/usr/bin/env python3
"""Local, no-GPU diagnostic for M1's flat PSNR across four training runs.

Four checkpoints (3968/5000/5866/8000 steps, all stable, all clean) landed
in the same ~13 dB / ~0.6 LPIPS band -- ruling out "just needs more
training." This isolates two independent questions without spending any
Kaggle budget, using an already-downloaded checkpoint and its training data:

1. One-step denoising near t=0: rectified flow's x_t = (1-t)*x0 + t*x1 is
   nearly clean at small t, so recovering x0 from x_t should be close to
   trivial for a model that has learned anything at all about this data.
   If this fails, the learned velocity field itself is bad -- training
   didn't work, regardless of the sampler. If this succeeds but the full
   50-step Euler sampler (scripts/run_m1_gate.py, integrating from pure
   noise) still produces ~13 dB, the sampler/integration path is the more
   likely fault, not the model.

2. Pose sensitivity: swap in a different frame's pose/intrinsics and check
   whether the model's velocity prediction actually changes. If it barely
   moves, Plucker conditioning has no real effect on the output -- the
   CLAUDE.md M1 gate's literal debug trigger ("if it can't get close,
   Plucker conditioning is broken").

Usage:
    python scripts/diagnose_m1_flat_psnr.py --checkpoint-dir <dir> --data-dir <dir>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "scripts"))
from run_m1_gate import load_ema_model  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--preset", default="5m")
    p.add_argument("--context-length", type=int, default=16)
    args = p.parse_args(argv)

    device = "cpu"
    npz_paths = sorted(Path(args.data_dir).glob("*.npz"))
    data = np.load(npz_paths[0])
    context_length = args.context_length
    latents = torch.from_numpy(data["latents"][:context_length]).float().unsqueeze(0)  # (1,T,N,C)
    poses = torch.from_numpy(data["poses"][:context_length]).float().unsqueeze(0)
    intrinsics = torch.from_numpy(
        np.broadcast_to(data["intrinsics"], (context_length, 4))
    ).float().unsqueeze(0)

    model, step = load_ema_model(Path(args.checkpoint_dir), args.preset, context_length, device)
    print(f"Loaded EMA weights from step {step}\n")

    torch.manual_seed(0)
    noise = torch.randn_like(latents)

    print("=== Test 1: one-step denoising accuracy vs noise level t ===")
    print(f"{'t':>6} {'x0_mse':>12} {'x0_mse_vs_random_baseline':>28}")
    baseline_mse = torch.nn.functional.mse_loss(noise, latents).item()
    for t_value in [0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 0.98]:
        t = torch.full((1, context_length), t_value)
        t_broadcast = t.view(1, context_length, 1, 1)
        x_t = (1 - t_broadcast) * latents + t_broadcast * noise
        with torch.no_grad():
            v_pred = model(x_t, poses, intrinsics, t)
        x0_pred = x_t - t_broadcast * v_pred
        mse = torch.nn.functional.mse_loss(x0_pred, latents).item()
        print(f"{t_value:>6.2f} {mse:>12.5f} {mse / baseline_mse:>28.4f}")
    print(f"(baseline: MSE(random_noise, x0) = {baseline_mse:.5f} -- a model predicting\n"
          f" pure noise as x0 would score ~1.0 in the last column at every t)\n")

    print("=== Test 2: does the prediction actually respond to pose? ===")
    print("(perturbing ONLY the last frame's pose, to cleanly separate\n"
          " 'does conditioning affect that frame's own prediction' from\n"
          " 'does a later frame leak into earlier ones' -- a full pose\n"
          " shuffle would conflate both, since every frame's own pose\n"
          " would change too.)\n")
    t_value = 0.3
    t = torch.full((1, context_length), t_value)
    t_broadcast = t.view(1, context_length, 1, 1)
    x_t = (1 - t_broadcast) * latents + t_broadcast * noise
    last = context_length - 1

    poses_perturbed = poses.clone()
    # A real rotation+translation perturbation of only the last frame,
    # nothing else -- swap two rotation axes and shift the camera far away,
    # not a subtle nudge.
    poses_perturbed[:, last, :3, :3] = poses[:, last, :3, [1, 0, 2]]
    poses_perturbed[:, last, :3, 3] += 5.0

    with torch.no_grad():
        v_pred_real_pose = model(x_t, poses, intrinsics, t)
        v_pred_perturbed_pose = model(x_t, poses_perturbed, intrinsics, t)

    v_pred_magnitude = v_pred_real_pose.pow(2).mean().item()
    diff_last_frame = torch.nn.functional.mse_loss(
        v_pred_real_pose[:, last], v_pred_perturbed_pose[:, last]
    ).item()
    print(f"velocity prediction magnitude (mean sq.):                 {v_pred_magnitude:.5f}")
    print(f"last frame's own prediction, change from its pose moving: {diff_last_frame:.5f}"
          f"  (ratio to magnitude: {diff_last_frame / v_pred_magnitude:.3f})")
    print("(near 0 ratio means pose changes barely move that frame's own prediction --\n"
          " weak/no Plucker conditioning. This is the literal debug trigger CLAUDE.md's\n"
          " M1 gate names: 'if it can't get close, Plucker conditioning is broken.')\n")

    earlier = last  # every frame strictly before the perturbed one
    causal_leak = torch.nn.functional.mse_loss(
        v_pred_real_pose[:, :earlier], v_pred_perturbed_pose[:, :earlier]
    ).item()
    print(f"causal leak check (earlier frames' predictions, before vs after\n"
          f"perturbing only the LAST frame's pose): {causal_leak:.10f}")
    print("(must be exactly 0.0 -- anything else means the causal mask itself is broken,\n"
          " which would invalidate every rollout/drift-curve result from M4 onward)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
