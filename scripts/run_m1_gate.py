#!/usr/bin/env python3
"""M1 gate: sample frames from a trained checkpoint, decode, compare to the
M0 AE ceiling.

A falling training loss (the smoke run's actual result) only shows the
optimization is stable; it says nothing about whether the model can
generate a usable frame. This closes that gap: load the checkpoint's EMA
weights, integrate the rectified-flow ODE from noise using the ground-truth
poses of the M1 overfit trajectory, decode through the chosen AE, and
report PSNR/LPIPS against ground truth and the M0 ceiling
(results/m0_ae_ceiling.json: PSNR 46.01 dB, LPIPS 0.0043 for this AE).

Usage:
    python scripts/run_m1_gate.py --checkpoint runs/m1_overfit_5m/checkpoints/latest.json \\
        --data-dir /kaggle/working/nanowm_data/m1 --out results/m1_gate.json
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

from src.eval.chosen_ae import decode_tokens, load_chosen_ae  # noqa: E402
from src.eval.metrics import LPIPS, psnr  # noqa: E402
from src.eval.sampler import sample_frames  # noqa: E402
from src.train.model_factory import build_causal_dit  # noqa: E402


def load_ema_model(checkpoint_dir: Path, preset: str, context_length: int, device: str):
    """`latest.json` (src/train/checkpoint.py's pointer file) names the
    actual checkpoint .pt; load that and copy its EMA shadow weights into a
    fresh model -- EMA, not the raw trained weights, is what a real rollout
    or eval should sample from."""
    pointer = json.loads((checkpoint_dir / "latest.json").read_text())
    ckpt_path = checkpoint_dir / pointer["path"]  # relative filename, per CheckpointManager.save
    # weights_only=True (torch>=2.6 default) rejects the numpy arrays our own
    # RNG-state capture stores (src/train/utils.py:capture_rng_state) --
    # numpy._core.multiarray._reconstruct isn't on the default allowlist.
    # Safe to disable here: this checkpoint was produced by our own
    # CheckpointManager, not an untrusted source.
    state = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = build_causal_dit(preset, context_length=context_length).to(device)
    ema_shadow = state["ema"]
    # torch.compile's OptimizedModule wrapper (scripts/train.py compiles
    # models by default) can rename or drop keys from .state_dict()
    # relative to the uncompiled module it wraps -- observed on the smoke
    # checkpoint: every key matched except frame_pos_embed, a bare
    # nn.Parameter assigned directly on CausalDiT rather than living inside
    # a submodule. Strip a "_orig_mod." prefix if present, and fall back to
    # the raw (non-EMA) trained weights for any key genuinely missing from
    # the EMA shadow, so a naming quirk degrades gracefully instead of
    # crashing the whole eval.
    def _strip_prefix(d, prefix="_orig_mod."):
        return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in d.items()}

    ema_shadow = _strip_prefix(ema_shadow)
    raw_model_sd = _strip_prefix(state["model"])
    msd = model.state_dict()
    missing = [k for k in msd if k not in ema_shadow]
    if missing:
        print(f"WARNING: {len(missing)} key(s) missing from EMA shadow, "
              f"falling back to raw trained weights: {missing}", file=sys.stderr)
    for k in msd:
        source = ema_shadow.get(k, raw_model_sd.get(k))
        if source is None:
            raise KeyError(f"'{k}' missing from both EMA shadow and raw model weights in checkpoint")
        msd[k].copy_(source.to(dtype=msd[k].dtype, device=device))
    model.eval()
    return model, state.get("step", -1)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-dir", required=True, help="Directory containing latest.json + ckpt_*.pt")
    p.add_argument("--data-dir", required=True, help="Directory of precomputed M1 .npz trajectories")
    p.add_argument("--preset", default="5m")
    p.add_argument("--num-sample-steps", type=int, default=50)
    p.add_argument("--out", default="results/m1_gate.json")
    p.add_argument("--skip-lpips", action="store_true")
    args = p.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    npz_paths = sorted(Path(args.data_dir).glob("*.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"no .npz trajectories in {args.data_dir}")
    data = np.load(npz_paths[0])
    poses_np, intrinsics_np = data["poses"], data["intrinsics"]
    context_length = poses_np.shape[0]

    model, step = load_ema_model(Path(args.checkpoint_dir), args.preset, context_length, device)
    print(f"Loaded EMA weights from step {step}")

    poses = torch.from_numpy(poses_np).unsqueeze(0).float().to(device)
    intrinsics = torch.from_numpy(intrinsics_np).unsqueeze(0).expand(1, context_length, 4).float().to(device)

    print(f"Sampling {context_length} frames ({args.num_sample_steps} Euler steps)...")
    generator = torch.Generator(device=device).manual_seed(0)
    sampled_latents = sample_frames(model, poses, intrinsics, num_steps=args.num_sample_steps, generator=generator)
    sampled_tokens = sampled_latents[0].cpu().numpy()  # (T, tokens_per_frame, latent_channels)

    ae = load_chosen_ae(device=device)
    sampled_frames = decode_tokens(ae, sampled_tokens, device=device)

    ground_truth_tokens = data["latents"]
    ground_truth_frames = decode_tokens(ae, ground_truth_tokens, device=device)

    psnr_values = [psnr(ground_truth_frames[i], sampled_frames[i]) for i in range(context_length)]
    finite = [v for v in psnr_values if np.isfinite(v)]
    result = {
        "checkpoint_step": step,
        "context_length": context_length,
        "num_sample_steps": args.num_sample_steps,
        "psnr_mean": float(np.mean(finite)) if finite else float("inf"),
        "psnr_min": float(np.min(finite)) if finite else float("inf"),
        "m0_ceiling_psnr": 46.01169080254894,
    }
    if not args.skip_lpips:
        lpips_metric = LPIPS()
        lpips_values = [lpips_metric.compute(ground_truth_frames[i], sampled_frames[i]) for i in range(context_length)]
        result["lpips_mean"] = float(np.mean(lpips_values))
        result["m0_ceiling_lpips"] = 0.004280852249430609

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\nSampled vs ground truth: PSNR {result['psnr_mean']:.2f} dB "
          f"(M0 ceiling: {result['m0_ceiling_psnr']:.2f} dB)")
    if "lpips_mean" in result:
        print(f"LPIPS {result['lpips_mean']:.4f} (M0 ceiling: {result['m0_ceiling_lpips']:.4f})")
    gap_db = result["m0_ceiling_psnr"] - result["psnr_mean"]
    print(f"Gap to ceiling: {gap_db:.2f} dB")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
