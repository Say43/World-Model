#!/usr/bin/env python3
"""M4 eval: context-conditioned future-frame prediction on held-out scenes.

run_m1_gate.py samples every frame of a window from pure noise given only
the poses. That answers M1's question (one memorised scene) but says nothing
about a model trained on 160 trajectories: without any clean frame the model
cannot know *which* scene it is in, so a PSNR against ground truth would
measure hallucination, not prediction. Here the first `context_frames` of
each window are held clean at t=0 (in-distribution for diffusion forcing,
which trains every frame at its own independent t) and only the remaining
frames are integrated from noise. The score is on those predicted frames.

Outputs, per split (heldout / train-reference):
  * PSNR / LPIPS per predicted frame, averaged, plus a per-horizon curve
    (frame k after the context -> mean PSNR), which is the drift curve at
    window scale.
  * one PNG grid per window: row 1 ground truth, row 2 prediction, the
    context frames marked; one GIF per window for scrolling through.
  * results JSON with everything above and the M0 ceiling for reference.

Usage:
    python scripts/run_m4_eval.py --checkpoint-dir runs/m4_main/checkpoints \\
        --heldout-dir /kaggle/working/nanowm_data/heldout \\
        --train-dir /kaggle/input/datasets/says43/nanowm-m3-latents \\
        --preset 40m --out-dir /kaggle/working/m4_eval
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

from scripts.run_m1_gate import load_ema_model  # noqa: E402
from src.eval.chosen_ae import decode_tokens, load_chosen_ae  # noqa: E402
from src.eval.metrics import LPIPS, psnr  # noqa: E402

M0_CEILING_PSNR = 46.01169080254894
M0_CEILING_LPIPS = 0.004280852249430609


@torch.no_grad()
def sample_future_frames(model, gt_latents: torch.Tensor, poses: torch.Tensor,
                         intrinsics: torch.Tensor, context_frames: int,
                         num_steps: int = 50, generator: torch.Generator = None) -> torch.Tensor:
    """Euler-integrate frames [context_frames:] from t=1 to t=0 while frames
    [:context_frames] stay at their clean latents with t=0 throughout.

    Same sign convention as src/eval/sampler.py: x_t = (1-t) x0 + t x1,
    v = x1 - x0, so dx/dt = -v integrated from t=1 down to t=0. The causal
    mask means the context frames' own predictions are ignored (they are
    overwritten after every step) and the predicted frames only ever see
    clean context plus their own noisy state -- the training condition.
    """
    device = gt_latents.device
    batch_size, context_length = gt_latents.shape[0], gt_latents.shape[1]
    x = torch.randn(gt_latents.shape, device=device, generator=generator)
    x[:, :context_frames] = gt_latents[:, :context_frames]
    dt = 1.0 / num_steps
    for step in range(num_steps):
        t_value = 1.0 - step * dt
        t = torch.full((batch_size, context_length), t_value, device=device)
        t[:, :context_frames] = 0.0
        velocity = model(x, poses, intrinsics, t)
        x = x - velocity * dt
        x[:, :context_frames] = gt_latents[:, :context_frames]
    return x


def windows_of(data, context_length: int, max_windows: int):
    """Non-overlapping windows of one trajectory, first `max_windows`."""
    total = data["poses"].shape[0]
    starts = list(range(0, total - context_length + 1, context_length))[:max_windows]
    for start in starts:
        yield start, {
            "latents": data["latents"][start:start + context_length],
            "poses": data["poses"][start:start + context_length],
            "intrinsics": data["intrinsics"],
        }


def save_grid_png(gt_frames: np.ndarray, pred_frames: np.ndarray, context_frames: int, path: Path) -> None:
    """Row 1 ground truth, row 2 prediction; context frames dimmed in row 2
    (they are copied from the ground truth, not predicted)."""
    from PIL import Image
    t, h, w, _ = gt_frames.shape
    canvas = np.ones((2 * h + 8, t * (w + 4), 3), dtype=np.float32)
    for i in range(t):
        x0 = i * (w + 4)
        canvas[0:h, x0:x0 + w] = gt_frames[i]
        pred = pred_frames[i] if i >= context_frames else 0.5 * gt_frames[i] + 0.5
        canvas[h + 8:2 * h + 8, x0:x0 + w] = pred
    Image.fromarray((canvas * 255).clip(0, 255).astype(np.uint8)).save(path)


def save_gif(gt_frames: np.ndarray, pred_frames: np.ndarray, context_frames: int, path: Path) -> None:
    from PIL import Image
    frames = []
    for i in range(gt_frames.shape[0]):
        pred = pred_frames[i] if i >= context_frames else 0.5 * gt_frames[i] + 0.5
        side = np.concatenate([gt_frames[i], np.ones((gt_frames.shape[1], 4, 3)), pred], axis=1)
        frames.append(Image.fromarray((side * 255).clip(0, 255).astype(np.uint8)))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=200, loop=0)


def evaluate_split(name: str, npz_paths, model, ae, lpips_metric, device: str, args, out_dir: Path) -> dict:
    per_window = []
    horizon_psnr = {}
    generator = torch.Generator(device=device).manual_seed(0)
    for path in npz_paths:
        data = np.load(path)
        for start, window in windows_of(data, args.context_length, args.max_windows_per_trajectory):
            gt_latents = torch.from_numpy(window["latents"]).unsqueeze(0).float().to(device)
            poses = torch.from_numpy(window["poses"]).unsqueeze(0).float().to(device)
            intrinsics = (torch.from_numpy(window["intrinsics"]).unsqueeze(0)
                          .expand(1, args.context_length, 4).float().to(device))
            pred_latents = sample_future_frames(
                model, gt_latents, poses, intrinsics, args.context_frames,
                num_steps=args.num_sample_steps, generator=generator,
            )
            gt_frames = decode_tokens(ae, window["latents"], device=device)
            pred_frames = decode_tokens(ae, pred_latents[0].cpu().numpy(), device=device)

            psnrs, lpipss = [], []
            for i in range(args.context_frames, args.context_length):
                value = psnr(gt_frames[i], pred_frames[i])
                psnrs.append(float(value))
                horizon_psnr.setdefault(i - args.context_frames + 1, []).append(float(value))
                if lpips_metric is not None:
                    lpipss.append(float(lpips_metric.compute(gt_frames[i], pred_frames[i])))

            tag = f"{name}_{path.stem}_w{start:03d}"
            save_grid_png(gt_frames, pred_frames, args.context_frames, out_dir / f"{tag}.png")
            save_gif(gt_frames, pred_frames, args.context_frames, out_dir / f"{tag}.gif")
            entry = {
                "trajectory": path.stem, "window_start": start,
                "psnr_mean": float(np.mean([v for v in psnrs if np.isfinite(v)])),
                "psnr_per_frame": psnrs,
            }
            if lpipss:
                entry["lpips_mean"] = float(np.mean(lpipss))
                entry["lpips_per_frame"] = lpipss
            per_window.append(entry)
            print(f"  {tag}: PSNR {entry['psnr_mean']:.2f} dB"
                  + (f", LPIPS {entry['lpips_mean']:.4f}" if lpipss else ""))

    summary = {
        "num_windows": len(per_window),
        "psnr_mean": float(np.mean([w["psnr_mean"] for w in per_window])),
        "psnr_min": float(np.min([w["psnr_mean"] for w in per_window])),
        "psnr_by_horizon": {k: float(np.mean(v)) for k, v in sorted(horizon_psnr.items())},
        "windows": per_window,
    }
    if per_window and "lpips_mean" in per_window[0]:
        summary["lpips_mean"] = float(np.mean([w["lpips_mean"] for w in per_window]))
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--heldout-dir", required=True, help="precompute_dataset.py --tier heldout output")
    p.add_argument("--train-dir", required=True, help="the training tier, for a train-vs-heldout reference")
    p.add_argument("--preset", default="40m")
    p.add_argument("--context-length", type=int, default=16)
    p.add_argument("--context-frames", type=int, default=8,
                   help="clean frames given to the model; the rest of the window is predicted")
    p.add_argument("--num-sample-steps", type=int, default=50)
    p.add_argument("--max-windows-per-trajectory", type=int, default=4)
    p.add_argument("--train-trajectories", type=int, default=4)
    p.add_argument("--out-dir", default="results/m4_eval")
    p.add_argument("--skip-lpips", action="store_true")
    args = p.parse_args(argv)
    if not 0 < args.context_frames < args.context_length:
        raise ValueError("context_frames must leave at least one frame to predict")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, step = load_ema_model(Path(args.checkpoint_dir), args.preset, args.context_length, device)
    model.eval()
    print(f"Loaded EMA weights from step {step}")
    ae = load_chosen_ae(device=device)
    lpips_metric = None if args.skip_lpips else LPIPS(device=device)

    heldout_paths = sorted(Path(args.heldout_dir).glob("*.npz"))
    train_paths = sorted(Path(args.train_dir).glob("*.npz"))[:args.train_trajectories]
    if not heldout_paths or not train_paths:
        raise FileNotFoundError("need .npz trajectories in both --heldout-dir and --train-dir")

    print(f"\nHeld-out scenes ({len(heldout_paths)} trajectories):")
    heldout = evaluate_split("heldout", heldout_paths, model, ae, lpips_metric, device, args, out_dir)
    print(f"\nTraining scenes, reference ({len(train_paths)} trajectories):")
    train_ref = evaluate_split("train", train_paths, model, ae, lpips_metric, device, args, out_dir)

    result = {
        "checkpoint_step": step,
        "preset": args.preset,
        "context_length": args.context_length,
        "context_frames": args.context_frames,
        "predicted_frames": args.context_length - args.context_frames,
        "num_sample_steps": args.num_sample_steps,
        "m0_ceiling_psnr": M0_CEILING_PSNR,
        "m0_ceiling_lpips": M0_CEILING_LPIPS,
        "heldout": heldout,
        "train_reference": train_ref,
    }
    (out_dir / "m4_eval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\n=== M4 eval @ step {step} ({args.context_frames} context -> "
          f"{args.context_length - args.context_frames} predicted frames) ===")
    for split_name, split in (("heldout", heldout), ("train_reference", train_ref)):
        line = f"{split_name:>15s}: PSNR {split['psnr_mean']:.2f} dB (min window {split['psnr_min']:.2f})"
        if "lpips_mean" in split:
            line += f", LPIPS {split['lpips_mean']:.4f}"
        print(line)
        print(f"{'':>15s}  by horizon: "
              + " ".join(f"+{k}:{v:.1f}" for k, v in split["psnr_by_horizon"].items()))
    print(f"M0 ceiling: PSNR {M0_CEILING_PSNR:.2f} dB, LPIPS {M0_CEILING_LPIPS:.4f}")
    print(f"Wrote {out_dir / 'm4_eval.json'} and {2 * (heldout['num_windows'] + train_ref['num_windows'])} images")
    return 0


if __name__ == "__main__":
    sys.exit(main())
