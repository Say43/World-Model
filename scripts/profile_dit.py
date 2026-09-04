#!/usr/bin/env python3
"""Real step-time / VRAM / MFU measurements for the CausalDiT presets.

Replaces the analytical estimate that M0.5 has been running on. That estimate
treated tokens-per-frame as the whole sequence length and so ignored the
context dimension entirely; the real sequence is tokens_per_frame *
context_length, which at 40M/256 tokens means 4096 positions and roughly
doubles cost per token. Ticket writing needs measured numbers, not that.

This is inference-side work: it costs Kaggle weekly quota, not the 10-hour
training budget. Each configuration is capped so the whole sweep stays short.

Usage:
    python scripts/profile_dit.py --out runs/profile.json
    python scripts/profile_dit.py --presets 5m --tokens 64 --no-compile
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.dit import (  # noqa: E402
    CausalDiT,
    preset_5m,
    preset_m2_proxy_5m,
    preset_15m,
    preset_40m,
)
from src.eval.chosen_ae import LATENT_CHANNELS  # noqa: E402
from src.train.mup import build_mup_optimizer  # noqa: E402

PRESETS = {
    "5m": preset_5m,
    "m2_proxy_5m": preset_m2_proxy_5m,
    "15m": preset_15m,
    "40m": preset_40m,
}
# T4 fp16 peak with tensor cores. Reported alongside MFU so the assumption is
# visible rather than baked into a single number.
T4_FP16_PEAK_TFLOPS = 65.0


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def make_batch(cfg, batch_size, device):
    """Random inputs matching CausalDiT's forward signature."""
    latents = torch.randn(batch_size, cfg.context_length, cfg.tokens_per_frame, cfg.latent_channels, device=device)
    poses = torch.eye(4, device=device).expand(batch_size, cfg.context_length, 4, 4).contiguous()
    # Spread the cameras out so the raymap is not degenerate.
    poses = poses.clone()
    poses[:, :, :3, 3] = torch.randn(batch_size, cfg.context_length, 3, device=device)
    intrinsics = torch.tensor(
        [cfg.raymap_resolution[1], cfg.raymap_resolution[0], cfg.raymap_resolution[1] / 2, cfg.raymap_resolution[0] / 2],
        device=device, dtype=torch.float32,
    ).expand(batch_size, cfg.context_length, 4).contiguous()
    t = torch.rand(batch_size, cfg.context_length, device=device)
    return latents, poses, intrinsics, t


def profile_one(
    preset_name,
    tokens_per_frame,
    batch_size,
    use_compile,
    steps,
    warmup,
    device,
    time_budget_s,
    num_heads=None,
    mup_base_dim=None,
):
    preset_fn = PRESETS[preset_name]
    side = int(round(tokens_per_frame ** 0.5))
    if side * side != tokens_per_frame:
        raise ValueError(f"tokens_per_frame={tokens_per_frame} is not a perfect square")
    overrides = {
        "tokens_per_frame": tokens_per_frame,
        "raymap_resolution": (side, side),
        "latent_channels": LATENT_CHANNELS,
    }
    if num_heads is not None:
        overrides["num_heads"] = num_heads
    cfg = preset_fn(**overrides)

    model = CausalDiT(cfg).to(device)
    n_params = count_parameters(model)
    optimizer = (
        build_mup_optimizer(model, cfg, base_dim=mup_base_dim, base_lr=1e-4)
        if mup_base_dim is not None
        else torch.optim.AdamW(model.parameters(), lr=1e-4)
    )
    scaler = torch.amp.GradScaler(device="cuda", enabled=device.type == "cuda")

    compiled = False
    compile_error = None
    if use_compile:
        try:
            model = torch.compile(model, mode="reduce-overhead", dynamic=False)
            compiled = True
        except Exception as exc:  # noqa: BLE001
            compile_error = repr(exc)

    latents, poses, intrinsics, t = make_batch(cfg, batch_size, device)
    target = torch.randn_like(latents)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    def one_step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(latents, poses, intrinsics, t)
            loss = torch.nn.functional.mse_loss(out.float(), target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return loss

    oom = False
    timings = []
    nonfinite_losses = 0
    scale_drops = 0
    try:
        for _ in range(warmup):
            one_step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        sweep_start = time.perf_counter()
        for _ in range(steps):
            scale_before = scaler.get_scale() if device.type == "cuda" else None
            step_start = time.perf_counter()
            loss = one_step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append(time.perf_counter() - step_start)
            if not torch.isfinite(loss).all():
                nonfinite_losses += 1
            if scale_before is not None and scaler.get_scale() < scale_before:
                scale_drops += 1
            if time.perf_counter() - sweep_start > time_budget_s:
                break
    except torch.cuda.OutOfMemoryError:
        oom = True
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        oom = True

    peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
    result = {
        "preset": preset_name,
        "params": n_params,
        "tokens_per_frame": tokens_per_frame,
        "context_length": cfg.context_length,
        "dim": cfg.dim,
        "depth": cfg.depth,
        "num_heads": cfg.num_heads,
        "mup_base_dim": mup_base_dim,
        "sequence_length": cfg.context_length * tokens_per_frame,
        "batch_size": batch_size,
        "compile_requested": use_compile,
        "compiled": compiled,
        "compile_error": compile_error,
        "oom": oom,
        "peak_vram_gb": round(peak_gb, 3),
        "measured_steps": len(timings),
        "nonfinite_losses": nonfinite_losses,
        "gradscaler_skips": scale_drops,
    }
    if timings and not oom:
        timings.sort()
        median = timings[len(timings) // 2]
        p90 = timings[min(len(timings) - 1, int(0.9 * len(timings)))]
        tokens_per_step = batch_size * cfg.context_length * tokens_per_frame
        # 6N per token for fwd+bwd on parameters; attention term added
        # separately since at these sequence lengths it is not negligible.
        seq = cfg.context_length * tokens_per_frame
        attn_flops_per_token = 6.0 * cfg.depth * 2 * cfg.dim * seq
        flops_per_token = 6.0 * n_params + attn_flops_per_token
        achieved = tokens_per_step * flops_per_token / median / 1e12
        result.update({
            "step_time_median_s": round(median, 5),
            "step_time_p90_s": round(p90, 5),
            "tokens_per_s": round(tokens_per_step / median, 1),
            "achieved_tflops": round(achieved, 2),
            "mfu_percent": round(100.0 * achieved / T4_FP16_PEAK_TFLOPS, 2),
            "flops_per_token": flops_per_token,
        })
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--presets", nargs="+", default=["5m", "15m", "40m"], choices=list(PRESETS))
    p.add_argument("--tokens", nargs="+", type=int, default=[16, 64, 256],
                   help="tokens per frame to sweep (must be perfect squares)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--no-compile", action="store_true", help="skip the torch.compile arm")
    p.add_argument("--compile-only", action="store_true",
                   help="profile only the mandatory compiled arm")
    p.add_argument("--time-budget", type=float, default=60.0,
                   help="seconds per configuration before cutting the sweep short")
    p.add_argument("--num-heads", type=int, default=None,
                   help="override head count; keep fixed for a width-only muP profile")
    p.add_argument("--mup-base-dim", type=int, default=None,
                   help="enable muP with this proxy width")
    p.add_argument("--out", default="runs/profile.json")
    args = p.parse_args(argv)
    if args.no_compile and args.compile_only:
        p.error("--no-compile and --compile-only are mutually exclusive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Profiling on {device}"
          f"{' (' + torch.cuda.get_device_name(0) + ')' if device.type == 'cuda' else ''}")

    compile_arms = [True] if args.compile_only else ([False] if args.no_compile else [False, True])
    results = []
    for preset in args.presets:
        for tokens in args.tokens:
            for use_compile in compile_arms:
                label = f"{preset} tokens/frame={tokens} compile={use_compile}"
                print(f"\n=== {label} ===", flush=True)
                try:
                    res = profile_one(
                        preset, tokens, args.batch_size, use_compile,
                        args.steps, args.warmup, device, args.time_budget,
                        num_heads=args.num_heads,
                        mup_base_dim=args.mup_base_dim,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"  FAILED: {exc!r}")
                    results.append({"preset": preset, "tokens_per_frame": tokens,
                                    "compile_requested": use_compile, "error": repr(exc)})
                    continue
                results.append(res)
                if res.get("oom"):
                    print("  OOM")
                else:
                    print(f"  step {res['step_time_median_s'] * 1000:.1f} ms | "
                          f"VRAM {res['peak_vram_gb']:.2f} GB | "
                          f"{res['tokens_per_s']:.0f} tok/s | MFU {res['mfu_percent']:.1f}% | "
                          f"compiled={res['compiled']} | "
                          f"nonfinite={res['nonfinite_losses']} scaler_skips={res['gradscaler_skips']}")
                del res
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "device_count": torch.cuda.device_count() if device.type == "cuda" else 0,
        "torch": torch.__version__,
        "t4_fp16_peak_tflops_assumed": T4_FP16_PEAK_TFLOPS,
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")

    print("\n=== Steps per budget (single device, measured) ===")
    for res in results:
        if res.get("oom") or "step_time_median_s" not in res:
            continue
        st = res["step_time_median_s"]
        print(f"{res['preset']:>4s} tok/frame={res['tokens_per_frame']:>4d} "
              f"compile={str(res['compiled']):>5s}: "
              f"{3600 / st:>9,.0f} steps/GPU-hour")
    return 0


if __name__ == "__main__":
    sys.exit(main())
