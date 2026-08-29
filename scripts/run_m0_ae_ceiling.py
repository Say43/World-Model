#!/usr/bin/env python3
"""M0: run every AE candidate on the same fixed frame set, tabulate PSNR/LPIPS.

Inference-side (frozen AE, no training): counts against Kaggle weekly quota,
not the 10-hour training budget. Requires `diffusers`+`torch` for the AE
weights and (optionally) `lpips`; both need network access on first run to
download weights, which is why this targets Kaggle with internet enabled
rather than this offline sandbox.

Usage:
    python scripts/run_m0_ae_ceiling.py --out results/m0_ae_ceiling.json
    python scripts/run_m0_ae_ceiling.py --skip-lpips   # PSNR only, faster
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.ae_ceiling import CANDIDATES, evaluate_candidate  # noqa: E402
from src.eval.test_frames import render_test_frames  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="results/m0_ae_ceiling.json")
    p.add_argument("--skip-lpips", action="store_true")
    args = p.parse_args(argv)

    lpips_metric = None
    if not args.skip_lpips:
        from src.eval.metrics import LPIPS
        lpips_metric = LPIPS()

    frames_by_resolution = {}
    results = []
    for candidate in CANDIDATES:
        if candidate.resolution not in frames_by_resolution:
            print(f"Rendering test frames at {candidate.resolution}px...")
            frames_by_resolution[candidate.resolution] = render_test_frames(candidate.resolution)
        frames = frames_by_resolution[candidate.resolution]

        print(f"\n=== {candidate.name} ({candidate.tokens_per_frame} tok/frame, {candidate.license}) ===")
        try:
            result = evaluate_candidate(candidate, frames, lpips_metric)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {exc!r}")
            results.append({"name": candidate.name, "error": repr(exc)})
            continue
        print(f"  PSNR mean={result['psnr_mean']:.2f} dB, min={result['psnr_min']:.2f} dB"
              + (f", LPIPS mean={result['lpips_mean']:.4f}" if "lpips_mean" in result else ""))
        results.append(result)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"candidates": results}, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")

    print("\n=== Summary (tokens/frame ascending) ===")
    ok = [r for r in results if "error" not in r]
    for r in sorted(ok, key=lambda r: r["tokens_per_frame"]):
        line = f"{r['name']:>22} | {r['tokens_per_frame']:>4} tok | PSNR {r['psnr_mean']:>6.2f} dB"
        if "lpips_mean" in r:
            line += f" | LPIPS {r['lpips_mean']:.4f}"
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
