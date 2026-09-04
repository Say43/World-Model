#!/usr/bin/env python3
"""M2 gate: LR sweep at 5M (muP base width), verify transfer to 15M.

CLAUDE.md's M2 gate: the optimal LR at 5M and 15M must land in the ratio
muP predicts. Under a correct muP implementation (src/train/mup.py), that
ratio is 1:1 for the swept *base_lr* hyperparameter itself -- the whole
point of "maximal update parametrization" is that the optimal base_lr is
width-invariant, so you tune once at small width and reuse it at large
width. Under standard (non-muP) parametrization, optimal LR typically
shrinks noticeably as width grows. If the two widths' optima disagree by
more than a small tolerance, muP is not correctly implemented.

Each (preset, base_lr) cell trains a fresh model for a fixed step budget on
the representative M3-tier data and records the trailing average loss. NaN
divergence at a given LR is recorded, not fatal to the sweep.

Usage:
    python scripts/run_m2_lr_sweep.py --config configs/m2_mup_lr_sweep.yaml
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.dataset import build_dataloader  # noqa: E402
from src.train.losses import rectified_flow_loss  # noqa: E402
from src.train.model_factory import build_causal_dit  # noqa: E402
from src.train.mup import build_mup_optimizer  # noqa: E402
from src.train.nan_watchdog import NaNDetected  # noqa: E402
from src.train.trainer import Trainer, TrainerConfig  # noqa: E402
from src.train.utils import seed_everything  # noqa: E402
from scripts.train import _iso, check_ticket, git_sha, log_run  # noqa: E402


class SweepDeadline(RuntimeError):
    """The approved sweep ticket expired between optimizer steps."""


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_one(
    preset: str,
    base_lr: float,
    steps: int,
    data_dir: str,
    device: str,
    *,
    seed: int,
    base_dim: int,
    num_heads: int,
    context_length: int,
    batch_size: int,
    trailing_window: int,
    compile_model: bool,
    deadline_monotonic: float,
):
    seed_everything(seed)
    model = build_causal_dit(
        preset,
        context_length=context_length,
        num_heads=num_heads,
    ).to(device)
    config = model.config
    optimizer = build_mup_optimizer(model, config, base_dim=base_dim, base_lr=base_lr)
    if device.startswith("cuda") and not compile_model:
        raise ValueError("M2 CUDA runs require compile=true (CLAUDE.md measured gate)")
    if device.startswith("cuda"):
        # CLAUDE.md makes compile mandatory after the measured 1.2x-2.6x
        # T4 speedup. A compile failure invalidates the profile/ticket and is
        # therefore fatal instead of silently falling back to eager.
        model = torch.compile(model, mode="reduce-overhead", dynamic=False)
    trainer_config = TrainerConfig(
        warmup_steps=max(1, steps // 20),
        total_steps=steps,
        min_lr_ratio=0.1,
        grad_clip=1.0,
        ema_decay=0.99,
        amp=(device == "cuda"),
        nan_watchdog_raise=False,  # a divergent LR is a sweep result, not a crash
        log_interval=max(1, steps // 5),
    )
    trainer = Trainer(model, optimizer, rectified_flow_loss, trainer_config, device=torch.device(device))
    dl = build_dataloader(data_dir, context_length=context_length, batch_size=batch_size)

    losses = []
    diverged_at = None

    def on_step(res):
        nonlocal diverged_at
        losses.append(res["loss"])
        if res["nan_event"] is not None and diverged_at is None:
            diverged_at = res["step"]
        if time.monotonic() >= deadline_monotonic:
            raise SweepDeadline(f"ticket deadline reached in {preset} lr={base_lr:.0e}")

    deadline_reached = False
    try:
        trainer.fit(dl, steps, on_step=on_step)
    except SweepDeadline:
        deadline_reached = True
    except NaNDetected:
        pass  # nan_watchdog_raise=False means this shouldn't fire, but stay defensive

    finite_losses = [l for l in losses if l == l]  # drop NaNs (float('nan') != float('nan'))
    trailing = finite_losses[-trailing_window:] if finite_losses else []
    trailing_avg = sum(trailing) / len(trailing) if trailing else float("inf")
    return {
        "preset": preset,
        "base_lr": base_lr,
        "steps_completed": len(losses),
        "diverged_at_step": diverged_at,
        "deadline_reached": deadline_reached,
        "trailing_avg_loss": trailing_avg,
        "final_loss": losses[-1] if losses else float("inf"),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Versioned YAML config under configs/")
    args = p.parse_args(argv)

    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    run_cfg = cfg["run"]
    sweep_cfg = cfg["sweep"]
    data_cfg = cfg["data"]
    runtime_cfg = cfg["runtime"]

    ticket_hours = run_cfg.get("ticket_hours")
    if not isinstance(ticket_hours, (int, float)) or ticket_hours <= 0:
        print(
            "ABORTED: run.ticket_hours must be filled only after T4 profiling "
            "and explicit user approval; no training started.",
            file=sys.stderr,
        )
        return 1
    if not check_ticket(run_cfg["milestone"], ticket_hours):
        return 1

    presets = sweep_cfg["presets"]
    lrs = [float(value) for value in sweep_cfg["lrs"]]
    steps = int(sweep_cfg["steps"])
    base_dim = int(sweep_cfg["base_dim"])
    num_heads = int(sweep_cfg["num_heads"])
    context_length = int(data_cfg["context_length"])
    batch_size = int(data_cfg["batch_size"])
    trailing_window = int(sweep_cfg["trailing_window"])
    seed = int(run_cfg.get("seed", 0))
    device = runtime_cfg.get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("M2 config requires CUDA, but torch.cuda.is_available() is false")

    configs = [
        build_causal_dit(preset, context_length=context_length, num_heads=num_heads).config
        for preset in presets
    ]
    topologies = {(config.depth, config.num_heads) for config in configs}
    if len(topologies) != 1:
        raise ValueError(
            "M2 is a width-transfer gate: all presets must have identical "
            f"(depth, num_heads), got {sorted(topologies)}"
        )

    out_path = Path(cfg["output"]["path"])
    start_ts = time.time()
    start_monotonic = time.monotonic()
    deadline_monotonic = start_monotonic + ticket_hours * 3600.0
    status = "ok"
    result_summary = "no result"
    results = []
    stopped_early = False
    print(f"Sweeping on {device}: presets={presets}, lrs={lrs}, steps={steps}\n")

    # Written after every single (preset, lr) arm, not just once at the end:
    # a sweep this size is exactly the kind of run a Kaggle ticket deadline
    # can cut off partway through (see CLAUDE.md's decision log on the
    # fit()-completion checkpoint bug this session already hit once), and
    # a sweep with 5-10 arms losing everything to the last unfinished one
    # would waste far more budget than a training run losing one checkpoint.
    try:
        for preset in presets:
            if stopped_early:
                break
            for lr in lrs:
                if time.monotonic() >= deadline_monotonic:
                    stopped_early = True
                    break
                arm_start = time.time()
                res = run_one(
                    preset,
                    lr,
                    steps,
                    data_cfg["data_dir"],
                    device,
                    seed=seed,
                    base_dim=base_dim,
                    num_heads=num_heads,
                    context_length=context_length,
                    batch_size=batch_size,
                    trailing_window=trailing_window,
                    compile_model=bool(runtime_cfg["compile"]),
                    deadline_monotonic=deadline_monotonic,
                )
                res["wall_seconds"] = time.time() - arm_start
                results.append(res)
                stopped_early = stopped_early or res["deadline_reached"]
                arm_status = (
                    "deadline" if res["deadline_reached"] else
                    f"diverged@{res['diverged_at_step']}" if res["diverged_at_step"] is not None else
                    "ok"
                )
                print(
                    f"{preset:>12s} lr={lr:.0e}  trailing_avg_loss="
                    f"{res['trailing_avg_loss']:.5f}  [{arm_status}]  "
                    f"({res['wall_seconds']:.1f}s)"
                )
                _write_json_atomic(out_path, {
                    "config": cfg,
                    "git_sha": git_sha(),
                    "results": results,
                    "complete": False,
                })
                if stopped_early:
                    break

        total_elapsed = time.time() - start_ts
        print(f"\nTotal sweep wall time: {total_elapsed / 60:.1f} min")
        print("\n=== Best LR per preset (lowest trailing-avg loss, excluding divergence) ===")
        best_lr = {}
        for preset in presets:
            candidates = [
                result for result in results
                if result["preset"] == preset
                and result["diverged_at_step"] is None
                and not result["deadline_reached"]
            ]
            if not candidates:
                print(f"{preset}: no complete finite candidate")
                continue
            best = min(candidates, key=lambda result: result["trailing_avg_loss"])
            best_lr[preset] = best["base_lr"]
            print(f"{preset:>12s}: lr={best['base_lr']:.0e}  loss={best['trailing_avg_loss']:.5f}")

        proxy_name = presets[0]
        if len(best_lr) >= 2 and proxy_name in best_lr:
            proxy_lr = best_lr[proxy_name]
            print("\n=== Transfer check (muP predicts base-LR ratio ~1.0) ===")
            for preset, lr in best_lr.items():
                if preset != proxy_name:
                    print(f"{preset} / {proxy_name}: {lr / proxy_lr:.3f}")

        status = "ticket_exceeded" if stopped_early else "ok"
        result_summary = (
            f"arms={len(results)}, complete={not stopped_early}, best_lr={best_lr}"
        )
        _write_json_atomic(out_path, {
            "config": cfg,
            "git_sha": git_sha(),
            "results": results,
            "best_lr": best_lr,
            "total_wall_seconds": total_elapsed,
            "complete": not stopped_early,
        })
        print(f"\nWrote {out_path}")
    except KeyboardInterrupt:
        status = "interrupted"
        result_summary = f"interrupted after {len(results)} arm(s)"
        raise
    except Exception as exc:
        status = "error"
        result_summary = f"aborted with exception: {exc!r}"
        raise
    finally:
        end_ts = time.time()
        try:
            log_run(
                run_id=run_cfg["run_id"],
                start=_iso(start_ts),
                end=_iso(end_ts),
                gpu_seconds=end_ts - start_ts,  # this sweep intentionally uses one GPU
                milestone=run_cfg["milestone"],
                purpose=run_cfg["purpose"],
                result=f"[{status}] {result_summary}",
                seed=seed,
                sha=git_sha(),
            )
        except Exception as log_exc:
            # Preserve the original sweep exception: a ledger write failure is
            # important, but must not disguise the actual training failure.
            print(f"WARNING: failed to write budget ledger: {log_exc!r}")
    return 2 if stopped_early else 0


if __name__ == "__main__":
    sys.exit(main())
