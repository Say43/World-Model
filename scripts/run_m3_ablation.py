#!/usr/bin/env python3
"""M3: the 2x2 ablation the project exists to measure -- REPA on/off crossed
with Muon vs AdamW, at >=2 seeds.

CLAUDE.md's framing: "Das eigentliche Ergebnis des Projekts ist nicht das
Modell, sondern die Messung." This script produces that measurement.

Three design points that make the numbers mean something:

  * Paired. Every arm at a given seed starts from the same seeded init and
    walks the same data order, so an arm-to-arm difference at one seed is
    a difference in the treatment, not in the draw. Comparisons are
    reported per seed, not only as pooled averages.

  * Curves, not endpoints. Each arm records a subsampled loss curve
    (`curve_interval`). At this budget an endpoint difference can easily be
    schedule noise; a curve shows whether an arm is genuinely ahead
    throughout or just landed well on the last step.

  * Equal wall-clock, not equal steps. Muon's Newton-Schulz iteration and
    REPA's auxiliary head both cost time per step. Ranking arms by loss at
    equal step count would credit a method for being slower. Each arm gets
    the same `seconds_per_arm` budget and the step count it reaches is part
    of the result.

Seed-major ordering is deliberate: if the ticket deadline truncates the
matrix, a completed seed 0 is a whole (single-seed) experiment, whereas
completing all seeds of some arms and none of others is nothing at all.

Usage:
    python scripts/run_m3_ablation.py --config configs/m3_ablation.yaml
"""
import argparse
import itertools
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.dataset import build_dataloader  # noqa: E402
from src.train.losses import rectified_flow_loss, rectified_flow_loss_with_repa  # noqa: E402
from src.train.model_factory import build_causal_dit  # noqa: E402
from src.train.muon import MuonWithAuxAdamW  # noqa: E402
from src.train.nan_watchdog import NaNDetected  # noqa: E402
from src.train.repa import attach_repa_head  # noqa: E402
from src.train.trainer import Trainer, TrainerConfig  # noqa: E402
from src.train.utils import seed_everything  # noqa: E402
from scripts.train import _iso, check_ticket, git_sha, log_run  # noqa: E402


class ArmDeadline(RuntimeError):
    """This arm's slice of the approved ticket ran out."""


def arm_name(use_repa: bool, optimizer: str) -> str:
    return f"{'repa' if use_repa else 'norepa'}_{optimizer}"


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def build_optimizer(model, optimizer: str, optim_cfg: dict):
    """AdamW at the LR M2 tuned, or Muon (+ aux AdamW) at the config's LRs.

    No muP here: scripts/train.py refuses to combine muP with Muon, and
    running only the AdamW arms under muP would make the optimizer
    comparison confound two changes at once.
    """
    if optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(optim_cfg["adamw_lr"]),
            weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
        )
    if optimizer == "muon":
        return MuonWithAuxAdamW(
            model,
            muon_lr=float(optim_cfg["muon_lr"]),
            adamw_lr=float(optim_cfg["muon_aux_adamw_lr"]),
            weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
        )
    raise ValueError(f"unknown optimizer {optimizer!r} (expected adamw or muon)")


def run_arm(
    *,
    use_repa: bool,
    optimizer: str,
    seed: int,
    data_cfg: dict,
    model_cfg: dict,
    optim_cfg: dict,
    repa_cfg: dict,
    trainer_cfg: dict,
    seconds_per_arm: float,
    total_steps: int,
    curve_interval: int,
    trailing_window: int,
    device: str,
    compile_model: bool,
    hard_deadline_monotonic: float,
) -> dict:
    # Seeded before construction so every arm at this seed shares an
    # identical initialization -- the pairing the comparison rests on.
    seed_everything(seed)
    model = build_causal_dit(
        model_cfg["preset"],
        context_length=int(data_cfg["context_length"]),
    )
    if use_repa:
        attach_repa_head(
            model,
            feature_dim=int(repa_cfg["feature_dim"]),
            align_layer=repa_cfg.get("align_layer"),
        )
        model.repa_weight = float(repa_cfg["weight"])
    model = model.to(device)

    # Built before compiling, as in scripts/train.py: torch.compile's wrapper
    # has already been seen to misreport parameters (CLAUDE.md's
    # frame_pos_embed checkpoint bug).
    opt = build_optimizer(model, optimizer, optim_cfg)
    if device.startswith("cuda"):
        if not compile_model:
            raise ValueError("M3 CUDA runs require compile=true (CLAUDE.md measured gate)")
        model = torch.compile(model, mode="reduce-overhead", dynamic=False)

    config = TrainerConfig(
        warmup_steps=int(trainer_cfg["warmup_steps"]),
        total_steps=total_steps,
        min_lr_ratio=float(trainer_cfg.get("min_lr_ratio", 0.1)),
        grad_clip=float(trainer_cfg.get("grad_clip", 1.0)),
        ema_decay=float(trainer_cfg.get("ema_decay", 0.99)),
        amp=device.startswith("cuda"),
        # A diverging arm is a result about that arm, not a crash of the
        # matrix -- same reasoning as the M2 sweep.
        nan_watchdog_raise=False,
        log_interval=max(1, total_steps // 5),
    )
    loss_fn = rectified_flow_loss_with_repa if use_repa else rectified_flow_loss
    trainer = Trainer(model, opt, loss_fn, config, device=torch.device(device))
    dataloader = build_dataloader(
        data_cfg["data_dir"],
        context_length=int(data_cfg["context_length"]),
        batch_size=int(data_cfg["batch_size"]),
        with_dinov2=use_repa,
    )

    arm_deadline = min(time.monotonic() + seconds_per_arm, hard_deadline_monotonic)
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    losses: list[float] = []       # what the optimizer minimizes
    flow_losses: list[float] = []  # the only term comparable across arms
    curve: list[dict] = []
    bucket: list[float] = []
    diverged_at = None

    def on_step(res):
        nonlocal diverged_at
        losses.append(res["loss"])
        # The REPA arm's total carries an extra non-negative alignment term,
        # so arms are ranked on the flow term alone. Without this the
        # ablation would report REPA's own auxiliary loss as REPA's effect.
        components = getattr(raw_model, "last_loss_components", None)
        flow = float(components["flow"]) if components is not None else res["loss"]
        flow_losses.append(flow)
        if flow == flow:  # NaN != NaN
            bucket.append(flow)
        if res["nan_event"] is not None and diverged_at is None:
            diverged_at = res["step"]
        if len(losses) % curve_interval == 0:
            curve.append({
                "step": res["step"] + 1,
                "seconds": time.monotonic() - arm_start_monotonic,
                "mean_flow_loss": statistics.fmean(bucket) if bucket else float("nan"),
            })
            bucket.clear()
        if time.monotonic() >= arm_deadline:
            raise ArmDeadline

    arm_start_monotonic = time.monotonic()
    stopped_by_deadline = False
    try:
        trainer.fit(dataloader, total_steps, on_step=on_step)
    except ArmDeadline:
        stopped_by_deadline = True
    except NaNDetected:
        pass  # nan_watchdog_raise=False, but stay defensive

    wall_seconds = time.monotonic() - arm_start_monotonic
    finite = [value for value in flow_losses if value == value]
    trailing = finite[-trailing_window:]
    finite_total = [value for value in losses if value == value]
    return {
        "arm": arm_name(use_repa, optimizer),
        "repa": use_repa,
        "optimizer": optimizer,
        "seed": seed,
        "steps_completed": len(losses),
        "wall_seconds": wall_seconds,
        "steps_per_second": len(losses) / wall_seconds if wall_seconds > 0 else 0.0,
        "diverged_at_step": diverged_at,
        "stopped_by_deadline": stopped_by_deadline,
        "planned_steps": total_steps,
        # True when the wall-clock slice ended before the LR schedule
        # finished decaying. Such an arm trained at a higher average LR than
        # intended and is not comparable to one that completed -- reported
        # rather than silently averaged in.
        "schedule_incomplete": len(losses) < total_steps,
        # The comparison metric: trailing average of the FLOW term only.
        "trailing_avg_loss": statistics.fmean(trailing) if trailing else float("inf"),
        # The objective the optimizer actually minimized -- equal to the
        # flow term in the non-REPA arms, larger in the REPA ones. Recorded
        # for diagnosis, never for ranking.
        "trailing_avg_total_objective": (
            statistics.fmean(finite_total[-trailing_window:]) if finite_total else float("inf")
        ),
        "final_loss": flow_losses[-1] if flow_losses else float("inf"),
        "curve": curve,
    }


def paired_deltas(results: list[dict]) -> dict:
    """Per-seed differences within each factor, holding the other fixed.

    Reported alongside the raw arms because a pooled mean over 2 seeds hides
    exactly the case that matters here: an effect smaller than the
    seed-to-seed spread. If the two seeds' deltas disagree in sign, the
    ablation has not measured anything and the honest report says so.
    """
    by_key = {(r["repa"], r["optimizer"], r["seed"]): r["trailing_avg_loss"] for r in results}
    seeds = sorted({r["seed"] for r in results})
    deltas: dict = {"repa_effect": {}, "muon_effect": {}}

    for optimizer in sorted({r["optimizer"] for r in results}):
        per_seed = {}
        for seed in seeds:
            on, off = by_key.get((True, optimizer, seed)), by_key.get((False, optimizer, seed))
            if on is not None and off is not None:
                per_seed[seed] = on - off  # negative = REPA helped
        if per_seed:
            deltas["repa_effect"][optimizer] = {
                "per_seed": per_seed,
                "mean": statistics.fmean(per_seed.values()),
                "consistent_sign": len({value > 0 for value in per_seed.values()}) == 1,
            }

    for use_repa in sorted({r["repa"] for r in results}):
        per_seed = {}
        for seed in seeds:
            muon = by_key.get((use_repa, "muon", seed))
            adamw = by_key.get((use_repa, "adamw", seed))
            if muon is not None and adamw is not None:
                per_seed[seed] = muon - adamw  # negative = Muon helped
        if per_seed:
            deltas["muon_effect"]["repa" if use_repa else "norepa"] = {
                "per_seed": per_seed,
                "mean": statistics.fmean(per_seed.values()),
                "consistent_sign": len({value > 0 for value in per_seed.values()}) == 1,
            }

    return deltas


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Versioned YAML config under configs/")
    args = parser.parse_args(argv)

    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    run_cfg, ablation_cfg = cfg["run"], cfg["ablation"]
    data_cfg, runtime_cfg = cfg["data"], cfg["runtime"]

    ticket_hours = run_cfg.get("ticket_hours")
    if not isinstance(ticket_hours, (int, float)) or ticket_hours <= 0:
        print(
            "ABORTED: run.ticket_hours is unset. It is filled in only after the "
            "user approves a concrete GPU-hour ticket; no training started.",
            file=sys.stderr,
        )
        return 1
    # Everything below validates the PLAN, and runs before check_ticket:
    # a matrix that cannot produce a valid result should never reach the
    # budget at all, let alone consume an approval.
    seeds = [int(value) for value in ablation_cfg["seeds"]]
    if len(seeds) < 2:
        raise ValueError(
            f"M3 requires >=2 seeds (CLAUDE.md: 'Ablationen brauchen >=2 Seeds'), got {seeds}"
        )
    repa_options = [bool(value) for value in ablation_cfg["repa"]]
    optimizers = list(ablation_cfg["optimizers"])
    arms = list(itertools.product(repa_options, optimizers))

    seconds_per_arm = float(ablation_cfg["seconds_per_arm"])

    # Each arm's LR schedule is shaped by its own step horizon, measured by
    # a profile run. Refusing to guess here is not pedantry: an arm whose
    # total_steps overshoots what its wall-clock slice can reach trains at
    # near-peak LR throughout, which is how an M1 run diverged.
    steps_per_arm = ablation_cfg.get("steps_per_arm") or {}
    missing = [
        arm_name(use_repa, optimizer)
        for use_repa, optimizer in arms
        if not isinstance(steps_per_arm.get(arm_name(use_repa, optimizer)), int)
    ]
    if missing:
        print(
            "ABORTED: ablation.steps_per_arm is unmeasured for "
            f"{missing}. Profile each arm's steps/s on the target hardware first "
            "(CLAUDE.md: 'Profilieren vor Trainieren') and set steps_per_arm to "
            "measured_steps_per_second * seconds_per_arm; otherwise the LR "
            "schedule will not decay within the arm's wall-clock slice.",
            file=sys.stderr,
        )
        return 1

    total_arm_seconds = seconds_per_arm * len(arms) * len(seeds)
    ticket_seconds = ticket_hours * 3600.0
    if total_arm_seconds > ticket_seconds:
        raise ValueError(
            f"{len(arms)} arms x {len(seeds)} seeds x {seconds_per_arm}s = "
            f"{total_arm_seconds / 3600:.2f} GPU-hours exceeds the "
            f"{ticket_hours}h ticket; the matrix would be truncated by design"
        )

    device = runtime_cfg.get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("M3 config requires CUDA, but torch.cuda.is_available() is false")

    # Plan is valid; now ask the budget.
    if not check_ticket(run_cfg["milestone"], ticket_hours):
        return 1

    out_path = Path(cfg["output"]["path"])
    start_ts = time.time()
    hard_deadline = time.monotonic() + ticket_seconds
    results: list[dict] = []
    status, summary, truncated = "ok", "no result", False

    print(f"M3 ablation on {device}: {len(arms)} arms x {len(seeds)} seeds, "
          f"{seconds_per_arm:.0f}s each, ticket {ticket_hours}h\n")

    try:
        # Seed-major: a truncated matrix should still contain whole seeds.
        for seed in seeds:
            for use_repa, optimizer in arms:
                if time.monotonic() >= hard_deadline:
                    truncated = True
                    break
                result = run_arm(
                    use_repa=use_repa,
                    optimizer=optimizer,
                    seed=seed,
                    data_cfg=data_cfg,
                    model_cfg=cfg["model"],
                    optim_cfg=cfg["optim"],
                    repa_cfg=cfg.get("repa", {}),
                    trainer_cfg=cfg["trainer"],
                    seconds_per_arm=seconds_per_arm,
                    total_steps=int(steps_per_arm[arm_name(use_repa, optimizer)]),
                    curve_interval=int(ablation_cfg["curve_interval"]),
                    trailing_window=int(ablation_cfg["trailing_window"]),
                    device=device,
                    compile_model=bool(runtime_cfg["compile"]),
                    hard_deadline_monotonic=hard_deadline,
                )
                results.append(result)
                flag = (
                    f"diverged@{result['diverged_at_step']}"
                    if result["diverged_at_step"] is not None
                    else "SCHEDULE INCOMPLETE" if result["schedule_incomplete"]
                    else "ok"
                )
                print(f"  seed {seed} {result['arm']:>14s}: "
                      f"steps={result['steps_completed']:>6d} "
                      f"({result['steps_per_second']:.1f}/s) "
                      f"trailing_loss={result['trailing_avg_loss']:.5f} [{flag}]")
                # Written after every arm: a deadline can cut the matrix at
                # any point and a partial result is still a result.
                _write_json_atomic(out_path, {
                    "config": cfg, "git_sha": git_sha(),
                    "results": results, "complete": False,
                })
            if truncated:
                break

        deltas = paired_deltas(results)
        complete = not truncated and len(results) == len(arms) * len(seeds)
        _write_json_atomic(out_path, {
            "config": cfg,
            "git_sha": git_sha(),
            "results": results,
            "paired_deltas": deltas,
            "total_wall_seconds": time.time() - start_ts,
            "complete": complete,
        })

        print("\n=== Paired effects (negative = the treatment lowered loss) ===")
        for factor, entries in deltas.items():
            for held_fixed, entry in entries.items():
                mark = "" if entry["consistent_sign"] else "  <- SIGN DISAGREES ACROSS SEEDS"
                print(f"{factor:>12s} @ {held_fixed:>7s}: mean {entry['mean']:+.5f} "
                      f"per_seed={ {k: round(v, 5) for k, v in entry['per_seed'].items()} }{mark}")
        print(f"\nWrote {out_path}")

        status = "ok" if complete else "ticket_exceeded"
        summary = f"arms={len(results)}/{len(arms) * len(seeds)}, complete={complete}"
    except KeyboardInterrupt:
        status, summary = "interrupted", f"interrupted after {len(results)} arm(s)"
        raise
    except Exception as exc:
        status, summary = "error", f"aborted with exception: {exc!r}"
        raise
    finally:
        end_ts = time.time()
        try:
            log_run(
                run_id=run_cfg["run_id"],
                start=_iso(start_ts),
                end=_iso(end_ts),
                gpu_seconds=end_ts - start_ts,  # single GPU by design, as in M2
                milestone=run_cfg["milestone"],
                purpose=run_cfg["purpose"],
                result=f"[{status}] {summary}",
                seed=seeds[0],
                sha=git_sha(),
            )
        except Exception as log_exc:
            print(f"WARNING: failed to write budget ledger: {log_exc!r}")

    return 2 if status == "ticket_exceeded" else 0


if __name__ == "__main__":
    sys.exit(main())
