#!/usr/bin/env python3
"""M3 pre-flight: measure what the ablation needs before it can be scheduled.

Two measurements, one ticket, one ledger entry -- they are both cheap and
both block the same run, so splitting them would spend two approvals on one
question.

  Phase 1, throughput. Each of the four arms is timed for a short burst
  after a warmup. This is not bookkeeping: `configs/m3_ablation.yaml`'s
  `steps_per_arm` shapes each arm's LR decay, and an arm whose horizon
  overshoots what its wall-clock slice reaches trains at near-peak LR the
  whole way -- how an M1 run diverged. "Profilieren vor Trainieren."

  Phase 2, a 3-point Muon LR probe. M2 tuned AdamW only, and Muon's useful
  LR is roughly two orders of magnitude away (its update is orthogonalized,
  so step size is decoupled from gradient magnitude). Running the matrix at
  a reference default risks all four Muon arms diverging -- ~2.4 GPU-hours
  of the M3 allocation measuring nothing, against ~0.15 for this probe. The
  probe is cheaper than the failure it prevents; that, not fairness alone,
  is why it exists. An AdamW reference arm at M2's LR runs alongside so the
  Muon losses have a scale to be read against.

Neither phase produces an ablation result. Its output is the input to
`scripts/run_m3_ablation.py`.

Usage:
    python scripts/run_m3_profile.py --config configs/m3_profile.yaml
"""
import argparse
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
from src.train.repa import attach_repa_head  # noqa: E402
from src.train.trainer import Trainer, TrainerConfig  # noqa: E402
from src.train.utils import seed_everything  # noqa: E402
from scripts.run_m3_ablation import arm_name  # noqa: E402
from scripts.train import _iso, check_ticket, git_sha, log_run  # noqa: E402


class ProfileDeadline(RuntimeError):
    """The approved profile ticket expired between steps."""


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def build_arm(use_repa: bool, optimizer: str, seed: int, cfg: dict, device: str,
              total_steps: int, muon_lr: float | None = None):
    """A model+optimizer+trainer+dataloader for one arm, as the ablation
    builds it. Shared by both phases so the profile times the same code the
    real run will execute."""
    data_cfg, optim_cfg, repa_cfg = cfg["data"], cfg["optim"], cfg.get("repa", {})
    seed_everything(seed)
    model = build_causal_dit(cfg["model"]["preset"], context_length=int(data_cfg["context_length"]))
    if use_repa:
        attach_repa_head(model, feature_dim=int(repa_cfg["feature_dim"]),
                         align_layer=repa_cfg.get("align_layer"))
        model.repa_weight = float(repa_cfg["weight"])
    model = model.to(device)

    if optimizer == "muon":
        opt = MuonWithAuxAdamW(
            model,
            muon_lr=float(muon_lr if muon_lr is not None else optim_cfg["muon_lr"]),
            adamw_lr=float(optim_cfg["muon_aux_adamw_lr"]),
            weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
        )
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=float(optim_cfg["adamw_lr"]),
                                weight_decay=float(optim_cfg.get("weight_decay", 0.0)))

    if device.startswith("cuda"):
        model = torch.compile(model, mode="reduce-overhead", dynamic=False)

    trainer_config = TrainerConfig(
        warmup_steps=max(1, total_steps // 20),
        total_steps=total_steps,
        min_lr_ratio=float(cfg["trainer"].get("min_lr_ratio", 0.1)),
        grad_clip=float(cfg["trainer"].get("grad_clip", 1.0)),
        ema_decay=float(cfg["trainer"].get("ema_decay", 0.99)),
        amp=device.startswith("cuda"),
        nan_watchdog_raise=False,  # divergence is a measurement here, not a crash
        log_interval=max(1, total_steps),
    )
    trainer = Trainer(model, opt, rectified_flow_loss_with_repa if use_repa else rectified_flow_loss,
                      trainer_config, device=torch.device(device))
    dataloader = build_dataloader(
        data_cfg["data_dir"],
        context_length=int(data_cfg["context_length"]),
        batch_size=int(data_cfg["batch_size"]),
        with_dinov2=use_repa,
    )
    return model, trainer, dataloader


def measure_throughput(use_repa: bool, optimizer: str, cfg: dict, device: str,
                       warmup_steps: int, timed_steps: int, deadline: float) -> dict:
    """Steps/s over `timed_steps`, after `warmup_steps` are discarded.

    The warmup is not optional on CUDA: torch.compile's first steps include
    compilation, and counting those would understate throughput badly enough
    to make every derived step horizon wrong in the safe-looking direction
    (too few steps, schedule decaying too early).
    """
    total = warmup_steps + timed_steps
    model, trainer, dataloader = build_arm(use_repa, optimizer, 0, cfg, device, total)
    timings: dict = {"start": None}

    def on_step(res):
        index = res["step"] + 1
        if index == warmup_steps:
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            timings["start"] = time.monotonic()
        if time.monotonic() >= deadline:
            raise ProfileDeadline(f"deadline during {arm_name(use_repa, optimizer)} throughput")

    trainer.fit(dataloader, total, on_step=on_step)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.monotonic() - timings["start"]
    return {
        "arm": arm_name(use_repa, optimizer),
        "repa": use_repa,
        "optimizer": optimizer,
        "warmup_steps": warmup_steps,
        "timed_steps": timed_steps,
        "seconds": elapsed,
        "steps_per_second": timed_steps / elapsed,
    }


def probe_lr(optimizer: str, lr: float | None, cfg: dict, device: str,
             steps: int, trailing_window: int, deadline: float) -> dict:
    """Train one short arm and report its trailing loss.

    REPA is off in every probe: the question here is the optimizer's step
    size, and adding the auxiliary term would only put a constant offset on
    the number being compared.
    """
    model, trainer, dataloader = build_arm(False, optimizer, 0, cfg, device, steps, muon_lr=lr)
    losses: list[float] = []
    diverged_at = None

    def on_step(res):
        nonlocal diverged_at
        losses.append(res["loss"])
        if res["nan_event"] is not None and diverged_at is None:
            diverged_at = res["step"]
        if time.monotonic() >= deadline:
            raise ProfileDeadline(f"deadline during {optimizer} lr={lr} probe")

    trainer.fit(dataloader, steps, on_step=on_step)
    finite = [value for value in losses if value == value]
    trailing = finite[-trailing_window:]
    return {
        "optimizer": optimizer,
        "lr": lr if lr is not None else float(cfg["optim"]["adamw_lr"]),
        "steps": len(losses),
        "diverged_at_step": diverged_at,
        "trailing_avg_loss": statistics.fmean(trailing) if trailing else float("inf"),
    }


def recommend_steps_per_arm(throughput: list[dict], seconds_per_arm: float,
                            safety_margin: float) -> dict:
    """Measured steps/s -> the step horizon each arm's LR schedule should use.

    Margin shaved off deliberately: overshooting the horizon is the harmful
    direction (the schedule never decays and the arm is flagged
    schedule_incomplete), while undershooting merely finishes the schedule
    slightly early and trains out the remainder at the LR floor.
    """
    return {
        entry["arm"]: int(entry["steps_per_second"] * seconds_per_arm * safety_margin)
        for entry in throughput
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    run_cfg, profile_cfg = cfg["run"], cfg["profile"]

    ticket_hours = run_cfg.get("ticket_hours")
    if not isinstance(ticket_hours, (int, float)) or ticket_hours <= 0:
        print("ABORTED: run.ticket_hours is unset; it is filled only after the user "
              "approves a concrete GPU-hour ticket. No training started.", file=sys.stderr)
        return 1

    device = cfg["runtime"].get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("M3 profile requires CUDA, but torch.cuda.is_available() is false")
    if not check_ticket(run_cfg["milestone"], ticket_hours):
        return 1

    out_path = Path(cfg["output"]["path"])
    start_ts = time.time()
    deadline = time.monotonic() + ticket_hours * 3600.0
    throughput: list[dict] = []
    probes: list[dict] = []
    status, summary = "ok", "no result"

    def snapshot(complete: bool) -> None:
        _write_json_atomic(out_path, {
            "config": cfg, "git_sha": git_sha(),
            "throughput": throughput, "lr_probes": probes,
            "recommended_steps_per_arm": recommend_steps_per_arm(
                throughput, float(profile_cfg["target_seconds_per_arm"]),
                float(profile_cfg["safety_margin"]),
            ) if throughput else {},
            "complete": complete,
        })

    try:
        print("=== Phase 1: throughput per arm ===")
        for use_repa in (False, True):
            for optimizer in ("adamw", "muon"):
                entry = measure_throughput(
                    use_repa, optimizer, cfg, device,
                    warmup_steps=int(profile_cfg["warmup_steps"]),
                    timed_steps=int(profile_cfg["timed_steps"]),
                    deadline=deadline,
                )
                throughput.append(entry)
                print(f"  {entry['arm']:>14s}: {entry['steps_per_second']:7.2f} steps/s")
                snapshot(False)

        print("\n=== Phase 2: Muon LR probe (REPA off) ===")
        for lr in [float(value) for value in profile_cfg["muon_lrs"]]:
            entry = probe_lr("muon", lr, cfg, device, int(profile_cfg["probe_steps"]),
                             int(profile_cfg["trailing_window"]), deadline)
            probes.append(entry)
            flag = f"diverged@{entry['diverged_at_step']}" if entry["diverged_at_step"] is not None else "ok"
            print(f"  muon  lr={lr:<8.4g} trailing_loss={entry['trailing_avg_loss']:.5f} [{flag}]")
            snapshot(False)

        # Reference point: without it the Muon numbers have no scale.
        entry = probe_lr("adamw", None, cfg, device, int(profile_cfg["probe_steps"]),
                         int(profile_cfg["trailing_window"]), deadline)
        probes.append(entry)
        print(f"  adamw lr={entry['lr']:<8.4g} trailing_loss={entry['trailing_avg_loss']:.5f}")

        usable = [p for p in probes if p["optimizer"] == "muon" and p["diverged_at_step"] is None]
        best = min(usable, key=lambda p: p["trailing_avg_loss"]) if usable else None
        recommended = recommend_steps_per_arm(
            throughput, float(profile_cfg["target_seconds_per_arm"]),
            float(profile_cfg["safety_margin"]),
        )
        snapshot(True)

        print("\n=== Values to fill into configs/m3_ablation.yaml ===")
        print(f"  optim.muon_lr: {best['lr'] if best else 'NO NON-DIVERGENT MUON LR -- widen the grid'}")
        for arm, steps in recommended.items():
            print(f"  ablation.steps_per_arm.{arm}: {steps}")
        if best is not None and best["lr"] in (min(p["lr"] for p in usable), max(p["lr"] for p in usable)):
            print("\n  NOTE: the best LR sits at an edge of the probed grid, so the "
                  "optimum may lie outside it. Read the M3 Muon result with that in mind.")
        print(f"\nWrote {out_path}")
        summary = (f"throughput={len(throughput)} arms, probes={len(probes)}, "
                   f"best_muon_lr={best['lr'] if best else None}")
    except ProfileDeadline as exc:
        status, summary = "ticket_exceeded", str(exc)
        snapshot(False)
        print(f"ABORTED: {exc}", file=sys.stderr)
    except KeyboardInterrupt:
        status, summary = "interrupted", f"interrupted after {len(throughput)} arm(s)"
        raise
    except Exception as exc:
        status, summary = "error", f"aborted with exception: {exc!r}"
        raise
    finally:
        end_ts = time.time()
        try:
            log_run(
                run_id=run_cfg["run_id"], start=_iso(start_ts), end=_iso(end_ts),
                gpu_seconds=end_ts - start_ts,  # single GPU by design, as in M2
                milestone=run_cfg["milestone"], purpose=run_cfg["purpose"],
                result=f"[{status}] {summary}", seed=0, sha=git_sha(),
            )
        except Exception as log_exc:
            print(f"WARNING: failed to write budget ledger: {log_exc!r}")

    return 2 if status == "ticket_exceeded" else 0


if __name__ == "__main__":
    sys.exit(main())
