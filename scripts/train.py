#!/usr/bin/env python3
"""nanoWM training entry point -- config-driven, budget-gated.

No run starts without an approved budget ticket, and every run -- success,
exception, or SIGTERM -- appends exactly one line to budget/ledger.jsonl.
See CLAUDE.md's "budget ledger" section for why this is non-negotiable.

Usage:
    python scripts/train.py --config configs/smoke_cpu.yaml

Config schema (see configs/smoke_cpu.yaml for a full example):
    run:        run_id, milestone, ticket_hours, purpose, seed, log_interval
    data:       target (dotted path to a callable returning a DataLoader), kwargs
    model:      target (dotted path to a callable returning an nn.Module), kwargs
    loss:       target (dotted path to loss_fn(model, batch) -> scalar tensor)
    optim:      lr, weight_decay (AdamW)
    trainer:    TrainerConfig fields (warmup_steps, total_steps, grad_clip, ...)
    checkpoint: dir, interval_steps
"""
import argparse
import datetime
import importlib
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.checkpoint import CheckpointManager, SigtermInterrupt
from src.train.ddp import cleanup_ddp, is_main_process, setup_ddp, wrap_model
from src.train.trainer import Trainer, TrainerConfig
from src.train.utils import seed_everything

BUDGET_SCRIPT = ROOT / "scripts" / "budget.py"


def _import_target(dotted_path: str):
    module_name, _, attr = dotted_path.rpartition(".")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def _iso(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).isoformat()


def check_ticket(milestone: str, hours: float) -> bool:
    """Refuse to let a run start without an approved budget ticket."""
    result = subprocess.run(
        [sys.executable, str(BUDGET_SCRIPT), "check-ticket", "--milestone", milestone, "--hours", str(hours)],
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.returncode != 0:
        if result.stderr.strip():
            print(result.stderr.strip(), file=sys.stderr)
        return False
    return True


def log_run(run_id, start, end, gpu_seconds, milestone, purpose, result, seed, sha) -> None:
    subprocess.run(
        [
            sys.executable, str(BUDGET_SCRIPT), "log",
            "--run-id", run_id,
            "--start", start,
            "--end", end,
            "--gpu-seconds", str(gpu_seconds),
            "--milestone", milestone,
            "--purpose", purpose,
            "--result", result,
            "--seed", str(seed),
            "--git-sha", sha,
        ],
        check=True,
    )


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), capture_output=True, text=True, check=True
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def build_dataloader(cfg: dict):
    target = _import_target(cfg["data"]["target"])
    return target(**cfg["data"].get("kwargs", {}))


def build_model(cfg: dict) -> torch.nn.Module:
    target = _import_target(cfg["model"]["target"])
    return target(**cfg["model"].get("kwargs", {}))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to a YAML config under configs/")
    args = parser.parse_args(argv)

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_cfg = cfg["run"]
    run_id = run_cfg["run_id"]
    milestone = run_cfg["milestone"]
    ticket_hours = run_cfg["ticket_hours"]
    seed = run_cfg.get("seed", 0)
    purpose = run_cfg.get("purpose", "")

    # 1. No ticket, no run. If rejected: abort immediately, nothing started,
    #    nothing logged (there is no run to log).
    if not check_ticket(milestone, ticket_hours):
        print(
            f"ABORTED: budget ticket for milestone '{milestone}' ({ticket_hours}h) was rejected -- "
            "no run started, nothing logged.",
            file=sys.stderr,
        )
        return 1

    seed_everything(seed)
    ddp_ctx = setup_ddp(device_override=run_cfg.get("device"))

    start_ts = time.time()
    start_iso = _iso(start_ts)
    status = "ok"
    result_str = "no result"
    trainer = None
    ckpt_manager = None

    try:
        model = build_model(cfg)
        model = wrap_model(model, ddp_ctx)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["optim"]["lr"],
            weight_decay=cfg["optim"].get("weight_decay", 0.0),
        )
        loss_fn = _import_target(cfg["loss"]["target"])
        trainer_config = TrainerConfig.from_dict(cfg["trainer"])
        trainer = Trainer(model, optimizer, loss_fn, trainer_config, device=ddp_ctx.device)

        ckpt_dir = ROOT / cfg["checkpoint"]["dir"]
        ckpt_manager = CheckpointManager(ckpt_dir)
        resume_state, resume_step = ckpt_manager.load_latest()
        if resume_state is not None:
            trainer.load_state_dict(resume_state)
            print(f"Resumed from checkpoint at step {resume_step}.")

        if is_main_process(ddp_ctx):
            ckpt_manager.register_sigterm_handler(lambda: (trainer.state_dict(), trainer.step))

        dataloader = build_dataloader(cfg)
        max_steps = trainer_config.total_steps
        log_interval = run_cfg.get("log_interval", trainer_config.log_interval)

        def on_step(res):
            if (res["step"] + 1) % log_interval == 0 or res["nan_event"] is not None:
                print(f"step {res['step']} loss {res['loss']:.6f} grad_norm {res['grad_norm']:.4f}")

        trainer.fit(
            dataloader,
            max_steps,
            on_step=on_step,
            checkpoint_manager=ckpt_manager,
            checkpoint_interval=cfg["checkpoint"].get("interval_steps"),
        )
        result_str = f"completed step={trainer.step}, nan_events={len(trainer.nan_events)}"

    except SigtermInterrupt as e:
        status = "sigterm"
        result_str = f"interrupted by SIGTERM at step {e.step}, checkpoint saved"
    except Exception as e:  # noqa: BLE001 -- must still log to the ledger before re-raising
        status = "error"
        result_str = f"aborted with exception: {e!r}"
        raise
    finally:
        cleanup_ddp(ddp_ctx)
        if is_main_process(ddp_ctx) and ckpt_manager is not None:
            ckpt_manager.restore_default_handler()
        end_ts = time.time()
        gpu_seconds = (end_ts - start_ts) * max(ddp_ctx.world_size, 1)
        try:
            log_run(
                run_id=run_id,
                start=start_iso,
                end=_iso(end_ts),
                gpu_seconds=gpu_seconds,
                milestone=milestone,
                purpose=purpose,
                result=f"[{status}] {result_str}",
                seed=seed,
                sha=git_sha(),
            )
        except Exception as log_exc:  # noqa: BLE001
            print(f"WARNING: failed to write budget ledger entry: {log_exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
