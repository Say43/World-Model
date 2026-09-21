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
    optim:      lr, weight_decay; optional optimizer: adamw (default) | muon,
                and muon_lr when optimizer is muon
    repa:       optional -- feature_dim, weight, align_layer (null = depth//3);
                requires a loss target that consumes the DINOv2 features and a
                dataloader built with with_dinov2: true
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
from src.train.utils import seed_everything, unwrap_compiled

BUDGET_SCRIPT = ROOT / "scripts" / "budget.py"


class TicketExceeded(RuntimeError):
    """Raised when a run outlives the budget ticket it was approved for."""


class ThroughputCollapse(RuntimeError):
    """Raised when the step rate falls far below the profiled rate."""


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

        # REPA's head must exist before DDP wrapping and before the optimizer
        # is built: DDP inspects the parameter set once, at construction, and
        # a head installed afterwards would train unsynchronized across ranks
        # and be missing from the optimizer entirely.
        repa_cfg = cfg.get("repa")
        if repa_cfg:
            from src.train.repa import attach_repa_head
            head = attach_repa_head(
                model,
                feature_dim=repa_cfg["feature_dim"],
                align_layer=repa_cfg.get("align_layer"),
            )
            model.repa_weight = float(repa_cfg["weight"])
            print(f"REPA: align_layer={model.repa_align_layer}, weight={model.repa_weight}, "
                  f"head_params={sum(p.numel() for p in head.parameters())}")

        model = wrap_model(model, ddp_ctx)
        # Optimizer param groups are built from the model BEFORE
        # torch.compile wraps it, same reasoning as src/train/utils.py's
        # unwrap_compiled: torch.compile's OptimizedModule previously
        # dropped/misshaped a bare top-level nn.Parameter in .state_dict()
        # (frame_pos_embed -- see CLAUDE.md's decision log), so anything
        # that inspects named_parameters() on a compiled model is not
        # trusted here even though .parameters() iteration specifically
        # hasn't shown that bug. The parameter tensor objects are identical
        # before and after compiling (compile wraps the module, it doesn't
        # replace its parameters), so gradients from the compiled forward/
        # backward still land on these exact objects.
        raw_model_for_optim = unwrap_compiled(model)
        # Measured on 2xT4 (CLAUDE.md "Measured on 2x Tesla T4"): compile
        # gives 1.2x-2.6x, not the 10-25% the pre-hardware estimate assumed,
        # and cuts peak VRAM by up to 40%. It compiled cleanly in every
        # configuration profiled, so default it on for CUDA runs; still
        # config-gated (trainer.compile: false) in case a future
        # architecture change hits a shape torch.compile can't handle, and
        # a compile failure degrades to eager rather than aborting the run.
        if ddp_ctx.device.type == "cuda" and cfg["trainer"].get("compile", True):
            # M4 part 1 (2026-09-19) died after 3.4 GPU-hours of an 8-hour
            # run: throughput collapsed right after the first checkpoint
            # save and ~12.5 GB of GPU memory outside PyTorch's allocator
            # piled up until OOM. CUDA-graph re-recording under
            # "reduce-overhead" is the prime suspect (graph executables are
            # exactly such non-allocator memory); the M3 arms never
            # checkpointed and never hit it. The mode is therefore
            # config-gated so a long checkpointed run can opt out of
            # cudagraphs ("default") without losing kernel fusion.
            compile_mode = cfg["trainer"].get("compile_mode", "reduce-overhead")
            try:
                model = torch.compile(model, mode=compile_mode, dynamic=False)
                print(f"torch.compile: enabled (mode={compile_mode})")
            except Exception as exc:  # noqa: BLE001
                print(f"torch.compile failed, continuing eager: {exc!r}", file=sys.stderr)

        mup_cfg = cfg.get("mup")
        optimizer_name = cfg["optim"].get("optimizer", "adamw")
        if optimizer_name == "muon":
            # Not combined with muP on purpose. muP's LR scaling rules are
            # derived for Adam-family per-coordinate updates; Muon's update
            # is orthogonalized, so its width scaling is a different (and
            # unsettled) question. Silently composing the two would make
            # M3's optimizer ablation measure an untested interaction rather
            # than Muon. Refuse instead of guessing (CLAUDE.md: "Bei
            # Unklarheit ueber eine Designentscheidung: fragen, nicht raten").
            if mup_cfg:
                raise ValueError(
                    "optim.optimizer=muon together with a mup block is not supported: "
                    "muP's LR scaling is derived for Adam-family updates and its "
                    "interaction with Muon's orthogonalized update is untested. "
                    "Pick one."
                )
            from src.train.muon import MuonWithAuxAdamW
            optimizer = MuonWithAuxAdamW(
                raw_model_for_optim,
                muon_lr=cfg["optim"]["muon_lr"],
                adamw_lr=cfg["optim"]["lr"],
                weight_decay=cfg["optim"].get("weight_decay", 0.0),
            )
            counts = {g["algorithm"]: sum(p.numel() for p in g["params"])
                      for g in optimizer.param_groups}
            print(f"Muon: muon_lr={cfg['optim']['muon_lr']}, adamw_lr={cfg['optim']['lr']}, "
                  f"params muon={counts['muon']:,} adamw={counts['adamw']:,}")
        elif optimizer_name != "adamw":
            raise ValueError(f"unknown optim.optimizer: {optimizer_name!r} (expected adamw or muon)")
        elif mup_cfg:
            from src.train.mup import build_mup_optimizer
            optimizer = build_mup_optimizer(
                raw_model_for_optim,
                raw_model_for_optim.config,
                base_dim=mup_cfg["base_dim"],
                base_lr=cfg["optim"]["lr"],
                weight_decay=cfg["optim"].get("weight_decay", 0.0),
            )
            print(f"muP optimizer: base_dim={mup_cfg['base_dim']}, "
                  f"width_mult={raw_model_for_optim.config.dim / mup_cfg['base_dim']:.3f}")
        else:
            optimizer = torch.optim.AdamW(
                raw_model_for_optim.parameters(),
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

        # "A run that exceeds its ticket is aborted. No extensions without a
        # new ticket." (CLAUDE.md §budget) Checking the ticket only at startup
        # enforces nothing -- the run could overshoot arbitrarily and the
        # ledger would just record the overrun after the fact, with the budget
        # already spent. ticket_hours is GPU-hours, so the wall-clock deadline
        # is that divided by world_size.
        deadline_wall_seconds = ticket_hours * 3600.0 / max(ddp_ctx.world_size, 1)

        # Throughput watchdog: a run whose step rate falls far below what
        # its profile measured is not training, it is dying slowly (M4 part
        # 1 spent three hours that way). Abort early, checkpoint, and let
        # the ledger say so, rather than burning the rest of the ticket.
        expected_sps = run_cfg.get("expected_steps_per_second")
        collapse_ratio = float(run_cfg.get("throughput_collapse_ratio", 0.3))
        window = {"t": time.time(), "step": 0}

        def on_step(res):
            step_now = res["step"] + 1
            if step_now % log_interval == 0 or res["nan_event"] is not None:
                now = time.time()
                sps = (step_now - window["step"]) / max(now - window["t"], 1e-9)
                window.update(t=now, step=step_now)
                mem = ""
                if ddp_ctx.device.type == "cuda":
                    free, total = torch.cuda.mem_get_info()
                    mem = (f" gpu_used {(total - free) / 2**30:.2f}GiB"
                           f" torch_reserved {torch.cuda.memory_reserved() / 2**30:.2f}GiB")
                print(f"step {res['step']} loss {res['loss']:.6f} grad_norm {res['grad_norm']:.4f}"
                      f" {sps:.2f} steps/s{mem}", flush=True)
                if (expected_sps and step_now % log_interval == 0 and step_now > log_interval
                        and sps < collapse_ratio * float(expected_sps)):
                    raise ThroughputCollapse(
                        f"throughput {sps:.2f} steps/s at step {res['step']} is below "
                        f"{collapse_ratio:.0%} of the profiled {float(expected_sps):.2f} steps/s"
                        f"{mem}; aborting and checkpointing rather than spending the ticket"
                    )
            elapsed = time.time() - start_ts
            if elapsed > deadline_wall_seconds:
                raise TicketExceeded(
                    f"run exceeded its {ticket_hours}h GPU-hour ticket at step "
                    f"{res['step']} ({elapsed / 60.0:.1f} wall-clock minutes on "
                    f"{ddp_ctx.world_size} device(s)); aborting per budget rules"
                )

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
    except (TicketExceeded, ThroughputCollapse) as e:
        # Not a crash: the budget rule (or its throughput cousin) working as
        # designed. Save what we have so the spent GPU-hours are not simply
        # lost, then record it honestly in the ledger.
        status = "ticket_exceeded" if isinstance(e, TicketExceeded) else "throughput_collapse"
        step_now = trainer.step if trainer is not None else -1
        if ckpt_manager is not None and trainer is not None:
            try:
                ckpt_manager.save(trainer.state_dict(), step_now)
            except Exception as save_exc:  # noqa: BLE001
                print(f"WARNING: could not checkpoint on ticket abort: {save_exc}", file=sys.stderr)
        result_str = f"{e}"
        print(f"ABORTED: {e}", file=sys.stderr)
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

    # Non-zero on a ticket abort so a driving notebook or shell script can tell
    # a completed run from one the budget rules cut short.
    return 2 if status == "ticket_exceeded" else 3 if status == "throughput_collapse" else 0


if __name__ == "__main__":
    sys.exit(main())
