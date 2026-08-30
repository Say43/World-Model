"""Model-agnostic trainer: fp16 AMP + GradScaler, NaN watchdog, grad
clipping, warmup+cosine LR, EMA in fp32 on CPU.

Deliberately knows nothing about DiT / the model's architecture: it
consumes any nn.Module, any torch.optim.Optimizer already constructed
over that module's parameters, a `loss_fn(model, batch) -> scalar loss
tensor` callable, and any torch DataLoader. model-architect owns what
`loss_fn` and the model actually compute; this module owns the loop
around them.

Resume semantics: one Trainer.step == one batch consumed (no gradient
accumulation in this version). `fit()` re-creates its dataloader iterator
and skips `self.step` batches before resuming, so as long as the
dataloader is deterministic and shuffle-free (see tests/train_fixtures.py
for the pattern used in tests), resuming from a mid-run checkpoint
reproduces the exact batch sequence the interrupted run would have seen.
"""
import logging
from dataclasses import dataclass, fields
from typing import Callable, Optional

import torch
import torch.nn as nn

from .ema import EMA
from .nan_watchdog import NaNEvent, NaNDetected, check_finite
from .scheduler import WarmupScheduler
from .utils import capture_rng_state, restore_rng_state, unwrap_compiled

logger = logging.getLogger("nanowm.train.trainer")


def _move_to_device(batch, device: torch.device):
    """Recursively move a batch (tensor, or list/tuple/dict of tensors) to
    `device`. Non-tensor leaves are passed through unchanged. Keeps the
    trainer usable with arbitrary batch structures without knowing
    anything about what a "batch" means for a given model.
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: _move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        moved = [_move_to_device(v, device) for v in batch]
        return type(batch)(moved) if not isinstance(batch, tuple) else tuple(moved)
    return batch


@dataclass
class TrainerConfig:
    warmup_steps: int = 10
    total_steps: int = 1000
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    amp: bool = True
    nan_watchdog_raise: bool = True
    log_interval: int = 50

    @classmethod
    def from_dict(cls, d: dict) -> "TrainerConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        config: TrainerConfig,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.config = config
        self.device = device if device is not None else next(model.parameters()).device
        # fp16 AMP is only meaningful on CUDA (the T4 target). On CPU
        # (all tests, this project's CI) autocast/GradScaler are disabled
        # so the code path is exercised without changing numerics needed
        # for bitwise determinism tests.
        self.amp_enabled = bool(config.amp and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler(device="cuda", enabled=self.amp_enabled)
        self.scheduler = WarmupScheduler(
            optimizer,
            warmup_steps=config.warmup_steps,
            total_steps=config.total_steps,
            min_lr_ratio=config.min_lr_ratio,
        )
        self.ema = EMA(unwrap_compiled(model), decay=config.ema_decay)
        self.step = 0
        self.nan_events: list[NaNEvent] = []

    def train_step(self, batch) -> dict:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        batch = _move_to_device(batch, self.device)

        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp_enabled):
            loss = self.loss_fn(self.model, batch)

        event = check_finite(loss, self.step, "loss")
        grad_norm_value = float("nan")
        skip_step = False

        if event is None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            grad_norm_value = float(grad_norm)
            event = check_finite(grad_norm, self.step, "grad")

        if event is not None:
            self._report_nan(event)
            self.optimizer.zero_grad(set_to_none=True)
            skip_step = True
            if self.config.nan_watchdog_raise:
                raise NaNDetected(event)

        if not skip_step:
            # GradScaler can still veto the step internally: if it found
            # inf/nan while unscaling it silently no-ops `step` and lowers the
            # scale in `update`. Our own checks above catch nearly all of that
            # (a non-finite grad makes clip_grad_norm_ non-finite), but the
            # scaler is the authority on its own decision, so ask it rather
            # than assume. A drop in scale means the step did not happen --
            # without this the run would count a skipped step as a real one
            # and the loss curve would quietly stall.
            scale_before = self.scaler.get_scale() if self.amp_enabled else None
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if scale_before is not None and self.scaler.get_scale() < scale_before:
                event = NaNEvent(
                    step=self.step,
                    kind="grad",
                    detail=(
                        f"GradScaler skipped the optimizer step (scale "
                        f"{scale_before} -> {self.scaler.get_scale()})"
                    ),
                )
                self._report_nan(event)
                skip_step = True
                if self.config.nan_watchdog_raise:
                    raise NaNDetected(event)
            else:
                self.scheduler.step()
                self.ema.update(unwrap_compiled(self.model))

        loss_detached = loss.detach()
        loss_value = float(loss_detached.item()) if torch.isfinite(loss_detached).all() else float("nan")

        result = {
            "step": self.step,
            "loss": loss_value,
            "grad_norm": grad_norm_value,
            "nan_event": event,
            "skipped": skip_step,
        }
        self.step += 1
        return result

    def _report_nan(self, event: NaNEvent) -> None:
        self.nan_events.append(event)
        logger.error("NaN watchdog triggered: %s", event)

    def fit(
        self,
        dataloader,
        max_steps: int,
        on_step: Optional[Callable[[dict], None]] = None,
        checkpoint_manager=None,
        checkpoint_interval: Optional[int] = None,
        skip_batches: Optional[int] = None,
    ) -> None:
        it = iter(dataloader)
        skip = self.step if skip_batches is None else skip_batches
        for _ in range(skip):
            try:
                next(it)
            except StopIteration:
                it = iter(dataloader)
                next(it)

        while self.step < max_steps:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dataloader)
                batch = next(it)

            result = self.train_step(batch)

            if on_step is not None:
                on_step(result)

            if checkpoint_manager is not None and checkpoint_interval and self.step % checkpoint_interval == 0:
                checkpoint_manager.save(self.state_dict(), self.step)

    def state_dict(self) -> dict:
        return {
            "step": self.step,
            # Always the uncompiled module's state_dict -- see
            # unwrap_compiled's docstring for why a checkpoint must not
            # depend on whether self.model happens to be torch.compile-wrapped.
            "model": unwrap_compiled(self.model).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "ema": self.ema.state_dict(),
            "rng": capture_rng_state(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.step = state["step"]
        unwrap_compiled(self.model).load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state["scaler"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.ema.load_state_dict(state["ema"])
        restore_rng_state(state["rng"])
