"""Step-based LR schedule: linear warmup, then cosine decay to a floor.

Deliberately not a torch.optim.lr_scheduler subclass -- those have version-
dependent state_dict quirks. This is a tiny, explicit, checkpoint-friendly
scheduler: state is just an integer step count plus the base LRs it read
from the optimizer at construction time.
"""
import math

import torch


class WarmupScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
    ):
        self.optimizer = optimizer
        self.warmup_steps = max(0, int(warmup_steps))
        self.total_steps = max(1, int(total_steps))
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.last_step = 0
        self._apply(0)

    def _lr_scale(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def _apply(self, step: int) -> None:
        scale = self._lr_scale(step)
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base * scale

    def step(self) -> None:
        self.last_step += 1
        self._apply(self.last_step)

    def state_dict(self) -> dict:
        return {"last_step": self.last_step, "base_lrs": list(self.base_lrs)}

    def load_state_dict(self, state: dict) -> None:
        self.last_step = state["last_step"]
        self.base_lrs = list(state["base_lrs"])
        self._apply(self.last_step)
