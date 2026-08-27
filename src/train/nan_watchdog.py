"""NaN/Inf detection that reports, never silently masks.

The failure mode this guards against: GradScaler's normal behavior on an
inf/nan gradient is to silently skip the optimizer step and reduce the
scale factor -- by design, so occasional fp16 overflow doesn't crash a
run. That's fine for a rare, self-healing overflow, but it means a *real*
instability (loss diverging, a bad batch, a data bug) can hide inside
"just another skipped step" with no visible signal. This module makes the
skip loud: every detected NaN/Inf is turned into a NaNEvent, logged at
ERROR level, and appended to Trainer.nan_events, and (by default) raised
as an exception so a training run stops rather than limping on silently.
"""
import logging
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger("nanowm.train.nan_watchdog")


@dataclass
class NaNEvent:
    step: int
    kind: str  # "loss" or "grad"
    detail: str


class NaNDetected(RuntimeError):
    """Raised when the watchdog is configured to abort on detection."""

    def __init__(self, event: NaNEvent):
        super().__init__(f"NaN/Inf detected at step {event.step} in {event.kind}: {event.detail}")
        self.event = event


def check_finite(tensor: Optional[torch.Tensor], step: int, kind: str) -> Optional[NaNEvent]:
    """Return a NaNEvent if `tensor` contains any non-finite value, else None."""
    if tensor is None:
        return None
    val = tensor.detach()
    if not torch.isfinite(val).all():
        bad = "nan" if torch.isnan(val).any() else "inf"
        sample = val.flatten()[:4].tolist() if val.numel() else []
        return NaNEvent(step=step, kind=kind, detail=f"{kind} contains {bad}; sample={sample}")
    return None
