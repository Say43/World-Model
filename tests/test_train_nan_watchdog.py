"""NaN watchdog: an artificially injected NaN in the loss (or, via the
lower-level helper, the gradient) must be detected and reported, never
silently masked.

"Pass" means:
  - with nan_watchdog_raise=True (the default), Trainer.train_step raises
    NaNDetected, and exactly one NaNEvent was recorded in
    trainer.nan_events before the exception propagated.
  - with nan_watchdog_raise=False, train_step does NOT raise, but still
    records the NaNEvent in trainer.nan_events, marks the step "skipped",
    and advances the step counter (the run continues, but the event is on
    the record -- it is reported, not hidden).
"""
import pytest
import torch

from src.train.nan_watchdog import NaNDetected, check_finite
from src.train.trainer import Trainer, TrainerConfig
from tests.train_fixtures import build_model


def nan_loss_fn(model, batch):
    """Deliberately injects a NaN into the loss without corrupting the
    model itself, to exercise the "loss" detection path in isolation.
    """
    x, y = batch
    pred = model(x)
    return (pred - y).pow(2).mean() * float("nan")


def test_nan_watchdog_raises_and_records_by_default():
    model = build_model(dim=16, hidden=32, dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = TrainerConfig(warmup_steps=1, total_steps=10, amp=False, nan_watchdog_raise=True)
    trainer = Trainer(model, optimizer, nan_loss_fn, config, device=torch.device("cpu"))
    batch = (torch.randn(4, 16), torch.randn(4, 16))

    with pytest.raises(NaNDetected):
        trainer.train_step(batch)

    assert len(trainer.nan_events) == 1
    assert trainer.nan_events[0].kind == "loss"
    assert "nan" in trainer.nan_events[0].detail


def test_nan_watchdog_reports_without_raising_when_configured_off():
    model = build_model(dim=16, hidden=32, dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = TrainerConfig(warmup_steps=1, total_steps=10, amp=False, nan_watchdog_raise=False)
    trainer = Trainer(model, optimizer, nan_loss_fn, config, device=torch.device("cpu"))
    batch = (torch.randn(4, 16), torch.randn(4, 16))

    result = trainer.train_step(batch)

    assert result["skipped"] is True
    assert result["nan_event"] is not None
    assert len(trainer.nan_events) == 1
    assert trainer.step == 1  # training continues, event is on the record


def test_grad_nan_detection_helper_directly():
    bad_inf = torch.tensor(float("inf"))
    event = check_finite(bad_inf, step=3, kind="grad")
    assert event is not None
    assert event.kind == "grad"
    assert event.step == 3

    bad_nan = torch.tensor(float("nan"))
    event_nan = check_finite(bad_nan, step=4, kind="grad")
    assert event_nan is not None

    good = torch.tensor(1.23)
    assert check_finite(good, step=5, kind="grad") is None
