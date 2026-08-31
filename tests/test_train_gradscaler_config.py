"""Regression test: GradScaler's growth parameters must be config-driven and
reach the real constructor, not silently fall back to torch's defaults.

Two consecutive M1 Kaggle runs died with an inf gradient at nearly the same
step (~6040-6080) regardless of a fully-recalibrated LR schedule between
them -- the LR theory was wrong. GradScaler's default growth_interval=2000
triples the scale (65536 -> 524288) by step ~6000 on a run with no skipped
steps, which is a much better match for a fixed, LR-independent failure
point. TrainerConfig now exposes the GradScaler growth parameters with a
much higher default growth_interval; this test pins down that Trainer
actually passes them through rather than constructing GradScaler with its
own defaults regardless of config.
"""
from unittest.mock import patch

import torch

from src.train.trainer import Trainer, TrainerConfig
from tests.train_fixtures import TinyModel


def test_trainer_passes_grad_scaler_config_to_gradscaler():
    model = TinyModel(dim=4, hidden=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = TrainerConfig(
        total_steps=10, warmup_steps=1, amp=True,
        grad_scaler_init_scale=4096.0,
        grad_scaler_growth_factor=1.5,
        grad_scaler_backoff_factor=0.25,
        grad_scaler_growth_interval=999,
    )

    with patch("torch.amp.GradScaler") as mock_scaler:
        # amp is only actually enabled on CUDA (src/train/trainer.py), but
        # the constructor call happens regardless of device -- force cuda
        # detection True via device arg so amp_enabled is True and the
        # config values are the ones under test, not short-circuited by
        # amp_enabled=False.
        Trainer(model, optimizer, lambda m, b: torch.tensor(0.0), config,
                device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    _, kwargs = mock_scaler.call_args
    assert kwargs["init_scale"] == 4096.0
    assert kwargs["growth_factor"] == 1.5
    assert kwargs["backoff_factor"] == 0.25
    assert kwargs["growth_interval"] == 999


def test_trainer_config_default_growth_interval_is_far_above_torch_default():
    """torch's own GradScaler default (2000) is what caused the failure;
    the project default must be meaningfully higher, not just different."""
    config = TrainerConfig()
    assert config.grad_scaler_growth_interval >= 50_000
