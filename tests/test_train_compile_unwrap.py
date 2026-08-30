"""Regression test for the M1-gate checkpoint bug: a checkpoint must always
store the uncompiled module's state, never a torch.compile wrapper's.

Doesn't need real torch.compile (CPU-only test environment; the bug was
observed on Kaggle's CUDA/compiled path) -- a minimal fake wrapper with the
same `_orig_mod` attribute torch.compile's OptimizedModule exposes is
enough to exercise unwrap_compiled and Trainer's use of it.
"""
import torch
import torch.nn as nn

from src.train.trainer import Trainer, TrainerConfig
from src.train.utils import unwrap_compiled
from tests.train_fixtures import DeterministicDataset, TinyModel


class _FakeCompiledWrapper(nn.Module):
    """Mimics torch.compile's OptimizedModule just enough for this test:
    wraps a module, forwards to it, and exposes `_orig_mod`. Deliberately
    does NOT define its own parameters/buffers matching the wrapped
    module's names 1:1 in .state_dict() -- like the real bug, its
    state_dict() is not trustworthy for checkpointing.
    """

    def __init__(self, wrapped: nn.Module):
        super().__init__()
        self._orig_mod = wrapped

    def forward(self, *args, **kwargs):
        return self._orig_mod(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        # A broken/incomplete state_dict, standing in for the real
        # torch.compile quirk (missing frame_pos_embed-equivalent, or a
        # shape mismatch) -- correctness here depends on this NEVER being
        # consulted by Trainer, not on it being self-consistent.
        return {"bogus_key": torch.zeros(1)}


def test_unwrap_compiled_returns_orig_mod_when_present():
    real = TinyModel(dim=4, hidden=8)
    wrapper = _FakeCompiledWrapper(real)
    assert unwrap_compiled(wrapper) is real


def test_unwrap_compiled_is_noop_for_uncompiled_model():
    real = TinyModel(dim=4, hidden=8)
    assert unwrap_compiled(real) is real


def test_trainer_checkpoint_uses_uncompiled_state_dict():
    real = TinyModel(dim=4, hidden=8)
    wrapped = _FakeCompiledWrapper(real)
    optimizer = torch.optim.AdamW(real.parameters(), lr=1e-3)
    config = TrainerConfig(total_steps=10, warmup_steps=1, amp=False)

    def loss_fn(model, batch):
        x, y = batch
        return torch.nn.functional.mse_loss(model(x), y)

    trainer = Trainer(wrapped, optimizer, loss_fn, config, device=torch.device("cpu"))
    state = trainer.state_dict()

    assert state["model"].keys() == real.state_dict().keys()
    assert "bogus_key" not in state["model"]
    assert state["ema"].keys() == real.state_dict().keys()
