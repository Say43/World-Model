"""Regression test: a gradient-level NaN with nan_watchdog_raise=False must
not leave GradScaler mid-protocol for the next step.

CUDA-only: GradScaler's "unscale_() already called" protocol check is
entirely bypassed when enabled=False (verified directly -- a second
unscale_() call raises nothing on a disabled scaler), which is exactly
the CPU test configuration every other trainer test uses. This bug is
real only when AMP is actually active, so it can only be exercised on
CUDA; skipped (not faked) where no GPU is available.

Found via scripts/run_m2_lr_sweep.py, the first real use of
nan_watchdog_raise=False across more than one step: a divergent LR
produced a grad-level NaN, and the *next* sweep step crashed with
RuntimeError("unscale_() has already been called on this optimizer since
the last update()."). Every M1 training run had nan_watchdog_raise=True,
which raises immediately and never reaches a next step, so this path was
never exercised until the sweep needed to survive a bad LR and keep going.
"""
import pytest
import torch
import torch.nn as nn

from src.train.trainer import Trainer, TrainerConfig

class _NaNGradOnFirstCall(torch.autograd.Function):
    """Forward returns the input unchanged; backward returns a NaN
    gradient exactly once (controlled by a class-level flag), then a
    normal gradient afterward -- deterministically exercises "loss is
    finite, gradient is not" (only reachable AFTER scaler.unscale_() has
    already run) on step 1, and a genuinely clean step 2.
    """

    inject_nan = True

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        if _NaNGradOnFirstCall.inject_nan:
            _NaNGradOnFirstCall.inject_nan = False
            return torch.full_like(grad_output, float("nan"))
        return grad_output


def _loss_fn(model, batch):
    # Applied to the model's OUTPUT, not its input: `batch` is a leaf
    # tensor with requires_grad=False, so a custom Function wrapped around
    # it would never have its backward() invoked at all (autograd only
    # calls backward along paths that actually need a gradient) -- the
    # first version of this test silently injected nothing and the
    # "skipped" assertion just failed outright, which is what caught it.
    pred = model(batch)
    pred = _NaNGradOnFirstCall.apply(pred)
    return pred.pow(2).mean()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GradScaler's protocol check is a no-op when disabled (CPU/amp=False)",
)
def test_survives_a_grad_level_nan_and_continues_training_on_cuda():
    device = torch.device("cuda")
    _NaNGradOnFirstCall.inject_nan = True
    model = nn.Linear(8, 8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = TrainerConfig(warmup_steps=1, total_steps=10, amp=True, nan_watchdog_raise=False)
    trainer = Trainer(model, optimizer, _loss_fn, config, device=device)
    batch = torch.randn(4, 8, device=device)

    bad_result = trainer.train_step(batch)
    assert bad_result["skipped"] is True
    assert bad_result["nan_event"] is not None
    assert bad_result["nan_event"].kind == "grad"

    # The real bug: this next call used to raise RuntimeError from
    # GradScaler's own protocol check ("unscale_() has already been
    # called..."), not from our NaN watchdog -- a confusing, unrelated
    # crash for what should just be "the run survives one bad step."
    good_result = trainer.train_step(batch)
    assert good_result["skipped"] is False
    assert good_result["nan_event"] is None
    assert trainer.step == 2


def test_disabled_scaler_does_not_apply_a_skipped_nan_gradient():
    """A disabled GradScaler passes step() straight to the optimizer.

    The CUDA protocol fix must therefore finalize a bad unscaled step only
    when AMP is enabled; doing so on CPU would corrupt weights while still
    reporting the step as skipped.
    """
    device = torch.device("cpu")
    _NaNGradOnFirstCall.inject_nan = True
    model = nn.Linear(8, 8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = TrainerConfig(warmup_steps=1, total_steps=10, amp=False, nan_watchdog_raise=False)
    trainer = Trainer(model, optimizer, _loss_fn, config, device=device)
    batch = torch.randn(4, 8, device=device)
    before = {name: param.detach().clone() for name, param in model.named_parameters()}

    bad_result = trainer.train_step(batch)

    assert bad_result["skipped"] is True
    for name, param in model.named_parameters():
        assert torch.equal(param, before[name]), name
        assert torch.isfinite(param).all(), name
