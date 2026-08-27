"""Determinism: two runs with the same seed must produce bitwise-identical
loss curves.

"Pass" means: run_once(seed=42) executed twice yields two Python lists of
per-step float losses that compare equal element-by-element with `==`
(exact float/bit equality, not `pytest.approx`). torch.set_num_threads(1)
is required for this to hold on CPU -- with more than one thread, MKL/
OpenMP can reduce matmuls in a different order between runs and perturb
the last bit of a float32 sum.
"""
import torch

from src.train.trainer import Trainer, TrainerConfig
from src.train.utils import seed_everything
from tests.train_fixtures import build_dataloader, build_model, mse_loss_fn


def run_once(seed: int, steps: int = 60):
    torch.set_num_threads(1)
    seed_everything(seed)
    model = build_model(dim=16, hidden=32, dropout=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = TrainerConfig(warmup_steps=5, total_steps=steps, grad_clip=1.0, ema_decay=0.99, amp=False)
    trainer = Trainer(model, optimizer, mse_loss_fn, config, device=torch.device("cpu"))
    dataloader = build_dataloader(size=2000, dim=16, seed=123, batch_size=8)

    losses = []
    trainer.fit(dataloader, steps, on_step=lambda r: losses.append(r["loss"]))
    return losses


def test_determinism_bitwise_identical_loss_curves():
    losses_a = run_once(seed=42)
    losses_b = run_once(seed=42)

    assert len(losses_a) == 60
    assert len(losses_b) == 60
    assert losses_a == losses_b, "identical seed must produce bitwise-identical loss curves"
    assert all(l == l for l in losses_a), "no NaNs should appear in this clean run"


def test_determinism_different_seeds_diverge():
    """Sanity check on the harness itself: different seeds must NOT
    accidentally collide, otherwise the equality test above would be
    vacuous (e.g. if seeding silently failed to reach model init).
    """
    losses_a = run_once(seed=1)
    losses_b = run_once(seed=2)
    assert losses_a != losses_b
