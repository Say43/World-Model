"""Regression test: a run that completes normally must leave behind a
usable checkpoint, even when total_steps isn't a multiple of
checkpoint_interval.

Real failure: an M1 Kaggle run trained cleanly to completion (step=8000,
zero NaN events) with checkpoint_interval=25000. The periodic save inside
fit()'s loop only fires when `step % checkpoint_interval == 0`, which
8000 never satisfies -- so nothing was ever written, and the ephemeral
Kaggle session ended with no checkpoint to evaluate. An entire successful
training run's compute was lost.
"""
import torch

from src.train.checkpoint import CheckpointManager
from src.train.trainer import Trainer, TrainerConfig
from src.train.utils import seed_everything
from tests.train_fixtures import build_dataloader, build_model, mse_loss_fn


def test_fit_saves_a_checkpoint_on_normal_completion_even_off_interval(tmp_path):
    seed_everything(0)
    model = build_model(dim=16, hidden=32, dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = TrainerConfig(warmup_steps=1, total_steps=8, amp=False)
    trainer = Trainer(model, optimizer, mse_loss_fn, config, device=torch.device("cpu"))
    dl = build_dataloader(size=2000, dim=16, seed=0, batch_size=8)

    ckpt_manager = CheckpointManager(tmp_path / "ckpt")
    # 8 total_steps, interval=25 -- the interval is never reached, matching
    # the real failure's total_steps=8000/interval=25000 shape.
    trainer.fit(dl, max_steps=8, checkpoint_manager=ckpt_manager, checkpoint_interval=25)

    assert (tmp_path / "ckpt" / "latest.json").exists(), (
        "no checkpoint was left behind after a normally-completed run"
    )
    state, step = ckpt_manager.load_latest()
    assert step == 8
    assert state is not None


def test_fit_does_not_double_save_when_last_step_hits_the_interval(tmp_path):
    """When the final step *does* land exactly on checkpoint_interval, the
    periodic save and the completion save both fire -- confirm this is
    harmless (same step, checkpoint still loads cleanly) rather than
    asserting call counts, since a redundant identical write is acceptable
    and simpler than adding bookkeeping to suppress it.
    """
    seed_everything(0)
    model = build_model(dim=16, hidden=32, dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = TrainerConfig(warmup_steps=1, total_steps=10, amp=False)
    trainer = Trainer(model, optimizer, mse_loss_fn, config, device=torch.device("cpu"))
    dl = build_dataloader(size=2000, dim=16, seed=0, batch_size=8)

    ckpt_manager = CheckpointManager(tmp_path / "ckpt")
    trainer.fit(dl, max_steps=10, checkpoint_manager=ckpt_manager, checkpoint_interval=10)

    _, step = ckpt_manager.load_latest()
    assert step == 10
