"""Resume: interrupt at step 100, resume, and the tail of the loss curve
must exactly match an uninterrupted run's steps [100, 150).

"Pass" means: `losses_a[:100] + losses_b == losses_full`, i.e. the loss at
every one of steps 100..149 produced by (checkpoint -> fresh Trainer/
model/optimizer -> load_state_dict -> continue) is bit-identical to the
loss produced at that same step by a Trainer that never stopped. The
resumed run is deliberately re-seeded with a *different* global seed
(999 instead of 7) before construction, so any match can only come from
state actually stored in and restored from the checkpoint -- not from
accidentally re-deriving the same trajectory via seeding. Since the dummy
model contains Dropout, this specifically exercises RNG-state save/
restore: without it, the resumed run's dropout masks diverge from the
uninterrupted run's and losses_b would not match.
"""
import torch

from src.train.checkpoint import CheckpointManager, SigtermInterrupt
from src.train.trainer import Trainer, TrainerConfig
from src.train.utils import seed_everything
from tests.train_fixtures import build_dataloader, build_model, mse_loss_fn


def _make_trainer(seed: int, steps: int) -> Trainer:
    torch.set_num_threads(1)
    seed_everything(seed)
    model = build_model(dim=16, hidden=32, dropout=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = TrainerConfig(warmup_steps=5, total_steps=steps, grad_clip=1.0, ema_decay=0.99, amp=False)
    return Trainer(model, optimizer, mse_loss_fn, config, device=torch.device("cpu"))


def test_resume_matches_uninterrupted_run(tmp_path):
    total_steps = 150
    interrupt_at = 100

    # Reference: a single uninterrupted run.
    trainer_full = _make_trainer(seed=7, steps=total_steps)
    dl_full = build_dataloader(size=2000, dim=16, seed=99, batch_size=8)
    losses_full = []
    trainer_full.fit(dl_full, total_steps, on_step=lambda r: losses_full.append(r["loss"]))
    assert len(losses_full) == total_steps

    # Interrupted run: train to `interrupt_at`, simulate SIGTERM (atomic
    # checkpoint write via handle_termination -- see checkpoint.py's
    # docstring on why tests call this directly rather than sending a real
    # OS signal), then build a brand-new Trainer/model/optimizer and
    # resume purely from the checkpoint.
    ckpt_manager = CheckpointManager(tmp_path / "ckpt")
    trainer_a = _make_trainer(seed=7, steps=total_steps)
    dl_a = build_dataloader(size=2000, dim=16, seed=99, batch_size=8)
    losses_a = []

    def stop_at_interrupt(res):
        losses_a.append(res["loss"])
        if res["step"] + 1 == interrupt_at:
            ckpt_manager.handle_termination(lambda: (trainer_a.state_dict(), trainer_a.step))

    try:
        trainer_a.fit(dl_a, total_steps, on_step=stop_at_interrupt)
        assert False, "expected SigtermInterrupt to abort the run at step 100"
    except SigtermInterrupt as e:
        assert e.step == interrupt_at

    assert trainer_a.step == interrupt_at
    assert len(losses_a) == interrupt_at

    # Fresh process-equivalent: different seed, new objects entirely.
    # Anything that matches from here on must come from the checkpoint.
    trainer_b = _make_trainer(seed=999, steps=total_steps)
    state, loaded_step = ckpt_manager.load_latest()
    assert loaded_step == interrupt_at
    trainer_b.load_state_dict(state)
    assert trainer_b.step == interrupt_at

    dl_b = build_dataloader(size=2000, dim=16, seed=99, batch_size=8)
    losses_b = []
    trainer_b.fit(dl_b, total_steps, on_step=lambda r: losses_b.append(r["loss"]))
    assert len(losses_b) == total_steps - interrupt_at

    resumed_curve = losses_a + losses_b
    assert resumed_curve == losses_full, "resume must reproduce the interrupted run's exact trajectory"


def test_resume_without_rng_restore_diverges(tmp_path):
    """Negative control: prove the RNG state is actually load-bearing. If we
    restore everything BUT the RNG state, the resumed run's dropout masks
    differ from the uninterrupted run's and the tail of the loss curve
    must NOT match -- otherwise the positive test above would be
    vacuously true (e.g. if this dummy model had no real randomness).
    """
    total_steps = 150
    interrupt_at = 100

    trainer_full = _make_trainer(seed=7, steps=total_steps)
    dl_full = build_dataloader(size=2000, dim=16, seed=99, batch_size=8)
    losses_full = []
    trainer_full.fit(dl_full, total_steps, on_step=lambda r: losses_full.append(r["loss"]))

    ckpt_manager = CheckpointManager(tmp_path / "ckpt")
    trainer_a = _make_trainer(seed=7, steps=total_steps)
    dl_a = build_dataloader(size=2000, dim=16, seed=99, batch_size=8)

    def stop_at_interrupt(res):
        if res["step"] + 1 == interrupt_at:
            ckpt_manager.handle_termination(lambda: (trainer_a.state_dict(), trainer_a.step))

    try:
        trainer_a.fit(dl_a, total_steps, on_step=stop_at_interrupt)
    except SigtermInterrupt:
        pass

    trainer_b = _make_trainer(seed=999, steps=total_steps)
    state, _ = ckpt_manager.load_latest()
    # Deliberately drop the RNG state before loading the rest.
    state = dict(state)
    del state["rng"]
    trainer_b.step = state["step"]
    trainer_b.model.load_state_dict(state["model"])
    trainer_b.optimizer.load_state_dict(state["optimizer"])
    trainer_b.scaler.load_state_dict(state["scaler"])
    trainer_b.scheduler.load_state_dict(state["scheduler"])
    trainer_b.ema.load_state_dict(state["ema"])
    # Note: RNG intentionally NOT restored here -- torch's global RNG is
    # left wherever seed_everything(999) put it.

    dl_b = build_dataloader(size=2000, dim=16, seed=99, batch_size=8)
    losses_b = []
    trainer_b.fit(dl_b, total_steps, on_step=lambda r: losses_b.append(r["loss"]))

    assert losses_b != losses_full[interrupt_at:], (
        "without RNG restore the resumed dropout masks should differ, "
        "proving RNG state is what makes the real resume test meaningful"
    )
