"""Seeding and RNG-state capture/restore.

Every source of randomness that can affect a training trajectory must be
seeded and its state must be checkpoint-able: Python's `random`, numpy,
torch's CPU generator, and torch's CUDA generators (per-device). The
resume test in tests/test_train_resume.py is only meaningful because the
dummy model contains Dropout: without restoring the exact RNG state at the
interruption point, a resumed run would silently diverge from the
uninterrupted one due to different dropout masks, even though every other
piece of state (weights, optimizer moments, step count) matched.
"""
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed every RNG this trainer touches. Call once, before model/optimizer
    construction, so weight init and any data setup are reproducible too.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict:
    """Snapshot all RNG state for inclusion in a checkpoint."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    """Restore RNG state previously captured by capture_rng_state().

    Restoring CUDA state is skipped (with no error) if the checkpoint was
    taken on a CUDA-enabled machine but is being resumed on CPU-only, or
    vice versa -- this keeps CPU-only tests able to load checkpoints
    produced (in principle) on GPU-equipped Kaggle sessions.
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def unwrap_compiled(model):
    """Return the original nn.Module underneath compile and DDP wrappers.

    Checkpoints must always be built from the uncompiled module: whether a
    model happens to be compiled is a runtime performance decision (the
    trainer compiles by default, see scripts/train.py), not something a
    checkpoint's *contents* should depend on. Observed in practice on a
    trained checkpoint's EMA shadow -- 'frame_pos_embed', a bare
    nn.Parameter assigned directly on CausalDiT rather than living inside a
    submodule, was dropped from a compiled OptimizedModule's .state_dict(),
    and other keys came back with mismatched shapes. Rather than have every
    checkpoint consumer work around torch.compile's .state_dict() quirks
    individually, this is applied once at the point state_dict()/
    load_state_dict() touch the model.
    """
    while hasattr(model, "_orig_mod"):
        model = model._orig_mod
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    return model
