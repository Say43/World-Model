"""Tests for src/eval/sampler.py's Euler integration, using the tiny CPU
model from tests/model_fixtures.py (no AE, no checkpoint needed -- just the
ODE loop itself)."""
import torch

from src.eval.sampler import sample_frames
from tests.model_fixtures import break_zero_init, build_model, random_batch, tiny_config


def test_sample_frames_shape():
    cfg = tiny_config()
    model = break_zero_init(build_model(cfg))
    _, poses, intrinsics, _ = random_batch(cfg, batch_size=2)
    out = sample_frames(model, poses, intrinsics, num_steps=4)
    assert out.shape == (2, cfg.context_length, cfg.tokens_per_frame, cfg.latent_channels)
    assert torch.isfinite(out).all()


def test_sample_frames_deterministic_with_generator():
    cfg = tiny_config()
    model = break_zero_init(build_model(cfg))
    _, poses, intrinsics, _ = random_batch(cfg, batch_size=1)
    a = sample_frames(model, poses, intrinsics, num_steps=4, generator=torch.Generator().manual_seed(0))
    b = sample_frames(model, poses, intrinsics, num_steps=4, generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(a, b)


def test_sample_frames_different_seed_differs():
    cfg = tiny_config()
    model = break_zero_init(build_model(cfg))
    _, poses, intrinsics, _ = random_batch(cfg, batch_size=1)
    a = sample_frames(model, poses, intrinsics, num_steps=4, generator=torch.Generator().manual_seed(0))
    b = sample_frames(model, poses, intrinsics, num_steps=4, generator=torch.Generator().manual_seed(1))
    assert not torch.allclose(a, b)


def test_sample_frames_more_steps_changes_output():
    """Not a correctness proof, just a sanity check that num_steps is
    actually wired into the integration rather than ignored."""
    cfg = tiny_config()
    model = break_zero_init(build_model(cfg))
    _, poses, intrinsics, _ = random_batch(cfg, batch_size=1)
    few = sample_frames(model, poses, intrinsics, num_steps=1, generator=torch.Generator().manual_seed(0))
    many = sample_frames(model, poses, intrinsics, num_steps=20, generator=torch.Generator().manual_seed(0))
    assert not torch.allclose(few, many)
