"""Tests for src/model/diffusion_forcing.py per-frame noise-level schedules."""
import pytest
import torch

from src.model.diffusion_forcing import (
    rectified_flow_target,
    sample_autoregressive_rollout,
    sample_uniform_independent,
)


def test_sample_uniform_independent_shape_and_range():
    t = sample_uniform_independent(batch_size=4, context_length=8)
    assert t.shape == (4, 8)
    assert (t >= 0.0).all() and (t < 1.0).all()


def test_sample_uniform_independent_is_decorrelated_across_frames():
    """Not a strict statistical test (would be flaky) -- just checks that
    frames within a single clip are not all forced identical, which would
    indicate the "independent per frame" contract silently regressed to
    the old shared-timestep-per-clip behavior."""
    torch.manual_seed(0)
    t = sample_uniform_independent(batch_size=1, context_length=32)
    assert t.unique().numel() > 1


def test_sample_autoregressive_rollout_schedule():
    t = sample_autoregressive_rollout(batch_size=2, context_length=5, num_clean=2, active_t=0.7)
    assert t.shape == (2, 5)
    assert torch.allclose(t[:, :2], torch.zeros(2, 2))
    assert torch.allclose(t[:, 2], torch.full((2,), 0.7))
    assert torch.allclose(t[:, 3:], torch.ones(2, 2))


def test_sample_autoregressive_rollout_invalid_num_clean_raises():
    with pytest.raises(ValueError):
        sample_autoregressive_rollout(batch_size=1, context_length=4, num_clean=4, active_t=0.5)
    with pytest.raises(ValueError):
        sample_autoregressive_rollout(batch_size=1, context_length=4, num_clean=-1, active_t=0.5)


def test_rectified_flow_target_endpoints():
    x0 = torch.randn(2, 3, 4)
    x1 = torch.randn(2, 3, 4)
    t0 = torch.zeros(2, 3)
    t1 = torch.ones(2, 3)

    x_at_0, v0 = rectified_flow_target(x0, x1, t0)
    x_at_1, v1 = rectified_flow_target(x0, x1, t1)

    assert torch.allclose(x_at_0, x0, atol=1e-6)
    assert torch.allclose(x_at_1, x1, atol=1e-6)
    assert torch.allclose(v0, x1 - x0)
    assert torch.allclose(v1, x1 - x0)  # velocity target is constant along the path


def test_rectified_flow_target_broadcast_over_token_axis():
    x0 = torch.randn(2, 3, 5, 4)  # (B, T, N, C)
    x1 = torch.randn(2, 3, 5, 4)
    t = torch.rand(2, 3)  # (B, T), broadcast over N and C
    x_t, v = rectified_flow_target(x0, x1, t)
    assert x_t.shape == x0.shape
    assert v.shape == x0.shape
