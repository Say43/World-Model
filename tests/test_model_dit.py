"""Shape, gradient, and preset parameter-count tests for src/model/dit.py."""
import pytest
import torch

from src.model.dit import (
    DiTConfig,
    CausalDiT,
    build_block_causal_mask,
    count_parameters,
    preset_5m,
    preset_15m,
    preset_40m,
)
from tests.model_fixtures import break_zero_init, build_model, random_batch, tiny_config


def test_forward_shape():
    cfg = tiny_config()
    model = build_model(cfg)
    latents, poses, intrinsics, t = random_batch(cfg, batch_size=2)
    out = model(latents, poses, intrinsics, t)
    assert out.shape == latents.shape


def test_forward_shape_partial_chunk_with_frame_start():
    """A sub-chunk starting mid-sequence (as used by incremental decode)
    must still produce a well-shaped output."""
    cfg = tiny_config()
    model = build_model(cfg)
    latents, poses, intrinsics, t = random_batch(cfg, batch_size=2, num_frames=1)
    out = model(latents, poses, intrinsics, t, frame_start=2)
    assert out.shape == latents.shape


def test_forward_start_plus_chunk_exceeds_context_raises():
    cfg = tiny_config()
    model = build_model(cfg)
    latents, poses, intrinsics, t = random_batch(cfg, batch_size=1, num_frames=2)
    with pytest.raises(ValueError):
        model(latents, poses, intrinsics, t, frame_start=cfg.context_length - 1)


def test_wrong_tokens_per_frame_raises():
    cfg = tiny_config()
    model = build_model(cfg)
    latents, poses, intrinsics, t = random_batch(cfg, batch_size=1)
    bad_latents = latents[:, :, :-1, :]  # drop one token
    with pytest.raises(ValueError):
        model(bad_latents, poses, intrinsics, t)


def test_gradient_flow_all_params_finite():
    """Every parameter must receive a finite gradient from a scalar loss on
    the model's output -- a dead/disconnected parameter (e.g. a module
    built but never wired into forward) would show up as `grad is None`
    here, and a numerically unstable path would show up as NaN/Inf."""
    cfg = tiny_config()
    model = build_model(cfg)
    # output_proj is zero-initialized by design (see dit.py docstring on
    # fp16-safety), which would make d(loss)/d(output_proj.weight) exactly
    # zero on the very first call -- still technically "finite", but not a
    # meaningful check that gradients actually propagate through the
    # zero-initialized modulation/output path. Break the symmetry first so
    # this test exercises the same path a real (partially trained) model
    # would.
    break_zero_init(model)

    latents, poses, intrinsics, t = random_batch(cfg, batch_size=2)
    out = model(latents, poses, intrinsics, t)
    loss = out.pow(2).mean()
    loss.backward()

    missing = []
    non_finite = []
    for name, p in model.named_parameters():
        if p.grad is None:
            missing.append(name)
        elif not torch.isfinite(p.grad).all():
            non_finite.append(name)
    assert not missing, f"parameters with no gradient: {missing}"
    assert not non_finite, f"parameters with non-finite gradient: {non_finite}"


def test_build_block_causal_mask_shape_and_values():
    mask = build_block_causal_mask(chunk_frames=3, tokens_per_frame=2, cache_len=0, device="cpu")
    assert mask.shape == (6, 6)
    # Frame 0's tokens (rows 0-1) may only see frame 0's tokens (cols 0-1).
    assert mask[0, :2].all()
    assert not mask[0, 2:].any()
    # Frame 2's tokens (rows 4-5) may see everything (frames 0,1,2).
    assert mask[4].all()


def test_build_block_causal_mask_with_cache():
    mask = build_block_causal_mask(chunk_frames=1, tokens_per_frame=2, cache_len=4, device="cpu")
    assert mask.shape == (2, 6)
    assert mask[:, :4].all()  # cached history always visible
    assert mask.all()  # single new frame attends to itself fully too


@pytest.mark.parametrize(
    "preset_fn,expected",
    [
        (preset_5m, 4_720_324),
        (preset_15m, 14_718_628),
        (preset_40m, 40_624_644),
    ],
)
def test_preset_param_counts(preset_fn, expected):
    model = CausalDiT(preset_fn())
    assert count_parameters(model) == expected


def test_preset_param_counts_independent_of_tokens_per_frame():
    """Total parameter count must not change when tokens_per_frame /
    raymap_resolution change (only frame_pos_embed depends on
    context_length, never on tokens_per_frame) -- this is what lets model
    size (M1-M4) and AE choice (M0) be decided independently."""
    base = CausalDiT(preset_5m())
    alt = CausalDiT(preset_5m(tokens_per_frame=256, raymap_resolution=(16, 16)))
    assert count_parameters(base) == count_parameters(alt)


def test_invalid_raymap_resolution_raises():
    with pytest.raises(ValueError):
        DiTConfig(tokens_per_frame=64, raymap_resolution=(7, 7))


def test_invalid_dim_heads_raises():
    with pytest.raises(ValueError):
        DiTConfig(dim=17, num_heads=4, tokens_per_frame=64, raymap_resolution=(8, 8))
