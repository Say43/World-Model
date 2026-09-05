"""REPA: the alignment loss, its projection head, and the DiT hook it needs.

The DINOv2 encoder itself is not exercised here (frozen pretrained weights,
network download -- same treatment as the AE in tests/test_eval_ae_ceiling.py);
these tests cover the parts that are ours: the intermediate-hidden-state
return path, the projection head's shapes, and the loss's behavior at known
inputs.
"""
import pytest
import torch

from src.model.dit import CausalDiT
from src.train.repa import RepaHead, default_align_layer, repa_loss
from tests.model_fixtures import break_zero_init, random_batch, tiny_config


def test_align_layer_sits_in_the_lower_third():
    assert default_align_layer(11) == 3
    assert default_align_layer(5) == 1
    assert default_align_layer(12) == 4
    # Degenerate but must not return a negative index.
    assert default_align_layer(1) == 0
    assert default_align_layer(2) == 0


def test_dit_returns_intermediate_hidden_state_with_right_shape():
    cfg = tiny_config()
    model = break_zero_init(CausalDiT(cfg))
    latents, poses, intrinsics, t = random_batch(cfg, batch_size=2)

    velocity, hidden = model(latents, poses, intrinsics, t, return_hidden_layer=1)

    assert velocity.shape == (2, cfg.context_length, cfg.tokens_per_frame, cfg.latent_channels)
    # Hidden state lives in the model's width, before the output projection.
    assert hidden.shape == (2, cfg.context_length, cfg.tokens_per_frame, cfg.dim)


def test_dit_velocity_is_identical_whether_or_not_hidden_is_requested():
    """Asking for the hidden state must not perturb the model's output --
    otherwise the REPA arm and the baseline arm of M3's ablation would not
    be comparing the same forward computation."""
    cfg = tiny_config()
    model = break_zero_init(CausalDiT(cfg))
    model.eval()
    latents, poses, intrinsics, t = random_batch(cfg, batch_size=2)

    with torch.no_grad():
        plain = model(latents, poses, intrinsics, t)
        velocity, _ = model(latents, poses, intrinsics, t, return_hidden_layer=1)

    torch.testing.assert_close(plain, velocity, rtol=0, atol=0)


def test_dit_rejects_out_of_range_align_layer():
    cfg = tiny_config()
    model = CausalDiT(cfg)
    latents, poses, intrinsics, t = random_batch(cfg, batch_size=1)
    with pytest.raises(ValueError, match="out of range"):
        model(latents, poses, intrinsics, t, return_hidden_layer=cfg.depth)


def test_repa_head_projects_to_feature_dim():
    head = RepaHead(dim=16, feature_dim=384)
    hidden = torch.randn(2, 4, 4, 16)
    assert head(hidden).shape == (2, 4, 4, 384)


def test_repa_loss_is_zero_for_identical_directions_and_one_for_orthogonal():
    features = torch.randn(2, 3, 4, 8)

    # Same direction (and scale-invariant: cosine ignores the 5x).
    assert repa_loss(features * 5.0, features).item() == pytest.approx(0.0, abs=1e-6)

    # Orthogonal: build a target orthogonal to the projection in every token.
    a = torch.zeros(1, 1, 1, 4)
    a[..., 0] = 1.0
    b = torch.zeros(1, 1, 1, 4)
    b[..., 1] = 1.0
    assert repa_loss(a, b).item() == pytest.approx(1.0, abs=1e-6)

    # Opposed directions are worse than orthogonal.
    assert repa_loss(features, -features).item() == pytest.approx(2.0, abs=1e-6)


def test_repa_loss_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        repa_loss(torch.randn(2, 3, 4, 8), torch.randn(2, 3, 4, 16))


def test_repa_gradient_reaches_the_dit_trunk():
    """The whole point of REPA is that the alignment signal trains the
    model's own blocks, not just the throwaway projection head."""
    # depth=4, not tiny_config's default 2: the point of this test is that
    # blocks AFTER the aligned layer get no REPA gradient, which is
    # vacuous if the aligned layer is already the last one.
    cfg = tiny_config(depth=4)
    model = break_zero_init(CausalDiT(cfg))
    head = RepaHead(dim=cfg.dim, feature_dim=32)
    latents, poses, intrinsics, t = random_batch(cfg, batch_size=2)
    target = torch.randn(2, cfg.context_length, cfg.tokens_per_frame, 32)

    _, hidden = model(latents, poses, intrinsics, t, return_hidden_layer=1)
    loss = repa_loss(head(hidden), target)
    loss.backward()

    aligned_block_grad = model.blocks[1].attn.qkv.weight.grad
    assert aligned_block_grad is not None
    assert aligned_block_grad.abs().sum() > 0

    # Blocks after the alignment point get no REPA gradient -- REPA only
    # constrains the trunk up to the layer it reads.
    later_block_grad = model.blocks[cfg.depth - 1].attn.qkv.weight.grad
    assert later_block_grad is None or later_block_grad.abs().sum() == 0
