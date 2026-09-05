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


def test_attach_repa_head_installs_head_into_the_model_parameters():
    """The head must be a submodule, not a sidecar: the optimizer is built
    from model.parameters(), and DDP fixes its parameter set at wrap time."""
    from src.train.repa import attach_repa_head

    cfg = tiny_config(depth=6)
    model = CausalDiT(cfg)
    before = {id(p) for p in model.parameters()}
    head = attach_repa_head(model, feature_dim=32)

    after = {id(p) for p in model.parameters()}
    assert {id(p) for p in head.parameters()} <= after
    assert len(after) > len(before)
    assert model.repa_align_layer == 2  # depth 6 // 3


def test_attach_repa_head_honors_an_explicit_align_layer_and_rejects_bad_ones():
    from src.train.repa import attach_repa_head

    model = CausalDiT(tiny_config(depth=6))
    attach_repa_head(model, feature_dim=32, align_layer=4)
    assert model.repa_align_layer == 4

    with pytest.raises(ValueError, match="already has a REPA head"):
        attach_repa_head(model, feature_dim=32)

    with pytest.raises(ValueError, match="out of range"):
        attach_repa_head(CausalDiT(tiny_config(depth=6)), feature_dim=32, align_layer=6)


def test_return_repa_projects_the_attached_layer():
    from src.train.repa import attach_repa_head

    cfg = tiny_config(depth=6)
    model = break_zero_init(CausalDiT(cfg))
    attach_repa_head(model, feature_dim=32, align_layer=2)
    latents, poses, intrinsics, t = random_batch(cfg, batch_size=2)
    model.eval()

    with torch.no_grad():
        velocity, projected = model(latents, poses, intrinsics, t, return_repa=True)
        _, hidden = model(latents, poses, intrinsics, t, return_hidden_layer=2)
        expected = model.repa_head(hidden)

    assert velocity.shape == (2, cfg.context_length, cfg.tokens_per_frame, cfg.latent_channels)
    assert projected.shape == (2, cfg.context_length, cfg.tokens_per_frame, 32)
    torch.testing.assert_close(projected, expected)


def test_return_repa_without_a_head_fails_rather_than_training_the_baseline():
    cfg = tiny_config()
    model = CausalDiT(cfg)
    latents, poses, intrinsics, t = random_batch(cfg, batch_size=1)
    with pytest.raises(ValueError, match="no REPA head"):
        model(latents, poses, intrinsics, t, return_repa=True)


def test_repa_loss_fn_trains_trunk_and_head_together():
    """End-to-end on the scripts/train.py contract: loss_fn(model, batch)."""
    from src.train.losses import rectified_flow_loss_with_repa
    from src.train.repa import attach_repa_head

    cfg = tiny_config(depth=6)
    model = break_zero_init(CausalDiT(cfg))
    attach_repa_head(model, feature_dim=32, align_layer=2)
    model.repa_weight = 0.5
    latents, poses, intrinsics, _ = random_batch(cfg, batch_size=2)
    batch = {
        "latents": latents, "poses": poses, "intrinsics": intrinsics,
        "dinov2": torch.randn(2, cfg.context_length, cfg.tokens_per_frame, 32),
    }

    loss = rectified_flow_loss_with_repa(model, batch)
    assert torch.isfinite(loss)
    loss.backward()

    assert model.repa_head.net[0].weight.grad.abs().sum() > 0
    assert model.blocks[2].attn.qkv.weight.grad.abs().sum() > 0
    assert model.output_proj.weight.grad.abs().sum() > 0


def test_repa_loss_fn_reports_a_missing_weight_and_missing_features_clearly():
    from src.train.losses import rectified_flow_loss_with_repa
    from src.train.repa import attach_repa_head

    cfg = tiny_config(depth=6)
    model = break_zero_init(CausalDiT(cfg))
    attach_repa_head(model, feature_dim=32)
    latents, poses, intrinsics, _ = random_batch(cfg, batch_size=1)
    batch = {"latents": latents, "poses": poses, "intrinsics": intrinsics}

    with pytest.raises(KeyError, match="--with-dinov2"):
        rectified_flow_loss_with_repa(model, batch)

    batch["dinov2"] = torch.randn(1, cfg.context_length, cfg.tokens_per_frame, 32)
    with pytest.raises(AttributeError, match="repa_weight"):
        rectified_flow_loss_with_repa(model, batch)


def test_repa_weight_zero_matches_the_plain_flow_loss():
    """A weight of 0 must reduce exactly to the baseline arm -- the sanity
    check that the two M3 arms differ only by the term under test."""
    from src.train.losses import rectified_flow_loss, rectified_flow_loss_with_repa
    from src.train.repa import attach_repa_head

    cfg = tiny_config(depth=6)
    model = break_zero_init(CausalDiT(cfg))
    attach_repa_head(model, feature_dim=32)
    model.repa_weight = 0.0
    latents, poses, intrinsics, _ = random_batch(cfg, batch_size=2)
    batch = {
        "latents": latents, "poses": poses, "intrinsics": intrinsics,
        "dinov2": torch.randn(2, cfg.context_length, cfg.tokens_per_frame, 32),
    }

    torch.manual_seed(0)
    with_repa = rectified_flow_loss_with_repa(model, batch)
    torch.manual_seed(0)
    plain = rectified_flow_loss(model, batch)

    torch.testing.assert_close(with_repa, plain)


def test_repa_loss_exposes_its_components_for_a_fair_comparison():
    """The REPA arm's total loss carries an extra non-negative term, so
    ranking M3's arms by total loss would report the alignment term itself
    as REPA's effect. The flow term must be recoverable without a second
    forward pass (a CPU smoke of the runner produced exactly that fake
    +0.49 'effect' before this was added)."""
    from src.train.losses import rectified_flow_loss_with_repa
    from src.train.repa import attach_repa_head

    cfg = tiny_config(depth=6)
    model = break_zero_init(CausalDiT(cfg))
    attach_repa_head(model, feature_dim=32)
    model.repa_weight = 0.5
    latents, poses, intrinsics, _ = random_batch(cfg, batch_size=2)
    batch = {
        "latents": latents, "poses": poses, "intrinsics": intrinsics,
        "dinov2": torch.randn(2, cfg.context_length, cfg.tokens_per_frame, 32),
    }

    total = rectified_flow_loss_with_repa(model, batch)
    components = model.last_loss_components

    assert set(components) == {"flow", "align"}
    assert not components["flow"].requires_grad, "components must be detached"
    torch.testing.assert_close(
        components["flow"] + 0.5 * components["align"], total.detach()
    )
    assert components["flow"] < total, "flow term must be strictly below the total"
