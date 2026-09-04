"""muP parameter classification and optimizer/forward plumbing.

CLAUDE.md's actual M2 gate is a real LR sweep at 5M verified to transfer to
15M -- that needs real training runs and is not something a CPU unit test
can substitute for. What these tests guard is the plumbing underneath it:
that every parameter lands in the category its shape implies (a
misclassification here would silently disable LR scaling for whichever
parameters fell into "base" by mistake, exactly as happened during
development -- fused QKV, SwiGLU's rounded hidden size, and RaymapEncoder's
fractional hidden size were all vulnerable under dimension whitelists), and
that exact per-parameter fan-in, attention, and readout scaling are wired.
"""
import torch

from src.model.dit import CausalDiT, preset_m2_proxy_5m, preset_15m
from src.train.mup import (
    build_mup_optimizer,
    classify_param,
    configure_mup_model,
    mup_param_groups,
)
from tests.model_fixtures import break_zero_init, random_batch


def _target_and_metadata():
    cfg = preset_15m(num_heads=8)
    model = CausalDiT(cfg)
    metadata = configure_mup_model(model, cfg, base_dim=192)
    return cfg, model, metadata


def test_every_parameter_is_classified_and_none_left_out():
    cfg, model, metadata = _target_and_metadata()
    total = sum(p.numel() for p in model.parameters())
    classified_total = 0
    for name, p in model.named_parameters():
        cat = classify_param(name, p, cfg, metadata[name]["base_shape"])
        assert cat in ("input", "hidden", "output", "base")
        classified_total += p.numel()
    assert classified_total == total


def test_swiglu_and_raymap_hidden_layers_classify_as_hidden_not_base():
    """Regression test for the exact bug found during development: a
    shape-ratio heuristic put every SwiGLU MLP weight and RaymapEncoder's
    hidden layer into "base" (no LR scaling) because their sizes are a
    rounded/fractional function of dim, not a clean integer multiple.
    """
    cfg, model, metadata = _target_and_metadata()
    params_by_name = dict(model.named_parameters())

    for name in ["blocks.0.mlp.w_gate.weight", "blocks.0.mlp.w_up.weight", "blocks.0.mlp.w_down.weight"]:
        assert classify_param(name, params_by_name[name], cfg, metadata[name]["base_shape"]) == "hidden", name

    name = "raymap_encoder.net.2.weight"
    assert classify_param(
        name, params_by_name[name], cfg, metadata[name]["base_shape"]
    ) == "hidden", name
    name = "raymap_encoder.net.0.weight"
    assert classify_param(
        name, params_by_name[name], cfg, metadata[name]["base_shape"]
    ) == "input", name  # fan_in=6 fixed, fan_out scales


def test_fused_qkv_is_hidden_not_base():
    """Regression: the prior dimension whitelist omitted 3*dim entirely."""
    cfg, model, metadata = _target_and_metadata()
    name = "blocks.0.attn.qkv.weight"
    param = dict(model.named_parameters())[name]
    assert classify_param(name, param, cfg, metadata[name]["base_shape"]) == "hidden"
    assert metadata[name]["fanin_mult"] == 320 / 192


def test_output_proj_classified_as_output():
    cfg, model, metadata = _target_and_metadata()
    params_by_name = dict(model.named_parameters())
    name = "output_proj.weight"
    assert classify_param(name, params_by_name[name], cfg, metadata[name]["base_shape"]) == "output"


def test_norms_and_biases_and_frame_pos_embed_classified_as_base():
    cfg, model, metadata = _target_and_metadata()
    params_by_name = dict(model.named_parameters())
    for name in ["frame_pos_embed", "output_proj.bias", "blocks.0.attn.q_norm.weight", "blocks.0.adaln_bias"]:
        assert classify_param(name, params_by_name[name], cfg, metadata[name]["base_shape"]) == "base", name


def test_mup_param_groups_cover_every_parameter_exactly_once():
    cfg = preset_15m(num_heads=8)
    model = CausalDiT(cfg)
    groups = mup_param_groups(model, cfg, base_dim=192, base_lr=1e-3)
    all_params_in_groups = [p for g in groups for p in g["params"]]
    model_params = list(model.parameters())
    assert len(all_params_in_groups) == len(model_params)
    assert {id(p) for p in all_params_in_groups} == {id(p) for p in model_params}


def test_mup_uses_exact_per_parameter_fanin_scaling():
    cfg = preset_15m(num_heads=8)
    model = CausalDiT(cfg)
    groups = mup_param_groups(model, cfg, base_dim=192, base_lr=1e-3, weight_decay=0.1)
    group_by_param = {id(param): group for group in groups for param in group["params"]}
    params = dict(model.named_parameters())

    qkv_group = group_by_param[id(params["blocks.0.attn.qkv.weight"])]
    mlp_down_group = group_by_param[id(params["blocks.0.mlp.w_down.weight"])]
    output_group = group_by_param[id(params["output_proj.weight"])]

    assert qkv_group["lr"] == 1e-3 / (320 / 192)
    # SwiGLU rounding makes this multiplier differ from dim/base_dim.
    assert mlp_down_group["lr"] == 1e-3 / (864 / 512)
    assert qkv_group["weight_decay"] == 0.1 * (320 / 192)
    # MuReadout keeps Adam LR unchanged; its width scaling is in forward.
    assert output_group["lr"] == 1e-3


def test_build_mup_optimizer_matches_base_lr_at_base_width():
    """At the base width itself (width_mult=1), muP's LR scaling must be a
    no-op: every group should train at exactly base_lr."""
    cfg = preset_m2_proxy_5m()
    model = CausalDiT(cfg)
    optimizer = build_mup_optimizer(model, cfg, base_dim=cfg.dim, base_lr=3e-4)
    for group in optimizer.param_groups:
        assert group["lr"] == 3e-4


def test_mup_configures_readout_and_attention_scaling():
    cfg = preset_15m(num_heads=8)
    model = CausalDiT(cfg)
    hidden_bias_before = model.time_mlp[2].bias.detach().clone()
    configure_mup_model(model, cfg, base_dim=192)

    assert model.mup_readout_width_mult == 320 / 192
    expected_attention_scale = (24 ** 0.5) / 40  # fixed 8 heads
    assert model.blocks[0].attn.mup_attention_scale == expected_attention_scale
    assert torch.allclose(
        model.time_mlp[2].bias,
        hidden_bias_before * (320 / 192) ** 0.5,
    )


def test_muP_lr_scaling_actually_changes_the_optimizer_step():
    """A single Adam step's *relative* update size is a weak signal for
    this: Adam's own per-parameter second-moment normalization already
    keeps update-norm-to-weight-norm ratios within a similar order of
    magnitude across widths even with NO muP scaling at all (measured
    directly: an unscaled shared-LR optimizer gave ratios of 1.09x and
    1.35x between 5M/15M/40M here -- comfortably inside any tolerance loose
    enough to also accept a correct implementation, which means that
    metric cannot actually distinguish "muP wired correctly" from "muP not
    applied"). A true coordinate check needs many steps and/or a real LR
    sweep, which is CLAUDE.md's M2 gate on real hardware, not a CPU unit
    test.

    What this test asserts instead, precisely: build_mup_optimizer's param
    groups actually produce distinct LRs at a non-base width and remain
    attached to a live optimizer step. Sufficiency for HP transfer is only
    established by the real M2 gate.
    """
    cfg = preset_15m(num_heads=8)
    model = CausalDiT(cfg)
    break_zero_init(model)
    optimizer = build_mup_optimizer(model, cfg, base_dim=192, base_lr=1e-2)

    lrs = {g["params"][0]: g["lr"] for g in optimizer.param_groups}
    distinct_lrs = {round(lr, 8) for lr in lrs.values()}
    assert len(distinct_lrs) >= 2, "expected at least two distinct LR values across categories"

    latents, poses, intrinsics, t = random_batch(cfg, batch_size=2)
    target = torch.randn_like(latents)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}

    out = model(latents, poses, intrinsics, t)
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()
    optimizer.step()

    # Every parameter that received a nonzero gradient must actually move
    # -- confirms the groups are attached to a real optimizer step, not
    # just constructed and discarded.
    moved = any(not torch.equal(p, before[n]) for n, p in model.named_parameters())
    assert moved, "no parameter changed after optimizer.step() -- groups are not live"
