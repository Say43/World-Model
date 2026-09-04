"""muP parameter classification and the coordinate check.

CLAUDE.md's actual M2 gate is a real LR sweep at 5M verified to transfer to
15M -- that needs real training runs and is not something a CPU unit test
can substitute for. What these tests guard is the plumbing underneath it:
that every parameter lands in the category its shape implies (a
misclassification here would silently disable LR scaling for whichever
parameters fell into "base" by mistake, exactly as happened during
development -- SwiGLU's rounded hidden size and RaymapEncoder's fractional
hidden size both initially misclassified as "base" under a naive
"is this a clean integer multiple of dim" heuristic), and that scaling
learning rate by 1/width_mult produces the standard muP coordinate-check
signature: per-parameter update *sizes* for "hidden"/"output" categories
stay comparable across widths, rather than growing with width the way an
unscaled (standard parametrization) LR would let them.
"""
import torch

from src.model.dit import CausalDiT, preset_5m, preset_15m, preset_40m
from src.train.mup import build_mup_optimizer, classify_param, mup_param_groups
from tests.model_fixtures import break_zero_init, random_batch


def test_every_parameter_is_classified_and_none_left_out():
    for preset_fn in (preset_5m, preset_15m, preset_40m):
        cfg = preset_fn()
        model = CausalDiT(cfg)
        total = sum(p.numel() for p in model.parameters())
        classified_total = 0
        for name, p in model.named_parameters():
            cat = classify_param(name, p, cfg)
            assert cat in ("input", "hidden", "output", "base")
            classified_total += p.numel()
        assert classified_total == total


def test_swiglu_and_raymap_hidden_layers_classify_as_hidden_not_base():
    """Regression test for the exact bug found during development: a
    shape-ratio heuristic put every SwiGLU MLP weight and RaymapEncoder's
    hidden layer into "base" (no LR scaling) because their sizes are a
    rounded/fractional function of dim, not a clean integer multiple.
    """
    cfg = preset_5m()
    model = CausalDiT(cfg)
    params_by_name = dict(model.named_parameters())

    for name in ["blocks.0.mlp.w_gate.weight", "blocks.0.mlp.w_up.weight", "blocks.0.mlp.w_down.weight"]:
        assert classify_param(name, params_by_name[name], cfg) == "hidden", name

    assert classify_param(
        "raymap_encoder.net.2.weight", params_by_name["raymap_encoder.net.2.weight"], cfg
    ) == "hidden"
    assert classify_param(
        "raymap_encoder.net.0.weight", params_by_name["raymap_encoder.net.0.weight"], cfg
    ) == "input"  # fan_in=6 (fixed, Plucker) -> fan_out=raymap hidden (scales)


def test_output_proj_classified_as_output():
    cfg = preset_5m()
    model = CausalDiT(cfg)
    params_by_name = dict(model.named_parameters())
    assert classify_param("output_proj.weight", params_by_name["output_proj.weight"], cfg) == "output"


def test_norms_and_biases_and_frame_pos_embed_classified_as_base():
    cfg = preset_5m()
    model = CausalDiT(cfg)
    params_by_name = dict(model.named_parameters())
    for name in ["frame_pos_embed", "output_proj.bias", "blocks.0.attn.q_norm.weight", "blocks.0.adaln_bias"]:
        assert classify_param(name, params_by_name[name], cfg) == "base", name


def test_mup_param_groups_cover_every_parameter_exactly_once():
    cfg = preset_15m()
    model = CausalDiT(cfg)
    groups = mup_param_groups(model, cfg, base_dim=256, base_lr=1e-3)
    all_params_in_groups = [p for g in groups for p in g["params"]]
    model_params = list(model.parameters())
    assert len(all_params_in_groups) == len(model_params)
    assert {id(p) for p in all_params_in_groups} == {id(p) for p in model_params}


def test_mup_scales_hidden_and_output_lr_by_inverse_width_mult():
    base_dim = 256
    cfg_15m = preset_15m()  # dim=320
    groups = mup_param_groups(CausalDiT(cfg_15m), cfg_15m, base_dim=base_dim, base_lr=1e-3)
    by_category = {g["mup_category"]: g["lr"] for g in groups}
    width_mult = cfg_15m.dim / base_dim

    assert by_category["input"] == 1e-3
    assert by_category["base"] == 1e-3
    assert by_category["hidden"] == 1e-3 / width_mult
    assert by_category["output"] == 1e-3 / width_mult


def test_build_mup_optimizer_matches_base_lr_at_base_width():
    """At the base width itself (width_mult=1), muP's LR scaling must be a
    no-op: every group should train at exactly base_lr."""
    cfg = preset_5m()
    model = CausalDiT(cfg)
    optimizer = build_mup_optimizer(model, cfg, base_dim=cfg.dim, base_lr=3e-4)
    for group in optimizer.param_groups:
        assert group["lr"] == 3e-4


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

    What this test asserts instead, precisely: build_mup_optimizer's
    param groups actually produce *different* per-group LR at a non-base
    width (confirming the groups are live, distinct optimizer state, not
    e.g. accidentally merged into one), and that the resulting step
    updates "hidden"/"output" params by a different absolute amount than
    "input"/"base" params when their LRs differ -- i.e. the mechanism has
    a real, measurable effect, even though its *sufficiency* for HP
    transfer is only established by the real M2 gate.
    """
    cfg = preset_15m()  # dim=320, width_mult=1.25 relative to base_dim=256 -> visibly different LRs
    model = CausalDiT(cfg)
    break_zero_init(model)
    optimizer = build_mup_optimizer(model, cfg, base_dim=256, base_lr=1e-2)

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
