"""muP (maximal update parametrization) for CausalDiT, via per-parameter-
group learning-rate scaling under Adam -- not the `mup` package, to keep
the project dependency-light, matching how the rest of src/train/ already
implements its own LR schedule rather than pulling in a scheduler library.

Why no init changes are needed: muP's two ingredients are (1) width-scaled
init variance and (2) width-scaled learning rate. CausalDiT's existing
init is already compatible with (1) by construction:
  - "hidden" matrices (both fan_in and fan_out scale with `dim`) use
    PyTorch's default Kaiming-style init, std ~ 1/sqrt(fan_in). Since
    fan_in itself scales with width, this already shrinks as
    1/sqrt(width_mult) relative to the base width -- exactly what muP
    prescribes for hidden layers. No change needed.
  - "output" matrices (fan_in scales, fan_out fixed -- just output_proj)
    are zero-initialized already (src/model/dit.py, for fp16-stability
    reasons unrelated to muP). Zero trivially satisfies "small enough";
    muP's output-layer init requirement is moot when starting at exactly
    zero.
  - "input" matrices (fan_in fixed, fan_out scales) and "base" params
    (norms, biases, frame_pos_embed) keep standard init; muP prescribes
    no width-dependent change for these either.

So the only remaining ingredient -- and the only thing this module does --
is classifying every parameter into one of muP's categories from its shape
relative to the model's own config, and building AdamW parameter groups
whose LR is divided by width_mult for "hidden" and "output" params, left
at base_lr for "input" and "base" params.

CLAUDE.md's M2 gate is the real verification: optimal LR at 5M and 15M
must land in the ratio muP predicts. tests/test_train_mup.py's coordinate
check is a much cheaper (CPU, seconds) regression guard that the group
classification and LR scaling are wired correctly -- it is not a
substitute for that gate.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from src.model.blocks import hidden_dim_swiglu


def _width_scaling_dims(config) -> set:
    """Every dimension that scales with model width, computed by calling
    the exact same formulas src/model/dit.py and its submodules use to
    build these layers -- not guessed from raw integer ratios.

    An earlier version tried to infer "scales with width" from whether a
    raw shape was a clean integer multiple of `config.dim`. That silently
    misclassified most of the model: RaymapEncoder's hidden size is
    `dim // hidden_mult` (a *fraction* of dim, e.g. 128 at dim=256, which
    is not a multiple of 256), and hidden_dim_swiglu's `multiple_of`
    rounding (256*4*2/3 = 682.67 -> rounded to 704) makes the SwiGLU MLP's
    hidden size scale with dim without being a clean ratio of it. Both
    landed in "base" (no LR scaling at all) instead of their correct
    category -- silently disabling muP's LR scaling for most of the
    model's parameters. Computing the exact values instead of guessing
    fixes this and keeps working if these formulas change, since it calls
    the same code the model itself does.
    """
    dim = config.dim
    raymap_hidden = max(dim // config.raymap_hidden_mult, 6)
    mlp_hidden = hidden_dim_swiglu(dim, config.mlp_mult, config.mlp_multiple_of)
    return {
        dim,
        raymap_hidden,
        mlp_hidden,
        2 * dim,  # final_mod: num_chunks=2
        6 * dim,  # adaln_single: num_chunks=6
    }


def _fixed_dims(config) -> set:
    """Dimensions that do NOT scale with model width, gathered from the
    config rather than hardcoded module names -- generalizes to any
    current or future input/output layer without listing them by name.
    """
    dims = {6, config.latent_channels, config.time_embed_dim}  # 6 = Plucker raymap width
    return {d for d in dims if d and d > 0}


def classify_param(name: str, tensor: torch.Tensor, config) -> str:
    """Returns one of "input", "hidden", "output", "base" for a named
    parameter, using only its shape and the model's own config -- no
    hardcoded module-name list, so it does not silently stop working if
    src/model/dit.py's internals change (as long as _width_scaling_dims
    is kept in sync with any new width-dependent formula).
    """
    if tensor.ndim != 2:
        # Norms, biases, frame_pos_embed (context_length, dim), adaln's
        # per-block bias (6*dim,): none of these have the fan-in blowup
        # that motivates muP's LR scaling. Standard muP treatment: base LR.
        return "base"

    out_features, in_features = tensor.shape
    width = _width_scaling_dims(config)
    fixed = _fixed_dims(config) - width  # a dim that coincides with both (only possible at base width) counts as width

    in_is_fixed = in_features in fixed
    out_is_fixed = out_features in fixed
    in_is_width = in_features in width
    out_is_width = out_features in width

    if in_is_fixed and out_is_width:
        return "input"
    if in_is_width and out_is_fixed:
        return "output"
    if in_is_width and out_is_width:
        return "hidden"
    return "base"


def mup_param_groups(model: nn.Module, config, base_dim: int, base_lr: float,
                      weight_decay: float = 0.0) -> List[Dict]:
    """Builds AdamW param groups with muP's LR scaling: "hidden" and
    "output" params get base_lr / width_mult; "input" and "base" params
    keep base_lr. `base_dim` is the reference width (this project's 5M
    preset, dim=256) that `base_lr` was tuned at.
    """
    width_mult = config.dim / base_dim
    buckets: Dict[str, list] = {"input": [], "hidden": [], "output": [], "base": []}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        buckets[classify_param(name, param, config)].append(param)

    scale = {"input": 1.0, "hidden": 1.0 / width_mult, "output": 1.0 / width_mult, "base": 1.0}
    groups = []
    for category, params in buckets.items():
        if not params:
            continue
        groups.append({
            "params": params,
            "lr": base_lr * scale[category],
            "weight_decay": weight_decay,
            "mup_category": category,  # informational only; AdamW ignores unknown keys? No -- see build_mup_optimizer
        })
    return groups


def build_mup_optimizer(model: nn.Module, config, base_dim: int, base_lr: float,
                         weight_decay: float = 0.0) -> torch.optim.AdamW:
    """Same as mup_param_groups but strips the informational "mup_category"
    key before constructing AdamW, which rejects unknown param-group keys.
    """
    groups = mup_param_groups(model, config, base_dim, base_lr, weight_decay)
    clean_groups = [{k: v for k, v in g.items() if k != "mup_category"} for g in groups]
    return torch.optim.AdamW(clean_groups)
