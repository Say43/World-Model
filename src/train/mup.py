"""CausalDiT-specific maximal-update parametrization (muP).

This implements the subset of ``microsoft/mup`` needed by nanoWM. Base and
target parameter shapes are compared directly, so fused QKV matrices and
rounded SwiGLU widths cannot silently fall out of muP scaling. AdamW LR is
divided by each matrix's exact fan-in multiplier when both dimensions scale;
linear biases, attention logits, and the final readout follow the reference
initialization/forward rules as well.

This is intentionally not a general replacement for the upstream package.
The proxy and target must have identical topology (notably depth and head
count); only width-shaped dimensions may differ.
"""
from __future__ import annotations

import dataclasses
import math
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from src.model.blocks import Attention
from src.model.dit import CausalDiT


def _bare_model(model: nn.Module) -> nn.Module:
    """Remove compile/DDP wrappers without depending on delegated attrs."""
    while hasattr(model, "_orig_mod"):
        model = model._orig_mod
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    return model


def _base_model(config, base_dim: int) -> CausalDiT:
    if base_dim <= 0:
        raise ValueError("base_dim must be positive")
    if base_dim % config.num_heads:
        raise ValueError(
            f"base_dim ({base_dim}) must be divisible by fixed num_heads "
            f"({config.num_heads})"
        )
    base_config = dataclasses.replace(config, dim=base_dim)
    # A shape-only proxy must not advance the training RNG. Diffusion-noise
    # sequences would otherwise depend on how many proxy parameters were
    # allocated before the run.
    with torch.random.fork_rng(devices=[]):
        return CausalDiT(base_config)


def _shape_metadata(model: nn.Module, config, base_dim: int):
    model = _bare_model(model)
    base = _base_model(config, base_dim)
    base_shapes = {name: tuple(param.shape) for name, param in base.named_parameters()}
    target_names = {name for name, _ in model.named_parameters()}
    if target_names != set(base_shapes):
        mismatch = sorted(target_names.symmetric_difference(base_shapes))
        raise ValueError(f"muP proxy/target topology differs: {mismatch}")

    metadata = {}
    for name, param in model.named_parameters():
        target_shape = tuple(param.shape)
        base_shape = base_shapes[name]
        if len(target_shape) != len(base_shape):
            raise ValueError(f"rank differs for {name}: target={target_shape}, base={base_shape}")
        infinite_dims = tuple(
            index
            for index, (target, proxy) in enumerate(zip(target_shape, base_shape))
            if target != proxy
        )
        fanin_mult = 1.0
        if param.ndim == 2 and len(infinite_dims) == 2:
            fanin_mult = target_shape[1] / base_shape[1]
        metadata[name] = {
            "base_shape": base_shape,
            "infinite_dims": infinite_dims,
            "fanin_mult": fanin_mult,
        }
    return metadata, base


def classify_param(
    name: str,
    tensor: torch.Tensor,
    config,
    base_shape: Tuple[int, ...],
) -> str:
    """Describe a parameter from its target and proxy shapes.

    Adam-like muP scaling itself is based on the number of changing
    dimensions. Only matrices with two changing dimensions get LR scaling.
    A readout weight therefore keeps base LR and is scaled in forward.
    """
    target_shape = tuple(tensor.shape)
    if len(target_shape) != len(base_shape):
        raise ValueError(f"rank differs for {name}: target={target_shape}, base={base_shape}")
    changed = tuple(
        index
        for index, (target, proxy) in enumerate(zip(target_shape, base_shape))
        if target != proxy
    )
    if tensor.ndim != 2 or not name.endswith(".weight"):
        return "base"
    if changed == (0,):
        return "input"
    if changed == (1,):
        return "output"
    if changed == (0, 1):
        return "hidden"
    return "base"


def configure_mup_model(model: nn.Module, config, base_dim: int):
    """Apply muP initialization/forward rules once and return shape metadata."""
    model = _bare_model(model)
    already = getattr(model, "_mup_base_dim", None)
    if already is not None:
        if already != base_dim:
            raise ValueError(f"model already configured for base_dim={already}, got {base_dim}")
        return model._mup_parameter_metadata

    metadata, base = _shape_metadata(model, config, base_dim)
    target_modules = dict(model.named_modules())
    base_modules = dict(base.named_modules())
    with torch.no_grad():
        # microsoft/mup's set_base_shapes rescales Linear biases by the
        # square root of the fan-in multiplier.
        for module_name, module in target_modules.items():
            if not isinstance(module, nn.Linear) or module.bias is None:
                continue
            if module is model.output_proj:
                continue  # MuReadout has its own parameter rescaling below.
            base_module = base_modules[module_name]
            fanin_mult = module.weight.shape[1] / base_module.weight.shape[1]
            module.bias.mul_(math.sqrt(fanin_mult))

        # MuReadout rescales parameters and divides the complete output by
        # width_mult. This project's readout starts at zero, but applying the
        # rule explicitly avoids relying on that detail.
        readout_mult = model.output_proj.weight.shape[1] / base.output_proj.weight.shape[1]
        model.output_proj.weight.mul_(math.sqrt(readout_mult))
        if model.output_proj.bias is not None:
            model.output_proj.bias.mul_(math.sqrt(readout_mult))

    base_head_dim = base.config.dim // base.config.num_heads
    target_head_dim = config.dim // config.num_heads
    attention_scale = math.sqrt(base_head_dim) / target_head_dim
    for module in target_modules.values():
        if isinstance(module, Attention):
            module.mup_attention_scale = attention_scale

    model.mup_readout_width_mult = readout_mult
    model._mup_base_dim = base_dim
    model._mup_parameter_metadata = metadata
    return metadata


def mup_param_groups(
    model: nn.Module,
    config,
    base_dim: int,
    base_lr: float,
    weight_decay: float = 0.0,
) -> List[Dict]:
    """Build AdamW groups matching ``microsoft/mup``'s MuAdamW rules."""
    model = _bare_model(model)
    metadata = configure_mup_model(model, config, base_dim)
    buckets = defaultdict(list)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        info = metadata[name]
        category = classify_param(name, param, config, info["base_shape"])
        matrix_like = param.ndim == 2 and len(info["infinite_dims"]) == 2
        width_mult = info["fanin_mult"] if matrix_like else 1.0
        buckets[(category, width_mult)].append(param)

    groups = []
    for (category, width_mult), params in buckets.items():
        groups.append({
            "params": params,
            "lr": base_lr / width_mult,
            # Compensate for AdamW multiplying its decoupled decay by LR;
            # this is the reference MuAdamW default.
            "weight_decay": weight_decay * width_mult,
            "mup_category": category,
            "mup_width_mult": width_mult,
        })
    return groups


def build_mup_optimizer(model: nn.Module, config, base_dim: int, base_lr: float,
                         weight_decay: float = 0.0) -> torch.optim.AdamW:
    """Construct AdamW after stripping informational muP group fields."""
    groups = mup_param_groups(model, config, base_dim, base_lr, weight_decay)
    clean_groups = [
        {key: value for key, value in group.items() if not key.startswith("mup_")}
        for group in groups
    ]
    return torch.optim.AdamW(clean_groups)
