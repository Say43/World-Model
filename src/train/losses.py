"""Rectified-flow training loss, wired to diffusion forcing.

`scripts/train.py`'s loss.target contract is `loss_fn(model, batch) ->
scalar tensor`; this module's `rectified_flow_loss` is that function.
"""
from __future__ import annotations

import torch

from src.model.diffusion_forcing import rectified_flow_target, sample_uniform_independent


def rectified_flow_loss(model, batch: dict) -> torch.Tensor:
    """batch: {"latents": (B,T,N,C), "poses": (B,T,4,4), "intrinsics": (B,T,4)}.

    Samples an independent per-frame timestep (diffusion forcing's training
    distribution), builds the noised input and velocity target via rectified
    flow, and returns the model's MSE against that target.
    """
    latents = batch["latents"]
    poses = batch["poses"]
    intrinsics = batch["intrinsics"]
    device = latents.device
    batch_size, context_length = latents.shape[0], latents.shape[1]

    t = sample_uniform_independent(batch_size, context_length, device=device)
    noise = torch.randn_like(latents)
    x_t, v_target = rectified_flow_target(latents, noise, t)

    v_pred = model(x_t, poses, intrinsics, t)
    return torch.nn.functional.mse_loss(v_pred, v_target)


def rectified_flow_loss_with_repa(model, batch: dict) -> torch.Tensor:
    """The REPA arm of M3's ablation: flow-matching loss plus a weighted
    alignment term against precomputed DINOv2 features.

    Same `loss_fn(model, batch)` contract, so the two arms differ only in
    the config's `loss.target` -- training loop, data, schedule and
    optimizer stay identical, which is what makes the comparison an ablation
    rather than two unrelated runs.

    The alignment weight lives on the model (set by scripts/train.py from the
    config) rather than as a constant here: it is a hyperparameter, and
    CLAUDE.md forbids those in code.
    """
    if "dinov2" not in batch:
        raise KeyError(
            "REPA loss needs 'dinov2' in the batch; build the dataloader with "
            "with_dinov2=True against a dataset precomputed using --with-dinov2"
        )
    latents = batch["latents"]
    poses = batch["poses"]
    intrinsics = batch["intrinsics"]
    batch_size, context_length = latents.shape[0], latents.shape[1]

    t = sample_uniform_independent(batch_size, context_length, device=latents.device)
    noise = torch.randn_like(latents)
    x_t, v_target = rectified_flow_target(latents, noise, t)

    v_pred, projected = model(x_t, poses, intrinsics, t, return_repa=True)
    flow = torch.nn.functional.mse_loss(v_pred, v_target)

    from src.train.repa import repa_loss  # local: baseline arm never imports REPA

    align = repa_loss(projected, batch["dinov2"].to(projected.dtype))
    return flow + _repa_weight(model) * align


def _repa_weight(model) -> float:
    from src.train.utils import unwrap_compiled

    weight = getattr(unwrap_compiled(model), "repa_weight", None)
    if weight is None:
        raise AttributeError(
            "model has no repa_weight; scripts/train.py sets it from the "
            "config's repa.weight when the REPA loss is selected"
        )
    return weight
