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
