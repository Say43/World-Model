"""Diffusion forcing: per-frame independent noise levels.

Standard video diffusion trains every frame in a clip at the *same* noise
level per step. Diffusion forcing instead samples an independent rectified-
flow timestep `t_i in [0, 1]` per frame, which is what lets the causal DiT
be rolled out autoregressively at inference: earlier frames can sit at
`t=0` (clean, already generated) while the frame currently being denoised
sits at some `t>0`, all in a single forward pass, with the model never
having seen a training distribution that assumes uniform noise across the
clip.

This module only produces the noise-level schedules; it does not touch the
DiT's parameters. `dit.CausalDiT.forward` takes `t` as `(B, T)` (one
timestep per frame) precisely so these schedules plug in directly.
"""
from __future__ import annotations

import torch


def sample_uniform_independent(
    batch_size: int, context_length: int, device=None, generator: torch.Generator = None
) -> torch.Tensor:
    """Fully independent per-frame, per-sample timesteps, uniform in
    [0, 1). This is the training-time distribution: every frame in every
    clip gets its own rectified-flow noise level, decorrelated from its
    neighbors, so the model cannot shortcut by assuming a shared clip-level
    noise level.
    """
    return torch.rand(batch_size, context_length, device=device, generator=generator)


def sample_autoregressive_rollout(
    batch_size: int,
    context_length: int,
    num_clean: int,
    active_t: float,
    device=None,
) -> torch.Tensor:
    """The inference-time schedule for rollout: the first `num_clean`
    frames are already-generated context (`t=0`, clean), the next frame is
    "being denoised" at `active_t`, and any frames beyond that (not yet
    reached in an autoregressive generation loop) are placeholders at
    `t=1` (pure noise) -- they contribute no information to earlier frames
    under the causal mask, but must still have a defined timestep since the
    model's forward pass is called on the full context window.

    Args:
        num_clean: number of frames already fully denoised (t=0), 0 <=
            num_clean < context_length.
        active_t: rectified-flow timestep of the frame currently being
            denoised (index `num_clean`).
    """
    if not (0 <= num_clean < context_length):
        raise ValueError(f"num_clean must be in [0, {context_length}), got {num_clean}")
    t = torch.ones(batch_size, context_length, device=device)
    t[:, :num_clean] = 0.0
    t[:, num_clean] = active_t
    return t


def rectified_flow_target(
    x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Rectified-flow (velocity parametrization) noised sample and target.

    `x0`: clean latents. `x1`: standard-normal noise, same shape as `x0`.
    `t`: per-frame timestep, broadcastable to `x0`'s shape (typically
    `(B, T)` broadcast against `(B, T, N, C)`).

    Interpolant: `x_t = (1 - t) * x0 + t * x1`. Velocity target:
    `v = x1 - x0` (constant along the straight-line path, independent of
    `t` -- this is what makes rectified flow's target simple to fit).
    Returns `(x_t, v_target)`.
    """
    while t.dim() < x0.dim():
        t = t.unsqueeze(-1)
    x_t = (1 - t) * x0 + t * x1
    v_target = x1 - x0
    return x_t, v_target
