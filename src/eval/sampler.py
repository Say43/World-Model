"""Rectified-flow sampling: integrate the learned velocity field from noise
to a clean latent.

Nothing in src/model/ or src/train/ actually generates a frame -- the
training loss only checks that the predicted velocity matches the
straight-line target at a random t. Whether the model is useful for
*generation* can only be checked by integrating that velocity field from
t=1 (pure noise) to t=0 (clean latent) and looking at what comes out. That
is what M1's gate needs: reconstruction quality against the M0 ceiling,
not a falling training loss.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def sample_frames(model, poses: torch.Tensor, intrinsics: torch.Tensor,
                   num_steps: int = 50, generator: torch.Generator = None) -> torch.Tensor:
    """Euler-integrates the model's velocity prediction from noise to a
    clean latent, non-causally: every frame is denoised together at the
    same shared timestep, one Euler step at a time. This is standard
    (non-autoregressive) rectified-flow sampling -- distinct from the
    causal, per-frame-independent-t rollout diffusion forcing enables,
    which src/model/diffusion_forcing.sample_autoregressive_rollout
    schedules for but which needs KV-cache-driven generation to exercise
    (a separate, later concern, not needed to answer M1's reconstruction
    question).

    poses, intrinsics: (B, T, ...) matching the trajectory whose ground
    truth latents this is meant to reconstruct.
    Returns (B, T, tokens_per_frame, latent_channels) clean latents at t=0.
    """
    device = poses.device
    batch_size, context_length = poses.shape[0], poses.shape[1]
    cfg = model.config if hasattr(model, "config") else model.module.config

    x = torch.randn(
        batch_size, context_length, cfg.tokens_per_frame, cfg.latent_channels,
        device=device, generator=generator,
    )
    dt = 1.0 / num_steps
    for step in range(num_steps):
        t_value = 1.0 - step * dt
        t = torch.full((batch_size, context_length), t_value, device=device)
        velocity = model(x, poses, intrinsics, t)
        # dx/dt = -v under this module's (x1 - x0) velocity convention
        # integrated from t=1 down to t=0 -- see rectified_flow_target's
        # docstring in diffusion_forcing.py for the sign convention.
        x = x - velocity * dt
    return x
