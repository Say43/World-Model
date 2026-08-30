"""The AE chosen by M0: dc-ae-f64c128-in-1.0-diffusers @ 256px, 16 tokens/frame.

CLAUDE.md decision log, 2026-08-30: beat every other M0 candidate on both
PSNR and LPIPS at a quarter of the tokens of the runner-up. This module is
the single place that encodes that choice, so the precompute pipeline and
any future AE swap touch one file, not every caller.
"""
from __future__ import annotations

MODEL_ID = "mit-han-lab/dc-ae-f64c128-in-1.0-diffusers"
RESOLUTION = 256
TOKENS_PER_FRAME = 16  # (256 / 64)^2
LATENT_CHANNELS = 128
LATENT_GRID = 4  # sqrt(TOKENS_PER_FRAME)


def load_chosen_ae(device: str = "cuda"):
    """Returns a frozen, eval-mode AutoencoderDC on `device`.

    Deferred import: this module can be imported without torch/diffusers
    installed; only calling this function requires them.
    """
    import torch
    from diffusers import AutoencoderDC

    model = AutoencoderDC.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def encode_frames(model, frames, device: str = "cuda"):
    """frames: (N, 256, 256, 3) float32 numpy in [0, 1].

    Returns (N, TOKENS_PER_FRAME, LATENT_CHANNELS) float32 numpy: the AE's
    (LATENT_GRID, LATENT_GRID, LATENT_CHANNELS) latent grid flattened to a
    token sequence, matching CausalDiT's (context_length, tokens_per_frame,
    latent_channels) input contract.
    """
    import torch

    with torch.no_grad():
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float().to(device) * 2.0 - 1.0
        latent = model.encode(x).latent  # (N, C, H, W)
        # Standard diffusers convention (see AutoencoderKL/SD-VAE usage):
        # raw encoder output is scaled to roughly unit variance before being
        # used as a diffusion target, and divided back out before decode.
        # The M1 smoke run's first real Kaggle attempt hit an inf gradient
        # at step 55 without this -- unscaled DC-AE latents at an unknown
        # magnitude, used directly as a rectified-flow target, is a
        # plausible fp16 instability source. M0's PSNR/LPIPS numbers are
        # unaffected: ae_ceiling.py round-trips encode->decode with no
        # scaling touched at all, which is scale-neutral by construction.
        scaling_factor = getattr(model.config, "scaling_factor", 1.0)
        latent = latent * scaling_factor
        n, c, h, w = latent.shape
        assert h * w == TOKENS_PER_FRAME, (
            f"expected {TOKENS_PER_FRAME} tokens/frame, got {h}x{w}={h * w} "
            f"-- MODEL_ID or RESOLUTION in this module is stale"
        )
        tokens = latent.permute(0, 2, 3, 1).reshape(n, h * w, c)
        return tokens.cpu().numpy().astype("float32")


def decode_tokens(model, tokens, device: str = "cuda"):
    """Inverse of encode_frames: (N, TOKENS_PER_FRAME, LATENT_CHANNELS) ->
    (N, RESOLUTION, RESOLUTION, 3) float32 numpy in [0, 1].

    Divides out the same scaling_factor encode_frames applied, so a
    round-trip through encode_frames -> decode_tokens reproduces the raw
    AE's own encode -> decode behavior (the M0 ceiling this is measured
    against was established that way, in ae_ceiling.py).
    """
    import torch

    with torch.no_grad():
        n = tokens.shape[0]
        latent = torch.from_numpy(tokens).float().to(device)
        latent = latent.reshape(n, LATENT_GRID, LATENT_GRID, LATENT_CHANNELS).permute(0, 3, 1, 2)
        scaling_factor = getattr(model.config, "scaling_factor", 1.0)
        latent = latent / scaling_factor
        out = model.decode(latent).sample
        out = ((out.clamp(-1, 1) + 1.0) / 2.0).permute(0, 2, 3, 1).cpu().numpy()
        return out.astype("float32")
