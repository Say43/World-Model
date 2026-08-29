"""M0: autoencoder-ceiling comparison.

The frozen AE is the hard upper bound for everything trained downstream of
it (CLAUDE.md M0 gate). This module reconstructs a fixed set of rendered
frames through each frozen AE candidate and reports PSNR/LPIPS, so the
choice of AE and resolution is made before any GPU-hour is spent training
against it.

Candidates are restricted to <=64 tokens/frame per the measured-on-T4
profiling in CLAUDE.md: 256 tokens/frame OOMs the 40M model and leaves too
few steps per M3 ablation arm even at 15M, so SD-VAE f8 @ 256px is excluded
before it is ever measured here.
"""
from __future__ import annotations

import dataclasses
from typing import Callable, List

import numpy as np

from .metrics import psnr


@dataclasses.dataclass(frozen=True)
class AECandidate:
    name: str
    resolution: int  # square: resolution x resolution
    tokens_per_frame: int  # spatial tokens after encoding, at this resolution
    license: str
    loader: Callable[[], "FrozenAE"]  # deferred: only imports torch/weights when actually used


class FrozenAE:
    """Interface every AE candidate's loader must return.

    Kept minimal on purpose: M0 only needs encode -> decode -> compare, not
    anything a training loop would need (no gradient, no batching contract
    beyond leading batch dim).
    """

    def reconstruct(self, frames: np.ndarray) -> np.ndarray:
        """frames: (N, H, W, 3) float32 in [0, 1]. Returns the same shape,
        each frame passed through encode then decode."""
        raise NotImplementedError


def _load_dc_ae(model_id: str, resolution: int) -> FrozenAE:
    """Loads a frozen DC-AE (Apache-2.0, https://github.com/mit-han-lab/efficientvit).

    Deferred import: this function is only called when a candidate is
    actually run, so importing ae_ceiling.py never requires torch/diffusers
    or a network connection.
    """
    import torch
    from diffusers import AutoencoderDC

    class _DCAE(FrozenAE):
        def __init__(self):
            self.model = AutoencoderDC.from_pretrained(model_id, torch_dtype=torch.float32)
            self.model.eval()
            self.resolution = resolution

        @torch.no_grad()
        def reconstruct(self, frames: np.ndarray) -> np.ndarray:
            x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() * 2.0 - 1.0
            latent = self.model.encode(x).latent
            out = self.model.decode(latent).sample
            out = ((out.clamp(-1, 1) + 1.0) / 2.0).permute(0, 2, 3, 1).numpy()
            return out.astype(np.float32)

    return _DCAE()


# Registry of candidates, restricted to <=64 tokens/frame (see module docstring).
# DC-AE compresses by 32x (f32) or 16x (f16) spatially; token grid is
# (resolution / factor)^2.
CANDIDATES: List[AECandidate] = [
    # Repo IDs must carry the "-diffusers" suffix: the plain mit-han-lab/*
    # repos ship the original (non-diffusers) checkpoint format and do not
    # load via AutoencoderDC.from_pretrained. There is also no f16 variant --
    # the DC-AE family is f32/f64/f128 (verified against the diffusers docs);
    # an earlier version of this list assumed f16 existed and it does not.
    AECandidate(
        name="dc-ae-f32-sana@128px", resolution=128, tokens_per_frame=16,
        license="Apache-2.0",
        loader=lambda: _load_dc_ae("mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers", 128),
    ),
    AECandidate(
        name="dc-ae-f32-sana@256px", resolution=256, tokens_per_frame=64,
        license="Apache-2.0",
        loader=lambda: _load_dc_ae("mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers", 256),
    ),
    AECandidate(
        # f64, trained on ImageNet reconstruction rather than as a
        # text-to-image diffusion backbone AE -- a useful contrast to the
        # sana variants above at the same 16 tok/frame budget.
        name="dc-ae-f64-in@256px", resolution=256, tokens_per_frame=16,
        license="Apache-2.0",
        loader=lambda: _load_dc_ae("mit-han-lab/dc-ae-f64c128-in-1.0-diffusers", 256),
    ),
]


def evaluate_candidate(candidate: AECandidate, frames: np.ndarray, lpips_metric=None) -> dict:
    """frames: (N, H, W, 3) float32 in [0, 1], H == W == candidate.resolution."""
    assert frames.shape[1] == candidate.resolution and frames.shape[2] == candidate.resolution, (
        f"{candidate.name} expects {candidate.resolution}px frames, got {frames.shape[1:3]}"
    )
    ae = candidate.loader()
    recon = ae.reconstruct(frames)

    psnr_values = [psnr(frames[i], recon[i]) for i in range(frames.shape[0])]
    finite = [v for v in psnr_values if np.isfinite(v)]
    result = {
        "name": candidate.name,
        "resolution": candidate.resolution,
        "tokens_per_frame": candidate.tokens_per_frame,
        "license": candidate.license,
        "n_frames": frames.shape[0],
        "psnr_mean": float(np.mean(finite)) if finite else float("inf"),
        "psnr_min": float(np.min(finite)) if finite else float("inf"),
    }
    if lpips_metric is not None:
        lpips_values = [lpips_metric.compute(frames[i], recon[i]) for i in range(frames.shape[0])]
        result["lpips_mean"] = float(np.mean(lpips_values))
    return result
