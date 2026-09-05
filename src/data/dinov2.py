"""Frozen DINOv2-small features, pooled to the DiT's token grid, for REPA.

REPA (representation alignment) trains the diffusion transformer's
intermediate hidden states to predict a frozen self-supervised encoder's
features for the same frame. That auxiliary target only makes sense if the
two live on the same spatial grid, so DINOv2's patch tokens are average-
pooled from its native grid down to the DiT's `LATENT_GRID` x `LATENT_GRID`
token layout (4x4 for the M0-chosen AE) before being stored.

Two sizing details that are easy to get wrong:
  - DINOv2 uses 14px patches, so its input side must be a multiple of 14.
    The project renders at 256px, which is not (256 = 14*18 + 4), so frames
    are resized to 252 = 14*18 first. That is an 8-pixel resize, chosen over
    padding so no synthetic border enters the features.
  - 18x18 patches pool evenly into neither 4x4 nor 16 tokens by simple
    reshaping, so adaptive average pooling handles the 18 -> 4 reduction.

Weights: facebook/dinov2-small, Apache-2.0 (the project's licensing
constraint rules out anything more restrictive; the M0 autoencoder choice
was made on the same basis).
"""
from __future__ import annotations

import numpy as np

MODEL_ID = "facebook/dinov2-small"
FEATURE_DIM = 384  # DINOv2-small hidden size
PATCH_SIZE = 14
INPUT_RESOLUTION = 252  # 14 * 18, the nearest multiple of 14 below the 256px render


def load_dinov2(device: str = "cuda"):
    """Returns a frozen, eval-mode DINOv2-small.

    Deferred import so this module can be imported (and the precompute
    script's other tiers used) without transformers installed.
    """
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(MODEL_ID)
    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def encode_frames(model, frames: np.ndarray, latent_grid: int, device: str = "cuda",
                   batch_size: int = 16) -> np.ndarray:
    """frames: (N, H, W, 3) float32 in [0, 1].

    Returns (N, latent_grid**2, FEATURE_DIM) float32 -- patch features
    average-pooled onto the DiT's token grid, in the same row-major token
    order `src/eval/chosen_ae.encode_frames` produces for the AE latents, so
    token i of one lines up with token i of the other.

    Batched for the same reason the AE encoder is: the M0 run OOM-killed a
    Kaggle session by pushing 200 frames through a frozen encoder at once
    (CLAUDE.md decision log).
    """
    import torch
    import torch.nn.functional as F

    # ImageNet statistics, which DINOv2 was trained with.
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    outputs = []
    with torch.no_grad():
        for start in range(0, frames.shape[0], batch_size):
            chunk = frames[start:start + batch_size]
            x = torch.from_numpy(chunk).permute(0, 3, 1, 2).float().to(device)
            x = F.interpolate(x, size=(INPUT_RESOLUTION, INPUT_RESOLUTION),
                              mode="bilinear", align_corners=False)
            x = (x - mean) / std

            hidden = model(pixel_values=x).last_hidden_state  # (B, 1 + num_patches, dim)
            patches = hidden[:, 1:, :]  # drop the CLS token; REPA aligns spatially

            side = INPUT_RESOLUTION // PATCH_SIZE
            batch, num_patches, dim = patches.shape
            if num_patches != side * side:
                raise ValueError(
                    f"expected {side*side} patch tokens at {INPUT_RESOLUTION}px, got {num_patches}"
                )
            grid = patches.transpose(1, 2).reshape(batch, dim, side, side)
            pooled = F.adaptive_avg_pool2d(grid, (latent_grid, latent_grid))
            tokens = pooled.reshape(batch, dim, latent_grid * latent_grid).transpose(1, 2)
            outputs.append(tokens.float().cpu().numpy())
            del x, hidden, patches, grid, pooled, tokens
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    return np.concatenate(outputs, axis=0).astype("float32")
