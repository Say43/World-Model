"""REPA: align a mid-depth DiT hidden state with frozen DINOv2 features.

The other half of M3's ablation. REPA's claim is that diffusion
transformers spend much of early training learning representations a
self-supervised encoder already has, and that regressing an intermediate
hidden state onto those features as an auxiliary loss removes that
duplicated work.

Implementation follows the paper's shape: take ONE intermediate layer's
output, project it through a small trainable MLP to the encoder's feature
dimension, and maximize cosine similarity with the frozen target,
token-by-token. The projection head is trainable and thrown away after
training -- it exists so the DiT's width and DINOv2's 384 dims need not
match, not to add capacity to the model.

Which layer: the paper aligns roughly a third to halfway up the stack, on
the reasoning that later layers specialize toward the denoising output and
no longer look like general-purpose features. `default_align_layer` uses
depth // 3, and it is a config parameter rather than a constant so M3 can
vary it if the ablation motivates that.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def default_align_layer(depth: int) -> int:
    """Index of the block whose output REPA aligns (0-based)."""
    return max(0, depth // 3)


class RepaHead(nn.Module):
    """Trainable projection from DiT hidden width to the frozen encoder's
    feature dimension.

    Deliberately small (one hidden layer): REPA's benefit is supposed to
    come from the alignment signal reaching the DiT's own trunk, not from a
    projection head powerful enough to paper over a mismatch by itself. A
    head with real capacity would make the ablation measure the head rather
    than the method.
    """

    def __init__(self, dim: int, feature_dim: int, hidden_mult: int = 2):
        super().__init__()
        hidden = dim * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, feature_dim),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states)


def attach_repa_head(model, feature_dim: int, align_layer: int | None = None,
                     hidden_mult: int = 2) -> RepaHead:
    """Install a REPA head on a CausalDiT, in place; returns the head.

    Attached as a submodule rather than kept beside the model so that (a) the
    optimizer picks it up through `model.parameters()` like everything else,
    (b) it is saved and resumed with the checkpoint, and (c) under DDP its
    gradients are all-reduced, which requires the head to be used inside the
    wrapped module's own forward (see `CausalDiT.forward`'s `return_repa`).

    Must be called before the model is wrapped for DDP or compiled: DDP
    inspects the parameter set once, at construction.
    """
    if model.repa_head is not None:
        raise ValueError("model already has a REPA head attached")
    depth = model.num_layers()
    layer = default_align_layer(depth) if align_layer is None else align_layer
    if not 0 <= layer < depth:
        raise ValueError(f"align_layer={layer} out of range for depth {depth}")
    head = RepaHead(model.config.dim, feature_dim, hidden_mult=hidden_mult)
    model.repa_head = head
    model.repa_align_layer = layer
    return head


def repa_loss(projected: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
    """Mean negative cosine similarity, per token.

    `projected`: (B, T, N, feature_dim) -- the DiT hidden state after
    RepaHead. `target_features`: (B, T, N, feature_dim) -- frozen DINOv2
    features on the same token grid.

    Cosine rather than MSE because the frozen features' scale carries no
    meaning the DiT should be forced to reproduce; only their direction
    does. Returned as a loss (lower is better), so 0 means perfectly
    aligned and 1 means orthogonal.
    """
    if projected.shape != target_features.shape:
        raise ValueError(
            f"REPA shape mismatch: projected {tuple(projected.shape)} vs "
            f"target {tuple(target_features.shape)}"
        )
    similarity = F.cosine_similarity(projected, target_features, dim=-1)
    return 1.0 - similarity.mean()
