"""KV-cache for incremental causal generation with the nanoWM DiT.

One cache instance holds, per transformer layer, the running key/value
tensors for every token generated so far (all frames up to and including
the most recently appended one). `Attention.forward` (see `blocks.py`)
appends this layer's new k/v into the cache and attends against the full
cached history, so autoregressive rollout never recomputes attention for
already-generated frames.

Deliberately dumb and stateful (a list of tensors per layer, resized by
concatenation) rather than a fixed pre-allocated ring buffer: the causal
DiT context length (16 frames, `DiTConfig.context_length`) already bounds
how much ever needs to be cached for the *local* attention window, and
`retrieval.py`'s frustum-overlap selection is a separate, explicit
mechanism for anything older than that -- this cache does not need to know
about either policy, it just stores whatever `append` is given.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch


class KVCache:
    """Per-layer running key/value cache.

    Shapes follow `Attention`'s internal layout: `(B, num_heads, L, head_dim)`
    for both k and v, concatenated along the sequence axis `L` on each
    `append`.
    """

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self._k: List[Optional[torch.Tensor]] = [None] * num_layers
        self._v: List[Optional[torch.Tensor]] = [None] * num_layers

    def append(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append this layer's freshly computed k/v (from the tokens in the
        current forward call) to the cache and return the full accumulated
        k/v (including what was just appended)."""
        if self._k[layer_idx] is None:
            self._k[layer_idx] = k
            self._v[layer_idx] = v
        else:
            self._k[layer_idx] = torch.cat([self._k[layer_idx], k], dim=2)
            self._v[layer_idx] = torch.cat([self._v[layer_idx], v], dim=2)
        return self._k[layer_idx], self._v[layer_idx]

    def length(self) -> int:
        """Number of cached tokens (0 if nothing appended yet)."""
        return 0 if self._k[0] is None else self._k[0].shape[2]

    def reset(self) -> None:
        self._k = [None] * self.num_layers
        self._v = [None] * self.num_layers

    def to(self, device) -> "KVCache":
        for i in range(self.num_layers):
            if self._k[i] is not None:
                self._k[i] = self._k[i].to(device)
                self._v[i] = self._v[i].to(device)
        return self
