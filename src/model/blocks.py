"""Low-level building blocks for the nanoWM causal DiT: RMSNorm, QK-norm,
SwiGLU MLP, and the adaLN-single modulation mechanism.

All modules are pure `nn.Module`s with no training-loop logic and no
hardcoded hyperparameters -- every size is a constructor argument, driven
by `src.model.dit.DiTConfig` at the call site.

fp16 / sm75 (T4) notes, since these primitives are what actually touches
numerics under AMP:
  - `RMSNorm` upcasts to fp32 internally for the mean-square reduction and
    casts back to the input dtype afterwards. Under fp16 AMP the squared
    activations can overflow fp16's ~65504 max well before they'd overflow
    fp32, so this upcast is not optional polish -- skipping it is a
    plausible source of silent NaNs during training.
  - QK-norm (RMSNorm applied to q/k along the head dimension, before the
    attention matmul) is exactly the mitigation the project decided on for
    attention-logit blowup; see the module docstring on `Attention` below.
  - The adaLN-single modulation produces a multiplicative `scale` that is
    used as `x * (1 + scale)`. Uninitialized, `scale` can be far from zero
    early in training (before the modulation MLP's output layer settles
    near its zero-init), which combined with fp16 can amplify activations.
    The output projection of the modulation MLP is zero-initialized (a
    standard DiT trick) specifically so `(1 + scale) == 1` and `gate == 0`
    at initialization, making the whole conditioning path a no-op at step 0
    and fp16-safe from the very first forward pass.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-mean-square layer norm, no mean-centering.

    `elementwise_affine=False` is the default here: in the DiT blocks below,
    scale/shift is supplied externally by adaLN-single modulation, so a
    second learnable affine transform inside the norm itself would be
    redundant parameters that the model has to learn to keep near identity.
    QK-norm (see `Attention`) does want its own tiny learnable scale (it is
    not adaLN-modulated), so it constructs this with `elementwise_affine=True`.
    """

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x_f32 = x.float()
        rms = torch.rsqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = (x_f32 * rms).to(in_dtype)
        if self.weight is not None:
            out = out * self.weight
        return out


class SwiGLU(nn.Module):
    """SwiGLU-gated MLP: `down(silu(gate(x)) * up(x))`.

    `hidden_dim` is a plain constructor argument (see `dit.hidden_dim_swiglu`
    for the conventional 8/3x sizing used by the presets, so a gated MLP
    costs about the same parameter count as a plain 4x MLP) -- never
    recomputed implicitly here, so callers stay in control of the exact
    parameter count.
    """

    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_up = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_down = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


def hidden_dim_swiglu(dim: int, mult: float = 4.0, multiple_of: int = 32) -> int:
    """Conventional SwiGLU hidden-size rule: scale the "equivalent" 4x plain
    MLP width down by 2/3 (since SwiGLU has 3 weight matrices instead of 2)
    so parameter count roughly matches a non-gated `mult`x MLP, then round
    up to `multiple_of` for clean matmul tiling on the T4 tensor cores.
    """
    hidden = int(2 * mult * dim / 3)
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


class Attention(nn.Module):
    """Multi-head self-attention with QK-norm, run through
    `F.scaled_dot_product_attention`.

    QK-norm (RMSNorm applied per-head to q and k before the dot product)
    bounds attention logits independently of activation scale, which is the
    project's chosen mitigation for fp16 attention-logit overflow (see
    CLAUDE.md hardware constraints: fp16 AMP, no bf16's wider dynamic
    range). Without it, q/k magnitudes drifting up during training can
    push `q @ k^T / sqrt(d)` outside fp16 range before the softmax ever
    sees it.

    An explicit boolean `attn_mask` (additive-bias-free, True = "allowed to
    attend") is accepted so the causal DiT can pass a block-causal
    (frame-granular, not just token-granular) mask -- see `dit.py`. SDPA is
    invoked without `is_causal` in that case since PyTorch's built-in causal
    mode is strictly token-lower-triangular and cannot express "attend to
    all tokens in all frames <= t".

    Backend: no explicit backend pin is done here (the mem-efficient
    kernel is what remains available and correct on sm75 once flash
    attention is unavailable/unsupported for the given dtype/mask
    combination); `torch.backends.cuda.sdp_kernel` context managers are a
    training-loop / t4-profiler concern, not this module's.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qk_norm_eps: float = 1e-6,
        attn_dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_dropout = attn_dropout
        # Set by src.train.mup.configure_mup_model. None preserves ordinary
        # PyTorch 1/sqrt(d_head) scaling for non-muP runs.
        self.mup_attention_scale = None

        self.qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.q_norm = RMSNorm(self.head_dim, eps=qk_norm_eps, elementwise_affine=True)
        self.k_norm = RMSNorm(self.head_dim, eps=qk_norm_eps, elementwise_affine=True)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        return x.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,L,Dh)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor = None,
        kv_cache=None,
        layer_idx: int = None,
    ) -> torch.Tensor:
        """`x`: (B, L, dim). `attn_mask`: broadcastable bool (..., Lq, Lk),
        True where attention is allowed. When `kv_cache` is given, this
        layer's freshly computed k/v are appended to the cache (see
        `kv_cache.py`) and attention is computed against the full cached
        history -- this is what makes incremental generation equivalent to
        a full causal forward pass.
        """
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if kv_cache is not None:
            k, v = kv_cache.append(layer_idx, k, v)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            scale=self.mup_attention_scale,
        )
        out = out.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.dim)
        return self.proj(out)


class AdaLNSingle(nn.Module):
    """Shared ("single") adaLN modulation, PixArt-alpha style.

    Instead of giving every transformer block its own `dim -> 6*dim` MLP
    (which would dominate the parameter budget at these small model
    sizes), a single global MLP maps the timestep/noise-level embedding to
    a base 6-way modulation vector (shift/scale/gate, for both the
    attention and MLP sub-layers). Each block then adds its own small
    per-block learnable bias (`6*dim` parameters, vs. `dim*6*dim` for a
    full per-block MLP) before splitting into the six components. This is
    the "-single" in adaLN-single: one MLP, many cheap per-block offsets.

    The global MLP's final linear is zero-initialized so every block starts
    at `shift=0, scale=0 (-> 1+scale=1), gate=0` -- an identity no-op at
    init, which matters for fp16 stability (see module docstring).

    Diffusion forcing gives every frame its own noise level, so the
    embedding this consumes has shape (B, T, embed_dim) (per-frame), not
    (B, embed_dim) (per-sequence) -- the modulation output is therefore
    (B, T, 6*dim) and must be broadcast across the `tokens_per_frame` axis
    by the caller before being applied to (B, T, N, dim) activations.
    """

    def __init__(self, embed_dim: int, dim: int, num_chunks: int = 6):
        super().__init__()
        self.dim = dim
        self.num_chunks = num_chunks
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, num_chunks * dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, t_embed: torch.Tensor) -> torch.Tensor:
        """`t_embed`: (..., embed_dim) -> (..., num_chunks*dim)."""
        return self.mlp(t_embed)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """`x * (1 + scale) + shift`, broadcasting `shift`/`scale` over any
    leading dims of `x` they don't already share (e.g. the token axis
    within a frame, when shift/scale come per-frame)."""
    return x * (1 + scale) + shift


def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Standard transformer/diffusion sinusoidal embedding.

    `t`: arbitrary-shaped float tensor (e.g. (B, T) per-frame noise
    levels/timesteps). Returns `(*t.shape, dim)`. Computed in fp32
    regardless of `t`'s dtype -- the frequency range spans many orders of
    magnitude and is exactly the kind of thing that loses precision in
    fp16.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float().unsqueeze(-1) * freqs
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb
