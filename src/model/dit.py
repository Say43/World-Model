"""Causal, pose-conditioned Diffusion Transformer for nanoWM.

Consumes per-frame latents (from the M0-selected autoencoder, channel
count and tokens-per-frame both config-driven -- see the M0 decision log
in CLAUDE.md, still open at the time this module was written) plus
per-frame camera poses/intrinsics (the `data-engineer` interface, see
`raymap.py`), and predicts the rectified-flow velocity field.

Causality is over the *frame* axis only: a token belonging to frame `t` may
attend to any token (including all spatial patches) in frames `<= t`, never
to a token in a frame `> t`. Within a frame, attention is unrestricted
(bidirectional over the `tokens_per_frame` patches) -- video generation
needs full spatial context within a frame, only the temporal/causal
structure needs to be autoregressive. See `build_block_causal_mask`.

Diffusion forcing (`diffusion_forcing.py`) means every frame in the context
window carries its *own* rectified-flow timestep `t`, not one shared value
-- `forward`'s `t` argument is `(B, T)`, and the adaLN-single modulation
this produces is per-frame, broadcast over the `tokens_per_frame` axis
before being applied (see `DiTBlock`).

## Parameter budget

Total parameter count is, to a very good approximation, INDEPENDENT of
`tokens_per_frame` and `context_length`: those only change the sequence
length fed to a fixed number of `nn.Linear` weights, not the weights
themselves (the sole exception is `frame_pos_embed`, sized
`context_length * dim`, which is tiny relative to the transformer body at
every preset below). This is what lets the M0 autoencoder decision
(tokens_per_frame in {16, 64, 256, 1024}) be made independently of the
M1-M4 model-size gates (5M / 15M / ~40M) -- see `count_parameters` and the
three presets for exact numbers, reverified by `tests/test_model_dit.py`.

## fp16 risk notes (sm75, no bf16, AMP)

  - RMSNorm upcasts to fp32 for its reduction (see `blocks.py`) -- required
    since squared fp16 activations can overflow well before fp32 would.
  - QK-norm bounds attention logit magnitude independent of activation
    drift; this is the project's chosen mitigation for fp16 softmax
    overflow (see `blocks.py:Attention`).
  - adaLN-single's modulation MLPs are zero-initialized (both the shared
    per-block one and the final layer's), so `scale=0, gate=0` at init --
    the conditioning path starts as an exact no-op. This matters
    specifically because rectified-flow timesteps near `t=0` or `t=1` are
    the two extremes of the interpolant, and an *untrained* modulation
    reacting sharply to the timestep embedding at those extremes is a
    plausible source of early-training fp16 blowup; zero-init defers any
    such sensitivity until the model has actually learned it should exist.
  - `sinusoidal_embedding` is computed in fp32 even when called under
    autocast, since its frequency range spans several orders of magnitude.
  - Residual streams (`x = x + gate * sublayer_out`) are left in whatever
    dtype autocast assigns to the surrounding matmuls (fp16 under AMP);
    this is standard and fine as long as the two points above hold, but if
    NaNs do appear during training, the gate values immediately after a
    fresh un-warmed-up adaLN update (i.e. right after the zero-init
    identity phase ends) are the first place to look -- t4-profiler /
    train-infra own diagnosing that at runtime, this module does not
    clamp or guard against it beyond the init-time mitigation above.
  - The block-causal attention mask is a full `(L, L)` (or `(L, L+cache)`)
    boolean tensor built fresh per forward call (see
    `build_block_causal_mask`); at the largest plausible configuration
    (`context_length=16`, `tokens_per_frame=1024` -> `L=16384`) that is
    ~268M bool elements (~268MB), on top of whatever SDPA's own
    mem-efficient backend allocates for the attention matrix itself. Not a
    correctness risk, but worth t4-profiler's attention if the 1024
    tokens/frame (SD-VAE f8 @ 256px) branch of the M0 decision is ever
    actually selected -- the smaller candidates (16/64/256) are all
    comfortably cheap.
"""
from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.model.blocks import (
    AdaLNSingle,
    Attention,
    RMSNorm,
    SwiGLU,
    hidden_dim_swiglu,
    modulate,
    sinusoidal_embedding,
)
from src.model.kv_cache import KVCache
from src.model.raymap import RaymapEncoder, plucker_raymap


@dataclasses.dataclass
class DiTConfig:
    """All hyperparameters for `CausalDiT`. Nothing below is read from
    anywhere but this dataclass -- no module-level constants stand in for
    a hyperparameter, per the project's config-driven rule.
    """

    context_length: int = 16  # frames; hard project decision, still a parameter here
    tokens_per_frame: int = 64  # open until M0 (16/64/256/1024 candidates)
    raymap_resolution: Tuple[int, int] = (8, 8)  # (H, W); H*W must == tokens_per_frame
    latent_channels: int = 4  # AE latent channel count, open until M0

    dim: int = 256
    depth: int = 6
    num_heads: int = 8
    mlp_mult: float = 4.0  # SwiGLU effective-width multiplier, see hidden_dim_swiglu
    mlp_multiple_of: int = 32

    qk_norm_eps: float = 1e-6
    norm_eps: float = 1e-6
    time_embed_dim: int = 256  # sinusoidal frequency dim fed into the timestep MLP
    raymap_hidden_mult: int = 2

    attn_dropout: float = 0.0
    mlp_dropout: float = 0.0

    def __post_init__(self) -> None:
        h, w = self.raymap_resolution
        if h * w != self.tokens_per_frame:
            raise ValueError(
                f"raymap_resolution {self.raymap_resolution} (H*W={h*w}) must equal "
                f"tokens_per_frame ({self.tokens_per_frame})"
            )
        if self.dim % self.num_heads != 0:
            raise ValueError(f"dim ({self.dim}) must be divisible by num_heads ({self.num_heads})")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")


def build_block_causal_mask(
    chunk_frames: int, tokens_per_frame: int, cache_len: int, device
) -> torch.Tensor:
    """Boolean attention mask, `True` = attention allowed.

    Shape `(chunk_frames * tokens_per_frame, cache_len + chunk_frames *
    tokens_per_frame)`. The first `cache_len` key columns (already-cached
    frames, all strictly earlier than anything in this chunk) are allowed
    unconditionally; the remaining `chunk_frames * tokens_per_frame`
    columns (this chunk's own keys) follow frame-block causality: query
    token in frame `i` (of this chunk) may attend to key token in frame `j`
    (of this chunk) iff `j <= i`. With `cache_len=0` and `chunk_frames ==
    context_length` this is exactly the full-sequence training-time mask;
    with `cache_len>0` and `chunk_frames==1` it is the incremental
    decode-step mask used with a `KVCache`.
    """
    new_frame_idx = torch.arange(chunk_frames, device=device).repeat_interleave(tokens_per_frame)
    new_allowed = new_frame_idx.unsqueeze(1) >= new_frame_idx.unsqueeze(0)  # (Lq, Lq_new)
    if cache_len > 0:
        lq = chunk_frames * tokens_per_frame
        old_allowed = torch.ones(lq, cache_len, dtype=torch.bool, device=device)
        return torch.cat([old_allowed, new_allowed], dim=1)
    return new_allowed


class DiTBlock(nn.Module):
    """One causal DiT transformer block: pre-norm attention + pre-norm
    SwiGLU MLP, both adaLN-single-modulated. See `blocks.AdaLNSingle` for
    why modulation is split into one shared MLP plus a cheap per-block
    additive offset (`self.adaln_bias`).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden: int,
        qk_norm_eps: float,
        norm_eps: float,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=norm_eps, elementwise_affine=False)
        self.attn = Attention(dim, num_heads, qk_norm_eps=qk_norm_eps, attn_dropout=attn_dropout)
        self.norm2 = RMSNorm(dim, eps=norm_eps, elementwise_affine=False)
        self.mlp = SwiGLU(dim, mlp_hidden)
        self.mlp_drop = nn.Dropout(mlp_dropout) if mlp_dropout > 0 else nn.Identity()
        # Per-block adaLN-single offset: added to the shared global
        # modulation vector before splitting into the six chunks. Zero
        # init keeps this a no-op contribution at the start of training,
        # same rationale as the shared MLP's zero-init output layer.
        self.adaln_bias = nn.Parameter(torch.zeros(6 * dim))

    def forward(
        self,
        x: torch.Tensor,
        global_mod: torch.Tensor,
        attn_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """`x`: (B, T, N, dim). `global_mod`: (B, T, 6*dim), per-frame
        (diffusion forcing), broadcast over the `N` spatial-token axis
        below via `unsqueeze(-2)`.
        """
        b, t, n, d = x.shape
        mod = global_mod + self.adaln_bias
        shift1, scale1, gate1, shift2, scale2, gate2 = mod.chunk(6, dim=-1)

        h = self.norm1(x)
        h = modulate(h, shift1.unsqueeze(-2), scale1.unsqueeze(-2))
        h_flat = h.reshape(b, t * n, d)
        attn_out = self.attn(h_flat, attn_mask=attn_mask, kv_cache=kv_cache, layer_idx=layer_idx)
        attn_out = attn_out.reshape(b, t, n, d)
        x = x + gate1.unsqueeze(-2) * attn_out

        h2 = self.norm2(x)
        h2 = modulate(h2, shift2.unsqueeze(-2), scale2.unsqueeze(-2))
        mlp_out = self.mlp_drop(self.mlp(h2))
        x = x + gate2.unsqueeze(-2) * mlp_out
        return x


class CausalDiT(nn.Module):
    """Causal, pose-conditioned DiT predicting rectified-flow velocity.

    Forward signature is deliberately chunk-based (`latents`/`poses`/
    `intrinsics`/`t` all carry their own frame axis `T`, which need not
    equal `config.context_length`): a full training forward passes
    `T == context_length`; incremental generation with a `KVCache` passes
    `T == 1` (or a small prefix) per call, with `frame_start` advancing to
    tell `frame_pos_embed` which absolute frame indices are being
    processed. See `kv_cache.py` and
    `tests/test_model_kv_cache.py` for the equivalence this is required to
    satisfy.
    """

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config
        c = config

        self.patch_embed = nn.Linear(c.latent_channels, c.dim)
        self.raymap_encoder = RaymapEncoder(c.dim, hidden_mult=c.raymap_hidden_mult)

        self.frame_pos_embed = nn.Parameter(torch.zeros(c.context_length, c.dim))
        nn.init.normal_(self.frame_pos_embed, std=0.02)

        self.time_mlp = nn.Sequential(
            nn.Linear(c.time_embed_dim, c.dim),
            nn.SiLU(),
            nn.Linear(c.dim, c.dim),
        )
        self.adaln_single = AdaLNSingle(c.dim, c.dim, num_chunks=6)

        mlp_hidden = hidden_dim_swiglu(c.dim, c.mlp_mult, c.mlp_multiple_of)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    c.dim,
                    c.num_heads,
                    mlp_hidden,
                    qk_norm_eps=c.qk_norm_eps,
                    norm_eps=c.norm_eps,
                    attn_dropout=c.attn_dropout,
                    mlp_dropout=c.mlp_dropout,
                )
                for _ in range(c.depth)
            ]
        )

        self.final_norm = RMSNorm(c.dim, eps=c.norm_eps, elementwise_affine=False)
        self.final_mod = AdaLNSingle(c.dim, c.dim, num_chunks=2)  # shift, scale only (no gate: not residual)
        self.output_proj = nn.Linear(c.dim, c.latent_channels)
        # Zero-init the output projection: at initialization the model
        # predicts an exact zero velocity field, a standard DiT trick that
        # keeps the very first training steps numerically tame under fp16.
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def num_layers(self) -> int:
        return len(self.blocks)

    def forward(
        self,
        latents: torch.Tensor,
        poses: torch.Tensor,
        intrinsics: torch.Tensor,
        t: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        frame_start: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            latents: (B, T, N, latent_channels), N == config.tokens_per_frame.
            poses: (B, T, 4, 4), camera-to-world (see raymap.py).
            intrinsics: (B, T, 4), (fx, fy, cx, cy) in the raymap grid's
                pixel units (`config.raymap_resolution`) -- rescale with
                `raymap.scale_intrinsics` before calling if they were
                computed at the render resolution.
            t: (B, T), per-frame rectified-flow timestep in [0, 1]
                (diffusion forcing -- independent per frame).
            kv_cache: optional `KVCache` for incremental generation. When
                given, `frame_start` must equal the number of frames
                already pushed through the cache.
            frame_start: absolute index of `latents[:, 0]` within the full
                `context_length`-frame sequence; used to index
                `frame_pos_embed` and to build the causal mask against
                `kv_cache`'s existing length.

        Returns:
            (B, T, N, latent_channels) predicted velocity `v = x1 - x0`.
        """
        c = self.config
        b, tt, n, _ = latents.shape
        if n != c.tokens_per_frame:
            raise ValueError(f"latents has {n} tokens/frame, config expects {c.tokens_per_frame}")
        if frame_start + tt > c.context_length:
            raise ValueError(
                f"frame_start ({frame_start}) + chunk length ({tt}) exceeds "
                f"context_length ({c.context_length})"
            )

        h, w = c.raymap_resolution
        x = self.patch_embed(latents)
        raymap = plucker_raymap(poses, intrinsics, h, w)
        x = x + self.raymap_encoder(raymap)

        frame_idx = torch.arange(frame_start, frame_start + tt, device=latents.device)
        x = x + self.frame_pos_embed[frame_idx].unsqueeze(0).unsqueeze(2)

        t_sin = sinusoidal_embedding(t, c.time_embed_dim)
        t_embed = self.time_mlp(t_sin.to(x.dtype))
        global_mod = self.adaln_single(t_embed)

        cache_len = kv_cache.length() if kv_cache is not None else 0
        mask = build_block_causal_mask(tt, n, cache_len, latents.device)
        mask = mask.unsqueeze(0).unsqueeze(0)  # broadcast over (B, H)

        for i, block in enumerate(self.blocks):
            x = block(x, global_mod, mask, kv_cache=kv_cache, layer_idx=i)

        hnorm = self.final_norm(x)
        shift, scale = self.final_mod(t_embed).chunk(2, dim=-1)
        hnorm = modulate(hnorm, shift.unsqueeze(-2), scale.unsqueeze(-2))
        output = self.output_proj(hnorm)
        # Set by src.train.mup.configure_mup_model. This is MuReadout's
        # width multiplier; ordinary models retain multiplier 1.
        return output / getattr(self, "mup_readout_width_mult", 1.0)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_parameters_by_component(model: "CausalDiT") -> dict:
    """Coarse component breakdown, for docstrings/reporting."""
    def n(mod):
        return sum(p.numel() for p in mod.parameters())

    return {
        "patch_embed": n(model.patch_embed),
        "raymap_encoder": n(model.raymap_encoder),
        "frame_pos_embed": model.frame_pos_embed.numel(),
        "time_mlp": n(model.time_mlp),
        "adaln_single": n(model.adaln_single),
        "blocks": n(model.blocks),
        "final_norm_mod_and_head": n(model.final_mod) + n(model.output_proj),
        "total": count_parameters(model),
    }


# ---------------------------------------------------------------------------
# Size presets for the M1 (5M) / M2 (15M) / M4 (~40M) milestone gates.
#
# `tokens_per_frame` / `raymap_resolution` are left at a placeholder
# (64 = 8x8) in every preset below since the AE choice is still open (M0);
# per the module docstring, total parameter count barely moves with this
# value (only `frame_pos_embed`, `context_length * dim`, depends on it --
# and not even on tokens_per_frame, only context_length). Callers should
# override `tokens_per_frame`/`raymap_resolution`/`latent_channels` once
# M0 lands, e.g. `dataclasses.replace(preset_5m(), tokens_per_frame=256,
# raymap_resolution=(16, 16))`.
#
# Exact parameter counts (measured by `count_parameters`, verified in
# tests/test_model_dit.py::test_preset_param_counts):
#   preset_5m:  dim=256, depth=5,  heads=8  -> 4,720,324 params
#     (blocks: 4,022,080 | patch_embed: 1,280 | raymap_encoder: 33,920 |
#      frame_pos_embed: 4,096 | time_mlp: 131,584 | adaln_single: 394,752 |
#      final_norm_mod_and_head: 132,612)
#   preset_m2_proxy_5m: dim=192, depth=11, heads=8 -> 5,286,196 params
#     (same topology as the M2 target; width-only muTransfer proxy)
#   preset_15m: dim=320, depth=11, heads=10 -> 14,718,628 params
#     (blocks: 13,651,264 | patch_embed: 1,600 | raymap_encoder: 52,640 |
#      frame_pos_embed: 5,120 | time_mlp: 184,960 | adaln_single: 616,320 |
#      final_norm_mod_and_head: 206,724)
#   preset_40m: dim=512, depth=12, heads=16 -> 40,624,644 params
#     (blocks: 37,982,976 | patch_embed: 2,560 | raymap_encoder: 133,376 |
#      frame_pos_embed: 8,192 | time_mlp: 394,240 | adaln_single: 1,575,936 |
#      final_norm_mod_and_head: 527,364)
# ---------------------------------------------------------------------------


def preset_5m(**overrides) -> DiTConfig:
    cfg = dict(
        context_length=16,
        tokens_per_frame=64,
        raymap_resolution=(8, 8),
        latent_channels=4,
        dim=256,
        depth=5,
        num_heads=8,
        mlp_mult=4.0,
    )
    cfg.update(overrides)
    return DiTConfig(**cfg)


def preset_m2_proxy_5m(**overrides) -> DiTConfig:
    """Width-only proxy for M2's muTransfer gate.

    The ordinary 5M and 15M presets have depths 5 and 11. That comparison
    mixes a depth change into a method whose guarantee is widthwise only.
    This proxy therefore shares the 15M target's depth and fixed head count.
    """
    cfg = dict(
        context_length=16,
        tokens_per_frame=64,
        raymap_resolution=(8, 8),
        latent_channels=4,
        dim=192,
        depth=11,
        num_heads=8,
        mlp_mult=4.0,
    )
    cfg.update(overrides)
    return DiTConfig(**cfg)


def preset_15m(**overrides) -> DiTConfig:
    cfg = dict(
        context_length=16,
        tokens_per_frame=64,
        raymap_resolution=(8, 8),
        latent_channels=4,
        dim=320,
        depth=11,
        num_heads=10,
        mlp_mult=4.0,
    )
    cfg.update(overrides)
    return DiTConfig(**cfg)


def preset_40m(**overrides) -> DiTConfig:
    cfg = dict(
        context_length=16,
        tokens_per_frame=64,
        raymap_resolution=(8, 8),
        latent_channels=4,
        dim=512,
        depth=12,
        num_heads=16,
        mlp_mult=4.0,
    )
    cfg.update(overrides)
    return DiTConfig(**cfg)
