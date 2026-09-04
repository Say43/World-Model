"""Builds a CausalDiT from a preset name, overridden for the M0-chosen AE.

scripts/train.py's model.target contract is `target(**kwargs) -> nn.Module`;
CausalDiT.__init__ takes a single DiTConfig, so this is the adapter between
the two. The presets in src/model/dit.py default to a generic 4-channel,
64-token/frame latent shape; the chosen AE (src/eval/chosen_ae.py) is
128-channel, 16 tokens/frame (4x4 grid), so those three fields are always
overridden here rather than left to preset defaults.
"""
from __future__ import annotations

from src.eval.chosen_ae import LATENT_CHANNELS, LATENT_GRID, TOKENS_PER_FRAME
from src.model.dit import CausalDiT, preset_5m, preset_m2_proxy_5m, preset_15m, preset_40m

_PRESETS = {
    "5m": preset_5m,
    "m2_proxy_5m": preset_m2_proxy_5m,
    "15m": preset_15m,
    "40m": preset_40m,
}


def build_causal_dit(preset: str = "5m", **overrides) -> CausalDiT:
    if preset not in _PRESETS:
        raise ValueError(f"unknown preset '{preset}', expected one of {list(_PRESETS)}")
    cfg_overrides = dict(
        tokens_per_frame=TOKENS_PER_FRAME,
        raymap_resolution=(LATENT_GRID, LATENT_GRID),
        latent_channels=LATENT_CHANNELS,
    )
    cfg_overrides.update(overrides)
    cfg = _PRESETS[preset](**cfg_overrides)
    return CausalDiT(cfg)
