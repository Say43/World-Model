"""Tests for scripts/run_m1_gate.py's checkpoint-loading logic.

Regression test for a real failure: the smoke run's checkpoint (trained
with torch.compile, on by default in scripts/train.py) was missing
'frame_pos_embed' from its EMA shadow -- a bare nn.Parameter assigned
directly on CausalDiT rather than living inside a submodule, apparently
dropped by torch.compile's OptimizedModule.state_dict(). Reproduced here
without needing torch.compile itself: a fabricated checkpoint with that
one key deliberately missing from "ema" but present in "model".
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from run_m1_gate import load_ema_model  # noqa: E402


def _write_fake_checkpoint(tmp_path, real_state_dict, missing_from_ema=()):
    ema = {k: v.clone() for k, v in real_state_dict.items() if k not in missing_from_ema}
    torch.save(
        {"step": 42, "model": real_state_dict, "ema": ema},
        tmp_path / "ckpt.pt",
    )
    (tmp_path / "latest.json").write_text('{"step": 42, "path": "ckpt.pt"}')


def test_load_ema_model_falls_back_to_raw_weights_for_missing_key(tmp_path):
    from src.train.model_factory import build_causal_dit

    reference = build_causal_dit("5m", context_length=16)
    real_state = reference.state_dict()
    torch.nn.init.normal_(reference.frame_pos_embed, std=1.0)  # distinct from a fresh model's init
    real_state = reference.state_dict()

    _write_fake_checkpoint(tmp_path, real_state, missing_from_ema=["frame_pos_embed"])

    model, step = load_ema_model(tmp_path, "5m", context_length=16, device="cpu")
    assert step == 42
    torch.testing.assert_close(model.frame_pos_embed, reference.frame_pos_embed)


def test_load_ema_model_raises_if_key_missing_from_both(tmp_path):
    from src.train.model_factory import build_causal_dit

    real_state = build_causal_dit("5m", context_length=16).state_dict()
    ema = {k: v for k, v in real_state.items() if k != "frame_pos_embed"}
    torch.save(
        {"step": 1, "model": {k: v for k, v in real_state.items() if k != "frame_pos_embed"}, "ema": ema},
        tmp_path / "ckpt.pt",
    )
    (tmp_path / "latest.json").write_text('{"step": 1, "path": "ckpt.pt"}')

    try:
        load_ema_model(tmp_path, "5m", context_length=16, device="cpu")
        assert False, "expected KeyError when a key is missing from both EMA and raw weights"
    except KeyError:
        pass


def test_load_ema_model_strips_orig_mod_prefix(tmp_path):
    """torch.compile can prefix every key with '_orig_mod.'; loading must
    tolerate that uniformly, not just the single-key EMA fallback case."""
    from src.train.model_factory import build_causal_dit

    reference = build_causal_dit("5m", context_length=16)
    real_state = {f"_orig_mod.{k}": v for k, v in reference.state_dict().items()}
    _write_fake_checkpoint(tmp_path, real_state)

    model, _ = load_ema_model(tmp_path, "5m", context_length=16, device="cpu")
    for k, v in reference.state_dict().items():
        torch.testing.assert_close(model.state_dict()[k], v)
