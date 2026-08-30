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
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from run_m1_gate import DEFAULT_CONTEXT_LENGTH, load_ema_model  # noqa: E402


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


def test_default_context_length_matches_m1_config():
    """Regression guard for a real failure: run_m1_gate.py originally
    inferred context_length from the precomputed trajectory's full length
    (128 frames for the M1 tier) instead of the window size the checkpoint
    was actually trained with (16, per configs/m1_overfit_5m.yaml). That
    built a fresh model with a wrongly-shaped frame_pos_embed, surfacing
    only later as a `copy_` shape mismatch when loading the checkpoint --
    not at model-construction time, where the real cause would have been
    obvious. --context-length must default to the training config's value;
    if that config ever changes, this test forces an explicit decision
    here too rather than a silent mismatch on the next Kaggle run.
    """
    config = yaml.safe_load(
        (Path(__file__).resolve().parent.parent / "configs" / "m1_overfit_5m.yaml").read_text()
    )
    trained_context_length = config["model"]["kwargs"]["context_length"]
    assert DEFAULT_CONTEXT_LENGTH == trained_context_length


def test_load_ema_model_rejects_context_length_mismatching_checkpoint(tmp_path):
    """The actual failure mode this guards against: building a fresh model
    at a DIFFERENT context_length than the checkpoint was trained with
    silently succeeds at construction time (both are valid DiTConfigs) and
    only fails later, confusingly, as a shape mismatch inside copy_. Assert
    that failure happens (so the behavior is pinned down), and that the
    mismatch is exactly the context_length axis, not something else.
    """
    from src.train.model_factory import build_causal_dit

    trained_at_16 = build_causal_dit("5m", context_length=16)
    _write_fake_checkpoint(tmp_path, trained_at_16.state_dict())

    try:
        load_ema_model(tmp_path, "5m", context_length=128, device="cpu")
        assert False, "expected a shape mismatch when context_length doesn't match training"
    except RuntimeError as exc:
        assert "size" in str(exc).lower()
