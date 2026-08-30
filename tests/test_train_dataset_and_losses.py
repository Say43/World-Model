"""End-to-end wiring test: precomputed .npz -> dataloader -> CausalDiT ->
rectified_flow_loss -> backward. Each piece has its own unit tests
elsewhere; this one exists because the M1 config wires them together via
dotted-path strings (scripts/train.py's target contract), where a mismatch
would only surface at the moment a real Kaggle run tries to start.
"""
import numpy as np
import pytest
import torch

from src.train.dataset import TrajectoryWindowDataset, build_dataloader
from src.train.losses import rectified_flow_loss
from src.train.model_factory import build_causal_dit


def _write_fake_trajectory(path, length=32, latent_channels=128, tokens_per_frame=16):
    np.savez(
        path,
        latents=np.random.default_rng(0).standard_normal((length, tokens_per_frame, latent_channels)).astype("float32"),
        poses=np.tile(np.eye(4, dtype="float32"), (length, 1, 1)),
        intrinsics=np.array([4.0, 4.0, 2.0, 2.0], dtype="float32"),
    )


def test_dataset_windows_shorter_trajectory_is_skipped(tmp_path):
    _write_fake_trajectory(tmp_path / "short.npz", length=8)
    with pytest.raises(ValueError):
        TrajectoryWindowDataset(str(tmp_path), context_length=16)


def test_dataset_produces_expected_number_of_windows(tmp_path):
    _write_fake_trajectory(tmp_path / "a.npz", length=32)
    ds = TrajectoryWindowDataset(str(tmp_path), context_length=16, stride=16)
    assert len(ds) == 2  # windows [0:16], [16:32]
    item = ds[0]
    assert item["latents"].shape == (16, 16, 128)
    assert item["poses"].shape == (16, 4, 4)
    assert item["intrinsics"].shape == (16, 4)


def test_full_pipeline_backward_pass(tmp_path):
    _write_fake_trajectory(tmp_path / "a.npz", length=32)
    model = build_causal_dit("5m", context_length=16)
    dl = build_dataloader(str(tmp_path), context_length=16, batch_size=2)
    batch = next(iter(dl))

    loss = rectified_flow_loss(model, batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_model_factory_overrides_match_chosen_ae():
    from src.eval.chosen_ae import LATENT_CHANNELS, TOKENS_PER_FRAME

    model = build_causal_dit("5m", context_length=16)
    assert model.config.tokens_per_frame == TOKENS_PER_FRAME
    assert model.config.latent_channels == LATENT_CHANNELS


def test_model_factory_rejects_unknown_preset():
    with pytest.raises(ValueError):
        build_causal_dit("100m")
