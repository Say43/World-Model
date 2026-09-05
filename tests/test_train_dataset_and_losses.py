"""End-to-end wiring test: precomputed .npz -> dataloader -> CausalDiT ->
rectified_flow_loss -> backward. Each piece has its own unit tests
elsewhere; this one exists because the M1 config wires them together via
dotted-path strings (scripts/train.py's target contract), where a mismatch
would only surface at the moment a real Kaggle run tries to start.
"""
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.train.dataset import TrajectoryWindowDataset, build_dataloader
from src.train.losses import rectified_flow_loss
from src.train.model_factory import build_causal_dit


def _write_fake_trajectory(path, length=32, latent_channels=128, tokens_per_frame=16,
                            dinov2_dim=None):
    arrays = dict(
        latents=np.random.default_rng(0).standard_normal((length, tokens_per_frame, latent_channels)).astype("float32"),
        poses=np.tile(np.eye(4, dtype="float32"), (length, 1, 1)),
        intrinsics=np.array([4.0, 4.0, 2.0, 2.0], dtype="float32"),
    )
    if dinov2_dim is not None:
        arrays["dinov2"] = np.random.default_rng(1).standard_normal(
            (length, tokens_per_frame, dinov2_dim)).astype("float32")
    np.savez(path, **arrays)


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


def test_dataset_loads_each_file_from_disk_only_once(tmp_path):
    """Regression test: __getitem__ originally called np.load() on every
    access -- decompressing the same savez_compressed file from disk on
    every single training step, for every item in the batch. Measured on
    Kaggle: real training throughput was ~11 steps/s against a ~98 steps/s
    synthetic profile of the identical model/batch config
    (results/profile_2xt4_v2.json used in-memory tensors, no file I/O) --
    an ~9x gap consistent with paying decompression on every fetch instead
    of caching once. Two trajectories, four windows each: np.load must be
    called exactly twice (once per file, at construction), never per-item.
    """
    _write_fake_trajectory(tmp_path / "a.npz", length=32)
    _write_fake_trajectory(tmp_path / "b.npz", length=32)

    real_load = np.load
    with patch("numpy.load", side_effect=real_load) as mock_load:
        ds = TrajectoryWindowDataset(str(tmp_path), context_length=16, stride=16)
        assert mock_load.call_count == 2  # once per file, during __init__

        for i in range(len(ds)):
            ds[i]
        for i in range(len(ds)):  # a second full pass, as a real epoch would do
            ds[i]

    assert mock_load.call_count == 2, "np.load must not be called again from __getitem__"


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


def test_dataset_omits_dinov2_unless_asked(tmp_path):
    _write_fake_trajectory(tmp_path / "a.npz", length=32, dinov2_dim=384)
    ds = TrajectoryWindowDataset(str(tmp_path), context_length=16, stride=16)
    assert "dinov2" not in ds[0]


def test_dataset_yields_dinov2_window_aligned_with_latents(tmp_path):
    """REPA's target must be the SAME frames as the window's latents --
    an off-by-one window slice would train the model to align frame t's
    hidden state with frame t+1's features and quietly degrade the arm."""
    _write_fake_trajectory(tmp_path / "a.npz", length=32, dinov2_dim=384)
    ds = TrajectoryWindowDataset(str(tmp_path), context_length=16, stride=16,
                                 with_dinov2=True)
    item = ds[1]
    assert item["dinov2"].shape == (16, 16, 384)
    assert item["dinov2"].dtype == torch.float32

    with np.load(tmp_path / "a.npz") as raw:
        expected = raw["dinov2"][16:32]
    # fp16 in the cache, so compare at fp16 tolerance rather than exactly.
    np.testing.assert_allclose(item["dinov2"].numpy(), expected, rtol=1e-2, atol=1e-2)


def test_dataset_fails_loudly_when_dinov2_is_requested_but_absent(tmp_path):
    """The M3 datasets precomputed before --with-dinov2 existed lack the
    array; silently training REPA on nothing would look like a null result."""
    _write_fake_trajectory(tmp_path / "a.npz", length=32)
    with pytest.raises(KeyError, match="--with-dinov2"):
        TrajectoryWindowDataset(str(tmp_path), context_length=16, with_dinov2=True)
