"""M4 eval: context-conditioned sampling and the image outputs, on a tiny
model and random data (no AE, no checkpoint, no GPU)."""
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_m4_eval import sample_future_frames, save_gif, save_grid_png, windows_of  # noqa: E402
from src.model.dit import CausalDiT  # noqa: E402
from tests.model_fixtures import random_batch, tiny_config  # noqa: E402


def test_context_frames_are_kept_and_future_frames_change():
    torch.manual_seed(0)
    cfg = tiny_config()
    model = CausalDiT(cfg).eval()
    gt, poses, intr, _ = random_batch(cfg, batch_size=1, num_frames=4)
    out = sample_future_frames(model, gt, poses, intr, context_frames=2, num_steps=4,
                               generator=torch.Generator().manual_seed(1))
    assert out.shape == gt.shape
    assert torch.equal(out[:, :2], gt[:, :2])
    assert torch.isfinite(out).all()
    assert not torch.allclose(out[:, 2:], gt[:, 2:])


def test_windows_are_non_overlapping_and_capped():
    data = {"latents": np.zeros((70, 16, 4)), "poses": np.zeros((70, 4, 4)), "intrinsics": np.zeros(4)}
    starts = [s for s, _ in windows_of(data, 16, max_windows=3)]
    assert starts == [0, 16, 32]
    _, w = next(windows_of(data, 16, max_windows=1))
    assert w["latents"].shape[0] == 16


def test_image_outputs_are_written(tmp_path):
    gt = np.random.rand(5, 8, 8, 3).astype(np.float32)
    pred = np.random.rand(5, 8, 8, 3).astype(np.float32)
    save_grid_png(gt, pred, 2, tmp_path / "g.png")
    save_gif(gt, pred, 2, tmp_path / "g.gif")
    from PIL import Image
    assert Image.open(tmp_path / "g.png").size == (5 * 12, 2 * 8 + 8)
    assert Image.open(tmp_path / "g.gif").n_frames == 5


def test_m4_config_is_not_runnable_as_committed():
    cfg = yaml.safe_load((ROOT / "configs" / "m4_main.yaml").read_text(encoding="utf-8"))
    assert cfg["run"]["ticket_hours"] is None
    assert cfg["trainer"]["total_steps"] is None
    assert cfg["optim"]["optimizer"] == "muon" and cfg["optim"]["muon_lr"] == 0.02
    assert cfg["loss"]["target"].endswith("rectified_flow_loss_with_repa")
    assert cfg["data"]["kwargs"]["with_dinov2"] is True
    assert cfg["model"]["kwargs"]["preset"] == "40m"
