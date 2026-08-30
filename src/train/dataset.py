"""Loads precomputed .npz trajectories (scripts/preprocess/precompute_dataset.py)
into fixed-length windows matching CausalDiT's (context_length, tokens_per_frame,
latent_channels) input.

Model-agnostic w.r.t. training internals (no loss, no diffusion forcing) --
this only produces windows of (latents, poses, intrinsics); losses.py turns
a window into a training example.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryWindowDataset(Dataset):
    """One item = one contiguous window of `context_length` frames from one
    precomputed trajectory .npz file. Windows are the unit CausalDiT trains
    on; a trajectory longer than context_length yields multiple windows.
    """

    def __init__(self, data_dir: str, context_length: int, stride: int = None):
        self.context_length = context_length
        self.stride = stride or context_length
        self._index: List[tuple] = []  # (npz_path, window_start)
        paths = sorted(Path(data_dir).glob("*.npz"))
        if not paths:
            raise FileNotFoundError(f"no .npz trajectories found in {data_dir}")
        for path in paths:
            with np.load(path) as data:
                length = data["latents"].shape[0]
            if length < context_length:
                continue  # too short to fill one window; skip rather than pad
            for start in range(0, length - context_length + 1, self.stride):
                self._index.append((path, start))
        if not self._index:
            raise ValueError(
                f"no trajectory in {data_dir} is long enough for context_length={context_length}"
            )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        path, start = self._index[idx]
        end = start + self.context_length
        with np.load(path) as data:
            latents = data["latents"][start:end]
            poses = data["poses"][start:end]
            intrinsics = data["intrinsics"]
        return {
            "latents": torch.from_numpy(latents.astype(np.float32)),
            "poses": torch.from_numpy(poses.astype(np.float32)),
            "intrinsics": torch.from_numpy(np.broadcast_to(intrinsics, (self.context_length, 4)).astype(np.float32)),
        }


def build_dataloader(data_dir: str, context_length: int, batch_size: int,
                      stride: int = None, shuffle: bool = True, num_workers: int = 0):
    dataset = TrajectoryWindowDataset(data_dir, context_length, stride)
    # Private generator: DataLoader.__iter__ draws from the global torch RNG
    # on every call otherwise, which perturbs whatever else reads that
    # stream (see src/train/trainer.py's resume-determinism fix for why
    # this matters).
    generator = torch.Generator()
    generator.manual_seed(0)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, generator=generator, drop_last=True,
    )
