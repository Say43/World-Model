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

    def __init__(self, data_dir: str, context_length: int, stride: int = None,
                 with_dinov2: bool = False):
        self.context_length = context_length
        self.stride = stride or context_length
        self.with_dinov2 = with_dinov2
        self._index: List[tuple] = []  # (traj_key, window_start)
        # Every trajectory's arrays are decompressed into memory once, here,
        # rather than in __getitem__. The first version of this class called
        # np.load() (on a savez_compressed file, so real zlib decompression,
        # not just a cheap mmap) inside __getitem__ -- once per item, every
        # batch, every step. Measured on Kaggle: real training reached
        # ~11 steps/s against a ~98 steps/s synthetic profile
        # (results/profile_2xt4_v2.json, same model/batch/tokens-per-frame,
        # in-memory random tensors) -- an ~9x gap consistent with paying
        # disk I/O + decompression on every single item fetch instead of
        # once. M1's tier is tiny (887KB total) and fits trivially in
        # memory; M3/M4's tiers (CLAUDE.md's ~160-trajectory estimate) are
        # still small enough to cache whole.
        self._cache: dict = {}
        paths = sorted(Path(data_dir).glob("*.npz"))
        if not paths:
            raise FileNotFoundError(f"no .npz trajectories found in {data_dir}")
        for path in paths:
            with np.load(path) as data:
                entry = {
                    "latents": data["latents"].astype(np.float32),
                    "poses": data["poses"].astype(np.float32),
                    "intrinsics": data["intrinsics"].astype(np.float32),
                }
                if with_dinov2:
                    if "dinov2" not in data:
                        raise KeyError(
                            f"{path.name} has no 'dinov2' array: this dataset was "
                            "precomputed without --with-dinov2, so REPA cannot train "
                            "on it. Re-run scripts/preprocess/precompute_dataset.py "
                            "with that flag."
                        )
                    # fp16 in the cache: these are ~10x the latents' volume
                    # (16 tokens x 384 dims vs 16 x 128) and only their
                    # direction is used, by a cosine loss. Widened back to
                    # fp32 per window in __getitem__.
                    entry["dinov2"] = data["dinov2"].astype(np.float16)
                self._cache[path] = entry
            length = self._cache[path]["latents"].shape[0]
            if length < context_length:
                del self._cache[path]
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
        cached = self._cache[path]
        intrinsics = cached["intrinsics"]
        item = {
            "latents": torch.from_numpy(cached["latents"][start:end].copy()),
            "poses": torch.from_numpy(cached["poses"][start:end].copy()),
            "intrinsics": torch.from_numpy(np.broadcast_to(intrinsics, (self.context_length, 4)).copy()),
        }
        if self.with_dinov2:
            item["dinov2"] = torch.from_numpy(
                cached["dinov2"][start:end].astype(np.float32)
            )
        return item


def build_dataloader(data_dir: str, context_length: int, batch_size: int,
                      stride: int = None, shuffle: bool = True, num_workers: int = 0,
                      with_dinov2: bool = False):
    dataset = TrajectoryWindowDataset(data_dir, context_length, stride, with_dinov2=with_dinov2)
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
