"""Tiny CPU-only model/data/loss fixtures for train-infra's own tests and
for configs/smoke_cpu.yaml.

Not a test file itself (no `test_` prefix, so pytest won't collect it and
it can't collide with other agents' `tests/test_data_*` files). It's a
stand-in for model-architect's DiT and data-engineer's real dataloader,
which don't exist yet -- scripts/train.py has zero import-time dependency
on this module; it only ever loads whatever dotted path a config's
`model.target` / `data.target` / `loss.target` point to.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class TinyModel(nn.Module):
    """A few Linear layers + Dropout: big enough to exercise AMP/grad-clip/
    EMA plausibly, tiny enough to run instantly on CPU. Dropout makes the
    forward pass depend on torch's global RNG stream, which is what makes
    RNG-state save/restore in checkpoint.py load-bearing for the resume
    test -- without restoring it correctly, a resumed run would silently
    diverge from the uninterrupted one via different dropout masks.
    """

    def __init__(self, dim: int = 16, hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return self.net(x)


class DeterministicDataset(Dataset):
    """Map-style dataset where sample `idx` is generated from a
    per-index-seeded torch.Generator, independent of any run-time global
    RNG state. With `shuffle=False`, iteration order and content are fully
    reproducible across process restarts -- required for the resume test,
    which re-creates the DataLoader from scratch after "restarting" and
    relies on Trainer.fit() skipping the already-consumed batches to
    realign with the interrupted run's position.
    """

    def __init__(self, size: int = 2000, dim: int = 16, seed: int = 0):
        self.size = size
        self.dim = dim
        self.seed = seed

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + idx)
        x = torch.randn(self.dim, generator=g)
        y = torch.randn(self.dim, generator=g)
        return x, y


def build_dataloader(size: int = 2000, dim: int = 16, seed: int = 0, batch_size: int = 8) -> DataLoader:
    dataset = DeterministicDataset(size=size, dim=dim, seed=seed)
    # DataLoader.__iter__() draws one value from its `generator` (a "shared
    # seed" mechanism used to sync samplers across distributed workers) on
    # every call, even with num_workers=0 and shuffle=False. If that
    # generator defaults to torch's *global* RNG, every re-creation of the
    # iterator (e.g. on resume, when the dataloader is rebuilt from scratch)
    # perturbs the global RNG stream the model's Dropout also draws from --
    # which would silently break checkpoint/resume determinism. Giving the
    # DataLoader its own private generator keeps that mechanism fully
    # decoupled from the model's RNG stream.
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, generator=generator)


def build_model(dim: int = 16, hidden: int = 32, dropout: float = 0.1) -> TinyModel:
    return TinyModel(dim=dim, hidden=hidden, dropout=dropout)


def mse_loss_fn(model: nn.Module, batch) -> torch.Tensor:
    x, y = batch
    pred = model(x)
    return torch.nn.functional.mse_loss(pred, y)
