"""DDP setup for 2xT4, degrading cleanly to single-process (CPU or one GPU)
when no distributed launch environment is present -- which is how every
test in this repo runs.

setup_ddp() only attempts torch.distributed.init_process_group() when
launched via a multi-process launcher (torchrun sets RANK/WORLD_SIZE/
LOCAL_RANK) AND there are at least world_size visible CUDA devices.
Otherwise it returns a DDPContext describing a single process on
whatever device is available, and wrap_model() / is_main_process() behave
correctly for that case without any special-casing at the call site.
"""
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class DDPContext:
    is_distributed: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device


def setup_ddp(backend: str = None, device_override: Optional[str] = None) -> DDPContext:
    """Set up DDP if launched via a multi-process launcher (torchrun sets
    RANK/WORLD_SIZE/LOCAL_RANK) with enough visible CUDA devices; otherwise
    degrade to a single process on whatever device is available.

    `device_override` (or the NANOWM_FORCE_DEVICE env var) forces the
    single-process device -- e.g. "cpu" -- regardless of what's visible.
    This exists so CPU-only infra tests and M0.5 smoke runs are actually
    CPU-only even on a dev box that happens to have a GPU: no test in this
    repo should burn GPU-hours that count against the project's budget.
    It has no effect once WORLD_SIZE>1 has selected the DDP branch.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1 and torch.cuda.is_available() and torch.cuda.device_count() >= world_size:
        import torch.distributed as dist

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend or "nccl", rank=rank, world_size=world_size)
        device = torch.device(f"cuda:{local_rank}")
        return DDPContext(is_distributed=True, rank=rank, world_size=world_size, local_rank=local_rank, device=device)

    forced = device_override or os.environ.get("NANOWM_FORCE_DEVICE")
    if forced:
        device = torch.device(forced)
    else:
        # Single-process degradation: CPU for tests, or a lone GPU otherwise.
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    return DDPContext(is_distributed=False, rank=0, world_size=1, local_rank=0, device=device)


def cleanup_ddp(ctx: DDPContext) -> None:
    if ctx.is_distributed:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()


def wrap_model(model: nn.Module, ctx: DDPContext) -> nn.Module:
    model = model.to(ctx.device)
    if ctx.is_distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = DDP(model, device_ids=[ctx.local_rank], output_device=ctx.local_rank)
    return model


def is_main_process(ctx: DDPContext) -> bool:
    return ctx.rank == 0
