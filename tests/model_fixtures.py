"""Small CPU-fast DiTConfig/inputs shared by tests/test_model_*.py. Not a
test file itself (no `test_` prefix), mirroring the pattern
`tests/train_fixtures.py` already uses.
"""
import torch

from src.model.dit import CausalDiT, DiTConfig


def tiny_config(**overrides) -> DiTConfig:
    cfg = dict(
        context_length=4,
        tokens_per_frame=4,
        raymap_resolution=(2, 2),
        latent_channels=3,
        dim=16,
        depth=2,
        num_heads=2,
        mlp_mult=2.0,
        mlp_multiple_of=8,
        time_embed_dim=16,
    )
    cfg.update(overrides)
    return DiTConfig(**cfg)


def random_batch(config: DiTConfig, batch_size: int = 2, num_frames: int = None, seed: int = 0):
    if num_frames is None:
        num_frames = config.context_length
    g = torch.Generator().manual_seed(seed)
    latents = torch.randn(batch_size, num_frames, config.tokens_per_frame, config.latent_channels, generator=g)

    poses = torch.eye(4).view(1, 1, 4, 4).repeat(batch_size, num_frames, 1, 1).clone()
    # Give each frame a distinct translation so poses aren't degenerate.
    offsets = torch.arange(num_frames, dtype=torch.float32).view(1, num_frames, 1)
    poses[..., :3, 3] += offsets * torch.tensor([0.1, 0.0, 0.0])

    intrinsics = torch.tensor([50.0, 50.0, 1.0, 1.0]).view(1, 1, 4).repeat(batch_size, num_frames, 1)
    t = torch.rand(batch_size, num_frames, generator=g)
    return latents, poses, intrinsics, t


def build_model(config: DiTConfig = None) -> CausalDiT:
    return CausalDiT(config or tiny_config())


def break_zero_init(model: CausalDiT, seed: int = 123) -> CausalDiT:
    """`CausalDiT` zero-initializes its output head and adaLN modulation
    output layers by design (see dit.py's fp16-stability notes), so a
    freshly constructed model always outputs an exact all-zeros tensor.
    That is correct production behavior but makes tests that check "does
    perturbing X change the output" (causality) or "do these two
    computation paths produce the same output" (KV-cache equivalence, in a
    way that would also trivially hold for an all-zero output) pass
    vacuously. This reinitializes every all-zero parameter tensor to small
    random values, in place, so such tests exercise the same information
    flow a partially-trained model actually would.
    """
    g = torch.Generator().manual_seed(seed)
    for p in model.parameters():
        if torch.count_nonzero(p) == 0:
            with torch.no_grad():
                p.copy_(torch.randn(p.shape, generator=g) * 0.02)
    return model


class _RandomBatchDataset(torch.utils.data.Dataset):
    """Wraps random_batch's tuple output as dict batches, matching
    src/train/losses.py's rectified_flow_loss contract -- for CPU
    integration tests of scripts/train.py that need a real dataloader
    target without precomputed .npz trajectory files on disk.
    """

    def __init__(self, config: DiTConfig, size: int, seed: int = 0):
        self.config = config
        self.size = size
        self.seed = seed

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        latents, poses, intrinsics, _ = random_batch(self.config, batch_size=1, seed=self.seed + idx)
        return {"latents": latents[0], "poses": poses[0], "intrinsics": intrinsics[0]}


def random_batch_dataloader(preset: str, batch_size: int, size: int, seed: int = 0,
                             context_length: int = 16):
    """`data.target` for a scripts/train.py config that needs a real
    dataloader but has no precomputed trajectory data -- e.g. a wiring
    test for something orthogonal to the data pipeline (src/train/mup.py).

    Builds its config via src.train.model_factory.build_causal_dit's own
    config-construction path (not a bare preset_*() call) so the latent
    shape here matches what that factory's model actually expects --
    build_causal_dit overrides tokens_per_frame/latent_channels/
    raymap_resolution to the M0-chosen AE's shape, which the raw presets'
    own defaults do not match.
    """
    from src.train.model_factory import build_causal_dit

    config = build_causal_dit(preset, context_length=context_length).config
    dataset = _RandomBatchDataset(config, size=size, seed=seed)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
