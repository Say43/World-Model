"""KV-cache equivalence: incremental, one-frame-at-a-time generation with a
KVCache must produce bit-for-bit (within float tolerance) the same output
as a single full causal forward pass over the whole context window. If
this ever drifts, autoregressive rollout (the only way the model is
actually used at inference/eval time) silently diverges from what training
optimizes.
"""
import torch

from src.model.kv_cache import KVCache
from tests.model_fixtures import break_zero_init, build_model, random_batch, tiny_config


def test_kv_cache_matches_full_forward():
    cfg = tiny_config()
    model = build_model(cfg)
    break_zero_init(model)
    model.eval()
    latents, poses, intrinsics, t = random_batch(cfg, batch_size=2)

    with torch.no_grad():
        full_out = model(latents, poses, intrinsics, t)

        cache = KVCache(num_layers=model.num_layers())
        incremental_outs = []
        for frame in range(cfg.context_length):
            out = model(
                latents[:, frame : frame + 1],
                poses[:, frame : frame + 1],
                intrinsics[:, frame : frame + 1],
                t[:, frame : frame + 1],
                kv_cache=cache,
                frame_start=frame,
            )
            incremental_outs.append(out)
        incremental_out = torch.cat(incremental_outs, dim=1)

    assert incremental_out.shape == full_out.shape
    assert torch.allclose(full_out, incremental_out, atol=1e-4, rtol=1e-4)


def test_kv_cache_matches_full_forward_with_multi_frame_chunks():
    """Equivalence must also hold when the cache is fed multi-frame chunks
    rather than strictly one frame at a time (e.g. an initial prompt of
    several frames, then one frame at a time after)."""
    cfg = tiny_config(context_length=6, tokens_per_frame=4, raymap_resolution=(2, 2))
    model = build_model(cfg)
    break_zero_init(model)
    model.eval()
    latents, poses, intrinsics, t = random_batch(cfg, batch_size=1)

    with torch.no_grad():
        full_out = model(latents, poses, intrinsics, t)

        cache = KVCache(num_layers=model.num_layers())
        prompt_len = 3
        out1 = model(
            latents[:, :prompt_len], poses[:, :prompt_len], intrinsics[:, :prompt_len], t[:, :prompt_len],
            kv_cache=cache, frame_start=0,
        )
        remaining = []
        for frame in range(prompt_len, cfg.context_length):
            out = model(
                latents[:, frame : frame + 1],
                poses[:, frame : frame + 1],
                intrinsics[:, frame : frame + 1],
                t[:, frame : frame + 1],
                kv_cache=cache,
                frame_start=frame,
            )
            remaining.append(out)
        incremental_out = torch.cat([out1] + remaining, dim=1)

    assert torch.allclose(full_out, incremental_out, atol=1e-4, rtol=1e-4)


def test_kv_cache_length_tracks_appended_tokens():
    cfg = tiny_config()
    model = build_model(cfg)
    model.eval()
    latents, poses, intrinsics, t = random_batch(cfg, batch_size=1, num_frames=1)

    cache = KVCache(num_layers=model.num_layers())
    assert cache.length() == 0
    with torch.no_grad():
        model(latents, poses, intrinsics, t, kv_cache=cache, frame_start=0)
    assert cache.length() == cfg.tokens_per_frame
    with torch.no_grad():
        model(latents, poses, intrinsics, t, kv_cache=cache, frame_start=1)
    assert cache.length() == 2 * cfg.tokens_per_frame


def test_kv_cache_reset():
    cache = KVCache(num_layers=2)
    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 4)
    cache.append(0, k, v)
    cache.append(1, k, v)
    assert cache.length() == 3
    cache.reset()
    assert cache.length() == 0
