"""The most important test in src/model/: causality.

A change to frame t's inputs (latents, pose, or noise level) must not
change the model's output for any frame < t. A leak here silently
invalidates every later autoregressive-rollout number (drift curves,
revisit-PSNR, etc.), since the model would effectively be cheating by
peeking at the future during training.

Every source of per-token information the forward pass consumes is
perturbed independently: patch latents, poses (-> Plücker raymap), and the
per-frame diffusion-forcing noise level `t`. All must respect causality.
"""
import torch

from tests.model_fixtures import break_zero_init, build_model, random_batch, tiny_config


def _perturb_and_compare(model, cfg, perturb_fn, seed=0):
    break_zero_init(model)
    latents, poses, intrinsics, t = random_batch(cfg, batch_size=2, seed=seed)
    model.eval()
    with torch.no_grad():
        out_before = model(latents, poses, intrinsics, t)

        latents2, poses2, intrinsics2, t2 = (
            latents.clone(),
            poses.clone(),
            intrinsics.clone(),
            t.clone(),
        )
        perturb_fn(latents2, poses2, intrinsics2, t2)
        out_after = model(latents2, poses2, intrinsics2, t2)

    cutoff = cfg.context_length - 1  # perturb the last frame only
    # Frames strictly before the perturbed frame must be byte-for-byte
    # unaffected.
    assert torch.equal(out_before[:, :cutoff], out_after[:, :cutoff])
    # Sanity: the perturbation must actually do something to the last
    # frame's output, otherwise this test would pass vacuously.
    assert not torch.allclose(out_before[:, cutoff], out_after[:, cutoff])


def test_causality_latent_perturbation():
    cfg = tiny_config()
    model = build_model(cfg)
    cutoff = cfg.context_length - 1

    def perturb(latents, poses, intrinsics, t):
        latents[:, cutoff] = torch.randn_like(latents[:, cutoff]) * 5.0

    _perturb_and_compare(model, cfg, perturb)


def test_causality_pose_perturbation():
    cfg = tiny_config()
    model = build_model(cfg)
    cutoff = cfg.context_length - 1

    def perturb(latents, poses, intrinsics, t):
        # Rotate the last frame's camera by swapping two axes (still a
        # valid rotation-ish perturbation for the purposes of this test --
        # only the raymap's sensitivity to it matters).
        poses[:, cutoff, :3, 3] += torch.tensor([3.0, -2.0, 1.0])

    _perturb_and_compare(model, cfg, perturb)


def test_causality_noise_level_perturbation():
    cfg = tiny_config()
    model = build_model(cfg)
    cutoff = cfg.context_length - 1

    def perturb(latents, poses, intrinsics, t):
        t[:, cutoff] = 1.0 - t[:, cutoff]  # flip toward the opposite extreme
        t[:, cutoff] = torch.clamp(t[:, cutoff] + 0.3, 0.0, 1.0)

    _perturb_and_compare(model, cfg, perturb)


def test_causality_middle_frame_only_affects_downstream():
    """Perturbing a frame in the middle of the context window must leave
    every earlier frame unchanged and may change every frame from that
    point on (it is not required to change all of them, only forbidden
    from changing anything earlier)."""
    cfg = tiny_config(context_length=6, tokens_per_frame=4, raymap_resolution=(2, 2))
    model = build_model(cfg)
    break_zero_init(model)
    model.eval()
    latents, poses, intrinsics, t = random_batch(cfg, batch_size=2)

    mid = 2
    with torch.no_grad():
        out_before = model(latents, poses, intrinsics, t)
        latents2 = latents.clone()
        latents2[:, mid] = torch.randn_like(latents2[:, mid]) * 5.0
        out_after = model(latents2, poses, intrinsics, t)

    assert torch.equal(out_before[:, :mid], out_after[:, :mid])


def test_causality_holds_with_dropout_in_train_mode_disabled_via_eval():
    """Dropout would make this test flaky (different masks each call) --
    confirm the fixture model has zero dropout by default so causality
    checks above are deterministic, not passing by chance."""
    cfg = tiny_config()
    assert cfg.attn_dropout == 0.0
    assert cfg.mlp_dropout == 0.0
