"""Guards and pure logic of the M3 pre-flight profile.

The measurements need a GPU; what is tested here is the ticket lock and the
step-horizon arithmetic the ablation depends on.
"""
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_m3_profile import recommend_steps_per_arm  # noqa: E402

SCRIPT = ROOT / "scripts" / "run_m3_profile.py"
CONFIG = ROOT / "configs" / "m3_profile.yaml"
ABLATION_CONFIG = ROOT / "configs" / "m3_ablation.yaml"


def _config(path=CONFIG):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_checked_in_config_is_locked_until_ticket_approval():
    assert _config()["run"]["ticket_hours"] is None
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(CONFIG)],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "ticket_hours is unset" in result.stderr


def test_profile_and_ablation_agree_on_model_and_data():
    """The profile only means anything for the run it is measured for: a
    different preset or batch size makes the steps/s numbers inapplicable."""
    profile, ablation = _config(), _config(ABLATION_CONFIG)
    assert profile["model"]["preset"] == ablation["model"]["preset"]
    assert profile["data"] == ablation["data"]
    assert profile["repa"] == ablation["repa"]
    assert profile["run"]["milestone"] == ablation["run"]["milestone"]


def test_muon_probe_grid_brackets_the_reference_default():
    """A grid that only goes downward could not detect an LR that is too
    small, and one that never includes the default cannot confirm it."""
    profile = _config()
    grid = profile["profile"]["muon_lrs"]
    default = profile["optim"]["muon_lr"]
    assert len(grid) >= 3
    assert min(grid) < default < max(grid)
    # Muon's LR lives far above AdamW's; a grid near 3e-4 would probe the
    # wrong region entirely.
    assert min(grid) > profile["optim"]["adamw_lr"] * 10


def test_target_seconds_stays_unset_until_the_ticket_is_sized():
    assert _config()["profile"]["target_seconds_per_arm"] is None


def test_recommended_horizons_scale_with_measured_throughput():
    throughput = [
        {"arm": "norepa_adamw", "steps_per_second": 10.0},
        {"arm": "repa_muon", "steps_per_second": 5.0},
    ]
    recommended = recommend_steps_per_arm(throughput, seconds_per_arm=600, safety_margin=0.9)

    assert recommended == {"norepa_adamw": 5400, "repa_muon": 2700}
    # The slower arm gets FEWER steps -- that is the point: it pays for its
    # cost per step instead of being handed extra wall-clock.
    assert recommended["repa_muon"] < recommended["norepa_adamw"]


def test_safety_margin_shortens_rather_than_extends_the_horizon():
    """Overshooting means the LR schedule never decays (the M1 divergence);
    undershooting just trains the tail at the floor. The margin must err in
    the harmless direction."""
    throughput = [{"arm": "a", "steps_per_second": 10.0}]
    margined = recommend_steps_per_arm(throughput, 600, 0.9)["a"]
    exact = recommend_steps_per_arm(throughput, 600, 1.0)["a"]
    assert margined < exact
    assert _config()["profile"]["safety_margin"] < 1.0


def _fake_dataset(tmp_path):
    import numpy as np

    rng = np.random.default_rng(0)
    for index in range(2):
        np.savez(
            tmp_path / f"t{index}.npz",
            latents=rng.standard_normal((32, 16, 128)).astype("float32"),
            poses=np.tile(np.eye(4, dtype="float32"), (32, 1, 1)),
            intrinsics=np.array([4.0, 4.0, 2.0, 2.0], dtype="float32"),
            dinov2=rng.standard_normal((32, 16, 384)).astype("float32"),
        )
    return str(tmp_path)


def test_checked_in_profile_config_has_every_key_the_runner_reads(tmp_path):
    """Builds all four arms from the REAL config file.

    The first Kaggle attempt died with KeyError('trainer') after the
    21-minute precompute had already run, because configs/m3_profile.yaml
    was missing a block build_arm reads. Asserting the presence of specific
    keys would only re-encode today's list; constructing the arms exercises
    whatever the code actually looks up.
    """
    from scripts.run_m3_profile import build_arm

    config = _config()
    config["data"] = dict(config["data"], data_dir=_fake_dataset(tmp_path), batch_size=2)
    config["model"]["preset"] = "5m"  # the 15M target is pointless on CPU

    for use_repa in (False, True):
        for optimizer in ("adamw", "muon"):
            model, trainer, dataloader = build_arm(
                use_repa, optimizer, seed=0, cfg=config, device="cpu", total_steps=4
            )
            assert (model.repa_head is not None) == use_repa
            assert trainer.optimizer is not None
            assert len(dataloader) > 0
