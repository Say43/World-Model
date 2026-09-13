"""Guards on the budgeted M3 ablation entry point.

The arms themselves need a GPU and a DINOv2-bearing dataset, so what is
tested here is everything that must hold BEFORE any budget is spent: the
config stays locked until a ticket is approved, the matrix cannot silently
shrink below two seeds, and the per-arm LR horizons cannot be guessed.
Plus the analysis function, which is pure and is what the milestone
actually reports.
"""
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_m3_ablation import arm_name, build_optimizer, paired_deltas  # noqa: E402

SCRIPT = ROOT / "scripts" / "run_m3_ablation.py"
CONFIG = ROOT / "configs" / "m3_ablation.yaml"


def _run(config_path):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(config_path)],
        cwd=str(ROOT), capture_output=True, text=True,
    )


def _config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_checked_in_config_is_locked_until_ticket_approval():
    assert _config()["run"]["ticket_hours"] is None
    result = _run(CONFIG)
    assert result.returncode == 1
    assert "ticket_hours is unset" in result.stderr


def test_config_declares_the_full_two_by_two_matrix_at_two_seeds():
    ablation = _config()["ablation"]
    assert sorted(ablation["repa"]) == [False, True]
    assert sorted(ablation["optimizers"]) == ["adamw", "muon"]
    assert len(ablation["seeds"]) >= 2
    assert len(set(ablation["seeds"])) == len(ablation["seeds"])


def test_config_leaves_every_arm_step_horizon_unmeasured():
    """steps_per_arm must stay null in the repository: filling it in with a
    guess is how an arm ends up training at near-peak LR the whole way."""
    steps = _config()["ablation"]["steps_per_arm"]
    assert set(steps) == {"norepa_adamw", "norepa_muon", "repa_adamw", "repa_muon"}
    assert all(value is None for value in steps.values())


def test_config_has_no_single_total_steps_for_all_arms():
    assert "total_steps" not in _config()["trainer"]


def test_runner_refuses_a_single_seed(tmp_path):
    config = _config()
    config["run"]["ticket_hours"] = 0.01
    config["ablation"]["seeds"] = [0]
    path = tmp_path / "one_seed.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = _run(path)
    assert result.returncode != 0
    assert ">=2 seeds" in (result.stderr + result.stdout)


def test_runner_refuses_unmeasured_step_horizons(tmp_path):
    config = _config()
    config["run"]["ticket_hours"] = 0.01
    config["ablation"]["seconds_per_arm"] = 5
    path = tmp_path / "no_steps.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = _run(path)
    assert result.returncode == 1
    assert "steps_per_arm is unmeasured" in result.stderr
    assert "Profilieren vor Trainieren" in result.stderr


def test_runner_refuses_a_matrix_that_cannot_fit_its_ticket(tmp_path):
    """Better to reject the plan than to run a matrix guaranteed to be
    truncated -- a half-finished ablation still spends the whole budget."""
    config = _config()
    config["run"]["ticket_hours"] = 0.1  # 360s
    config["ablation"]["seconds_per_arm"] = 600  # 8 arms x 600s
    config["ablation"]["steps_per_arm"] = {name: 100 for name in config["ablation"]["steps_per_arm"]}
    path = tmp_path / "too_big.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = _run(path)
    assert result.returncode != 0
    assert "exceeds the" in (result.stderr + result.stdout)


def test_arm_name_matches_the_config_keys():
    assert arm_name(False, "adamw") == "norepa_adamw"
    assert arm_name(True, "muon") == "repa_muon"
    assert set(_config()["ablation"]["steps_per_arm"]) == {
        arm_name(r, o) for r in (False, True) for o in ("adamw", "muon")
    }


def test_build_optimizer_routes_muon_and_adamw():
    import torch

    from src.train.model_factory import build_causal_dit
    from src.train.muon import MuonWithAuxAdamW

    model = build_causal_dit("5m", context_length=16)
    optim_cfg = {"adamw_lr": 3e-4, "muon_lr": 0.02, "muon_aux_adamw_lr": 3e-4}

    assert isinstance(build_optimizer(model, "adamw", optim_cfg), torch.optim.AdamW)
    assert isinstance(build_optimizer(model, "muon", optim_cfg), MuonWithAuxAdamW)
    with pytest.raises(ValueError, match="unknown optimizer"):
        build_optimizer(model, "lion", optim_cfg)


def _result(repa, optimizer, seed, loss):
    return {"repa": repa, "optimizer": optimizer, "seed": seed, "trailing_avg_loss": loss}


def test_paired_deltas_are_computed_per_seed_within_each_cell():
    results = [
        _result(False, "adamw", 0, 1.00), _result(True, "adamw", 0, 0.90),
        _result(False, "adamw", 1, 1.10), _result(True, "adamw", 1, 0.95),
        _result(False, "muon", 0, 0.98), _result(True, "muon", 0, 0.94),
        _result(False, "muon", 1, 1.02), _result(True, "muon", 1, 1.00),
    ]
    deltas = paired_deltas(results)

    repa_on_adamw = deltas["repa_effect"]["adamw"]
    assert repa_on_adamw["per_seed"] == {0: pytest.approx(-0.10), 1: pytest.approx(-0.15)}
    assert repa_on_adamw["mean"] == pytest.approx(-0.125)
    assert repa_on_adamw["consistent_sign"] is True

    muon_no_repa = deltas["muon_effect"]["norepa"]
    assert muon_no_repa["per_seed"] == {0: pytest.approx(-0.02), 1: pytest.approx(-0.08)}


def test_paired_deltas_flag_an_effect_the_seeds_disagree_about():
    """The result this project most needs to report honestly: an effect
    smaller than the seed-to-seed spread is not an effect."""
    results = [
        _result(False, "adamw", 0, 1.00), _result(True, "adamw", 0, 0.95),  # REPA helps
        _result(False, "adamw", 1, 1.00), _result(True, "adamw", 1, 1.05),  # REPA hurts
    ]
    entry = paired_deltas(results)["repa_effect"]["adamw"]

    assert entry["consistent_sign"] is False
    assert entry["mean"] == pytest.approx(0.0)


def test_paired_deltas_skip_cells_a_truncated_run_never_reached():
    results = [_result(False, "adamw", 0, 1.0), _result(True, "adamw", 0, 0.9),
               _result(False, "adamw", 1, 1.0)]  # seed 1's REPA arm missing
    entry = paired_deltas(results)["repa_effect"]["adamw"]
    assert list(entry["per_seed"]) == [0]
    assert "muon" not in paired_deltas(results)["repa_effect"]


def test_checked_in_ablation_config_has_every_key_the_runner_reads(tmp_path):
    """Same guard as the profile's: run one real arm end to end from the
    checked-in config, so a missing block fails here and not after a
    21-minute precompute on Kaggle."""
    import numpy as np

    from scripts.run_m3_ablation import run_arm

    rng = np.random.default_rng(0)
    for index in range(2):
        np.savez(
            tmp_path / f"t{index}.npz",
            latents=rng.standard_normal((32, 16, 128)).astype("float32"),
            poses=np.tile(np.eye(4, dtype="float32"), (32, 1, 1)),
            intrinsics=np.array([4.0, 4.0, 2.0, 2.0], dtype="float32"),
            dinov2=rng.standard_normal((32, 16, 384)).astype("float32"),
        )

    config = _config()
    result = run_arm(
        use_repa=True, optimizer="muon", seed=0,
        data_cfg=dict(config["data"], data_dir=str(tmp_path), batch_size=2),
        model_cfg={"preset": "5m"},
        optim_cfg=config["optim"],
        repa_cfg=config["repa"],
        trainer_cfg=config["trainer"],
        seconds_per_arm=600.0, total_steps=4, curve_interval=2,
        trailing_window=2, device="cpu", compile_model=False,
        hard_deadline_monotonic=__import__("time").monotonic() + 600,
    )
    assert result["arm"] == "repa_muon"
    assert result["steps_completed"] == 4
    assert result["schedule_incomplete"] is False
    # Ranked on the flow term, which must be below the total it minimized.
    assert result["trailing_avg_loss"] < result["trailing_avg_total_objective"]
