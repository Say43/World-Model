"""End-to-end test that scripts/train.py actually builds a muP optimizer
when a config sets `mup:`, using a real CausalDiT preset on CPU -- not
just the unit-level tests in test_train_mup.py, which never exercise
scripts/train.py's own wiring (raw_model_for_optim, the pre-compile
capture, the config.get("mup") branch).
"""
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = ROOT / "scripts" / "train.py"


def _write_mup_config(path: Path, ckpt_dir: Path) -> None:
    config = {
        "run": {
            "run_id": "mup_wiring_test",
            "milestone": "M1_smoke_overfit",
            "ticket_hours": 0.01,
            "purpose": "mup wiring test",
            "seed": 0,
            "device": "cpu",
        },
        "data": {
            "target": "tests.model_fixtures.random_batch_dataloader",
            "kwargs": {"preset": "5m", "batch_size": 2, "size": 8, "seed": 0},
        },
        "model": {
            "target": "src.train.model_factory.build_causal_dit",
            "kwargs": {"preset": "5m", "context_length": 16},
        },
        "loss": {"target": "src.train.losses.rectified_flow_loss"},
        "optim": {"lr": 0.001},
        "mup": {"base_dim": 256},
        "trainer": {"warmup_steps": 1, "total_steps": 3, "grad_clip": 1.0, "amp": False},
        "checkpoint": {"dir": str(ckpt_dir), "interval_steps": 100},
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


def test_train_script_builds_mup_optimizer_and_completes(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    allocation = tmp_path / "allocation.yaml"
    allocation.write_text(yaml.safe_dump({
        "total_gpu_hours": 1.0,
        "milestones": {"M1_smoke_overfit": {"budget_gpu_hours": 1.0, "purpose": "test"}},
    }), encoding="utf-8")

    ckpt_dir = tmp_path / "checkpoints"
    config_path = tmp_path / "config.yaml"
    _write_mup_config(config_path, ckpt_dir)

    env = os.environ.copy()
    env["NANOWM_LEDGER_PATH"] = str(ledger)
    env["NANOWM_ALLOCATION_PATH"] = str(allocation)

    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), "--config", str(config_path)],
        capture_output=True, text=True, env=env, cwd=str(ROOT),
    )
    combined = result.stdout + result.stderr

    assert result.returncode == 0, combined
    assert "muP optimizer: base_dim=256" in combined, combined
    assert (ckpt_dir / "latest.json").exists(), "run should still checkpoint on completion"
