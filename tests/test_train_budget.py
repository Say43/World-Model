"""Budget enforcement: a ticket over the remaining budget must prevent a
run from starting at all, and scripts/train.py must never write a
checkpoint or a ledger entry for a rejected ticket.

Uses NANOWM_LEDGER_PATH / NANOWM_ALLOCATION_PATH (scripts/budget.py) to
point every subprocess at throwaway files under tmp_path -- these tests
never touch the real budget/ledger.jsonl or budget/allocation.yaml.

"Pass" means:
  - `budget.py check-ticket` exits non-zero with "REJECTED" on stderr when
    the requested hours exceed the milestone's (or total's) remaining
    budget, and exits 0 with "APPROVED" on stdout otherwise.
  - `scripts/train.py` exits non-zero, prints "ABORTED", never creates the
    configured checkpoint directory, and never appends to the ledger, when
    its ticket is rejected.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BUDGET_SCRIPT = ROOT / "scripts" / "budget.py"
TRAIN_SCRIPT = ROOT / "scripts" / "train.py"


def _write_allocation(path: Path, total_hours: float, milestone_budget: float) -> None:
    data = {
        "total_gpu_hours": total_hours,
        "milestones": {
            "M1_smoke_overfit": {"budget_gpu_hours": milestone_budget, "purpose": "test"},
        },
    }
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_check_ticket_rejects_over_budget(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    allocation = tmp_path / "allocation.yaml"
    _write_allocation(allocation, total_hours=1.0, milestone_budget=0.1)

    env = os.environ.copy()
    env["NANOWM_LEDGER_PATH"] = str(ledger)
    env["NANOWM_ALLOCATION_PATH"] = str(allocation)

    over_budget = subprocess.run(
        [sys.executable, str(BUDGET_SCRIPT), "check-ticket", "--milestone", "M1_smoke_overfit", "--hours", "5.0"],
        capture_output=True, text=True, env=env,
    )
    assert over_budget.returncode != 0
    assert "REJECTED" in over_budget.stderr

    within_budget = subprocess.run(
        [sys.executable, str(BUDGET_SCRIPT), "check-ticket", "--milestone", "M1_smoke_overfit", "--hours", "0.05"],
        capture_output=True, text=True, env=env,
    )
    assert within_budget.returncode == 0
    assert "APPROVED" in within_budget.stdout


def _write_config(path: Path, ckpt_dir: Path, ticket_hours: float) -> None:
    config = {
        "run": {
            "run_id": "over_budget_test",
            "milestone": "M1_smoke_overfit",
            "ticket_hours": ticket_hours,
            "purpose": "budget enforcement test",
            "seed": 0,
            "device": "cpu",  # this test must never spend GPU-hours, even on a GPU-equipped host
        },
        "data": {
            "target": "tests.train_fixtures.build_dataloader",
            "kwargs": {"size": 100, "dim": 8, "seed": 0, "batch_size": 4},
        },
        "model": {"target": "tests.train_fixtures.build_model", "kwargs": {"dim": 8, "hidden": 16}},
        "loss": {"target": "tests.train_fixtures.mse_loss_fn"},
        "optim": {"lr": 0.001},
        "trainer": {"warmup_steps": 2, "total_steps": 10, "grad_clip": 1.0, "amp": False},
        "checkpoint": {"dir": str(ckpt_dir), "interval_steps": 5},
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


def test_train_script_aborts_without_starting_when_ticket_rejected(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    allocation = tmp_path / "allocation.yaml"
    _write_allocation(allocation, total_hours=1.0, milestone_budget=0.01)

    ckpt_dir = tmp_path / "checkpoints"
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, ckpt_dir, ticket_hours=5.0)

    env = os.environ.copy()
    env["NANOWM_LEDGER_PATH"] = str(ledger)
    env["NANOWM_ALLOCATION_PATH"] = str(allocation)

    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), "--config", str(config_path)],
        capture_output=True, text=True, env=env, cwd=str(ROOT),
    )

    assert result.returncode != 0
    assert "ABORTED" in (result.stdout + result.stderr)
    assert not ckpt_dir.exists(), "no checkpoint should be written for a rejected ticket"
    assert not ledger.exists() or ledger.read_text(encoding="utf-8").strip() == "", (
        "no ledger entry should be appended for a rejected ticket"
    )


def test_train_script_runs_and_logs_ledger_when_ticket_approved(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    allocation = tmp_path / "allocation.yaml"
    _write_allocation(allocation, total_hours=1.0, milestone_budget=0.5)

    ckpt_dir = tmp_path / "checkpoints"
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, ckpt_dir, ticket_hours=0.001)

    env = os.environ.copy()
    env["NANOWM_LEDGER_PATH"] = str(ledger)
    env["NANOWM_ALLOCATION_PATH"] = str(allocation)

    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), "--config", str(config_path)],
        capture_output=True, text=True, env=env, cwd=str(ROOT),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert ledger.exists()
    lines = [line for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1, "exactly one ledger entry for exactly one run"
    entry = json.loads(lines[0])
    assert entry["run_id"] == "over_budget_test"
    assert entry["milestone"] == "M1_smoke_overfit"
    assert (ckpt_dir / "latest.json").exists()


def test_run_is_aborted_when_it_outlives_its_ticket(tmp_path):
    """Approving a ticket is not enough -- the run must be cut off at it.

    Checking only at startup would let a run overshoot arbitrarily, with the
    ledger recording the overrun after the GPU-hours are already gone. The
    ticket here is small enough that the deadline passes during the first
    step, so the abort path has to fire.
    """
    ledger = tmp_path / "ledger.jsonl"
    allocation = tmp_path / "allocation.yaml"
    _write_allocation(allocation, total_hours=1.0, milestone_budget=0.5)

    ckpt_dir = tmp_path / "checkpoints"
    config_path = tmp_path / "config.yaml"
    # Approved (well under the 0.5h milestone budget) but far too short to
    # actually finish: ~4 microseconds of wall-clock.
    _write_config(config_path, ckpt_dir, ticket_hours=1e-9)

    env = os.environ.copy()
    env["NANOWM_LEDGER_PATH"] = str(ledger)
    env["NANOWM_ALLOCATION_PATH"] = str(allocation)

    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), "--config", str(config_path)],
        capture_output=True, text=True, env=env, cwd=str(ROOT),
    )
    combined = result.stdout + result.stderr

    assert result.returncode == 2, f"expected ticket-abort exit code 2:\n{combined}"
    assert "APPROVED" in combined, "the ticket must have been approved at startup"
    assert "exceeded its" in combined, combined

    lines = [line for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1, "an aborted run still owes the ledger exactly one entry"
    entry = json.loads(lines[0])
    assert entry["result"].startswith("[ticket_exceeded]"), entry["result"]
    assert entry["gpu_seconds"] > 0, "the GPU time actually spent must be recorded"
