"""Safety and topology checks for the budgeted M2 sweep entry point."""
import subprocess
import sys
from pathlib import Path

import yaml

from src.train.model_factory import build_causal_dit


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "run_m2_lr_sweep.py"
CONFIG = ROOT / "configs" / "m2_mup_lr_sweep.yaml"


def test_checked_in_config_is_locked_until_ticket_approval():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["run"]["ticket_hours"] is None

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(CONFIG)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "explicit user approval" in result.stderr


def test_m2_pair_changes_width_but_not_topology():
    proxy = build_causal_dit("m2_proxy_5m", num_heads=8).config
    target = build_causal_dit("15m", num_heads=8).config

    assert proxy.dim == 192
    assert target.dim == 320
    assert proxy.depth == target.depth == 11
    assert proxy.num_heads == target.num_heads == 8
