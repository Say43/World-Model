"""Shared pytest configuration: ensure the repo root is importable as `src`.

There is no packaging config (setup.py/pyproject) yet, so pytest's default
import mode won't put the repo root on sys.path on its own. This conftest
lives at the top of tests/ so pytest picks it up for every test module,
regardless of which subagent's test_*.py files are being collected.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
