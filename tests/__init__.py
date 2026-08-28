"""Test package marker.

Required, not incidental: the fixture modules are imported as
`tests.model_fixtures` / `tests.train_fixtures`. Without this file that works
locally (conftest.py puts the repo root on sys.path and pytest's rootdir
inference covers the rest) but fails collection on Kaggle's Python 3.12 /
newer pytest with ModuleNotFoundError.
"""
