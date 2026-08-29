"""Tests for src/eval/metrics.py: PSNR math, isolated from any AE weights."""
import numpy as np
import pytest

from src.eval.metrics import psnr


def test_psnr_identical_images_is_infinite():
    img = np.random.default_rng(0).random((8, 8, 3), dtype=np.float32)
    assert psnr(img, img) == float("inf")


def test_psnr_decreases_with_more_noise():
    rng = np.random.default_rng(1)
    img = rng.random((16, 16, 3), dtype=np.float32)
    small_noise = img + rng.normal(0, 0.01, img.shape).astype(np.float32)
    big_noise = img + rng.normal(0, 0.2, img.shape).astype(np.float32)
    assert psnr(img, small_noise) > psnr(img, big_noise)


def test_psnr_known_value():
    # MSE = 0.01 exactly -> PSNR = 20*log10(1) - 10*log10(0.01) = 20 dB.
    a = np.zeros((4, 4, 1), dtype=np.float32)
    b = np.full((4, 4, 1), 0.1, dtype=np.float32)
    assert psnr(a, b) == pytest.approx(20.0, abs=1e-4)


def test_psnr_respects_data_range():
    a = np.zeros((2, 2, 1), dtype=np.float32)
    b = np.full((2, 2, 1), 25.5, dtype=np.float32)  # MSE = 650.25 in [0,255] range
    val = psnr(a, b, data_range=255.0)
    assert val > 0  # would be nonsensical (large negative) with the wrong data_range
