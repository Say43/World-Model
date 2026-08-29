"""Tests for src/eval/ae_ceiling.py using a fake AE (no network, no torch AE
weights needed) plus real rendered frames from src/data/."""
import numpy as np

from src.eval.ae_ceiling import AECandidate, FrozenAE, evaluate_candidate
from src.eval.test_frames import render_test_frames


class _PerfectAE(FrozenAE):
    def reconstruct(self, frames: np.ndarray) -> np.ndarray:
        return frames.copy()


class _NoisyAE(FrozenAE):
    def __init__(self, sigma: float):
        self.sigma = sigma

    def reconstruct(self, frames: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(0)
        return np.clip(frames + rng.normal(0, self.sigma, frames.shape), 0.0, 1.0).astype(np.float32)


def test_evaluate_candidate_perfect_reconstruction_is_infinite_psnr():
    frames = render_test_frames(resolution=32, scene_seeds=[0], frames_per_scene=4)
    candidate = AECandidate(name="perfect", resolution=32, tokens_per_frame=4,
                             license="test", loader=lambda: _PerfectAE())
    result = evaluate_candidate(candidate, frames)
    assert result["psnr_mean"] == float("inf")
    assert result["n_frames"] == frames.shape[0]


def test_evaluate_candidate_noisier_ae_scores_lower():
    frames = render_test_frames(resolution=32, scene_seeds=[0], frames_per_scene=4)
    low_noise = AECandidate(name="low", resolution=32, tokens_per_frame=4,
                             license="test", loader=lambda: _NoisyAE(0.01))
    high_noise = AECandidate(name="high", resolution=32, tokens_per_frame=4,
                              license="test", loader=lambda: _NoisyAE(0.2))
    r_low = evaluate_candidate(low_noise, frames)
    r_high = evaluate_candidate(high_noise, frames)
    assert r_low["psnr_mean"] > r_high["psnr_mean"]


def test_evaluate_candidate_rejects_wrong_resolution():
    frames = render_test_frames(resolution=32, scene_seeds=[0], frames_per_scene=2)
    candidate = AECandidate(name="mismatched", resolution=64, tokens_per_frame=4,
                             license="test", loader=lambda: _PerfectAE())
    try:
        evaluate_candidate(candidate, frames)
        assert False, "expected an assertion error on resolution mismatch"
    except AssertionError:
        pass
