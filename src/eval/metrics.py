"""PSNR / LPIPS for M0's autoencoder-ceiling comparison.

Every model number in this project is reported relative to this ceiling
(see CLAUDE.md's eval-harness rules) -- so these two functions are the
foundation the rest of eval/ builds on. Kept dependency-light: LPIPS needs
`lpips` (torch + AlexNet/VGG weights, requires internet on first use);
PSNR needs nothing beyond numpy.
"""
from __future__ import annotations

import numpy as np


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio between two (H, W, C) float arrays in [0, data_range].

    Returns +inf for a perfect match rather than raising -- an autoencoder
    that reconstructs exactly is a legitimate (if suspicious) result, not
    an error.
    """
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse == 0.0:
        return float("inf")
    return 20.0 * np.log10(data_range) - 10.0 * np.log10(mse)


class LPIPS:
    """Thin wrapper so callers don't need `lpips` installed to import this
    module -- only to actually call `.compute()`. Model download requires
    internet; that only matters on Kaggle, not for testing the metric math.
    """

    def __init__(self, net: str = "alex", device: str = "cpu"):
        import lpips as _lpips  # deferred: heavy import, network on first use

        self._model = _lpips.LPIPS(net=net).to(device)
        self._model.eval()
        self.device = device

    @staticmethod
    def _to_lpips_input(img: np.ndarray):
        """(H, W, 3) float32 in [0, 1] -> (1, 3, H, W) float32 in [-1, 1]."""
        import torch

        t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
        return t * 2.0 - 1.0

    def compute(self, a: np.ndarray, b: np.ndarray) -> float:
        import torch

        with torch.no_grad():
            ta = self._to_lpips_input(a).to(self.device)
            tb = self._to_lpips_input(b).to(self.device)
            return float(self._model(ta, tb).item())
