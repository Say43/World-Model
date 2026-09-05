"""Muon: momentum SGD whose update is orthogonalized via Newton-Schulz.

One half of M3's optimizer ablation (Muon vs AdamW). Muon applies momentum
SGD to 2D parameters, then replaces the update matrix with its closest
semi-orthogonal matrix via a quintic Newton-Schulz iteration -- the claim
being that equalizing the update's singular values uses each step's
"budget" better than Adam's per-coordinate rescaling.

Only 2D parameters get this treatment. Biases, norm gains, and the
frame position embedding have no meaningful singular-value structure, so
they fall back to AdamW, matching the reference implementation.

fp32, not bf16: the reference runs Newton-Schulz in bf16 for speed, but
this project targets T4/sm75, which has no bf16 (CLAUDE.md's hard
constraints). The iteration is numerically delicate -- it relies on the
spectral norm staying inside the polynomial's convergence basin -- so
running it in fp32 rather than fp16 is a deliberate stability choice,
also spelled out in the project's kickoff constraints.
"""
from __future__ import annotations

from typing import Iterable, List

import torch

# Quintic coefficients from the reference implementation. Tuned so the
# iteration converges fast for singular values in [0, 1] while tolerating a
# flat region near zero, which matters because normalizing by the Frobenius
# norm leaves most singular values well below 1.
_NS_COEFFS = (3.4445, -4.7750, 2.0315)


def zeropower_via_newtonschulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Return an approximately semi-orthogonal matrix sharing G's row/column
    space -- i.e. G's singular values flattened toward 1.

    Computed in fp32 regardless of G's dtype (see module docstring), then
    cast back, so an fp16 AMP training step still gets a stable iteration.
    """
    if G.ndim != 2:
        raise ValueError(f"Newton-Schulz orthogonalization needs a 2D matrix, got shape {tuple(G.shape)}")

    a, b, c = _NS_COEFFS
    original_dtype = G.dtype
    X = G.float()
    X = X / (X.norm() + eps)

    # The iteration is written for wide matrices; transpose tall ones so the
    # X @ X.T products stay on the smaller dimension.
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T
    return X.to(original_dtype)


class Muon(torch.optim.Optimizer):
    """Muon for 2D parameters only.

    Callers are expected to route non-2D parameters to a separate optimizer
    (see `split_muon_adamw_params`); passing a non-2D parameter here raises
    rather than silently falling back, so a routing mistake surfaces at
    construction instead of quietly training part of the model differently
    than intended.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        params = list(params)
        for param in params:
            if param.ndim != 2:
                raise ValueError(
                    f"Muon only handles 2D parameters, got shape {tuple(param.shape)}; "
                    "route biases/norms/embeddings to AdamW instead"
                )
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov,
            ns_steps=ns_steps, weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group["momentum"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)
                update = grad.add(buf, alpha=momentum) if group["nesterov"] else buf

                update = zeropower_via_newtonschulz(update, steps=group["ns_steps"])

                # The orthogonalized update has singular values ~1 regardless
                # of the matrix's shape, so a non-square matrix would
                # otherwise take a systematically larger or smaller effective
                # step than a square one. This is the reference scaling.
                scale = max(1.0, param.size(0) / param.size(1)) ** 0.5

                if group["weight_decay"]:
                    param.mul_(1 - group["lr"] * group["weight_decay"])
                param.add_(update, alpha=-group["lr"] * scale)

        return loss


def split_muon_adamw_params(model: torch.nn.Module) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Split a model's trainable parameters into (2D -> Muon, rest -> AdamW).

    Deliberately shape-based rather than name-based, matching how
    src/train/mup.py classifies parameters: a name-based rule silently stops
    working when the architecture changes, which already bit this project
    once (see mup.py's docstring on the SwiGLU/raymap misclassification).
    """
    muon_params, adamw_params = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        (muon_params if param.ndim == 2 else adamw_params).append(param)
    return muon_params, adamw_params
