"""Muon: Newton-Schulz orthogonalization and the optimizer around it.

The property that matters is spectral: the iteration must flatten a
matrix's singular values toward 1 (that is the entire premise of the
optimizer), and it must do so in fp32 even when handed fp16 input, since
T4/sm75 has no bf16 and the iteration is not fp16-stable.
"""
import pytest
import torch

from src.train.muon import Muon, split_muon_adamw_params, zeropower_via_newtonschulz


def _singular_values(matrix: torch.Tensor) -> torch.Tensor:
    return torch.linalg.svdvals(matrix.float())


def test_newtonschulz_flattens_singular_values():
    torch.manual_seed(0)
    # Deliberately ill-conditioned: singular values spanning three orders of
    # magnitude, so "flattened toward 1" is a real change, not a no-op.
    u, _ = torch.linalg.qr(torch.randn(64, 64))
    v, _ = torch.linalg.qr(torch.randn(64, 64))
    spectrum = torch.logspace(0, -3, 64)
    matrix = u @ torch.diag(spectrum) @ v.T

    before = _singular_values(matrix)
    after = _singular_values(zeropower_via_newtonschulz(matrix, steps=5))

    assert before.max() / before.min() > 100, "test matrix should start ill-conditioned"
    # Not exactly 1 -- five quintic steps is an approximation, and the
    # smallest input singular values are hardest -- but the spread must
    # collapse dramatically.
    assert after.max() / after.min() < before.max() / before.min() / 10
    assert 0.5 < after.max() < 1.5


def test_newtonschulz_preserves_shape_for_tall_and_wide():
    for shape in [(32, 8), (8, 32), (16, 16)]:
        result = zeropower_via_newtonschulz(torch.randn(*shape), steps=5)
        assert result.shape == shape


def test_newtonschulz_runs_in_fp32_and_returns_input_dtype():
    """fp16 in, fp16 out -- but the iteration itself must run in fp32.

    Checked behaviorally rather than by inspecting internals: the same
    ill-conditioned matrix in fp16 must produce a result close to the fp32
    result. An fp16 iteration would accumulate visibly different (or
    non-finite) values on a matrix whose norm-normalized entries are small.
    """
    torch.manual_seed(0)
    u, _ = torch.linalg.qr(torch.randn(64, 64))
    v, _ = torch.linalg.qr(torch.randn(64, 64))
    matrix = u @ torch.diag(torch.logspace(0, -3, 64)) @ v.T

    result_fp32 = zeropower_via_newtonschulz(matrix, steps=5)
    result_fp16 = zeropower_via_newtonschulz(matrix.half(), steps=5)

    assert result_fp16.dtype == torch.float16
    assert torch.isfinite(result_fp16).all()
    torch.testing.assert_close(result_fp16.float(), result_fp32, atol=2e-2, rtol=2e-2)


def test_newtonschulz_rejects_non_2d():
    with pytest.raises(ValueError, match="2D"):
        zeropower_via_newtonschulz(torch.randn(4, 4, 4))


def test_muon_rejects_non_2d_parameters_at_construction():
    """A routing mistake must fail loudly at construction, not silently
    train part of the model with the wrong optimizer."""
    bias = torch.nn.Parameter(torch.zeros(8))
    with pytest.raises(ValueError, match="2D"):
        Muon([bias])


def test_muon_step_decreases_a_simple_quadratic():
    """Muon takes fixed-size steps, so budget the step count accordingly.

    Because the update is orthogonalized, its singular values are ~1
    regardless of the gradient's magnitude -- the step length is
    approximately `lr * sqrt(min(rows, cols))` in Frobenius norm every
    iteration, not proportional to the gradient. Convergence is therefore
    roughly linear in distance/step, not the exponential decay plain
    gradient descent shows on a quadratic. A first version of this test
    asked for a 50% loss drop in 30 steps at lr=0.05, which is ~6 units of
    travel against a ~22 unit starting distance -- it failed on correct
    code purely because the expectation was calibrated for gradient-
    proportional dynamics.
    """
    torch.manual_seed(0)
    weight = torch.nn.Parameter(torch.randn(16, 16))
    target = torch.randn(16, 16)
    optimizer = Muon([weight], lr=0.05)

    losses = []
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        loss = (weight - target).pow(2).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert all(torch.isfinite(torch.tensor(value)) for value in losses)
    assert losses[-1] < losses[0] * 0.05, f"did not converge: {losses[0]:.4f} -> {losses[-1]:.4f}"
    # Fixed-size steps overshoot near the optimum, so allow a noisy tail but
    # require the trajectory to be monotone over its bulk.
    assert losses[100] < losses[50] < losses[10] < losses[0]


def test_muon_momentum_buffer_persists_in_state():
    weight = torch.nn.Parameter(torch.randn(8, 8))
    optimizer = Muon([weight], lr=0.01)
    weight.grad = torch.randn(8, 8)
    optimizer.step()
    assert "momentum_buffer" in optimizer.state[weight]
    assert optimizer.state[weight]["momentum_buffer"].shape == (8, 8)


def test_split_routes_matrices_to_muon_and_the_rest_to_adamw():
    from src.train.model_factory import build_causal_dit

    model = build_causal_dit("5m", context_length=16)
    muon_params, adamw_params = split_muon_adamw_params(model)

    assert all(param.ndim == 2 for param in muon_params)
    assert all(param.ndim != 2 for param in adamw_params)
    total = sum(param.numel() for param in muon_params) + sum(param.numel() for param in adamw_params)
    assert total == sum(param.numel() for param in model.parameters())
    # The matrices should dominate: if this ever inverts, the split is wrong.
    assert sum(p.numel() for p in muon_params) > sum(p.numel() for p in adamw_params)
