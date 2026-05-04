"""
test_analytic_base_v3.py — Phase 1 kill-signal gate.

A5 mandate: if these tests fail, the hybrid paradigm dies. Find out now,
not after writing 2000 lines of model code.

Tests:
  1. Parity vs `synthetic/ground_truth.py` to < 1e-4 relative error
  2. Autograd flows through w, h, d, eps_r (gradcheck)
  3. Module wrapper produces same numbers as functional API
  4. Mode B (stacked dielectric) collapses to Mode A in single-layer limit
  5. Mode B series formula correct on a 3-layer test case
  6. CUDA equivalence (skip if no CUDA available)
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.analytic_base_v3 import (
    EPS0_FF_UM,
    analytic_parallel_plate,
    analytic_stacked_dielectric,
    AnalyticBase,
    validate_against_ground_truth,
    validate_autograd,
)
from src.synthetic.ground_truth import (
    parallel_plate_capacitance_fF,
    stacked_dielectric_capacitance_fF,
)


# ============================================================================
# 1. Parity vs ground_truth.py
# ============================================================================


def test_parallel_plate_parity_100_samples():
    """A5 mandate: 100 samples from synthetic/ground_truth.py, max rel err < 1e-4."""
    out = validate_against_ground_truth(n_samples=100, seed=42, tolerance_rel=1e-4)
    assert out["passed"], f"Parity FAILED: {out}"
    assert out["max_rel_err"] < 1e-4, f"max_rel_err = {out['max_rel_err']}"
    assert out["mean_rel_err"] < 1e-5


def test_eps0_ff_um_value():
    """Constant must match synthetic/ground_truth.py."""
    from src.synthetic.ground_truth import EPS0_FF_UM as gt_eps
    assert abs(EPS0_FF_UM - gt_eps) < 1e-15


# ============================================================================
# 2. Autograd
# ============================================================================


def test_parallel_plate_autograd_gradcheck():
    """A5 mandate: gradcheck passes — gradients flow correctly."""
    out = validate_autograd(seed=42, n_samples=5)
    assert out["passed"], f"gradcheck FAILED: {out}"


def test_parallel_plate_grad_signs():
    """∂C/∂w > 0, ∂C/∂h > 0, ∂C/∂d < 0, ∂C/∂ε_r > 0."""
    w = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    h = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)
    d = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
    eps = torch.tensor([4.0], dtype=torch.float64, requires_grad=True)
    c = analytic_parallel_plate(w, h, d, eps)
    c.sum().backward()
    assert w.grad.item() > 0, f"∂C/∂w should be > 0, got {w.grad}"
    assert h.grad.item() > 0, f"∂C/∂h should be > 0, got {h.grad}"
    assert d.grad.item() < 0, f"∂C/∂d should be < 0, got {d.grad}"
    assert eps.grad.item() > 0, f"∂C/∂ε should be > 0, got {eps.grad}"


# ============================================================================
# 3. Module wrapper
# ============================================================================


def test_module_wrapper_matches_functional():
    """`AnalyticBase` module API matches functional API."""
    base = AnalyticBase()
    rng = np.random.default_rng(0)
    w = torch.tensor(rng.uniform(0.5, 5.0, 20))
    h = torch.tensor(rng.uniform(0.5, 5.0, 20))
    d = torch.tensor(rng.uniform(0.05, 0.5, 20))
    eps = torch.tensor(rng.uniform(2.0, 8.0, 20))

    c_func = analytic_parallel_plate(w, h, d, eps)
    c_mod = base.forward_parallel_plate(w, h, d, eps)
    assert torch.allclose(c_func, c_mod), f"module ≠ functional"


# ============================================================================
# 4. Mode B (stacked dielectric)
# ============================================================================


def test_stacked_dielectric_single_layer_collapses_to_pp():
    """1-layer stack with thickness d, ε_r should == parallel-plate(w, h, d, ε_r)."""
    w = torch.tensor([1.5])
    h = torch.tensor([2.0])
    d = torch.tensor([0.3])
    eps = torch.tensor([4.0])
    c_pp = analytic_parallel_plate(w, h, d, eps)
    c_st = analytic_stacked_dielectric(
        w, h,
        layer_thicknesses_um=d.unsqueeze(-1),
        layer_eps_r=eps.unsqueeze(-1),
    )
    assert torch.allclose(c_pp, c_st, rtol=1e-12), f"pp={c_pp}, st={c_st}"


def test_stacked_dielectric_3layer_series():
    """3-layer stack matches Σ d_i / ε_i formula."""
    w = torch.tensor([2.0])
    h = torch.tensor([2.0])
    thickness = torch.tensor([[0.1, 0.2, 0.15]])
    eps = torch.tensor([[3.0, 4.0, 5.0]])
    c_torch = analytic_stacked_dielectric(w, h, thickness, eps).item()
    c_gt = stacked_dielectric_capacitance_fF(
        w_um=2.0, h_um=2.0,
        layer_thicknesses_um=[0.1, 0.2, 0.15],
        layer_eps_r=[3.0, 4.0, 5.0],
    )
    rel = abs(c_torch - c_gt) / c_gt
    assert rel < 1e-6, f"rel_err = {rel}, torch={c_torch}, gt={c_gt}"


def test_stacked_dielectric_uniform_layers_collapse():
    """N layers all same ε, total thickness d, should == parallel plate."""
    w = torch.tensor([1.0])
    h = torch.tensor([1.0])
    thickness = torch.tensor([[0.2, 0.2, 0.2]])  # total = 0.6
    eps = torch.tensor([[3.5, 3.5, 3.5]])  # uniform
    c_st = analytic_stacked_dielectric(w, h, thickness, eps)
    c_pp = analytic_parallel_plate(w, h, torch.tensor([0.6]), torch.tensor([3.5]))
    assert torch.allclose(c_st, c_pp, rtol=1e-12)


# ============================================================================
# 5. CUDA equivalence (optional)
# ============================================================================


def test_parallel_plate_cuda_matches_cpu():
    """If CUDA available, GPU result matches CPU within tolerance."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    rng = np.random.default_rng(1)
    n = 50
    args_cpu = [torch.tensor(rng.uniform(0.5, 5.0, n)) for _ in range(4)]
    args_cuda = [a.cuda() for a in args_cpu]
    c_cpu = analytic_parallel_plate(*args_cpu)
    c_cuda = analytic_parallel_plate(*args_cuda).cpu()
    assert torch.allclose(c_cpu, c_cuda, rtol=1e-6)


# ============================================================================
# 6. Edge cases
# ============================================================================


def test_d_clamp_prevents_division_by_zero():
    """d=0 should not produce inf/nan."""
    w = torch.tensor([1.0])
    h = torch.tensor([1.0])
    d = torch.tensor([0.0])
    eps = torch.tensor([4.0])
    c = analytic_parallel_plate(w, h, d, eps, d_clamp_um=1e-3)
    assert torch.isfinite(c).all(), f"got non-finite: {c}"


def test_broadcasting():
    """w (B,) × h (B,) × d (1,) × eps (1,) broadcasts correctly."""
    w = torch.tensor([1.0, 2.0, 3.0])
    h = torch.tensor([1.0, 2.0, 3.0])
    d = torch.tensor([0.5])
    eps = torch.tensor([4.0])
    c = analytic_parallel_plate(w, h, d, eps)
    assert c.shape == (3,)
    expected = EPS0_FF_UM * 4.0 * w * h / 0.5
    assert torch.allclose(c, expected)
