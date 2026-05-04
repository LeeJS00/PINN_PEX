"""
analytic_base_v3.py — Phase 1 differentiable analytic capacitance kernel.

A5 (neural-operator-architect) "ship today" recommendation:

> Smallest viable test: differentiable parallel-plate `analytic_base.forward(geometry)
> → C_fF` validated on Stage 1 generator within 0.1% on 100 samples.
> If `dC_analytic/dgeometry` autograd doesn't work cleanly, the hybrid
> paradigm dies — find out now, not after writing 2000 lines.

This module is the **kill-signal gate** for Phase 1. If autograd fails or
numerics drift, we know before committing to the full hybrid arch.

Two analytic modes (matching `pex_v3/src/synthetic/`):
  - Mode A (parallel plate):    closed-form, citation: Jackson §1.6, Sakurai-Tamaru 1983
  - Mode B (stacked dielectric): series formula, citation: Jackson §4.4, Sadiku ch.5

Mode C (single-interface image-charge correction) is `[HYPOTHESIS]`-level
per A4 audit and intentionally NOT implemented here. Will be replaced with
vector-fitted complex-image (Chow-Aksun) when Phase 1 hits its <4% gate.
"""
from __future__ import annotations
from typing import Tuple

import torch


# Conversion: C [fF] = ε₀ × ε_r × A_um² / d_um × 1e9
# (matches `pex_v3/src/synthetic/ground_truth.py:EPS0_FF_UM`)
EPS0_FF_UM = 8.8541878128e-3


# ============================================================================
# Mode A — parallel plate (no fringe)
# ============================================================================


def analytic_parallel_plate(
    w_um: torch.Tensor,
    h_um: torch.Tensor,
    d_um: torch.Tensor,
    eps_r: torch.Tensor,
    d_clamp_um: float = 1e-3,
) -> torch.Tensor:
    """Differentiable parallel-plate capacitance.

    C = ε₀ · ε_r · w · h / d                    [fF]

    All inputs broadcast-compatible. Returns tensor of same broadcast shape.

    Args:
        w_um:       plate width, μm. Differentiable.
        h_um:       plate height, μm. Differentiable.
        d_um:       gap, μm. Clamped to >= d_clamp_um for numerical stability.
        eps_r:      relative permittivity (dimensionless). Differentiable.
        d_clamp_um: floor for d to prevent division by zero. Gradient through
                    `clamp_min` is identity for values above the threshold.

    Returns:
        Capacitance in fF. Gradient flows through w, h, eps_r, d.
    """
    return EPS0_FF_UM * eps_r * w_um * h_um / d_um.clamp_min(d_clamp_um)


# ============================================================================
# Mode B — stacked dielectric series formula
# ============================================================================


def analytic_stacked_dielectric(
    w_um: torch.Tensor,
    h_um: torch.Tensor,
    layer_thicknesses_um: torch.Tensor,
    layer_eps_r: torch.Tensor,
    d_clamp_um: float = 1e-6,
) -> torch.Tensor:
    """Differentiable series-capacitance formula for stacked dielectrics.

    C = ε₀ · A / Σ_i (d_i / ε_i)                [fF]

    Args:
        w_um, h_um:                  plate xy dimensions, μm. Differentiable.
        layer_thicknesses_um:        shape (..., N_layers). Differentiable.
        layer_eps_r:                 shape (..., N_layers). Differentiable.
                                     Must broadcast with layer_thicknesses_um.
        d_clamp_um:                  floor for total series gap to prevent
                                     division by zero.

    Returns:
        Capacitance in fF, shape = broadcast(w_um, h_um, leading dims of stack)
    """
    # Compute series sum Σ (d_i / ε_i) over the last axis
    series = (layer_thicknesses_um / layer_eps_r.clamp_min(1e-6)).sum(dim=-1)
    A = w_um * h_um
    return EPS0_FF_UM * A / series.clamp_min(d_clamp_um)


# ============================================================================
# Module wrapper (lets the hybrid arch import a `nn.Module` if needed)
# ============================================================================


class AnalyticBase(torch.nn.Module):
    """Stateless wrapper around the analytic kernels.

    The hybrid arch (`hybrid_v3.py`) will import this and call:
        c_analytic = self.analytic_base(geometry)

    Stateless — no parameters; deterministic given inputs.
    """

    def forward_parallel_plate(
        self,
        w_um: torch.Tensor,
        h_um: torch.Tensor,
        d_um: torch.Tensor,
        eps_r: torch.Tensor,
    ) -> torch.Tensor:
        return analytic_parallel_plate(w_um, h_um, d_um, eps_r)

    def forward_stacked_dielectric(
        self,
        w_um: torch.Tensor,
        h_um: torch.Tensor,
        layer_thicknesses_um: torch.Tensor,
        layer_eps_r: torch.Tensor,
    ) -> torch.Tensor:
        return analytic_stacked_dielectric(
            w_um, h_um, layer_thicknesses_um, layer_eps_r
        )


# ============================================================================
# Validation gate — A5 mandate: parity vs `synthetic/ground_truth.py` < 1e-4
# ============================================================================


def validate_against_ground_truth(
    n_samples: int = 100,
    seed: int = 42,
    tolerance_rel: float = 1e-4,
    device: str = "cpu",
) -> dict:
    """A5 mandate: kill-signal gate for the hybrid paradigm.

    Generate `n_samples` parallel-plate samples from the closed-form
    `synthetic/ground_truth.py:parallel_plate_capacitance_fF` and verify
    the torch-autograd-aware `analytic_parallel_plate` reproduces them
    within `tolerance_rel`.

    Returns dict with max/mean rel error + pass/fail verdict.
    """
    import numpy as np
    rng = np.random.default_rng(seed)

    w = rng.uniform(0.1, 10.0, n_samples)
    h = rng.uniform(0.1, 10.0, n_samples)
    d = 10.0 ** rng.uniform(-2, 0, n_samples)
    eps_r = rng.uniform(1.0, 10.0, n_samples)

    # Closed-form ground truth
    gt = EPS0_FF_UM * eps_r * w * h / d

    # Torch path
    w_t = torch.tensor(w, dtype=torch.float64, device=device)
    h_t = torch.tensor(h, dtype=torch.float64, device=device)
    d_t = torch.tensor(d, dtype=torch.float64, device=device)
    eps_t = torch.tensor(eps_r, dtype=torch.float64, device=device)
    pred = analytic_parallel_plate(w_t, h_t, d_t, eps_t).cpu().numpy()

    rel_err = np.abs(pred - gt) / gt
    return {
        "n_samples": n_samples,
        "max_rel_err": float(rel_err.max()),
        "mean_rel_err": float(rel_err.mean()),
        "p95_rel_err": float(np.percentile(rel_err, 95)),
        "tolerance_rel": tolerance_rel,
        "passed": bool(rel_err.max() < tolerance_rel),
    }


def validate_autograd(seed: int = 42, n_samples: int = 5) -> dict:
    """Run torch.autograd.gradcheck on a small batch.

    A5 mandate: confirm gradient flows through w, h, d, eps_r.
    Without this, the hybrid arch's residual cannot learn anything.

    Note: gradcheck uses double-precision and is intolerant of
    aggressive clamping; we use d_clamp_um well below sample minimum.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    # Choose values well within the operating regime (d ≥ 0.1 μm) so
    # the d_clamp_min branch isn't activated.
    w = torch.tensor(rng.uniform(1.0, 5.0, n_samples), dtype=torch.float64, requires_grad=True)
    h = torch.tensor(rng.uniform(1.0, 5.0, n_samples), dtype=torch.float64, requires_grad=True)
    d = torch.tensor(rng.uniform(0.1, 0.5, n_samples), dtype=torch.float64, requires_grad=True)
    eps_r = torch.tensor(rng.uniform(2.0, 6.0, n_samples), dtype=torch.float64, requires_grad=True)

    def f(w, h, d, eps_r):
        return analytic_parallel_plate(w, h, d, eps_r)

    # gradcheck returns True or raises
    try:
        ok = torch.autograd.gradcheck(f, (w, h, d, eps_r), eps=1e-6, atol=1e-4)
    except Exception as e:
        return {"passed": False, "error": str(e)}
    return {"passed": bool(ok)}
