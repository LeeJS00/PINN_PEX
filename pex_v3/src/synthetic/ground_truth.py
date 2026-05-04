"""
ground_truth.py — analytic verifiers for synthetic curriculum.

Validation discipline (synthetic-data-pipeline-owner mandate):
    - Every stage must reproduce known closed-form in limit cases
    - No oracle (Q3D, FastCap) is trusted without spot-check vs analytic

This module collects the analytic / spot-check helpers used by
stage1, stage2, and (later) the stage3-4-4.5 cross-validation harness.

All capacitance values are returned in **fF** (1e-15 F).
All length inputs are in **μm** unless explicitly noted.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np


# ============================================================================
# physical constants
# ============================================================================

EPS0_F_PER_M = 8.8541878128e-12  # vacuum permittivity, F/m

# Conversion: C [fF] from ε [F/m] × A [μm²] / d [μm]
#   C [F] = ε × A_m² / d_m = ε × (A_um² × 1e-12) / (d_um × 1e-6)
#         = ε × A_um² / d_um × 1e-6
#   C [fF] = C [F] × 1e15 = ε × A_um² / d_um × 1e9
# So:
#   C [fF] = (EPS0_F_PER_M × 1e9) × eps_r × A_um² / d_um
# Define EPS0_FF_UM such that C_fF = EPS0_FF_UM × eps_r × A_um² / d_um:
EPS0_FF_UM = EPS0_F_PER_M * 1.0e9  # ≈ 8.8541878e-3   fF·μm⁻¹ per (μm²·dimensionless)


# ============================================================================
# Closed-form capacitance helpers
# ============================================================================

def parallel_plate_capacitance_fF(
    w_um: float,
    h_um: float,
    d_um: float,
    eps_r: float,
) -> float:
    """Analytic parallel-plate capacitance in fF (no fringe).

    C = ε₀ · ε_r · (w · h) / d

    Args:
        w_um, h_um: plate xy dimensions, μm
        d_um:       plate separation, μm
        eps_r:      relative permittivity of dielectric between plates

    Returns:
        Capacitance in fF.
    """
    if d_um <= 0:
        raise ValueError(f"d_um must be > 0, got {d_um}")
    return EPS0_FF_UM * eps_r * (w_um * h_um) / d_um


def stacked_dielectric_capacitance_fF(
    w_um: float,
    h_um: float,
    layer_thicknesses_um: Sequence[float],
    layer_eps_r: Sequence[float],
) -> float:
    """Capacitance of a stack of dielectrics between two parallel plates.

    Series-capacitance formula:
        1 / C_total = Σ_i (1 / C_i)   where C_i = ε₀ · ε_i · A / d_i

    Equivalently:
        C_total = ε₀ · A / Σ_i (d_i / ε_i)

    Args:
        w_um, h_um: plate xy dimensions, μm
        layer_thicknesses_um: per-layer thickness (μm), summed = total d
        layer_eps_r: per-layer relative permittivity, same length as thicknesses

    Returns:
        Capacitance in fF.
    """
    if len(layer_thicknesses_um) != len(layer_eps_r):
        raise ValueError(
            f"thickness/eps length mismatch: "
            f"{len(layer_thicknesses_um)} vs {len(layer_eps_r)}"
        )
    if len(layer_thicknesses_um) == 0:
        raise ValueError("Empty stack")
    if any(t <= 0 for t in layer_thicknesses_um):
        raise ValueError("All layer thicknesses must be positive")
    if any(e <= 0 for e in layer_eps_r):
        raise ValueError("All layer permittivities must be positive")

    series = sum(t / e for t, e in zip(layer_thicknesses_um, layer_eps_r))
    A_um2 = w_um * h_um
    return EPS0_FF_UM * A_um2 / series


def half_space_image_correction_factor(
    eps1: float,
    eps2: float,
) -> float:
    """Image-charge reflection coefficient at a flat dielectric interface.

    For a charge q in medium 1 (ε₁) at distance z₀ from a planar interface
    with medium 2 (ε₂) below, the image charge in medium 1's continuation
    is:
        q' = ((ε₁ - ε₂) / (ε₁ + ε₂)) · q

    Returns:
        The dimensionless reflection coefficient k = (ε₁ − ε₂) / (ε₁ + ε₂).
        k > 0 if ε₁ > ε₂ (image attracts opposite-sign);
        k = 0 when ε₁ = ε₂ (no interface).
    """
    return (eps1 - eps2) / (eps1 + eps2)


def interface_corrected_capacitance_fF(
    w_um: float,
    h_um: float,
    d_um: float,
    eps_r_between: float,
    eps_r_below: float,
) -> float:
    """[HYPOTHESIS] Heuristic correction; NOT a derived physics formula.

    ⚠️ Phase C physics audit (2026-05-02) found this formula has NO canonical
    citation. The expression:

        correction = 1 + (-1) · k · (d/√A),  k = (ε_between − ε_below)/(ε_between + ε_below)

    looks like an image-charge leading-order correction, but the real
    layered-media plate-near-halfspace problem uses Sommerfeld / complex-image
    (Chow 1991, Aksun 1996) and is **nonlinear in d/√A**. The α=−1 sign
    silently encodes a geometry assumption (bottom plate transparent /
    image-method valid) that is generally false when the bottom is a BEOL
    ground plate screening the half-space below.

    Use only:
      - As a Stage 2 SANITY check on `eps_below = eps_between` (collapses
        to parallel plate exactly — this part IS correct).
      - For Phase 1 pretraining IF `d/√A < 0.05` where the correction → 0
        anyway and the formula reduces to parallel-plate harmlessly.

    Do NOT use as ground truth for Stage 2 Mode B production samples until
    replaced with vector-fitted complex-image kernel.

    Args:
        w_um, h_um: plate xy dimensions, μm
        d_um: gap between plates, μm
        eps_r_between: dielectric between the plates
        eps_r_below: dielectric below the bottom (ground) plate
                     (= eps_r_between → parallel-plate exactly, no correction)

    Returns:
        Capacitance in fF (HYPOTHESIS-level for d/√A > 0.05).
    """
    base = parallel_plate_capacitance_fF(w_um, h_um, d_um, eps_r_between)
    k = half_space_image_correction_factor(eps_r_between, eps_r_below)
    sqrtA = math.sqrt(w_um * h_um)
    correction = 1.0 + (-1.0) * k * (d_um / sqrtA)
    # Clamp to physical positive value (defensive):
    correction = max(correction, 0.1)
    return base * correction


# ============================================================================
# Cross-validation helpers (used to spot-check Q3D vs FastCap later)
# ============================================================================

@dataclass(frozen=True)
class OracleAgreementSummary:
    n_samples: int
    mean_disagreement_pct: float
    median_disagreement_pct: float
    p95_disagreement_pct: float
    max_disagreement_pct: float
    n_outliers: int  # samples where disagreement > tolerance_pct


def cross_validate_oracles(
    samples: Iterable,
    oracle_a,
    oracle_b,
    tolerance_pct: float = 1.0,
) -> OracleAgreementSummary:
    """Cross-check two oracles (e.g., Q3D vs FastCap) on a shared sample set.

    Args:
        samples:         iterable of geometry inputs (oracle-specific)
        oracle_a:        callable(sample) -> capacitance in fF
        oracle_b:        callable(sample) -> capacitance in fF
        tolerance_pct:   relative disagreement above which sample is flagged outlier

    Returns:
        OracleAgreementSummary with mean/median/p95/max disagreement and
        outlier count.
    """
    diffs_pct = []
    for s in samples:
        c_a = oracle_a(s)
        c_b = oracle_b(s)
        avg = 0.5 * (c_a + c_b)
        if avg <= 0:
            continue
        diffs_pct.append(100.0 * abs(c_a - c_b) / avg)

    if not diffs_pct:
        raise ValueError("cross_validate_oracles: no valid samples")

    arr = np.asarray(diffs_pct)
    return OracleAgreementSummary(
        n_samples=len(arr),
        mean_disagreement_pct=float(arr.mean()),
        median_disagreement_pct=float(np.median(arr)),
        p95_disagreement_pct=float(np.percentile(arr, 95)),
        max_disagreement_pct=float(arr.max()),
        n_outliers=int((arr > tolerance_pct).sum()),
    )


# ============================================================================
# Verification helpers (Phase 1 sanity gates)
# ============================================================================

def verify_module_against_parallel_plate(
    module_fn,
    tolerance_rel: float = 1e-3,
    n_test: int = 50,
    seed: int = 17,
) -> Tuple[bool, dict]:
    """Sweep parallel-plate geometry, compare to closed-form.

    Args:
        module_fn: callable(w, h, d, eps_r) -> predicted_C_fF
        tolerance_rel: max relative error allowed (default 0.1%)
        n_test: number of samples to check
        seed: RNG seed

    Returns:
        (pass: bool, details: dict)
    """
    rng = np.random.default_rng(seed)
    w = rng.uniform(0.1, 10.0, n_test)
    h = rng.uniform(0.1, 10.0, n_test)
    d = 10.0 ** rng.uniform(-2, 0, n_test)  # 0.01-1 μm log-uniform
    eps_r = rng.uniform(1.0, 10.0, n_test)

    rels = []
    for i in range(n_test):
        gold = parallel_plate_capacitance_fF(w[i], h[i], d[i], eps_r[i])
        pred = module_fn(w[i], h[i], d[i], eps_r[i])
        rels.append(abs(pred - gold) / gold)

    rels = np.asarray(rels)
    passed = bool(rels.max() < tolerance_rel)
    return passed, {
        "n_test": n_test,
        "max_rel_err": float(rels.max()),
        "mean_rel_err": float(rels.mean()),
        "p95_rel_err": float(np.percentile(rels, 95)),
        "tolerance_rel": tolerance_rel,
    }


def verify_module_against_stacked_dielectric(
    module_fn,
    tolerance_rel: float = 1e-3,
    n_test: int = 50,
    seed: int = 19,
) -> Tuple[bool, dict]:
    """Sweep stacked-dielectric geometry; verify module reproduces series formula."""
    rng = np.random.default_rng(seed)

    rels = []
    for i in range(n_test):
        n_layers = int(rng.integers(1, 6))
        thicknesses = rng.uniform(0.01, 0.5, n_layers).tolist()
        eps_r = rng.uniform(1.5, 8.0, n_layers).tolist()
        w = float(rng.uniform(0.5, 5.0))
        h = float(rng.uniform(0.5, 5.0))

        gold = stacked_dielectric_capacitance_fF(w, h, thicknesses, eps_r)
        pred = module_fn(w, h, thicknesses, eps_r)
        rels.append(abs(pred - gold) / gold)

    rels = np.asarray(rels)
    passed = bool(rels.max() < tolerance_rel)
    return passed, {
        "n_test": n_test,
        "max_rel_err": float(rels.max()),
        "mean_rel_err": float(rels.mean()),
        "p95_rel_err": float(np.percentile(rels, 95)),
        "tolerance_rel": tolerance_rel,
    }
