"""
test_synthetic_stages.py — Stage 1 + Stage 2 generator + ground-truth invariants.

These tests gate the synthetic-data pipeline. If any fails, the curriculum
is producing data that disagrees with closed-form physics — abort before
training on it.
"""
from __future__ import annotations
import math

import numpy as np
import pytest

from src.synthetic.ground_truth import (
    EPS0_FF_UM,
    parallel_plate_capacitance_fF,
    stacked_dielectric_capacitance_fF,
    half_space_image_correction_factor,
    interface_corrected_capacitance_fF,
    verify_module_against_parallel_plate,
    verify_module_against_stacked_dielectric,
)
from src.synthetic.stage1_parallel_plate import (
    ParallelPlateSample,
    generate_parallel_plate_stream,
)
from src.synthetic.stage2_layered_image import (
    StackedDielectricSample,
    SingleInterfaceSample,
    generate_stacked_dielectric_stream,
    generate_single_interface_stream,
    verify_uniform_halfspace_limit,
)


# ============================================================================
# Constants + closed-form physics
# ============================================================================


def test_eps0_ff_um_value():
    """EPS0_FF_UM = ε₀ × 1e9 ≈ 8.854e-3."""
    assert abs(EPS0_FF_UM - 8.8541878e-3) < 1e-7


def test_parallel_plate_unit_consistency():
    """1 μm × 1 μm plate, 1 μm gap, ε=1: C = ε₀ × 1 μm² / 1 μm.

    Numerically: C = 8.854e-3 fF. (Sanity check the unit conversion.)
    """
    c = parallel_plate_capacitance_fF(w_um=1.0, h_um=1.0, d_um=1.0, eps_r=1.0)
    assert abs(c - EPS0_FF_UM) < 1e-9


def test_parallel_plate_scaling_in_area():
    """C ∝ w·h."""
    c1 = parallel_plate_capacitance_fF(1.0, 1.0, 0.1, 4.0)
    c4 = parallel_plate_capacitance_fF(2.0, 2.0, 0.1, 4.0)  # 4× area
    assert abs(c4 - 4 * c1) < 1e-12


def test_parallel_plate_scaling_in_d():
    """C ∝ 1/d."""
    c1 = parallel_plate_capacitance_fF(1.0, 1.0, 0.1, 4.0)
    c2 = parallel_plate_capacitance_fF(1.0, 1.0, 0.2, 4.0)  # 2× gap
    assert abs(c1 - 2 * c2) < 1e-12


def test_parallel_plate_scaling_in_eps():
    """C ∝ ε_r."""
    c1 = parallel_plate_capacitance_fF(1.0, 1.0, 0.1, 1.0)
    c4 = parallel_plate_capacitance_fF(1.0, 1.0, 0.1, 4.0)
    assert abs(c4 - 4 * c1) < 1e-12


def test_parallel_plate_d_zero_raises():
    with pytest.raises(ValueError):
        parallel_plate_capacitance_fF(1.0, 1.0, 0.0, 4.0)


# ============================================================================
# Stacked dielectric
# ============================================================================


def test_stacked_dielectric_single_layer_equals_parallel_plate():
    """1-layer stacked = parallel-plate."""
    c_pp = parallel_plate_capacitance_fF(1.0, 1.0, 0.5, 4.0)
    c_st = stacked_dielectric_capacitance_fF(
        w_um=1.0, h_um=1.0,
        layer_thicknesses_um=[0.5],
        layer_eps_r=[4.0],
    )
    assert abs(c_pp - c_st) < 1e-12


def test_stacked_dielectric_uniform_layers_equals_parallel_plate():
    """N layers, all same ε, total d = sum: collapses to parallel plate."""
    c_pp = parallel_plate_capacitance_fF(1.0, 1.0, 0.6, 3.0)
    c_st = stacked_dielectric_capacitance_fF(
        w_um=1.0, h_um=1.0,
        layer_thicknesses_um=[0.2, 0.2, 0.2],
        layer_eps_r=[3.0, 3.0, 3.0],
    )
    assert abs(c_pp - c_st) < 1e-12


def test_stacked_dielectric_series_formula():
    """Two layers, half-half: 1/C = 1/C1 + 1/C2."""
    c1 = parallel_plate_capacitance_fF(1.0, 1.0, 0.3, 4.0)
    c2 = parallel_plate_capacitance_fF(1.0, 1.0, 0.3, 2.0)
    c_series = 1.0 / (1.0 / c1 + 1.0 / c2)
    c_st = stacked_dielectric_capacitance_fF(
        w_um=1.0, h_um=1.0,
        layer_thicknesses_um=[0.3, 0.3],
        layer_eps_r=[4.0, 2.0],
    )
    assert abs(c_st - c_series) < 1e-9


def test_stacked_dielectric_lengths_must_match():
    with pytest.raises(ValueError):
        stacked_dielectric_capacitance_fF(1.0, 1.0, [0.1, 0.2], [4.0])


def test_stacked_dielectric_empty_raises():
    with pytest.raises(ValueError):
        stacked_dielectric_capacitance_fF(1.0, 1.0, [], [])


# ============================================================================
# Half-space image correction
# ============================================================================


def test_image_factor_uniform_is_zero():
    """ε₁ = ε₂ → no interface, k = 0."""
    assert half_space_image_correction_factor(4.0, 4.0) == 0.0


def test_image_factor_sign():
    """ε₁ > ε₂ → k > 0; ε₁ < ε₂ → k < 0."""
    assert half_space_image_correction_factor(8.0, 2.0) > 0
    assert half_space_image_correction_factor(2.0, 8.0) < 0


def test_image_factor_perfect_metal_limit():
    """ε₂ → ∞ (perfect metal ground): k → -1."""
    k = half_space_image_correction_factor(4.0, 1e9)
    assert abs(k - (-1.0)) < 1e-3


def test_interface_corrected_collapses_to_parallel_plate():
    """When eps_below = eps_between, Mode B = parallel plate exactly."""
    c_pp = parallel_plate_capacitance_fF(2.0, 2.0, 0.1, 4.0)
    c_int = interface_corrected_capacitance_fF(
        w_um=2.0, h_um=2.0, d_um=0.1,
        eps_r_between=4.0, eps_r_below=4.0,
    )
    assert abs(c_pp - c_int) < 1e-12


def test_interface_correction_larger_than_pp_for_higher_eps_below():
    """ε_below > ε_between → k < 0 → with our α=-1 convention, correction > 1.

    NOTE (Phase C audit, 2026-05-02): The Mode B `interface_corrected`
    formula `1 + (-1)·k·d/√A` is `[HYPOTHESIS]`-level (no Jackson/Sadiku
    citation). The α=-1 sign was chosen to produce attraction-like
    enhancement when k<0, but this implicitly assumes the bottom plate is
    transparent (image-method valid). For a real BEOL ground plate that
    screens the half-space below, the correction should be ≈ 0.

    Until the formula is replaced with a derived Sommerfeld/complex-image
    (Chow-Aksun) approximation or cross-validated against FEM, this test
    documents the CURRENT IMPLEMENTATION'S behavior, not the physics.

    See `pex_v3/docs/AGENT_INFRA_GAP.md` for the audit trail.
    """
    c_pp = parallel_plate_capacitance_fF(2.0, 2.0, 0.5, 3.0)
    c_int = interface_corrected_capacitance_fF(
        w_um=2.0, h_um=2.0, d_um=0.5,
        eps_r_between=3.0, eps_r_below=8.0,
    )
    # Implementation choice: with α=-1 and k<0 (eps_below > eps_between),
    # factor > 1, so C_int > C_pp. Test verifies the implementation matches
    # the documented α=-1 convention (NOT that this is physically right).
    assert c_int > c_pp


# ============================================================================
# Stage 1 generator
# ============================================================================


def test_stage1_generator_count():
    samples = list(generate_parallel_plate_stream(n_samples=50, seed=1))
    assert len(samples) == 50


def test_stage1_generator_zero_samples():
    samples = list(generate_parallel_plate_stream(n_samples=0, seed=1))
    assert samples == []


def test_stage1_generator_deterministic():
    a = list(generate_parallel_plate_stream(n_samples=10, seed=42))
    b = list(generate_parallel_plate_stream(n_samples=10, seed=42))
    assert a == b


def test_stage1_generator_matches_analytic():
    """Every generated sample must satisfy the closed-form formula."""
    for s in generate_parallel_plate_stream(n_samples=200, seed=7):
        gold = parallel_plate_capacitance_fF(s.w_um, s.h_um, s.d_um, s.eps_r)
        rel = abs(s.c_fF - gold) / gold
        assert rel < 1e-12


def test_stage1_generator_respects_ranges():
    samples = list(generate_parallel_plate_stream(
        n_samples=200, seed=1,
        d_range=(0.05, 0.5),
        w_range=(1.0, 4.0),
        h_range=(1.0, 4.0),
        eps_range=(2.0, 5.0),
    ))
    for s in samples:
        assert 0.05 <= s.d_um <= 0.5
        assert 1.0 <= s.w_um <= 4.0
        assert 1.0 <= s.h_um <= 4.0
        assert 2.0 <= s.eps_r <= 5.0


def test_stage1_materialize_writes_csv(tmp_path):
    from src.synthetic.stage1_parallel_plate import materialize_parallel_plate_dataset
    import pandas as pd

    out = tmp_path / "stage1.csv"
    materialize_parallel_plate_dataset(n_samples=20, seed=3, out_path=out)
    assert out.exists()
    df = pd.read_csv(out)
    assert len(df) == 20
    assert set(df.columns) == {"w_um", "h_um", "d_um", "eps_r", "c_fF"}


# ============================================================================
# Stage 2 generators
# ============================================================================


def test_stage2_stacked_generator_count():
    samples = list(generate_stacked_dielectric_stream(n_samples=30, seed=1))
    assert len(samples) == 30


def test_stage2_stacked_generator_matches_analytic():
    for s in generate_stacked_dielectric_stream(n_samples=100, seed=2):
        gold = stacked_dielectric_capacitance_fF(
            s.w_um, s.h_um,
            list(s.layer_thicknesses_um),
            list(s.layer_eps_r),
        )
        rel = abs(s.c_fF - gold) / gold
        assert rel < 1e-12


def test_stage2_stacked_total_thickness_sums():
    for s in generate_stacked_dielectric_stream(n_samples=20, seed=3):
        assert abs(s.total_thickness_um - sum(s.layer_thicknesses_um)) < 1e-12


def test_stage2_interface_generator_collapses_to_pp():
    """For samples where eps_between == eps_below, c equals parallel plate."""
    # Force eps ranges to produce many overlap cases; here we use the
    # `interface_corrected_capacitance_fF` direct call which is guaranteed
    # to collapse exactly when eps_below == eps_between.
    rng = np.random.default_rng(5)
    for _ in range(50):
        w = float(rng.uniform(0.5, 5.0))
        h = float(rng.uniform(0.5, 5.0))
        d = float(10 ** rng.uniform(-2, 0))
        eps = float(rng.uniform(1.5, 8.0))
        c_int = interface_corrected_capacitance_fF(w, h, d, eps, eps)
        c_pp = parallel_plate_capacitance_fF(w, h, d, eps)
        assert abs(c_int - c_pp) < 1e-12


def test_verify_uniform_halfspace_limit_with_correct_module():
    """verify_uniform_halfspace_limit accepts a callable that collapses correctly."""
    def correct_module(w, h, d, eps_above, eps_below):
        return interface_corrected_capacitance_fF(w, h, d, eps_above, eps_below)
    passed, details = verify_uniform_halfspace_limit(correct_module, tolerance_rel=1e-6)
    assert passed, details


def test_verify_uniform_halfspace_limit_with_wrong_module():
    """verify_uniform_halfspace_limit catches a module that doesn't collapse."""
    def wrong_module(w, h, d, eps_above, eps_below):
        return interface_corrected_capacitance_fF(w, h, d, eps_above, eps_below) * 1.5
    passed, _ = verify_uniform_halfspace_limit(wrong_module, tolerance_rel=1e-3)
    assert not passed


# ============================================================================
# Module verifiers (used by Phase 1 architecture as a CI gate)
# ============================================================================


def test_verify_module_against_parallel_plate_passes_for_correct():
    def correct(w, h, d, eps_r):
        return parallel_plate_capacitance_fF(w, h, d, eps_r)
    passed, details = verify_module_against_parallel_plate(correct, tolerance_rel=1e-6)
    assert passed, details


def test_verify_module_against_stacked_dielectric_passes_for_correct():
    def correct(w, h, t, e):
        return stacked_dielectric_capacitance_fF(w, h, t, e)
    passed, details = verify_module_against_stacked_dielectric(correct, tolerance_rel=1e-6)
    assert passed, details
