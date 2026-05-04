"""
test_calibration_v3.py — Tier 3 NNLS calibration invariants.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from src.baselines.calibration_v3 import (
    fit_scalar_calibration,
    apply_scalar_calibration,
    fit_per_layer_calibration,
    apply_per_layer_calibration,
    validate_calibration,
)


def _make_fixture_df(n: int = 1000, seed: int = 0, gnd_bias: float = 0.35,
                     cpl_bias: float = 1.81) -> pd.DataFrame:
    """Synthetic features with known analytic/golden bias."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        layer = int(rng.integers(2, 6))   # M2..M5
        c_gnd = float(np.abs(rng.normal(1.0, 1.0)) + 0.1)
        c_cpl = float(np.abs(rng.normal(0.5, 0.5)) + 0.05)
        # Apply known bias + log-normal noise
        compact_gnd = c_gnd * gnd_bias * float(np.exp(rng.normal(0, 0.3)))
        compact_cpl = c_cpl * cpl_bias * float(np.exp(rng.normal(0, 0.3)))
        d = {
            "split": "train" if i < n * 0.7 else ("valid" if i < n * 0.85 else "test"),
            "c_gnd_fF": c_gnd,
            "c_cpl_total_fF": c_cpl,
            "compact_gnd_estimate_fF": compact_gnd,
            "compact_cpl_estimate_total_fF": compact_cpl,
        }
        for L in range(1, 9):
            d[f"layer_hist_M{L}"] = 10.0 if L == layer else 0.0
        d["layer_hist_M9_plus"] = 0.0
        rows.append(d)
    return pd.DataFrame(rows)


# ============================================================================
# Scalar calibration
# ============================================================================


def test_scalar_calibration_fits_known_bias():
    """When fixture has gnd_bias=0.35, fit s_gnd ≈ 1/0.35 = 2.86."""
    df = _make_fixture_df(n=2000, seed=42, gnd_bias=0.35, cpl_bias=1.81)
    train = df[df["split"] == "train"]
    calib = fit_scalar_calibration(train)
    # s_gnd = median(golden / analytic) ≈ 1/0.35 = 2.86
    assert 2.5 < calib.s_gnd < 3.3, f"s_gnd={calib.s_gnd}"
    assert 0.45 < calib.s_cpl < 0.65, f"s_cpl={calib.s_cpl}"


def test_scalar_calibration_apply_brings_median_to_1():
    """After applying scalar calibration, median ratio ≈ 1.0 on valid."""
    df = _make_fixture_df(n=2000, seed=43, gnd_bias=0.35, cpl_bias=1.81)
    train = df[df["split"] == "train"]
    valid = df[df["split"] == "valid"]
    calib = fit_scalar_calibration(train)
    valid_calibrated = apply_scalar_calibration(valid, calib)
    v = validate_calibration(valid_calibrated)
    assert 0.85 < v["median_ratio_gnd"] < 1.15, f"after calib gnd: {v}"
    assert 0.85 < v["median_ratio_cpl"] < 1.15, f"after calib cpl: {v}"


def test_scalar_calibration_no_inplace_default():
    """fit_scalar_calibration does not mutate input DataFrame."""
    df = _make_fixture_df(n=200, seed=44)
    train = df[df["split"] == "train"]
    orig_gnd = train["compact_gnd_estimate_fF"].copy()
    calib = fit_scalar_calibration(train)
    _ = apply_scalar_calibration(train, calib)
    pd.testing.assert_series_equal(train["compact_gnd_estimate_fF"], orig_gnd)


def test_scalar_calibration_in_place_mutates():
    df = _make_fixture_df(n=200, seed=45)
    train = df[df["split"] == "train"].copy()
    orig = train["compact_gnd_estimate_fF"].copy()
    calib = fit_scalar_calibration(train)
    apply_scalar_calibration(train, calib, in_place=True)
    # Mutated
    assert not np.allclose(train["compact_gnd_estimate_fF"], orig)


def test_scalar_calibration_missing_columns_raises():
    bad = pd.DataFrame({"split": ["train"], "c_gnd_fF": [1.0]})
    with pytest.raises(KeyError, match="compact_gnd"):
        fit_scalar_calibration(bad)


# ============================================================================
# Per-layer calibration
# ============================================================================


def test_per_layer_calibration_fits_per_layer():
    df = _make_fixture_df(n=4000, seed=50)
    train = df[df["split"] == "train"]
    calib = fit_per_layer_calibration(train, min_nets_per_layer=50)
    # Should have entries for M2..M5
    assert len(calib.s_gnd_per_layer) >= 2
    assert len(calib.s_cpl_per_layer) >= 2


def test_per_layer_calibration_apply_brings_median_to_1():
    df = _make_fixture_df(n=4000, seed=51, gnd_bias=0.35, cpl_bias=1.81)
    train = df[df["split"] == "train"]
    valid = df[df["split"] == "valid"]
    calib = fit_per_layer_calibration(train, min_nets_per_layer=50)
    valid_calibrated = apply_per_layer_calibration(valid, calib)
    v = validate_calibration(valid_calibrated)
    assert 0.85 < v["median_ratio_gnd"] < 1.15, v
    assert 0.85 < v["median_ratio_cpl"] < 1.15, v


def test_per_layer_calibration_uses_default_for_unknown_layer():
    """A net with unrecognized layer histogram falls back to default scaling."""
    df = _make_fixture_df(n=2000, seed=52)
    train = df[df["split"] == "train"]
    calib = fit_per_layer_calibration(train, min_nets_per_layer=50)
    # Net with all-zero layer_hist
    bad_net = train.iloc[:1].copy()
    for L in range(1, 9):
        bad_net[f"layer_hist_M{L}"] = 0.0
    bad_net["layer_hist_M9_plus"] = 0.0
    out = apply_per_layer_calibration(bad_net, calib)
    # Should multiply by s_gnd_default
    assert out["compact_gnd_estimate_fF"].iloc[0] == pytest.approx(
        bad_net["compact_gnd_estimate_fF"].iloc[0] * calib.s_gnd_default
    )


# ============================================================================
# Validation function
# ============================================================================


def test_validate_calibration_returns_dict():
    df = _make_fixture_df(n=200, seed=60)
    out = validate_calibration(df)
    for k in ["median_ratio_gnd", "p5_ratio_gnd", "p95_ratio_gnd",
              "iqr_ratio_gnd", "median_ratio_cpl", "p5_ratio_cpl"]:
        assert k in out
