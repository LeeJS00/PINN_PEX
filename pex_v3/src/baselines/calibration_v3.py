"""
calibration_v3.py — Tier 3 NNLS-style analytic prior calibration.

Maps the poorly-calibrated `compact_gnd_estimate_fF` and
`compact_cpl_estimate_total_fF` to median-ratio-≈-1 prior so the
bounded-residual paradigm (`hybrid_v3` with clamp=log(1.5)) becomes
viable.

Two calibration variants:
    1. Scalar global rescale (1 parameter per channel):
         s_ch = median(golden_ch / analytic_ch) on TRAIN data
         calibrated_analytic = analytic × s_ch
       Easy, robust, removes median bias but keeps relative spread.

    2. Per-layer rescale (one parameter per dominant layer):
         s_ch_L = median(golden_ch / analytic_ch) on TRAIN ∩ {dom_layer == L}
         calibrated_analytic[i] = analytic[i] × s_ch_dom_layer[i]
       Slight improvement for designs with non-uniform layer mix.

Per-cuboid Sakurai-Tamaru rebuild is the "right" fix but ~3-day work
and gated on better z-position parsing. Out of scope for this turn.

A2 audit + Phase 1 Tier 2 first result (`PHASE1_TIER2_FIRST_RESULT.md`)
called this out: median ratio gnd=0.35, cpl=1.81 forces the residual to
fight the bias instead of learning physics. After scalar calibration,
median ratio → 1.0; clamp=log(1.5) covers the IQR.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================================
# Helpers
# ============================================================================


def _safe_ratio(num: pd.Series, den: pd.Series, eps: float = 1e-3) -> pd.Series:
    return num / den.clip(lower=eps)


def _dominant_layer(df: pd.DataFrame, n_layers: int = 9) -> pd.Series:
    """Identify dominant metal layer per net (argmax over layer histogram)."""
    cols = [f"layer_hist_M{i}" for i in range(1, n_layers)] + ["layer_hist_M9_plus"]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.Series([0] * len(df), index=df.index)
    arr = df[cols].to_numpy()
    # Layer index is 1-based; if all zero, return 0
    has_data = arr.sum(axis=1) > 0
    out = np.where(has_data, np.argmax(arr, axis=1) + 1, 0)
    return pd.Series(out, index=df.index)


# ============================================================================
# Scalar calibration (1 param per channel)
# ============================================================================


@dataclass
class ScalarCalibration:
    """Single global scaling factor per channel."""
    s_gnd: float            # multiply compact_gnd_estimate by this
    s_cpl: float
    median_ratio_gnd_before: float
    median_ratio_cpl_before: float
    median_ratio_gnd_after: float = 1.0
    median_ratio_cpl_after: float = 1.0
    n_train_nets: int = 0


def fit_scalar_calibration(
    train_df: pd.DataFrame,
    eps: float = 1e-3,
) -> ScalarCalibration:
    """Fit 1 scalar per channel via median ratio.

    The optimal scalar (under MAPE-loss) is approximately
    `median(golden / analytic)`. Using median, not mean, for outlier
    robustness.

    NO TEST DATA must be passed in — fit on TRAIN split only to avoid
    leakage. Caller's responsibility.
    """
    if "compact_gnd_estimate_fF" not in train_df.columns:
        raise KeyError("train_df missing compact_gnd_estimate_fF")
    if "c_gnd_fF" not in train_df.columns:
        raise KeyError("train_df missing c_gnd_fF")

    ratio_gnd = _safe_ratio(train_df["c_gnd_fF"], train_df["compact_gnd_estimate_fF"], eps)
    ratio_cpl = _safe_ratio(train_df["c_cpl_total_fF"], train_df["compact_cpl_estimate_total_fF"], eps)

    s_gnd = float(ratio_gnd.median())
    s_cpl = float(ratio_cpl.median())

    # Sanity: median(analytic / golden) before
    inv_ratio_gnd = _safe_ratio(train_df["compact_gnd_estimate_fF"], train_df["c_gnd_fF"], eps)
    inv_ratio_cpl = _safe_ratio(train_df["compact_cpl_estimate_total_fF"], train_df["c_cpl_total_fF"], eps)

    return ScalarCalibration(
        s_gnd=s_gnd,
        s_cpl=s_cpl,
        median_ratio_gnd_before=float(inv_ratio_gnd.median()),
        median_ratio_cpl_before=float(inv_ratio_cpl.median()),
        n_train_nets=len(train_df),
    )


def apply_scalar_calibration(
    df: pd.DataFrame,
    calib: ScalarCalibration,
    in_place: bool = False,
) -> pd.DataFrame:
    """Multiply compact estimates by scaling factors. Returns new (or modified) df."""
    if not in_place:
        df = df.copy()
    df["compact_gnd_estimate_fF"] = df["compact_gnd_estimate_fF"] * calib.s_gnd
    df["compact_cpl_estimate_total_fF"] = df["compact_cpl_estimate_total_fF"] * calib.s_cpl
    return df


# ============================================================================
# Per-layer calibration (one param per dominant layer per channel)
# ============================================================================


@dataclass
class PerLayerCalibration:
    """Per-dominant-layer scaling factor per channel."""
    s_gnd_per_layer: dict   # {layer_idx: float}
    s_cpl_per_layer: dict
    s_gnd_default: float    # fallback for nets with unknown layer
    s_cpl_default: float
    n_train_nets: int = 0


def fit_per_layer_calibration(
    train_df: pd.DataFrame,
    min_nets_per_layer: int = 200,
    eps: float = 1e-3,
) -> PerLayerCalibration:
    """Fit per-layer scaling. Layers with too few nets fall back to default."""
    train_df = train_df.copy()
    train_df["dom_layer"] = _dominant_layer(train_df)

    # Default = global median (same as scalar calibration)
    scalar = fit_scalar_calibration(train_df, eps=eps)
    s_gnd_default = scalar.s_gnd
    s_cpl_default = scalar.s_cpl

    s_gnd_per_layer: dict = {}
    s_cpl_per_layer: dict = {}
    for L, sub in train_df.groupby("dom_layer"):
        if L == 0 or len(sub) < min_nets_per_layer:
            continue
        ratio_gnd = _safe_ratio(sub["c_gnd_fF"], sub["compact_gnd_estimate_fF"], eps)
        ratio_cpl = _safe_ratio(sub["c_cpl_total_fF"], sub["compact_cpl_estimate_total_fF"], eps)
        s_gnd_per_layer[int(L)] = float(ratio_gnd.median())
        s_cpl_per_layer[int(L)] = float(ratio_cpl.median())

    return PerLayerCalibration(
        s_gnd_per_layer=s_gnd_per_layer,
        s_cpl_per_layer=s_cpl_per_layer,
        s_gnd_default=s_gnd_default,
        s_cpl_default=s_cpl_default,
        n_train_nets=len(train_df),
    )


def apply_per_layer_calibration(
    df: pd.DataFrame,
    calib: PerLayerCalibration,
    in_place: bool = False,
) -> pd.DataFrame:
    """Multiply compact estimates by per-layer factor (or default)."""
    if not in_place:
        df = df.copy()
    df["dom_layer"] = _dominant_layer(df)
    df["s_gnd"] = df["dom_layer"].map(calib.s_gnd_per_layer).fillna(calib.s_gnd_default)
    df["s_cpl"] = df["dom_layer"].map(calib.s_cpl_per_layer).fillna(calib.s_cpl_default)
    df["compact_gnd_estimate_fF"] = df["compact_gnd_estimate_fF"] * df["s_gnd"]
    df["compact_cpl_estimate_total_fF"] = df["compact_cpl_estimate_total_fF"] * df["s_cpl"]
    df = df.drop(columns=["s_gnd", "s_cpl"])
    return df


# ============================================================================
# Validation
# ============================================================================


def validate_calibration(
    valid_df: pd.DataFrame,
    eps: float = 1e-3,
) -> dict:
    """Verify median ratio after calibration ≈ 1.0 and spread."""
    ratio_gnd = _safe_ratio(valid_df["compact_gnd_estimate_fF"], valid_df["c_gnd_fF"], eps)
    ratio_cpl = _safe_ratio(valid_df["compact_cpl_estimate_total_fF"], valid_df["c_cpl_total_fF"], eps)
    return {
        "median_ratio_gnd": float(ratio_gnd.median()),
        "p5_ratio_gnd": float(ratio_gnd.quantile(0.05)),
        "p95_ratio_gnd": float(ratio_gnd.quantile(0.95)),
        "iqr_ratio_gnd": float(ratio_gnd.quantile(0.75) - ratio_gnd.quantile(0.25)),
        "median_ratio_cpl": float(ratio_cpl.median()),
        "p5_ratio_cpl": float(ratio_cpl.quantile(0.05)),
        "p95_ratio_cpl": float(ratio_cpl.quantile(0.95)),
        "iqr_ratio_cpl": float(ratio_cpl.quantile(0.75) - ratio_cpl.quantile(0.25)),
    }
