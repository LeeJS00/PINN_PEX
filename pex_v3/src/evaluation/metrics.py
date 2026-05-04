"""
metrics.py — primary + derived metrics for net-level capacitance prediction.

The Strategy v3 paper is required to report (per `classical-baseline-owner`
mandate, Codex round 1):

    1. Cap MAPE       — direct |pred - golden| / |golden|, per-net
    2. Delay error    — RC-product equivalent (downstream timing impact)
    3. Power error    — switching power equivalent (downstream)
    4. RC percentile  — chip-level distribution agreement (Wasserstein-style)

This module gives clean implementations of all four. Callers pass either
NumPy arrays of (predicted, golden) per-net values, or a richer dict
including per-net resistance and switching activity.

Numeric conventions:
    - Capacitance values in fF
    - Resistance values in Ω
    - Voltage default to 1.0 V (delay/power are unitless ratios anyway)
    - Switching activity (alpha) default 0.5 (uniform)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ============================================================================
# Primary: cap MAPE
# ============================================================================

def cap_mape(
    pred_fF: np.ndarray,
    golden_fF: np.ndarray,
    eps_fF: float = 1e-3,
) -> np.ndarray:
    """Per-net relative absolute error, dropping near-zero golden targets.

    Returns a 1-D array of shape (n_nets,). Targets with |golden| < eps_fF
    yield np.nan in the output (caller decides whether to ignore or
    replace with a zero-target supervision metric).
    """
    pred_fF = np.asarray(pred_fF, dtype=np.float64)
    golden_fF = np.asarray(golden_fF, dtype=np.float64)
    out = np.full_like(golden_fF, np.nan, dtype=np.float64)
    mask = np.abs(golden_fF) >= eps_fF
    out[mask] = np.abs(pred_fF[mask] - golden_fF[mask]) / np.abs(golden_fF[mask])
    return out


def cap_mape_summary(
    pred_fF: np.ndarray,
    golden_fF: np.ndarray,
    eps_fF: float = 1e-3,
) -> dict:
    """Reduce per-net MAPE array to summary stats (median, mean, p95, etc.)."""
    arr = cap_mape(pred_fF, golden_fF, eps_fF=eps_fF)
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return {
            "n_valid": 0,
            "n_zero_target": int(np.isnan(arr).sum()),
        }
    return {
        "n_valid": int(len(valid)),
        "n_zero_target": int(np.isnan(arr).sum()),
        "median_mape": float(np.median(valid)),
        "mean_mape": float(np.mean(valid)),
        "stdev_mape": float(np.std(valid)),
        "p25_mape": float(np.percentile(valid, 25)),
        "p75_mape": float(np.percentile(valid, 75)),
        "p95_mape": float(np.percentile(valid, 95)),
        "max_mape": float(np.max(valid)),
        "iqr_mape": float(np.percentile(valid, 75) - np.percentile(valid, 25)),
    }


# ============================================================================
# Derived: delay error
# ============================================================================

def delay_error(
    pred_fF: np.ndarray,
    golden_fF: np.ndarray,
    res_ohm: np.ndarray,
    voltage: float = 1.0,
) -> dict:
    """Per-net delay error = relative |pred_RC - golden_RC|.

    Delay (Elmore) ≈ R · C; first-order timing surrogate.
    Returns dict with summary stats.
    """
    pred_fF = np.asarray(pred_fF, dtype=np.float64)
    golden_fF = np.asarray(golden_fF, dtype=np.float64)
    res_ohm = np.asarray(res_ohm, dtype=np.float64)

    pred_rc = pred_fF * res_ohm  # fF·Ω = ps×k (just a scale; ratio invariant)
    gold_rc = golden_fF * res_ohm
    eps = 1e-12
    err = np.abs(pred_rc - gold_rc) / (np.abs(gold_rc) + eps)
    return {
        "median_delay_err": float(np.median(err)),
        "mean_delay_err": float(np.mean(err)),
        "p95_delay_err": float(np.percentile(err, 95)),
        "max_delay_err": float(np.max(err)),
    }


# ============================================================================
# Derived: switching power error
# ============================================================================

def power_error(
    pred_fF: np.ndarray,
    golden_fF: np.ndarray,
    activity_alpha: Optional[np.ndarray] = None,
    voltage: float = 1.0,
    freq_ghz: float = 1.0,
) -> dict:
    """Per-net switching-power error.

    P = α · C · V² · f
    Relative error in P collapses to relative error in C when V, f, α match,
    so this metric ends up close to cap_mape but is reported separately
    because the reviewer expects it (ResCap, ParaFormer convention).

    Pass `activity_alpha` if known; otherwise uniform 0.5.
    """
    pred_fF = np.asarray(pred_fF, dtype=np.float64)
    golden_fF = np.asarray(golden_fF, dtype=np.float64)
    if activity_alpha is None:
        activity_alpha = np.full_like(golden_fF, 0.5)
    activity_alpha = np.asarray(activity_alpha, dtype=np.float64)

    pred_p = activity_alpha * pred_fF * voltage * voltage * freq_ghz
    gold_p = activity_alpha * golden_fF * voltage * voltage * freq_ghz
    eps = 1e-12
    err = np.abs(pred_p - gold_p) / (np.abs(gold_p) + eps)
    return {
        "median_power_err": float(np.median(err)),
        "mean_power_err": float(np.mean(err)),
        "p95_power_err": float(np.percentile(err, 95)),
    }


# ============================================================================
# Derived: chip-level RC percentile distribution match
# ============================================================================

def rc_percentile_metrics(
    pred_fF: np.ndarray,
    golden_fF: np.ndarray,
    res_ohm: Optional[np.ndarray] = None,
    percentiles: tuple = (10, 25, 50, 75, 90, 95, 99),
) -> dict:
    """Distribution-level chip ratio at multiple percentiles.

    For each percentile p, compute (pred RC at p / golden RC at p). Should
    be close to 1.0 across all percentiles if the model is calibrated. A
    skewed ratio at high percentiles indicates large-net under-prediction
    (the heteroscedastic problem); skewed at low percentiles indicates
    small-net over-prediction.
    """
    pred_fF = np.asarray(pred_fF, dtype=np.float64)
    golden_fF = np.asarray(golden_fF, dtype=np.float64)
    if res_ohm is None:
        # Use cap directly (RES factors out evenly)
        pred_rc = pred_fF
        gold_rc = golden_fF
    else:
        res_ohm = np.asarray(res_ohm, dtype=np.float64)
        pred_rc = pred_fF * res_ohm
        gold_rc = golden_fF * res_ohm

    out = {}
    for p in percentiles:
        gp = float(np.percentile(gold_rc, p))
        pp = float(np.percentile(pred_rc, p))
        ratio = pp / gp if gp != 0 else float("nan")
        out[f"chip_ratio_p{p:02d}"] = ratio
    # Worst ratio across percentiles
    ratios = [out[f"chip_ratio_p{p:02d}"] for p in percentiles
              if not np.isnan(out[f"chip_ratio_p{p:02d}"])]
    out["max_abs_log_ratio"] = float(max(abs(np.log(r)) for r in ratios)) if ratios else float("nan")
    return out


# ============================================================================
# Combined paper-grade row
# ============================================================================

@dataclass(frozen=True)
class MetricsRow:
    """Single row in the paper-grade comparison table.

    All four metrics in one place, suitable for direct CSV write.
    """
    method: str
    seed: int
    cap_mape_median: float
    cap_mape_mean: float
    cap_mape_p95: float
    delay_err_median: float
    delay_err_p95: float
    power_err_median: float
    rc_chip_ratio_p50: float
    rc_chip_ratio_p95: float
    n_valid_nets: int


def build_metrics_row(
    method: str,
    seed: int,
    pred_fF: np.ndarray,
    golden_fF: np.ndarray,
    res_ohm: Optional[np.ndarray] = None,
) -> MetricsRow:
    """One-shot computation of all four metric families. Returns a flat dataclass."""
    cap = cap_mape_summary(pred_fF, golden_fF)
    if res_ohm is None:
        delay = {"median_delay_err": float("nan"), "p95_delay_err": float("nan")}
        rc = rc_percentile_metrics(pred_fF, golden_fF, res_ohm=None)
    else:
        delay = delay_error(pred_fF, golden_fF, res_ohm)
        rc = rc_percentile_metrics(pred_fF, golden_fF, res_ohm=res_ohm)
    power = power_error(pred_fF, golden_fF)

    return MetricsRow(
        method=method,
        seed=seed,
        cap_mape_median=cap.get("median_mape", float("nan")),
        cap_mape_mean=cap.get("mean_mape", float("nan")),
        cap_mape_p95=cap.get("p95_mape", float("nan")),
        delay_err_median=delay["median_delay_err"],
        delay_err_p95=delay["p95_delay_err"],
        power_err_median=power["median_power_err"],
        rc_chip_ratio_p50=rc["chip_ratio_p50"],
        rc_chip_ratio_p95=rc["chip_ratio_p95"],
        n_valid_nets=cap.get("n_valid", 0),
    )
