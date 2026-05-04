"""
seed_aggregator.py — 5-seed result aggregation per benchmarking-statistician spec.

Inputs: a directory containing N seed subdirectories, each with a single
`metrics_row.csv` produced by the standard runner.

Outputs:
    - per_run.csv       — N rows, one per seed
    - per_method.csv    — one row per method, mean/median/stdev across seeds
    - mwu_pairs.csv     — Mann-Whitney U for every method pair
    - bootstrap_ci.csv  — 95% bootstrap CI on median MAPE per method
    - cohens_d.csv      — effect size between every method pair

This is the ONLY way an "improvement claim" reaches `PHASE_STATUS.md` or
the paper. Per Loss Rule 5 + benchmarking-statistician.md.
"""
from __future__ import annotations
import itertools
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# ============================================================================
# Mann-Whitney U + Cohen's d
# ============================================================================


def mann_whitney_u_two_sided(a: np.ndarray, b: np.ndarray) -> dict:
    """Mann-Whitney U two-sided test. Pure NumPy/SciPy.

    Returns dict with U statistic, p-value, n_a, n_b. Falls back to
    a NaN result if SciPy not available.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    try:
        from scipy.stats import mannwhitneyu
        res = mannwhitneyu(a, b, alternative="two-sided")
        return {
            "U": float(res.statistic),
            "p_value": float(res.pvalue),
            "n_a": int(len(a)),
            "n_b": int(len(b)),
        }
    except ImportError:
        return {"U": float("nan"), "p_value": float("nan"),
                "n_a": int(len(a)), "n_b": int(len(b))}


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size — pooled-stdev normalization."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    var_a = float(np.var(a, ddof=1)) if len(a) > 1 else 0.0
    var_b = float(np.var(b, ddof=1)) if len(b) > 1 else 0.0
    pooled = np.sqrt((var_a + var_b) / 2.0)
    if pooled == 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled)


def cohens_d_label(d: float) -> str:
    """Describe a Cohen's d magnitude qualitatively (Cohen 1988)."""
    if np.isnan(d):
        return "n/a"
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    if abs_d < 0.5:
        return "small"
    if abs_d < 0.8:
        return "medium"
    return "large"


# ============================================================================
# Bootstrap CI
# ============================================================================


def bootstrap_median_ci(
    values: np.ndarray,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
    method: str = "percentile",
) -> tuple:
    """Bootstrap CI on the median.

    `method = "percentile"` returns the simple percentile interval. For BCa
    (bias-corrected accelerated), pass `method = "bca"` and rely on SciPy.
    """
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(values)

    if method == "bca":
        try:
            from scipy.stats import bootstrap
            res = bootstrap((values,), np.median, confidence_level=confidence,
                            n_resamples=n_resamples, method="BCa", random_state=rng)
            return float(np.median(values)), float(res.confidence_interval.low), float(res.confidence_interval.high)
        except Exception:
            pass

    medians = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        medians[i] = np.median(values[idx])
    alpha = (1.0 - confidence) / 2.0
    low = float(np.quantile(medians, alpha))
    high = float(np.quantile(medians, 1.0 - alpha))
    return float(np.median(values)), low, high


# ============================================================================
# Aggregator
# ============================================================================


def aggregate_per_method(per_run: pd.DataFrame, metric_col: str = "cap_mape_median") -> pd.DataFrame:
    """Reduce per-seed runs to per-method summary stats."""
    rows = []
    for method, sub in per_run.groupby("method"):
        vals = sub[metric_col].to_numpy()
        med, lo, hi = bootstrap_median_ci(vals, n_resamples=2000, seed=0)
        rows.append({
            "method": method,
            "n_seeds": int(len(vals)),
            "median": float(np.median(vals)),
            "mean": float(np.mean(vals)),
            "stdev": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
            "ci95_low": lo,
            "ci95_high": hi,
        })
    return pd.DataFrame(rows).sort_values("median").reset_index(drop=True)


def aggregate_mwu_pairs(per_run: pd.DataFrame, metric_col: str = "cap_mape_median") -> pd.DataFrame:
    """Mann-Whitney U + Cohen's d for every method pair."""
    methods = sorted(per_run["method"].unique())
    rows = []
    for a, b in itertools.combinations(methods, 2):
        va = per_run.loc[per_run["method"] == a, metric_col].to_numpy()
        vb = per_run.loc[per_run["method"] == b, metric_col].to_numpy()
        mwu = mann_whitney_u_two_sided(va, vb)
        d = cohens_d(va, vb)
        rows.append({
            "method_a": a,
            "method_b": b,
            "n_a": mwu["n_a"],
            "n_b": mwu["n_b"],
            "U": mwu["U"],
            "p_value": mwu["p_value"],
            "cohens_d": d,
            "cohens_d_label": cohens_d_label(d),
            "support": (
                "supported" if (mwu["p_value"] < 0.05 and abs(d) >= 0.5) else
                "small_effect" if (mwu["p_value"] < 0.05) else
                "ns"
            ),
        })
    return pd.DataFrame(rows)


def collect_per_run_csvs(root_dir: Path, glob: str = "*/metrics_row.csv") -> pd.DataFrame:
    """Walk `root_dir/seed*/metrics_row.csv` and concatenate."""
    rows = []
    for csv in root_dir.rglob(glob):
        df = pd.read_csv(csv)
        df["__source"] = str(csv.relative_to(root_dir))
        rows.append(df)
    if not rows:
        raise RuntimeError(f"No metrics_row.csv files under {root_dir}")
    return pd.concat(rows, ignore_index=True)


def write_aggregation(
    per_run: pd.DataFrame,
    out_dir: Path,
    metric_col: str = "cap_mape_median",
) -> dict:
    """Write per-run, per-method, MWU-pair tables under `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    paths["per_run"] = str(out_dir / "per_run.csv")
    per_run.to_csv(paths["per_run"], index=False)

    per_method = aggregate_per_method(per_run, metric_col=metric_col)
    paths["per_method"] = str(out_dir / "per_method.csv")
    per_method.to_csv(paths["per_method"], index=False)

    mwu = aggregate_mwu_pairs(per_run, metric_col=metric_col)
    paths["mwu_pairs"] = str(out_dir / "mwu_pairs.csv")
    mwu.to_csv(paths["mwu_pairs"], index=False)

    return paths
