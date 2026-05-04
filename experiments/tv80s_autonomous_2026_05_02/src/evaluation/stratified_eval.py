"""
stratified_eval.py — paper-grade stratified error reporting.

Reviewers (per Codex round 2 Q6) expect error breakdown by:
    - cap magnitude quartile  (heteroscedastic effect)
    - layer depth bucket      (M1-M3 / M4-M5 / M6-M9)
    - design                  (per-design generalization)
    - net length / fanout     (topology effect)
    - net class               (clock vs signal vs power)

Aggregate single-number MAPE hides where the model fails. This module
produces stratified tables suitable for direct paper inclusion.

Input: long-format DataFrame with required columns:
    design_name, net_name, pred_fF, golden_fF
Optional columns:
    layer_top, layer_bot, length_um, fanout, net_class, res_ohm

The output is a dict of pandas DataFrames, one per stratification axis.
"""
from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd

from src.evaluation.metrics import cap_mape, cap_mape_summary


# ============================================================================
# axes
# ============================================================================

# Default cap magnitude quartile boundaries (in fF).
# Choose data-driven quartiles at runtime, not these defaults, when possible.
DEFAULT_CAP_QUARTILE_BOUNDS_fF = (0.0, 0.05, 0.5, 5.0, np.inf)

# Layer-depth bucket assignments. Adjust per PDK.
DEFAULT_LAYER_BUCKETS = {
    1: "M1-M3", 2: "M1-M3", 3: "M1-M3",
    4: "M4-M5", 5: "M4-M5",
    6: "M6-M9", 7: "M6-M9", 8: "M6-M9", 9: "M6-M9",
}


def _eps_safe_relative(pred: np.ndarray, gold: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    out = np.full_like(gold, np.nan, dtype=np.float64)
    mask = np.abs(gold) >= eps
    out[mask] = np.abs(pred[mask] - gold[mask]) / np.abs(gold[mask])
    return out


def _summarize_group(group: pd.DataFrame, eps_fF: float) -> pd.Series:
    """Summary stats for a single stratum's per-net error array."""
    pred = group["pred_fF"].to_numpy()
    gold = group["golden_fF"].to_numpy()
    err = _eps_safe_relative(pred, gold, eps=eps_fF)
    valid = err[~np.isnan(err)]
    if len(valid) == 0:
        return pd.Series({
            "n_nets": int(len(group)),
            "n_valid": 0,
            "median_mape": np.nan,
            "mean_mape": np.nan,
            "p95_mape": np.nan,
            "chip_ratio": np.nan,
        })
    chip_ratio = float(pred.sum() / gold.sum()) if abs(gold.sum()) > 0 else float("nan")
    return pd.Series({
        "n_nets": int(len(group)),
        "n_valid": int(len(valid)),
        "median_mape": float(np.median(valid)),
        "mean_mape": float(np.mean(valid)),
        "p95_mape": float(np.percentile(valid, 95)),
        "chip_ratio": chip_ratio,
    })


# ============================================================================
# stratifiers
# ============================================================================

def stratify_by_cap_quartile(
    df: pd.DataFrame,
    cap_quartile_bounds_fF: tuple = DEFAULT_CAP_QUARTILE_BOUNDS_fF,
    eps_fF: float = 1e-3,
) -> pd.DataFrame:
    """MAPE by magnitude quartile (small / medium / large / huge nets).

    Computes per-quartile median, mean, p95, chip ratio. Quartile boundaries
    can be passed explicitly; otherwise defaults from the legacy CTS-heavy
    distribution are used.
    """
    df = df.copy()
    labels = [f"Q{i+1}" for i in range(len(cap_quartile_bounds_fF) - 1)]
    df["cap_quartile"] = pd.cut(
        df["golden_fF"],
        bins=list(cap_quartile_bounds_fF),
        labels=labels,
        right=False,
        include_lowest=True,
    )
    out = df.groupby("cap_quartile", observed=True).apply(
        _summarize_group, eps_fF=eps_fF, include_groups=False,
    ).reset_index()
    return out


def stratify_by_layer_bucket(
    df: pd.DataFrame,
    layer_buckets: dict = DEFAULT_LAYER_BUCKETS,
    eps_fF: float = 1e-3,
) -> pd.DataFrame:
    """MAPE by layer-depth bucket. Requires `layer_top` integer column."""
    if "layer_top" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["layer_bucket"] = df["layer_top"].map(layer_buckets).fillna("other")
    return df.groupby("layer_bucket", observed=True).apply(
        _summarize_group, eps_fF=eps_fF, include_groups=False,
    ).reset_index()


def stratify_by_design(df: pd.DataFrame, eps_fF: float = 1e-3) -> pd.DataFrame:
    """Per-design MAPE table — generalization breadth."""
    return df.groupby("design_name", observed=True).apply(
        _summarize_group, eps_fF=eps_fF, include_groups=False,
    ).reset_index()


def stratify_by_length_bucket(
    df: pd.DataFrame,
    bins_um: tuple = (0, 5, 50, 500, np.inf),
    eps_fF: float = 1e-3,
) -> pd.DataFrame:
    """MAPE by net total wire length bucket. Requires `length_um` column."""
    if "length_um" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    labels = [f"L{i+1}" for i in range(len(bins_um) - 1)]
    df["length_bucket"] = pd.cut(
        df["length_um"], bins=list(bins_um), labels=labels,
        right=False, include_lowest=True,
    )
    return df.groupby("length_bucket", observed=True).apply(
        _summarize_group, eps_fF=eps_fF, include_groups=False,
    ).reset_index()


def stratify_by_net_class(df: pd.DataFrame, eps_fF: float = 1e-3) -> pd.DataFrame:
    """MAPE by net class (clock / signal / power). Requires `net_class` column."""
    if "net_class" not in df.columns:
        return pd.DataFrame()
    return df.groupby("net_class", observed=True).apply(
        _summarize_group, eps_fF=eps_fF, include_groups=False,
    ).reset_index()


# ============================================================================
# all-axes report
# ============================================================================


def full_stratified_report(
    df: pd.DataFrame,
    eps_fF: float = 1e-3,
    cap_quartile_bounds_fF: Optional[tuple] = None,
    layer_buckets: Optional[dict] = None,
) -> dict:
    """Run every applicable stratification axis. Returns dict of DataFrames.

    Schema check: `df` must have columns {design_name, net_name, pred_fF, golden_fF}.
    Optional: layer_top, length_um, fanout, net_class, res_ohm.
    """
    required = {"design_name", "net_name", "pred_fF", "golden_fF"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"stratified report missing required columns: {missing}")

    out = {
        "by_design": stratify_by_design(df, eps_fF=eps_fF),
        "by_cap_quartile": stratify_by_cap_quartile(
            df,
            cap_quartile_bounds_fF=(
                cap_quartile_bounds_fF or DEFAULT_CAP_QUARTILE_BOUNDS_fF
            ),
            eps_fF=eps_fF,
        ),
        "overall": pd.DataFrame([_summarize_group(df, eps_fF=eps_fF)]),
    }

    if "layer_top" in df.columns:
        out["by_layer"] = stratify_by_layer_bucket(
            df,
            layer_buckets=(layer_buckets or DEFAULT_LAYER_BUCKETS),
            eps_fF=eps_fF,
        )
    if "length_um" in df.columns:
        out["by_length"] = stratify_by_length_bucket(df, eps_fF=eps_fF)
    if "net_class" in df.columns:
        out["by_class"] = stratify_by_net_class(df, eps_fF=eps_fF)

    return out


def write_stratified_report(
    report: dict,
    out_dir,
    fmt: str = "csv",
) -> dict:
    """Write each DataFrame in the report to a file in `out_dir`. Returns paths."""
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for k, df in report.items():
        if df is None or len(df) == 0:
            continue
        path = out_dir / f"stratified_{k}.{fmt}"
        if fmt == "csv":
            df.to_csv(path, index=False)
        elif fmt == "parquet":
            df.to_parquet(path, index=False)
        written[k] = str(path)
    return written
