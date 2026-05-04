"""
eval_logger.py — per-net prediction logger for stratified MAPE analysis.

Writes a parquet file with one row per evaluated net containing predictions,
golden values, and stratification covariates. Downstream `stratify_eval.py`
consumes the parquet to produce per-design / per-quartile / per-fanout /
per-dominant-layer MAPE tables.

Schema (Week-1 sprint contract — all subsequent ablation variants MUST emit
these columns):

    net_id                str   # f"{design}::{net_name}"  (uniqueness key)
    design                str
    net_name              str
    fanout                int
    bbox_xy_um2           float
    compact_gnd_estimate_fF  float   # the analytic prior used as Q1..Q4 axis
    gnd_pred              float   # model prediction (calibrated, in fF)
    gnd_gold              float   # StarRC golden (in fF)
    cpl_pred              float
    cpl_gold              float
    total_pred            float   # gnd + cpl
    total_gold            float
    dominant_layer        str     # e.g. "M2", "M3" — argmax(layer_hist_*)

Two ways to populate:

    1. From a model + DataLoader (training-time):
        rows = collect_per_net_predictions(model, loader, device, df_split)
        write_eval_parquet(rows, out_path)

    2. From an existing ensemble predictions CSV joined with v3 features:
        df = load_ensemble_with_features(pred_csv, features_csv, designs)
        write_eval_parquet_from_df(df, out_path)

Both paths produce the same schema. Stratifier doesn't care which one you used.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Iterable

import numpy as np
import pandas as pd

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# Required output columns — stratifier checks this contract on read.
EVAL_LOGGER_SCHEMA = [
    "net_id",
    "design",
    "net_name",
    "fanout",
    "bbox_xy_um2",
    "compact_gnd_estimate_fF",
    "gnd_pred",
    "gnd_gold",
    "cpl_pred",
    "cpl_gold",
    "total_pred",
    "total_gold",
    "dominant_layer",
]

LAYER_HIST_COLS = [
    "layer_hist_M1", "layer_hist_M2", "layer_hist_M3",
    "layer_hist_M4", "layer_hist_M5", "layer_hist_M6",
    "layer_hist_M7", "layer_hist_M8", "layer_hist_M9_plus",
]


def _dominant_layer(df: pd.DataFrame) -> pd.Series:
    """Compute argmax(layer_hist_*) → 'Mk' string per row.

    If no layer_hist columns exist, returns 'unknown'.
    """
    present = [c for c in LAYER_HIST_COLS if c in df.columns]
    if not present:
        return pd.Series(["unknown"] * len(df), index=df.index)
    sub = df[present].fillna(0.0)
    # Tie-break by column order (idxmax returns first max)
    idx = sub.idxmax(axis=1).str.replace("layer_hist_", "")
    return idx


def collect_per_net_predictions(
    model,
    loader,
    device: str,
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """Run model over loader, join with features_df, return schema DataFrame.

    Args:
        model: callable with predict_gnd/predict_cpl OR a function that takes
               a batch dict and returns (pred_gnd, pred_cpl). We assume the
               HybridPexV3Mesh-compatible API.
        loader: PyTorch DataLoader yielding the cuboid_set_dataset batches.
        device: 'cuda:N' or 'cpu'.
        features_df: full features DataFrame for the eval split (must contain
                     fanout, bbox_xy_um2, compact_gnd_estimate_fF, layer_hist_*).
    """
    if not _HAS_TORCH:
        raise RuntimeError("torch required for collect_per_net_predictions")

    model = model.to(device)
    model.eval()

    pred_rows: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            ag = batch["analytic_gnd"].to(device)
            ac = batch["analytic_cpl"].to(device)
            sf = batch["self_features"].to(device)
            pf = batch["pair_features"].to(device)
            cb = batch["cuboids"].to(device)
            mk = batch["padding_mask"].to(device)
            gg = batch["golden_gnd"].cpu().numpy()
            gc = batch["golden_cpl"].cpu().numpy()

            pg = model.predict_gnd(ag, sf, cb, mk).cpu().numpy()
            pc = model.predict_cpl(ac, pf, cb, mk).cpu().numpy()

            for i, (d, n) in enumerate(zip(batch["design_name"], batch["net_name"])):
                pred_rows.append({
                    "design": str(d),
                    "net_name": str(n),
                    "gnd_pred": float(pg[i]),
                    "cpl_pred": float(pc[i]),
                    "gnd_gold": float(gg[i]),
                    "cpl_gold": float(gc[i]),
                })
    pred_df = pd.DataFrame(pred_rows)
    return _join_with_features(pred_df, features_df)


def _join_with_features(pred_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """Inner-join predictions with features; assemble final schema columns."""
    f = features_df.rename(columns={"design_name": "design"})
    keep_cols = ["design", "net_name", "fanout", "bbox_xy_um2",
                 "compact_gnd_estimate_fF"] + [c for c in LAYER_HIST_COLS if c in f.columns]
    f_slim = f[keep_cols].drop_duplicates(subset=["design", "net_name"])

    df = pred_df.merge(f_slim, on=["design", "net_name"], how="inner")
    df["net_id"] = df["design"].astype(str) + "::" + df["net_name"].astype(str)
    df["total_pred"] = df["gnd_pred"] + df["cpl_pred"]
    df["total_gold"] = df["gnd_gold"] + df["cpl_gold"]
    df["dominant_layer"] = _dominant_layer(df)

    # Order columns per schema; drop layer_hist_* (now collapsed into dominant_layer)
    out = df[EVAL_LOGGER_SCHEMA].copy()
    return out


def load_ensemble_with_features(
    ensemble_csv: Path,
    features_csv: Path,
    designs_filter: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Build the schema DataFrame from an existing ensemble predictions CSV.

    The ensemble CSV (e.g. `phase1_mesh_5seed_ensemble/ensemble_predictions_test.csv`)
    has columns:
        design_name, net_name, pred_gnd_fF, pred_cpl_fF, pred_total_fF,
        golden_gnd_fF, golden_cpl_fF, golden_total_fF
    """
    ens = pd.read_csv(ensemble_csv)
    pred = pd.DataFrame({
        "design": ens["design_name"].astype(str),
        "net_name": ens["net_name"].astype(str),
        "gnd_pred": ens["pred_gnd_fF"].astype(float),
        "cpl_pred": ens["pred_cpl_fF"].astype(float),
        "gnd_gold": ens["golden_gnd_fF"].astype(float),
        "cpl_gold": ens["golden_cpl_fF"].astype(float),
    })
    feats = pd.read_csv(features_csv)
    if designs_filter is not None:
        feats = feats[feats["design_name"].isin(list(designs_filter))]
    return _join_with_features(pred, feats)


def write_eval_parquet(df: pd.DataFrame, out_path: Path) -> Path:
    """Validate schema and write parquet."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    missing = [c for c in EVAL_LOGGER_SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(f"eval_logger schema missing columns: {missing}")

    df = df[EVAL_LOGGER_SCHEMA].copy()
    try:
        df.to_parquet(out_path, index=False)
    except Exception as e:
        # Fallback to CSV if pyarrow/fastparquet missing — note path change
        alt = out_path.with_suffix(".csv")
        df.to_csv(alt, index=False)
        print(f"[eval_logger] parquet write failed ({e}); wrote CSV fallback: {alt}")
        return alt
    return out_path


def read_eval_parquet(path: Path) -> pd.DataFrame:
    """Read parquet (or csv fallback) and validate schema."""
    path = Path(path)
    if not path.exists():
        # Try .csv fallback
        alt = path.with_suffix(".csv")
        if alt.exists():
            df = pd.read_csv(alt)
        else:
            raise FileNotFoundError(f"eval_logger: not found {path} or {alt}")
    else:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
    missing = [c for c in EVAL_LOGGER_SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(f"eval_logger schema missing columns in {path}: {missing}")
    return df


# ---------------------------------------------------------------------------
# Stratification primitives — used by both stratify_eval.py and ad-hoc analysis.
# ---------------------------------------------------------------------------


EPS_FF = 1e-3


def add_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add gnd_rel_err / cpl_rel_err / total_rel_err in-place."""
    df = df.copy()
    df["gnd_rel_err"] = (df["gnd_pred"] - df["gnd_gold"]).abs() / df["gnd_gold"].clip(lower=EPS_FF)
    df["cpl_rel_err"] = (df["cpl_pred"] - df["cpl_gold"]).abs() / df["cpl_gold"].clip(lower=EPS_FF)
    df["total_rel_err"] = (df["total_pred"] - df["total_gold"]).abs() / df["total_gold"].clip(lower=EPS_FF)
    return df


def fanout_bucket(f: int) -> str:
    if f <= 1:
        return "1"
    if f <= 5:
        return "2-5"
    if f <= 20:
        return "6-20"
    return ">20"


def stratify_per_design(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for design, sub in df.groupby("design"):
        rows.append({
            "design": design,
            "n_nets": len(sub),
            "gnd_mape_median": float(sub["gnd_rel_err"].median()),
            "gnd_mape_mean": float(sub["gnd_rel_err"].mean()),
            "cpl_mape_median": float(sub["cpl_rel_err"].median()),
            "cpl_mape_mean": float(sub["cpl_rel_err"].mean()),
            "total_mape_median": float(sub["total_rel_err"].median()),
            "total_mape_mean": float(sub["total_rel_err"].mean()),
        })
    return pd.DataFrame(rows).sort_values("design").reset_index(drop=True)


def stratify_per_quartile(df: pd.DataFrame, axis: str = "compact_gnd_estimate_fF") -> pd.DataFrame:
    """Stratify by quartile of `axis` (default: compact_gnd_estimate_fF).

    Q1 = lowest, Q4 = highest. By default this stratifies on the analytic
    prior; passing axis='gnd_rel_err' stratifies by error magnitude (Q4
    surfaces the top-25% giant-CTS / Mode-B failures — paper §7).
    """
    df = df.copy()
    df["__q"] = pd.qcut(df[axis], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    rows = []
    for q, sub in df.groupby("__q", observed=True):
        rows.append({
            "quartile": str(q),
            "axis": axis,
            "axis_min": float(sub[axis].min()),
            "axis_max": float(sub[axis].max()),
            "n_nets": len(sub),
            "gnd_mape_median": float(sub["gnd_rel_err"].median()),
            "gnd_mape_mean": float(sub["gnd_rel_err"].mean()),
            "cpl_mape_median": float(sub["cpl_rel_err"].median()),
            "cpl_mape_mean": float(sub["cpl_rel_err"].mean()),
            "total_mape_median": float(sub["total_rel_err"].median()),
        })
    return pd.DataFrame(rows)


def stratify_per_fanout(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["__f"] = df["fanout"].astype(int).map(fanout_bucket)
    rows = []
    order = {"1": 0, "2-5": 1, "6-20": 2, ">20": 3}
    for f, sub in df.groupby("__f"):
        rows.append({
            "fanout_bucket": f,
            "n_nets": len(sub),
            "gnd_mape_median": float(sub["gnd_rel_err"].median()),
            "gnd_mape_mean": float(sub["gnd_rel_err"].mean()),
            "cpl_mape_median": float(sub["cpl_rel_err"].median()),
            "cpl_mape_mean": float(sub["cpl_rel_err"].mean()),
            "total_mape_median": float(sub["total_rel_err"].median()),
        })
    out = pd.DataFrame(rows)
    out["__order"] = out["fanout_bucket"].map(order).fillna(99)
    return out.sort_values("__order").drop(columns="__order").reset_index(drop=True)


def stratify_per_layer(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for layer, sub in df.groupby("dominant_layer"):
        rows.append({
            "dominant_layer": str(layer),
            "n_nets": len(sub),
            "gnd_mape_median": float(sub["gnd_rel_err"].median()),
            "gnd_mape_mean": float(sub["gnd_rel_err"].mean()),
            "cpl_mape_median": float(sub["cpl_rel_err"].median()),
            "cpl_mape_mean": float(sub["cpl_rel_err"].mean()),
            "total_mape_median": float(sub["total_rel_err"].median()),
        })
    return pd.DataFrame(rows).sort_values("dominant_layer").reset_index(drop=True)


def top_outliers(df: pd.DataFrame, n: int = 50, by: str = "gnd_rel_err") -> pd.DataFrame:
    return df.nlargest(n, by)[["net_id", "design", "fanout", "bbox_xy_um2",
                               "compact_gnd_estimate_fF", "gnd_gold", "gnd_pred",
                               "cpl_gold", "cpl_pred", "dominant_layer", by]]
