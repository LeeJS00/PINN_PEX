"""
Load cached per-design feature parquets and assemble the cross-design split.

Train designs (9): aes, gcd, ibex, ldpc, mc, spi, usbf, vga, wb_conmax
Validation:        nova
Test:              tv80s

For total_cap < 1e-4 fF nets we drop (numerical floor; SPEF artifacts).
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd

from configs import cfg


CAP_FLOOR_FF = 1e-3   # nets with total_cap < this are excluded from MAPE eval


def _load_design(design: str) -> pd.DataFrame:
    p = cfg.CACHE_DIR / "features" / f"{design}.parquet"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_parquet(p)
    df["design_name"] = design
    return df


def _select_feature_cols(df: pd.DataFrame) -> List[str]:
    # n_aggressors_spef and cpl_p95_fF are SPEF-derived (label leakage).
    # total_res_ohm is also from SPEF; leaky if used as input feature.
    drop = {
        "design_name", "net_name", "split",
        "total_cap_fF", "c_gnd_fF", "c_cpl_total_fF",
        "total_res_ohm",
        "n_aggressors_spef", "cpl_p95_fF",
    }
    return [c for c in df.columns if c not in drop]


def load_split(targets: List[str] = None) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """Load and concat the train/val/test design parquets.

    Returns:
        splits: {"train": df, "val": df, "test": df}
        feature_cols: ordered list of input columns
    """
    targets = targets or cfg.TARGETS
    train = pd.concat([_load_design(d) for d in cfg.TRAIN_DESIGNS], ignore_index=True)
    val   = pd.concat([_load_design(d) for d in cfg.VAL_DESIGNS],   ignore_index=True)
    test  = pd.concat([_load_design(d) for d in cfg.TEST_DESIGNS],  ignore_index=True)

    # Drop rows below cap floor (we will not score MAPE on these anyway)
    for name, df in [("train", train), ("val", val), ("test", test)]:
        before = len(df)
        df.drop(df.index[df["total_cap_fF"] < CAP_FLOOR_FF], inplace=True)
        df.reset_index(drop=True, inplace=True)
        if before != len(df):
            print(f"  {name}: dropped {before - len(df)} rows below {CAP_FLOOR_FF}fF floor → {len(df)}")

    feature_cols = _select_feature_cols(train)
    print(f"Train: {len(train):,} nets across {len(cfg.TRAIN_DESIGNS)} designs")
    print(f"Val:   {len(val):,} nets   ({cfg.VAL_DESIGNS[0]})")
    print(f"Test:  {len(test):,} nets   ({cfg.TEST_DESIGNS[0]})")
    print(f"Feature dim: {len(feature_cols)}")
    return {"train": train, "val": val, "test": test}, feature_cols


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def mape_per_net(y_true: np.ndarray, y_pred: np.ndarray, floor_fF: float = CAP_FLOOR_FF) -> np.ndarray:
    """Per-net APE (%). Mask nets with target below floor."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = y_true > floor_fF
    ape = np.full_like(y_true, np.nan)
    ape[mask] = 100.0 * np.abs(y_pred[mask] - y_true[mask]) / np.abs(y_true[mask])
    return ape


def report_mape(y_true, y_pred, label: str = "") -> dict:
    ape = mape_per_net(y_true, y_pred)
    finite = ape[np.isfinite(ape)]
    out = {
        "n": int(len(finite)),
        "mape_mean":   float(np.mean(finite)),
        "mape_median": float(np.median(finite)),
        "mape_p90":    float(np.percentile(finite, 90)),
        "mape_p99":    float(np.percentile(finite, 99)),
    }
    if label:
        print(f"  [{label}]  n={out['n']:,}  mean={out['mape_mean']:.3f}%  median={out['mape_median']:.3f}%  p90={out['mape_p90']:.2f}%  p99={out['mape_p99']:.2f}%")
    return out
