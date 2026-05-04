"""
xgboost_baseline.py — Phase 0.5 B1.

XGBoost on hand-engineered NetFeatureVector. Predicts:
    - C_gnd (per-net self-capacitance to ground)
    - C_cpl_total (per-net total coupling, summed across all aggressors)

Per-pair C_cpl[t, a] is more demanding (requires pair-level features); we
defer that variant — `C_cpl_total` is what reviewers care about for paper-grade
comparison.

This module is consumed by `pex_v3/scripts/05_5seed_runner.py` via the
`run_one_seed` entrypoint contract.

Required inputs at runtime (built by feature_dataset.py, separate concern):
    train_features.parquet  — columns: net_id + NetFeatureVector fields + targets
        Required target columns: c_gnd_fF, c_cpl_total_fF
    valid_features.parquet  — same schema
    test_features.parquet   — same schema (used at eval time)

Synthetic-mode fallback: if the real dataset paths don't exist, fall back to
a synthetic regression task driven by the feature vector itself, so smoke
tests can run end-to-end.

Output schema per seed (under output_dir):
    eval_predictions_valid.csv     — per-net pred + golden, valid split
    eval_predictions_test.csv      — per-net pred + golden, test (OOD) split
    eval_predictions.csv           — alias of eval_predictions_valid.csv
                                     (kept for backward compatibility with
                                      `13_cross_design_acc_runtime.py` loader)
    per_channel_summary.json       — gnd/cpl/total median + mean + p95 on
                                     valid + test, per-design breakdown
    metrics_row.csv                — single-row MetricsRow built on VALID
                                     (eval_on respected for legacy callers;
                                      defaults to valid)
    model_gnd.json, model_cpl.json — saved XGBoost models
"""
from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .features import NetFeatureVector
from ..evaluation.metrics import build_metrics_row, MetricsRow
from ..utils.seeds import set_all_seeds


# ============================================================================
# Helpers
# ============================================================================


def _feature_columns() -> list[str]:
    """Column names XGBoost will consume, in the locked NetFeatureVector order."""
    return NetFeatureVector.field_names()


def _train_xgb_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: Optional[np.ndarray],
    y_valid: Optional[np.ndarray],
    seed: int,
    n_estimators: int = 500,
    max_depth: int = 8,
    learning_rate: float = 0.05,
    early_stopping_rounds: int = 50,
):
    """Fit an XGBoost regressor on (X_train, y_train) targeting log1p(target).

    Uses log1p target transform because cap distribution is power-law.
    """
    import xgboost as xgb

    # FIX (Phase B audit, 2026-05-02): without `subsample` and `colsample_*`,
    # XGBoost with `tree_method="hist"` is fully deterministic given the data,
    # so 5-seed runs return IDENTICAL values (stdev=0). Adding 0.8 row-subsample
    # and 0.8 column-subsample restores per-seed variability so the 5-seed
    # protocol estimates true variance.
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=seed,
        tree_method="hist",
        objective="reg:squarederror",
        subsample=0.8,
        colsample_bytree=0.8,
        verbosity=0,
        early_stopping_rounds=(early_stopping_rounds if X_valid is not None else None),
    )
    fit_kwargs = {}
    if X_valid is not None and y_valid is not None:
        fit_kwargs["eval_set"] = [(X_valid, np.log1p(y_valid))]
        fit_kwargs["verbose"] = False
    model.fit(X_train, np.log1p(y_train), **fit_kwargs)
    return model


def _xgb_predict(model, X: np.ndarray) -> np.ndarray:
    """Predict in original cap units (undo log1p)."""
    log_pred = model.predict(X)
    return np.expm1(log_pred)


def _load_real_features_df(features_root: Path) -> pd.DataFrame:
    """Load all per-design feature CSVs (or parquet) from a v3 features dir.

    Looks for `<features_root>/all_designs.csv` first (concatenated by the
    feature_dataset orchestrator), then falls back to globbing per-design
    files.
    """
    all_csv = features_root / "all_designs.csv"
    all_parquet = features_root / "all_designs.parquet"
    if all_csv.exists():
        return pd.read_csv(all_csv)
    if all_parquet.exists():
        return pd.read_parquet(all_parquet)

    # Fallback: glob per-design files
    csvs = sorted(features_root.glob("intel22_*.csv"))
    parquets = sorted(features_root.glob("intel22_*.parquet"))
    if csvs:
        return pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
    if parquets:
        return pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)

    raise FileNotFoundError(
        f"No feature files found under {features_root}. "
        f"Run pex_v3/scripts/04_build_feature_dataset.py first."
    )


def _split_by_manifest_column(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Use the `split` column (from v3 manifest H1 hash) to partition rows.

    This preserves the H1 net-level split discipline — every (design, net)
    is in exactly one of {train, valid, test}.
    """
    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_valid = df[df["split"] == "valid"].reset_index(drop=True)
    df_test = df[df["split"] == "test"].reset_index(drop=True)
    return df_train, df_valid, df_test


def _make_synthetic_features_df(seed: int, n_rows: int = 500) -> pd.DataFrame:
    """Synthetic fallback for smoke testing only. Used when real features
    are unavailable (no v3 features dir). Real runs must use
    `_load_real_features_df`.
    """
    rng = np.random.default_rng(seed)
    cols = _feature_columns()
    df = pd.DataFrame({c: rng.normal(0, 1, n_rows) for c in cols})
    base_gnd = (
        2.0
        + 0.5 * df["total_metal_area_um2"]
        + 0.3 * df["compact_gnd_estimate_fF"]
        + 0.05 * rng.normal(0, 1, n_rows)
    )
    base_cpl = (
        1.0
        + 0.4 * df["broadside_overlap_total_um2"]
        + 0.25 * df["compact_cpl_estimate_total_fF"]
        + 0.05 * rng.normal(0, 1, n_rows)
    )
    df["c_gnd_fF"] = np.exp(base_gnd) / np.exp(base_gnd).mean()
    df["c_cpl_total_fF"] = np.exp(base_cpl) / np.exp(base_cpl).mean()
    df["design_name"] = "synthetic"
    df["net_name"] = [f"synth_{i:05d}" for i in range(n_rows)]
    df["split"] = ["train"] * (n_rows - 100) + ["valid"] * 50 + ["test"] * 50
    df["total_res_ohm"] = 1.0 + 0.5 * rng.uniform(0, 1, n_rows)
    return df


# ============================================================================
# Public entrypoint
# ============================================================================


def run_one_seed(
    seed: int,
    train_manifest_path: Path,
    golden_spef_dir: Path,
    output_dir: Path,
    config_snapshot: dict,
) -> MetricsRow:
    """Train + evaluate XGBoost baseline for one seed.

    Contract: matches `scripts/05_5seed_runner.py:run_one_seed` signature.

    Currently uses synthetic features as a smoke test. Real-data path is gated
    on `feature_dataset.py` (DEF→features pipeline) which is the next
    integration step.

    Output written to `output_dir`:
        - features_used.csv (metadata only, not the raw values)
        - model_gnd.json, model_cpl.json (XGBoost saved models)
        - eval_predictions.csv (per-net pred + golden columns)
        - metrics_row.csv (single row with summary stats)

    Returns:
        MetricsRow dataclass for the orchestrator to aggregate.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    set_all_seeds(seed, deterministic=True)

    # ---- Load feature dataset -----------------------------------------
    # Real-data path (Phase B): read pre-built features from v3 features dir.
    # Falls back to synthetic only if no features available (smoke testing).
    features_root = Path(train_manifest_path).parent / "features"
    use_synthetic = config_snapshot.get("use_synthetic_features", False)
    if use_synthetic or not features_root.exists():
        print(f"  >>> seed {seed}: using SYNTHETIC features (smoke mode)")
        df = _make_synthetic_features_df(seed=seed, n_rows=2000)
    else:
        print(f"  >>> seed {seed}: loading real features from {features_root}")
        df = _load_real_features_df(features_root)
        print(f"  loaded {len(df):,} rows across "
              f"{df['design_name'].nunique()} designs")
        # Filter to nets with non-zero golden cap (drop rows where SPEF total = 0)
        before = len(df)
        df = df[(df["c_gnd_fF"] + df["c_cpl_total_fF"]) > 1e-4].reset_index(drop=True)
        if len(df) < before:
            print(f"  filtered {before - len(df)} zero-cap rows")

    df_train, df_valid, df_test = _split_by_manifest_column(df)
    print(f"  splits: train={len(df_train):,}  valid={len(df_valid):,}  "
          f"test={len(df_test):,}")
    if len(df_train) == 0:
        raise RuntimeError(f"Empty train ({len(df_train)}) split.")

    # P2 (2026-05-03): always evaluate on BOTH valid AND test, write per-split
    # prediction CSVs. The MetricsRow aggregated by `05_5seed_runner.py` is
    # built from `xgb_eval_on` (default "valid"), preserving the existing
    # 4.66% ± 0.026pp valid headline. The test (OOD) numbers are downstream-
    # computable from `eval_predictions_test.csv`.
    eval_on = config_snapshot.get("xgb_eval_on", "valid")
    if eval_on not in ("valid", "test"):
        raise ValueError(f"xgb_eval_on must be 'valid' or 'test', got {eval_on}")

    feat_cols = _feature_columns()
    X_train = df_train[feat_cols].fillna(0.0).to_numpy(dtype=np.float64)
    X_valid = df_valid[feat_cols].fillna(0.0).to_numpy(dtype=np.float64) if len(df_valid) else None
    X_test  = df_test[feat_cols].fillna(0.0).to_numpy(dtype=np.float64)  if len(df_test)  else None

    y_train_gnd = df_train["c_gnd_fF"].to_numpy(dtype=np.float64)
    y_valid_gnd = df_valid["c_gnd_fF"].to_numpy(dtype=np.float64) if len(df_valid) else None
    y_test_gnd  = df_test["c_gnd_fF"].to_numpy(dtype=np.float64)  if len(df_test)  else None

    y_train_cpl = df_train["c_cpl_total_fF"].to_numpy(dtype=np.float64)
    y_valid_cpl = df_valid["c_cpl_total_fF"].to_numpy(dtype=np.float64) if len(df_valid) else None
    y_test_cpl  = df_test["c_cpl_total_fF"].to_numpy(dtype=np.float64)  if len(df_test)  else None

    # ---- Train two regressors -----------------------------------------
    model_gnd = _train_xgb_regressor(X_train, y_train_gnd, X_valid, y_valid_gnd, seed=seed)
    model_cpl = _train_xgb_regressor(X_train, y_train_cpl, X_valid, y_valid_cpl, seed=seed)

    model_gnd.save_model(str(output_dir / "model_gnd.json"))
    model_cpl.save_model(str(output_dir / "model_cpl.json"))

    # ---- Evaluate on BOTH splits + write per-split CSVs ---------------
    def _eval_and_write(
        df_split: pd.DataFrame,
        X_split: np.ndarray,
        y_gnd: np.ndarray,
        y_cpl: np.ndarray,
        out_name: str,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        pred_gnd_s = _xgb_predict(model_gnd, X_split)
        pred_cpl_s = _xgb_predict(model_cpl, X_split)
        pred_tot_s = pred_gnd_s + pred_cpl_s
        gold_tot_s = y_gnd + y_cpl
        res_col = "total_res_ohm" if "total_res_ohm" in df_split.columns else "res_ohm"
        keep = ["design_name", "net_name"]
        if res_col in df_split.columns:
            keep.append(res_col)
        out_df = df_split[keep].copy()
        out_df["pred_gnd_fF"] = pred_gnd_s
        out_df["pred_cpl_fF"] = pred_cpl_s
        out_df["pred_total_fF"] = pred_tot_s
        out_df["golden_gnd_fF"] = y_gnd
        out_df["golden_cpl_fF"] = y_cpl
        out_df["golden_total_fF"] = gold_tot_s
        out_df.to_csv(output_dir / out_name, index=False)
        return out_df, pred_tot_s, gold_tot_s

    eval_dfs: dict[str, pd.DataFrame] = {}
    pred_total_for_metrics: Optional[np.ndarray] = None
    gold_total_for_metrics: Optional[np.ndarray] = None
    res_for_metrics: Optional[np.ndarray] = None

    if X_valid is not None and len(df_valid) > 0:
        v_df, v_pred, v_gold = _eval_and_write(
            df_valid, X_valid, y_valid_gnd, y_valid_cpl, "eval_predictions_valid.csv"
        )
        eval_dfs["valid"] = v_df
        if eval_on == "valid":
            pred_total_for_metrics = v_pred
            gold_total_for_metrics = v_gold
    if X_test is not None and len(df_test) > 0:
        t_df, t_pred, t_gold = _eval_and_write(
            df_test, X_test, y_test_gnd, y_test_cpl, "eval_predictions_test.csv"
        )
        eval_dfs["test"] = t_df
        if eval_on == "test":
            pred_total_for_metrics = t_pred
            gold_total_for_metrics = t_gold

    if pred_total_for_metrics is None:
        raise RuntimeError(
            f"Empty {eval_on} split — cannot build MetricsRow on requested eval_on='{eval_on}'."
        )

    # Backward-compat alias (script 13's loader reads `eval_predictions.csv`)
    if eval_on in eval_dfs:
        eval_dfs[eval_on].to_csv(output_dir / "eval_predictions.csv", index=False)

    # res for the MetricsRow corresponds to whichever split eval_on points at
    metrics_split_df = eval_dfs[eval_on]
    res_col = "total_res_ohm" if "total_res_ohm" in metrics_split_df.columns else "res_ohm"
    if res_col in metrics_split_df.columns:
        res_for_metrics = metrics_split_df[res_col].fillna(1.0).to_numpy(dtype=np.float64)
    else:
        res_for_metrics = np.ones(len(metrics_split_df), dtype=np.float64)

    # ---- Per-channel + per-design summary (parity with Option F + B4) ----
    summary: dict = {
        "eval_on_for_metrics_row": eval_on,
        "n_train": int(len(df_train)),
    }
    for split_name, split_df in eval_dfs.items():
        summary[split_name] = _per_channel_mape_dict(split_df)
        summary[f"{split_name}_per_design"] = _per_design_mape_dict(split_df)
    with open(output_dir / "per_channel_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Build MetricsRow ---------------------------------------------
    row = build_metrics_row(
        method="B1_xgboost",
        seed=seed,
        pred_fF=pred_total_for_metrics,
        golden_fF=gold_total_for_metrics,
        res_ohm=res_for_metrics,
    )
    return row


# ============================================================================
# Per-channel + per-design helpers (mirror script 14's pattern)
# ============================================================================


_EPS_FF = 1e-3


def _mape_arr(pred: np.ndarray, gold: np.ndarray) -> np.ndarray:
    return np.abs(pred - gold) / np.clip(np.abs(gold), _EPS_FF, None)


def _per_channel_mape_dict(pred_df: pd.DataFrame) -> dict:
    out: dict = {}
    for ch in ["gnd", "cpl", "total"]:
        rel = _mape_arr(
            pred_df[f"pred_{ch}_fF"].to_numpy(),
            pred_df[f"golden_{ch}_fF"].to_numpy(),
        )
        out[ch] = {
            "n_nets": int(len(rel)),
            "median": float(np.median(rel)),
            "mean": float(np.mean(rel)),
            "p95": float(np.percentile(rel, 95)),
        }
    return out


def _per_design_mape_dict(pred_df: pd.DataFrame) -> dict:
    return {
        str(design): _per_channel_mape_dict(sub)
        for design, sub in pred_df.groupby("design_name")
    }
