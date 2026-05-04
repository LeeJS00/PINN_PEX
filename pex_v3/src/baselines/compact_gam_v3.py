"""
compact_gam_v3.py — Phase 0.5 B4 baseline.

Sakurai-Tamaru analytic compact-model + per-channel residual via GBDT.
Closest paradigm match to ResCap (ASPDAC 2025): physics-guided linear
base + ML residual.

Three variants in the "compact + ML" family:

    Variant 1 — Linear-only baseline:
        pred = a * compact_estimate + b
        (purely affine adjustment of analytic prior)

    Variant 2 — GBDT-residual:
        pred = compact_estimate + GBDT(features) — additive residual
        Uses XGBoost as residual learner

    Variant 3 — log-space GBDT-residual (multiplicative):
        log_pred = log(compact_estimate) + GBDT(features)
        Robust to power-law cap distribution

A2 audit estimated B4 effort at 3 days; in practice the Sakurai features
are already in NetFeatureVector, so most work is wrapping the ML layer.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# A2 audit: same hand features as B1 — paper-grade comparison demands shared inputs


@dataclass
class CompactGAMResult:
    """Per-channel result of B4 compact-plus-GBDT residual."""
    method: str
    seed: int
    pred_gnd: np.ndarray            # (n_eval,)
    pred_cpl: np.ndarray
    pred_total: np.ndarray
    golden_gnd: np.ndarray
    golden_cpl: np.ndarray
    golden_total: np.ndarray
    train_seconds: float
    inference_seconds: float
    n_train: int
    n_eval: int


def _load_xgb_or_die():
    try:
        import xgboost as xgb
        return xgb
    except ImportError as e:
        raise ImportError(
            "xgboost not installed. "
            "Run: /tool/etc/python/install/3.11.9/bin/python3 -m pip install --user xgboost"
        ) from e


def _build_feature_matrix(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Project DataFrame to (n, dim) numpy array, log1p-compressed positives."""
    out = np.zeros((len(df), len(feature_cols)), dtype=np.float32)
    for i, c in enumerate(feature_cols):
        if c in df.columns:
            v = df[c].fillna(0.0).to_numpy(dtype=np.float32)
            v = np.log1p(np.clip(v, 0, None))
            out[:, i] = v
    return out


# ============================================================================
# Variant 1 — Linear-only (sanity floor)
# ============================================================================


def linear_compact_baseline(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
) -> CompactGAMResult:
    """Per-channel affine fit on the Sakurai compact estimate.

    pred_gnd = a_gnd * compact_gnd + b_gnd
    Fitted via least squares on TRAIN only (no leakage).
    Sanity floor — should be close to compact estimate's median ratio.
    """
    import time
    t0 = time.time()

    # Per-channel 1-D linear regression
    a_gnd, b_gnd = np.polyfit(
        train_df["compact_gnd_estimate_fF"].to_numpy(dtype=np.float64),
        train_df["c_gnd_fF"].to_numpy(dtype=np.float64),
        1,
    )
    a_cpl, b_cpl = np.polyfit(
        train_df["compact_cpl_estimate_total_fF"].to_numpy(dtype=np.float64),
        train_df["c_cpl_total_fF"].to_numpy(dtype=np.float64),
        1,
    )
    train_t = time.time() - t0

    t1 = time.time()
    pred_gnd = a_gnd * eval_df["compact_gnd_estimate_fF"].to_numpy() + b_gnd
    pred_cpl = a_cpl * eval_df["compact_cpl_estimate_total_fF"].to_numpy() + b_cpl
    inf_t = time.time() - t1
    pred_total = pred_gnd + pred_cpl

    return CompactGAMResult(
        method="B4_linear_compact",
        seed=0,
        pred_gnd=pred_gnd,
        pred_cpl=pred_cpl,
        pred_total=pred_total,
        golden_gnd=eval_df["c_gnd_fF"].to_numpy(),
        golden_cpl=eval_df["c_cpl_total_fF"].to_numpy(),
        golden_total=(eval_df["c_gnd_fF"] + eval_df["c_cpl_total_fF"]).to_numpy(),
        train_seconds=train_t,
        inference_seconds=inf_t,
        n_train=len(train_df),
        n_eval=len(eval_df),
    )


# ============================================================================
# Variant 2 — Compact + GBDT residual (additive in fF)
# ============================================================================


def compact_plus_gbdt_residual(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_cols: list[str],
    seed: int = 0,
    n_estimators: int = 300,
    max_depth: int = 6,
    lr: float = 0.05,
) -> CompactGAMResult:
    """Compact estimate + GBDT residual on hand features.

    Per channel:
        target_residual = c_ch_fF - compact_estimate
        GBDT learns:    residual ≈ f(features)
        pred =          compact_estimate + GBDT.predict(features)

    Per-channel separation matches B1 + β-strategy.
    """
    import time
    xgb = _load_xgb_or_die()

    X_train = _build_feature_matrix(train_df, feature_cols)
    X_eval = _build_feature_matrix(eval_df, feature_cols)

    # Residual targets
    y_train_gnd_resid = (
        train_df["c_gnd_fF"] - train_df["compact_gnd_estimate_fF"]
    ).to_numpy(dtype=np.float64)
    y_train_cpl_resid = (
        train_df["c_cpl_total_fF"] - train_df["compact_cpl_estimate_total_fF"]
    ).to_numpy(dtype=np.float64)

    t0 = time.time()
    model_gnd = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=lr,
        random_state=seed,
        tree_method="hist",
        objective="reg:squarederror",
        subsample=0.8,
        colsample_bytree=0.8,
        verbosity=0,
    )
    model_cpl = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=lr,
        random_state=seed + 1,
        tree_method="hist",
        objective="reg:squarederror",
        subsample=0.8,
        colsample_bytree=0.8,
        verbosity=0,
    )
    model_gnd.fit(X_train, y_train_gnd_resid)
    model_cpl.fit(X_train, y_train_cpl_resid)
    train_t = time.time() - t0

    t1 = time.time()
    resid_gnd = model_gnd.predict(X_eval)
    resid_cpl = model_cpl.predict(X_eval)
    pred_gnd = eval_df["compact_gnd_estimate_fF"].to_numpy() + resid_gnd
    pred_cpl = eval_df["compact_cpl_estimate_total_fF"].to_numpy() + resid_cpl
    inf_t = time.time() - t1
    pred_total = pred_gnd + pred_cpl

    return CompactGAMResult(
        method="B4_compact_gbdt_resid",
        seed=seed,
        pred_gnd=pred_gnd,
        pred_cpl=pred_cpl,
        pred_total=pred_total,
        golden_gnd=eval_df["c_gnd_fF"].to_numpy(),
        golden_cpl=eval_df["c_cpl_total_fF"].to_numpy(),
        golden_total=(eval_df["c_gnd_fF"] + eval_df["c_cpl_total_fF"]).to_numpy(),
        train_seconds=train_t,
        inference_seconds=inf_t,
        n_train=len(train_df),
        n_eval=len(eval_df),
    )


# ============================================================================
# Variant 3 — Compact + log-space GBDT residual (multiplicative)
# ============================================================================


def compact_plus_log_gbdt_residual(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_cols: list[str],
    seed: int = 0,
    eps_fF: float = 1e-3,
    n_estimators: int = 300,
    max_depth: int = 6,
    lr: float = 0.05,
) -> CompactGAMResult:
    """Multiplicative residual in log space (more robust on power-law data).

    Per channel:
        log_target = log(c_ch_fF) - log(compact_estimate_fF)   [residual in log space]
        GBDT learns log_residual ≈ f(features)
        pred = compact_estimate × exp(GBDT.predict(features))
    """
    import time
    xgb = _load_xgb_or_die()

    X_train = _build_feature_matrix(train_df, feature_cols)
    X_eval = _build_feature_matrix(eval_df, feature_cols)

    log_compact_gnd_train = np.log(
        train_df["compact_gnd_estimate_fF"].clip(lower=eps_fF).to_numpy(dtype=np.float64)
    )
    log_compact_cpl_train = np.log(
        train_df["compact_cpl_estimate_total_fF"].clip(lower=eps_fF).to_numpy(dtype=np.float64)
    )
    log_y_gnd = np.log(train_df["c_gnd_fF"].clip(lower=eps_fF).to_numpy(dtype=np.float64))
    log_y_cpl = np.log(train_df["c_cpl_total_fF"].clip(lower=eps_fF).to_numpy(dtype=np.float64))

    y_train_log_resid_gnd = log_y_gnd - log_compact_gnd_train
    y_train_log_resid_cpl = log_y_cpl - log_compact_cpl_train

    t0 = time.time()
    model_gnd = xgb.XGBRegressor(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr,
        random_state=seed, tree_method="hist", objective="reg:squarederror",
        subsample=0.8, colsample_bytree=0.8, verbosity=0,
    )
    model_cpl = xgb.XGBRegressor(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr,
        random_state=seed + 1, tree_method="hist", objective="reg:squarederror",
        subsample=0.8, colsample_bytree=0.8, verbosity=0,
    )
    model_gnd.fit(X_train, y_train_log_resid_gnd)
    model_cpl.fit(X_train, y_train_log_resid_cpl)
    train_t = time.time() - t0

    t1 = time.time()
    log_resid_gnd = model_gnd.predict(X_eval)
    log_resid_cpl = model_cpl.predict(X_eval)
    compact_gnd_eval = eval_df["compact_gnd_estimate_fF"].clip(lower=eps_fF).to_numpy()
    compact_cpl_eval = eval_df["compact_cpl_estimate_total_fF"].clip(lower=eps_fF).to_numpy()
    pred_gnd = compact_gnd_eval * np.exp(log_resid_gnd)
    pred_cpl = compact_cpl_eval * np.exp(log_resid_cpl)
    inf_t = time.time() - t1
    pred_total = pred_gnd + pred_cpl

    return CompactGAMResult(
        method="B4_compact_log_gbdt_resid",
        seed=seed,
        pred_gnd=pred_gnd,
        pred_cpl=pred_cpl,
        pred_total=pred_total,
        golden_gnd=eval_df["c_gnd_fF"].to_numpy(),
        golden_cpl=eval_df["c_cpl_total_fF"].to_numpy(),
        golden_total=(eval_df["c_gnd_fF"] + eval_df["c_cpl_total_fF"]).to_numpy(),
        train_seconds=train_t,
        inference_seconds=inf_t,
        n_train=len(train_df),
        n_eval=len(eval_df),
    )


# ============================================================================
# Per-channel MAPE summary
# ============================================================================


def per_channel_mape(result: CompactGAMResult, eps_fF: float = 1e-3) -> dict:
    """Summarize per-channel MAPE for a CompactGAMResult."""
    def _stats(pred: np.ndarray, gold: np.ndarray) -> dict:
        gold_safe = np.clip(gold, eps_fF, None)
        rel = np.abs(pred - gold) / gold_safe
        return {
            "median": float(np.median(rel)),
            "mean": float(np.mean(rel)),
            "p95": float(np.percentile(rel, 95)),
            "n": len(rel),
        }

    return {
        "method": result.method,
        "seed": result.seed,
        "gnd": _stats(result.pred_gnd, result.golden_gnd),
        "cpl": _stats(result.pred_cpl, result.golden_cpl),
        "total": _stats(result.pred_total, result.golden_total),
        "train_seconds": result.train_seconds,
        "inference_seconds": result.inference_seconds,
        "n_train": result.n_train,
        "n_eval": result.n_eval,
    }
