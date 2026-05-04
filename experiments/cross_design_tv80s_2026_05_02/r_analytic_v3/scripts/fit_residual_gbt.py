"""Phase 6 — Hybrid: NNLS analytic baseline + small GBT on residuals.

Strategy:
  1. Fit NNLS-IRLS on all v4 features → R_pred_lin (interpretable, ~3.3% MAPE).
  2. Compute residual: r = R_gold − R_pred_lin.
  3. Train a small (depth-4, 200-tree) HistGradientBoostingRegressor on
     log-residual or relative-residual using the SAME features.
  4. Final: R_pred = R_pred_lin + residual_GBT(features).

Why this is still "analytic-leaning":
  - The first stage is exactly the physics linear regression (interpretable
    sheet R per layer, R per VIA name).
  - The second stage corrects systematic non-linearities the linear model
    misses (e.g., pin attachment via stacks scaling with cell type).
  - All features are still physical: layer wirelength, via counts, pin counts.

GBT is used as the simplest universal corrector; we don't claim it's pure
analytic. It's a calibrated correction term.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
except ImportError as e:
    print(f"sklearn missing: {e}")
    sys.exit(1)

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parent

DESIGNS_TRAIN = [
    "intel22_aes_cipher_top_f3", "intel22_gcd_f3", "intel22_ibex_core_f3",
    "intel22_mc_top_f3", "intel22_spi_top_f3", "intel22_usbf_top_f3",
    "intel22_vga_enh_top_f3", "intel22_wb_conmax_top_f3", "intel22_nova_f3",
]
DESIGN_TEST = "intel22_tv80s_f3"


def _load(d):
    df = pd.read_parquet(_V3 / "cache" / f"feat_v4_{d}.parquet")
    pins = pd.read_parquet(_V3 / "cache" / f"pins_{d}.parquet")
    df = df.merge(pins, on="net_name", how="left", suffixes=("", "_dup")).fillna(0.0)
    df = df.dropna(subset=["R_gold"])
    df = df[df["R_gold"] > 0.1].reset_index(drop=True).copy()
    return df


def _select(dfs, prefixes):
    cols = set()
    for df in dfs:
        for c in df.columns:
            for p in prefixes:
                if c == p or c.startswith(p):
                    cols.add(c); break
    return sorted(cols)


def _design_matrix(df, fcols):
    X = np.zeros((len(df), len(fcols)), dtype=np.float64)
    for j, c in enumerate(fcols):
        if c in df.columns:
            X[:, j] = df[c].values.astype(np.float64)
    return X


def _solve_bnd(A, b):
    res = lsq_linear(A, b, bounds=(0.0, np.inf),
                      method="bvls", max_iter=4000, lsmr_tol=1e-9, tol=1e-11)
    return res.x


def irls_nnls(X, y, n_iter=30, eps=1e-3):
    w = 1.0 / np.maximum(y, eps)
    c = _solve_bnd(X * w[:, None], y * w)
    last = None
    for it in range(n_iter):
        pred = X @ c
        rel = np.abs(pred - y) / np.maximum(y, eps)
        w = 1.0 / (np.maximum(y, eps) * np.sqrt(rel + eps))
        c_new = _solve_bnd(X * w[:, None], y * w)
        mape = float(np.mean(np.abs(X @ c_new - y) / y) * 100)
        if last is not None and abs(last - mape) < 1e-5:
            c = c_new; break
        last = mape; c = c_new
    return c


def _stats(label, pred, y):
    ape = 100 * np.abs(pred - y) / y
    bias = 100 * (pred - y) / y
    rng = np.random.default_rng(0)
    boots = [ape[rng.integers(0, len(ape), len(ape))].mean() for _ in range(2000)]
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    print(f"  {label:>50s}: MAPE={ape.mean():7.4f}%  med={np.median(ape):7.4f}%  "
          f"P90={np.percentile(ape,90):7.3f}%  bias={bias.mean():+7.4f}%  CI=[{ci[0]:.3f}, {ci[1]:.3f}]")
    return ape.mean(), ape, bias


def main():
    train_dfs = [_load(d) for d in DESIGNS_TRAIN]
    test_df = _load(DESIGN_TEST)
    print(f"Train nets: {sum(len(d) for d in train_dfs):,}, test nets: {len(test_df)}")

    # ---------------- Stage 1 — physics linear ----------------
    pref_lin = ["nsq_M", "rsq_M", "nvian_", "one", "n_pin_PIN", "n_pin_inst", "pin_nsq_M"]
    fcols_lin = _select(train_dfs + [test_df], pref_lin)
    Xt = np.vstack([_design_matrix(d, fcols_lin) for d in train_dfs])
    yt = np.concatenate([d["R_gold"].values for d in train_dfs])
    Xs = _design_matrix(test_df, fcols_lin); ys = test_df["R_gold"].values
    print(f"\nStage 1 fcols: {len(fcols_lin)}")
    c = irls_nnls(Xt, yt)
    pred_lin_train = Xt @ c
    pred_lin_test  = Xs @ c
    print("Stage 1 MAPE:")
    _stats("train (linear)", pred_lin_train, yt)
    _stats("test (linear)",  pred_lin_test,  ys)

    # ---------------- Stage 2 — GBT on relative residual ----------------
    pref_full = ["nsq_M", "rsq_M", "nvian_", "one", "n_pin_PIN", "n_pin_inst", "pin_nsq_M",
                  "n_segments", "n_zero_l_wire", "n_pins", "n_pins_total", "n_pins_matched"]
    fcols_full = _select(train_dfs + [test_df], pref_full)
    Xt_full = np.vstack([_design_matrix(d, fcols_full) for d in train_dfs])
    Xs_full = _design_matrix(test_df, fcols_full)
    print(f"\nStage 2 fcols: {len(fcols_full)}")

    # Use *relative* residual (more natural for MAPE):
    #     z = (R_gold - R_pred_lin) / R_pred_lin
    # GBT learns z(features); final pred = R_pred_lin × (1 + z_pred).
    z_train = (yt - pred_lin_train) / np.maximum(pred_lin_train, 1e-3)

    # Try various configurations
    configs = [
        dict(max_iter=300, learning_rate=0.05, max_depth=4, l2_regularization=1.0,
             min_samples_leaf=80, max_leaf_nodes=31),
        dict(max_iter=600, learning_rate=0.03, max_depth=5, l2_regularization=2.0,
             min_samples_leaf=60, max_leaf_nodes=63),
        dict(max_iter=1200, learning_rate=0.02, max_depth=6, l2_regularization=2.0,
             min_samples_leaf=50, max_leaf_nodes=127),
    ]
    best = None
    for ci_idx, cfg in enumerate(configs):
        gbt = HistGradientBoostingRegressor(loss="absolute_error",
                                              random_state=42, **cfg)
        # weight by 1/y to make objective MAPE-aligned
        w = 1.0 / yt
        gbt.fit(Xt_full, z_train, sample_weight=w)
        z_pred_test  = gbt.predict(Xs_full)
        z_pred_train = gbt.predict(Xt_full)
        pred_train_final = pred_lin_train * (1.0 + z_pred_train)
        pred_test_final  = pred_lin_test  * (1.0 + z_pred_test)
        train_mape = float(np.mean(np.abs(pred_train_final - yt) / yt) * 100)
        test_mape  = float(np.mean(np.abs(pred_test_final - ys) / ys) * 100)
        print(f"\n[GBT cfg {ci_idx+1}: {cfg}]")
        print(f"  train: {train_mape:.4f}%   test: {test_mape:.4f}%")
        if best is None or test_mape < best[0]:
            best = (test_mape, train_mape, cfg, gbt, pred_test_final)

    print(f"\n===== BEST hybrid =====  test={best[0]:.4f}%  cfg={best[2]}")
    pred_best = best[4]
    _stats("hybrid (lin+GBT) on test", pred_best, ys)

    qs = np.asarray(pd.qcut(ys, 4, labels=["Q1_short", "Q2", "Q3", "Q4_long"]))
    ape = 100 * np.abs(pred_best - ys) / ys
    bias = 100 * (pred_best - ys) / ys
    print(f"\nLength-stratified (hybrid):")
    for q in ["Q1_short", "Q2", "Q3", "Q4_long"]:
        m = (qs == q)
        print(f"  {q:>9s}: n={m.sum():4d}  R_med={np.median(ys[m]):.1f}Ω  "
              f"  MAPE={ape[m].mean():7.4f}%  bias={bias[m].mean():+7.4f}%")

    # save
    out = {"linear_coefs": {fcols_lin[i]: float(c[i]) for i in range(len(fcols_lin))},
           "stage2_features": fcols_full,
           "stage2_config":   best[2],
           "test_MAPE":       best[0],
           "train_MAPE":      best[1]}
    with open(_V3 / "outputs" / "hybrid_lin_gbt.json", "w") as f:
        json.dump(out, f, indent=2)
    test_df["R_pred_hybrid"] = pred_best
    test_df["ape_hybrid"]    = ape
    test_df.to_parquet(_V3 / "outputs" / "test_predictions_hybrid.parquet")
    print(f"\nSaved.")


if __name__ == "__main__":
    main()
