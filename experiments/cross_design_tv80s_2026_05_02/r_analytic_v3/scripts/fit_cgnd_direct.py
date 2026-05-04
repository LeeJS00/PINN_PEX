"""Phase 13c — c_gnd DIRECT GBT (not residual).

c_gnd has multiplicative noise + heavy tail (median 0.2fF, max 14fF).
The Stage 1+2 residual approach fails because (y - pred_lin) / pred_lin is
unstable for small c_gnd nets.

Strategy:
  Stage 1: NNLS linear (interpretable, ~26% MAPE).
  Stage 2: LGBM 5-seed ensemble predicting LOG(c_gnd) directly.
           log-target naturally MAPE-friendly.
  Stage 3: blend Stage 1 (linear) and Stage 2 (LGBM) via val-fit α.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import lsq_linear, minimize_scalar

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parent

DESIGNS_TRAIN = [
    "intel22_aes_cipher_top_f3", "intel22_gcd_f3", "intel22_ibex_core_f3",
    "intel22_mc_top_f3", "intel22_spi_top_f3", "intel22_usbf_top_f3",
    "intel22_vga_enh_top_f3", "intel22_wb_conmax_top_f3", "intel22_nova_f3",
]
DESIGN_TEST = "intel22_tv80s_f3"

CGND_FLOOR_FF = 1e-3


def _load(d):
    df = pd.read_parquet(_V3 / "cache" / f"feat_v4_{d}.parquet")
    pins = pd.read_parquet(_V3 / "cache" / f"pins_{d}.parquet")
    v6 = pd.read_parquet(_V3 / "cache" / f"feat_v6_{d}.parquet")
    cgnd = pd.read_parquet(_V3 / "cache" / f"cgnd_{d}.parquet")
    df = df.merge(pins, on="net_name", how="left")
    df = df.merge(v6, on="net_name", how="left")
    df = df.merge(cgnd, on="net_name", how="left").fillna(0.0)
    df = df.dropna(subset=["c_gnd_gold"])
    df = df[df["c_gnd_gold"] > CGND_FLOOR_FF].reset_index(drop=True).copy()
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
    res = lsq_linear(A, b, bounds=(0.0, np.inf), method="bvls",
                      max_iter=4000, lsmr_tol=1e-9, tol=1e-11)
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
    boots = [ape[rng.integers(0, len(ape), len(ape))].mean() for _ in range(1000)]
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    print(f"  {label:>50s}: MAPE={ape.mean():7.4f}%  med={np.median(ape):7.4f}%  "
          f"P90={np.percentile(ape,90):7.3f}%  bias={bias.mean():+7.4f}%  CI=[{ci[0]:.3f}, {ci[1]:.3f}]", flush=True)
    return ape.mean(), ape, bias


def main():
    print("Loading ...", flush=True)
    train_dfs = [_load(d) for d in DESIGNS_TRAIN]
    test_df = _load(DESIGN_TEST)
    print(f"  train: {sum(len(d) for d in train_dfs):,} test: {len(test_df)}", flush=True)

    # Stage 1 — NNLS with all features
    pref_lin = ["nsq_M", "rsq_M", "nvian_", "one", "n_pin_PIN", "n_pin_inst", "pin_nsq_M",
                  "v6_obs_signal_nsq_M", "v6_obs_signal_area_M", "v6_obs_n_via_v",
                  "v6_cell_size_w", "v6_cell_size_h", "v6_cell_area"]
    fcols_lin = _select(train_dfs + [test_df], pref_lin)
    Xt = np.vstack([_design_matrix(d, fcols_lin) for d in train_dfs])
    yt = np.concatenate([d["c_gnd_gold"].values for d in train_dfs])
    Xs = _design_matrix(test_df, fcols_lin)
    ys = test_df["c_gnd_gold"].values
    c_lin = irls_nnls(Xt, yt)
    pred_lin_train = np.maximum(Xt @ c_lin, 1e-4)
    pred_lin_test  = np.maximum(Xs @ c_lin, 1e-4)
    print("\nStage 1 (NNLS linear, c_gnd):", flush=True)
    _stats("train", pred_lin_train, yt)
    s1_mape, _, _ = _stats("test",  pred_lin_test,  ys)

    # Stage 2 — LGBM DIRECT prediction of log(c_gnd)
    pref_full = ["nsq_M", "rsq_M", "nvian_", "one", "n_pin_PIN", "n_pin_inst", "pin_nsq_M",
                  "v6_", "n_segments", "n_zero_l_wire", "n_pins", "n_pins_total", "n_pins_matched"]
    fcols_full = _select(train_dfs + [test_df], pref_full)
    Xt_full = np.vstack([_design_matrix(d, fcols_full) for d in train_dfs])
    Xs_full = _design_matrix(test_df, fcols_full)
    n_des = len(DESIGNS_TRAIN)
    one_hot_train = np.zeros((Xt_full.shape[0], n_des))
    cum = 0
    for di, df in enumerate(train_dfs):
        one_hot_train[cum:cum+len(df), di] = 1.0
        cum += len(df)
    one_hot_test = np.full((Xs_full.shape[0], n_des), 1.0 / n_des)
    Xt_full = np.hstack([Xt_full, one_hot_train, pred_lin_train[:, None],
                          np.log(pred_lin_train)[:, None]])
    Xs_full = np.hstack([Xs_full, one_hot_test, pred_lin_test[:, None],
                          np.log(pred_lin_test)[:, None]])
    print(f"\nStage 2 fcols: {Xt_full.shape[1]}", flush=True)

    # Direct y target with L1 + 1/y weighting (MAPE-aligned)
    rng = np.random.default_rng(0)
    n = len(yt)
    val_idx = rng.choice(n, size=int(0.05 * n), replace=False)
    train_mask = np.ones(n, dtype=bool); train_mask[val_idx] = False
    w_full = 1.0 / yt   # MAPE-aligned weights

    cfg2 = dict(n_estimators=400, learning_rate=0.04, num_leaves=15, max_depth=3,
                min_child_samples=120, reg_lambda=2.0,
                objective="regression_l1", metric="l1",
                feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=5)
    n_seeds = 10  # more seeds, smaller trees
    pred_test_seeds = []
    pred_train_seeds = []
    print(f"\n=== Stage 2: LGBM direct y (L1 + 1/y weights) {n_seeds} seeds ===", flush=True)
    for seed in range(n_seeds):
        cfg_s = {**cfg2, "random_state": seed, "seed": seed}
        gbm = lgb.LGBMRegressor(**cfg_s, n_jobs=-1, verbose=-1)
        gbm.fit(Xt_full[train_mask], yt[train_mask], sample_weight=w_full[train_mask],
                  eval_set=[(Xt_full[val_idx], yt[val_idx])],
                  eval_sample_weight=[w_full[val_idx]],
                  callbacks=[lgb.early_stopping(50)])
        pred_test_seeds.append(np.maximum(gbm.predict(Xs_full), 1e-4))
        pred_train_seeds.append(np.maximum(gbm.predict(Xt_full), 1e-4))
        print(f"  S2 seed {seed}: test MAPE = {np.mean(np.abs(pred_test_seeds[-1]-ys)/ys)*100:.4f}%", flush=True)

    pred_s2_train = np.mean(pred_train_seeds, axis=0)
    pred_s2_test  = np.mean(pred_test_seeds, axis=0)
    print(f"\n=== Stage 2 (log-target) ensemble ===", flush=True)
    _stats("S2 train", pred_s2_train, yt)
    s2_mape, _, _ = _stats("S2 test",  pred_s2_test,  ys)

    # Stage 3 — blend (linear + log-LGBM) via val-fit α
    # Use train val_idx to find optimal α
    pred_train_blend = lambda a: a * pred_lin_train[val_idx] + (1 - a) * pred_s2_train[val_idx]
    def cost(a):
        p = pred_train_blend(a)
        return float(np.mean(np.abs(p - yt[val_idx]) / yt[val_idx]))
    res = minimize_scalar(cost, bounds=(0.0, 1.0), method="bounded")
    a_opt = float(res.x)
    print(f"\nStage 3: optimal val-fit α = {a_opt:.4f}  val MAPE = {cost(a_opt)*100:.4f}%", flush=True)

    pred_blend_test = a_opt * pred_lin_test + (1 - a_opt) * pred_s2_test
    print(f"\n=== Stage 3 (linear + log-LGBM blend) ===", flush=True)
    _stats("Final test", pred_blend_test, ys)

    qs = np.asarray(pd.qcut(ys, 4, labels=["Q1", "Q2", "Q3", "Q4"]))
    ape = 100 * np.abs(pred_blend_test - ys) / ys
    bias = 100 * (pred_blend_test - ys) / ys
    print(f"\nLength-stratified (c_gnd quartiles):", flush=True)
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        m = (qs == q)
        print(f"  {q}: n={m.sum():4d}  med={np.median(ys[m]):.4f}fF  "
              f"  MAPE={ape[m].mean():7.4f}%  bias={bias[m].mean():+7.4f}%")

    test_df["c_gnd_pred"] = pred_blend_test
    test_df["ape_cgnd"] = ape
    test_df.to_parquet(_V3 / "outputs" / "test_predictions_cgnd_direct.parquet")
    out = {"stage1_test_MAPE": s1_mape,
           "stage2_test_MAPE": s2_mape,
           "stage3_test_MAPE": float(np.mean(ape)),
           "stage3_median":    float(np.median(ape)),
           "stage3_p90":       float(np.percentile(ape, 90)),
           "blend_alpha":      a_opt,
           "stage1_features":  fcols_lin,
           "n_seeds":          n_seeds}
    with open(_V3 / "outputs" / "cgnd_direct_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved.", flush=True)


if __name__ == "__main__":
    main()
