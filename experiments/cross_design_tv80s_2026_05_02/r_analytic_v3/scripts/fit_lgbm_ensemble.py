"""Phase 7 — Hybrid linear + LGBM ensemble (5 seeds, cfg1 small + cfg2 medium).

Stage 1: NNLS-IRLS linear (interpretable physics).
Stage 2: 5 LGBM seeds × 2 configs = 10 trees → ensemble (mean) on relative residual.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import lsq_linear

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
    df = df.merge(pins, on="net_name", how="left").fillna(0.0)
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


def _stats(label, pred, y, n_boot=1000):
    ape = 100 * np.abs(pred - y) / y
    bias = 100 * (pred - y) / y
    rng = np.random.default_rng(0)
    boots = [ape[rng.integers(0, len(ape), len(ape))].mean() for _ in range(n_boot)]
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    print(f"  {label:>50s}: MAPE={ape.mean():7.4f}%  med={np.median(ape):7.4f}%  "
          f"P90={np.percentile(ape,90):7.3f}%  bias={bias.mean():+7.4f}%  CI=[{ci[0]:.3f}, {ci[1]:.3f}]", flush=True)
    return ape.mean(), ape, bias


def main():
    train_dfs = [_load(d) for d in DESIGNS_TRAIN]
    test_df = _load(DESIGN_TEST)
    print(f"Train nets: {sum(len(d) for d in train_dfs):,}, test nets: {len(test_df)}", flush=True)

    # Stage 1
    pref_lin = ["nsq_M", "rsq_M", "nvian_", "one", "n_pin_PIN", "n_pin_inst", "pin_nsq_M"]
    fcols_lin = _select(train_dfs + [test_df], pref_lin)
    Xt = np.vstack([_design_matrix(d, fcols_lin) for d in train_dfs])
    yt = np.concatenate([d["R_gold"].values for d in train_dfs])
    Xs = _design_matrix(test_df, fcols_lin); ys = test_df["R_gold"].values
    c = irls_nnls(Xt, yt)
    pred_lin_train = Xt @ c
    pred_lin_test  = Xs @ c
    print("Stage 1:", flush=True)
    _stats("train", pred_lin_train, yt)
    _stats("test",  pred_lin_test,  ys)

    # Stage 2
    pref_full = ["nsq_M", "rsq_M", "nvian_", "one", "n_pin_PIN", "n_pin_inst", "pin_nsq_M",
                  "n_segments", "n_zero_l_wire", "n_pins", "n_pins_total", "n_pins_matched"]
    fcols_full = _select(train_dfs + [test_df], pref_full)

    # add per-design 1-hot dummy features (helps capture design-level systematic offset)
    design_id_per_train = []
    for di, df in enumerate(train_dfs):
        design_id_per_train.append(np.full(len(df), di, dtype=np.int32))
    design_ids_train = np.concatenate(design_id_per_train)
    # for test: assign design_id = -1 (out of train); we'll predict with the
    # MOST COMMON one-hot pattern (mean of train distribution) at inference.
    Xt_full = np.vstack([_design_matrix(d, fcols_full) for d in train_dfs])
    Xs_full = _design_matrix(test_df, fcols_full)
    # one-hot encode design_id
    n_des = len(DESIGNS_TRAIN)
    one_hot_train = np.zeros((Xt_full.shape[0], n_des))
    for i, did in enumerate(design_ids_train):
        one_hot_train[i, did] = 1.0
    # for test: use uniform distribution across train designs
    one_hot_test = np.full((Xs_full.shape[0], n_des), 1.0 / n_des)
    Xt_full = np.hstack([Xt_full, one_hot_train])
    Xs_full = np.hstack([Xs_full, one_hot_test])
    fcols_full = fcols_full + [f"des_{d}" for d in DESIGNS_TRAIN]
    print(f"\nStage 2 fcols: {len(fcols_full)} (incl per-design 1-hot)", flush=True)

    z_train = (yt - pred_lin_train) / np.maximum(pred_lin_train, 1e-3)

    # validation split for early stopping
    rng = np.random.default_rng(0)
    n = len(yt)
    val_idx = rng.choice(n, size=int(0.05 * n), replace=False)
    train_mask = np.ones(n, dtype=bool); train_mask[val_idx] = False

    cfg = dict(n_estimators=500, learning_rate=0.05, num_leaves=31, max_depth=4,
               min_child_samples=80, reg_lambda=1.0,
               objective="regression_l1", metric="l1",
               feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5)

    n_seeds = 5
    z_pred_test_seeds = []
    z_pred_train_seeds = []
    for seed in range(n_seeds):
        cfg_s = {**cfg, "random_state": seed, "seed": seed}
        gbm = lgb.LGBMRegressor(**cfg_s, n_jobs=-1, verbose=-1)
        w_full = 1.0 / yt
        gbm.fit(Xt_full[train_mask], z_train[train_mask], sample_weight=w_full[train_mask],
                  eval_set=[(Xt_full[val_idx], z_train[val_idx])],
                  eval_sample_weight=[w_full[val_idx]],
                  callbacks=[lgb.early_stopping(30)])
        zp_te = gbm.predict(Xs_full)
        zp_tr = gbm.predict(Xt_full)
        z_pred_test_seeds.append(zp_te)
        z_pred_train_seeds.append(zp_tr)
        pred_test_seed = pred_lin_test * (1.0 + zp_te)
        test_mape = float(np.mean(np.abs(pred_test_seed - ys) / ys) * 100)
        print(f"  seed {seed}: test MAPE = {test_mape:.4f}%, best_iter = {gbm.best_iteration_}", flush=True)

    # ensemble
    z_pred_test_mean  = np.mean(z_pred_test_seeds,  axis=0)
    z_pred_train_mean = np.mean(z_pred_train_seeds, axis=0)
    pred_test_ens  = pred_lin_test  * (1.0 + z_pred_test_mean)
    pred_train_ens = pred_lin_train * (1.0 + z_pred_train_mean)

    print(f"\n=== ENSEMBLE ({n_seeds} seeds) ===", flush=True)
    _stats("ensemble train", pred_train_ens, yt)
    _stats("ensemble test",  pred_test_ens,  ys)

    qs = np.asarray(pd.qcut(ys, 4, labels=["Q1_short", "Q2", "Q3", "Q4_long"]))
    ape = 100 * np.abs(pred_test_ens - ys) / ys
    bias = 100 * (pred_test_ens - ys) / ys
    print(f"\nLength-stratified (ensemble):", flush=True)
    for q in ["Q1_short", "Q2", "Q3", "Q4_long"]:
        m = (qs == q)
        print(f"  {q:>9s}: n={m.sum():4d}  R_med={np.median(ys[m]):.1f}Ω  "
              f"  MAPE={ape[m].mean():7.4f}%  bias={bias[m].mean():+7.4f}%")

    test_df["R_pred_ens"] = pred_test_ens
    test_df["ape_ens"] = ape
    test_df.to_parquet(_V3 / "outputs" / "test_predictions_ensemble.parquet")

    out = {"linear_coefs":     {fcols_lin[i]: float(c[i]) for i in range(len(fcols_lin))},
           "stage2_features":  fcols_full,
           "stage2_config":    cfg,
           "n_seeds":          n_seeds,
           "test_MAPE_ens":    float(np.mean(ape))}
    with open(_V3 / "outputs" / "ensemble_summary.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
