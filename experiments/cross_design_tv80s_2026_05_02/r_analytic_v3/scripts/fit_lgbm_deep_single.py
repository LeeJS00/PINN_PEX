"""Phase 9 — Deep LGBM single seed, hybrid linear + GBT.

cfg2 from earlier (n_estimators=2000, depth=6, num_leaves=63) was hitting
single-seed potential. Run with FULL convergence + L1 + log-target hybrid.
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
    train_dfs = [_load(d) for d in DESIGNS_TRAIN]
    test_df = _load(DESIGN_TEST)
    print(f"Train: {sum(len(d) for d in train_dfs):,}, test: {len(test_df)}", flush=True)

    pref_lin = ["nsq_M", "rsq_M", "nvian_", "one", "n_pin_PIN", "n_pin_inst", "pin_nsq_M"]
    fcols_lin = _select(train_dfs + [test_df], pref_lin)
    Xt = np.vstack([_design_matrix(d, fcols_lin) for d in train_dfs])
    yt = np.concatenate([d["R_gold"].values for d in train_dfs])
    Xs = _design_matrix(test_df, fcols_lin); ys = test_df["R_gold"].values
    c_lin = irls_nnls(Xt, yt)
    pred_lin_train = np.clip(Xt @ c_lin, 1e-3, None)
    pred_lin_test  = np.clip(Xs @ c_lin, 1e-3, None)
    print("Stage 1:", flush=True)
    _stats("train", pred_lin_train, yt)
    _stats("test",  pred_lin_test,  ys)

    pref_full = ["nsq_M", "rsq_M", "nvian_", "one", "n_pin_PIN", "n_pin_inst", "pin_nsq_M",
                  "n_segments", "n_zero_l_wire", "n_pins", "n_pins_total", "n_pins_matched"]
    fcols_full = _select(train_dfs + [test_df], pref_full)
    Xt_full = np.vstack([_design_matrix(d, fcols_full) for d in train_dfs])
    Xs_full = _design_matrix(test_df, fcols_full)
    Xt_full = np.column_stack([Xt_full, pred_lin_train, np.log(pred_lin_train), np.sqrt(pred_lin_train)])
    Xs_full = np.column_stack([Xs_full, pred_lin_test,  np.log(pred_lin_test),  np.sqrt(pred_lin_test)])
    fcols_full = fcols_full + ["R_lin", "log_R_lin", "sqrt_R_lin"]
    print(f"\nStage 2 fcols: {len(fcols_full)}", flush=True)

    z_train = (yt - pred_lin_train) / np.maximum(pred_lin_train, 1e-3)

    rng = np.random.default_rng(0)
    n = len(yt)
    val_idx = rng.choice(n, size=int(0.05 * n), replace=False)
    train_mask = np.ones(n, dtype=bool); train_mask[val_idx] = False

    cfg = dict(n_estimators=3000, learning_rate=0.015, num_leaves=127, max_depth=8,
               min_child_samples=30, reg_lambda=1.5, reg_alpha=0.5,
               objective="regression_l1", metric="l1",
               feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
               max_bin=511, n_jobs=-1, verbose=-1)
    print(f"\n[Deep LGBM cfg: n_est={cfg['n_estimators']}, depth={cfg['max_depth']}, leaves={cfg['num_leaves']}]", flush=True)
    gbm = lgb.LGBMRegressor(random_state=0, seed=0, **cfg)
    w_full = 1.0 / yt
    gbm.fit(Xt_full[train_mask], z_train[train_mask], sample_weight=w_full[train_mask],
              eval_set=[(Xt_full[val_idx], z_train[val_idx])],
              eval_sample_weight=[w_full[val_idx]],
              callbacks=[lgb.early_stopping(60), lgb.log_evaluation(200)])
    z_pred_test  = gbm.predict(Xs_full)
    z_pred_train = gbm.predict(Xt_full)
    pred_test_final  = pred_lin_test  * (1.0 + z_pred_test)
    pred_train_final = pred_lin_train * (1.0 + z_pred_train)
    train_mape = float(np.mean(np.abs(pred_train_final - yt) / yt) * 100)
    test_mape  = float(np.mean(np.abs(pred_test_final - ys) / ys) * 100)
    print(f"\n  best_iter: {gbm.best_iteration_}", flush=True)
    _stats("Stage 2 train", pred_train_final, yt)
    _stats("Stage 2 test",  pred_test_final,  ys)

    qs = np.asarray(pd.qcut(ys, 4, labels=["Q1_short", "Q2", "Q3", "Q4_long"]))
    ape = 100 * np.abs(pred_test_final - ys) / ys
    bias = 100 * (pred_test_final - ys) / ys
    print(f"\nLength-stratified (deep single):", flush=True)
    for q in ["Q1_short", "Q2", "Q3", "Q4_long"]:
        m = (qs == q)
        print(f"  {q:>9s}: n={m.sum():4d}  R_med={np.median(ys[m]):.1f}Ω  "
              f"  MAPE={ape[m].mean():7.4f}%  bias={bias[m].mean():+7.4f}%")

    test_df["R_pred_deep_single"] = pred_test_final
    test_df["ape_deep_single"] = ape
    test_df.to_parquet(_V3 / "outputs" / "test_predictions_deep_single.parquet")
    out = {"test_MAPE":          test_mape,
           "train_MAPE":         train_mape,
           "test_median_ape":    float(np.median(ape)),
           "test_p90_ape":       float(np.percentile(ape, 90)),
           "best_iter":          int(gbm.best_iteration_),
           "config":             cfg,
           "linear_coefs":       {fcols_lin[i]: float(c_lin[i]) for i in range(len(fcols_lin))},
           "stage2_features":    fcols_full}
    with open(_V3 / "outputs" / "deep_single_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved.", flush=True)


if __name__ == "__main__":
    main()
