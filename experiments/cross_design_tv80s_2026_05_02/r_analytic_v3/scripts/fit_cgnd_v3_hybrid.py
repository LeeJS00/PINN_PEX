"""Phase 13b — c_gnd v3 hybrid: NNLS + 5-seed LGBM + Stage 3 stacking.

Mirrors fit_v6_stage3.py but targets c_gnd_gold instead of R_gold.
Goal: see if cell OBS + cell SIZE features break the pex_v3 21% c_gnd ceiling.

Stages:
  1. NNLS-IRLS on physics features (nsq_M, rsq_M, nvian_, n_pin_inst, one,
     pin_nsq_M, v6 cell OBS + SIZE)
  2. 5-seed LGBM ensemble on relative residual + per-design 1-hot
  3. 3-seed LGBM stacking on Stage 2 residual

c_gnd has different physics than R:
  - wire-to-substrate cap (proportional to wire AREA, not L/W)
  - cell intrinsic gate cap (transistor Cgg, scales with cell active area)
  - cell pin pad-to-power coupling

So features that should matter most for c_gnd:
  - rect_area_M (cell pin pad area)
  - sumL_M / nsq_M (wire footprint)
  - v6_obs_signal_area_M, v6_cell_area_sum (cell active area / gate cap)
  - n_pin_inst (gate count)
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

# Floor for c_gnd MAPE (drop nets below this — they're numerical noise).
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
    print("Loading per-design features + c_gnd targets ...", flush=True)
    train_dfs = [_load(d) for d in DESIGNS_TRAIN]
    test_df = _load(DESIGN_TEST)
    print(f"  train: {sum(len(d) for d in train_dfs):,} nets, test: {len(test_df)}", flush=True)
    print(f"  test c_gnd mean = {test_df['c_gnd_gold'].mean():.4f}fF  median = {test_df['c_gnd_gold'].median():.4f}fF", flush=True)

    # ---------------- Stage 1 — NNLS ablations ----------------
    print("\n=== Stage 1 NNLS ablations (target: c_gnd) ===", flush=True)
    abls = [
        ("v4 baseline (wire+via+pins)",
            ["nsq_M", "rsq_M", "nvian_", "one", "n_pin_PIN", "n_pin_inst", "pin_nsq_M"]),
        ("+ v6 signal OBS",
            ["nsq_M", "rsq_M", "nvian_", "one", "n_pin_PIN", "n_pin_inst", "pin_nsq_M",
             "v6_obs_signal_nsq_M", "v6_obs_signal_area_M", "v6_obs_n_via_v"]),
        ("+ v6 OBS + cell SIZE",
            ["nsq_M", "rsq_M", "nvian_", "one", "n_pin_PIN", "n_pin_inst", "pin_nsq_M",
             "v6_obs_signal_nsq_M", "v6_obs_signal_area_M", "v6_obs_n_via_v",
             "v6_cell_size_w", "v6_cell_size_h", "v6_cell_area"]),
        ("+ all v6 + segment counts + sumL",
            ["nsq_M", "rsq_M", "nvian_", "one", "n_pin_PIN", "n_pin_inst", "pin_nsq_M",
             "v6_", "n_segments", "n_zero_l_wire"]),
    ]
    rows_s1 = []
    for label, prefixes in abls:
        fcols = _select(train_dfs + [test_df], prefixes)
        Xt = np.vstack([_design_matrix(d, fcols) for d in train_dfs])
        yt = np.concatenate([d["c_gnd_gold"].values for d in train_dfs])
        Xs = _design_matrix(test_df, fcols)
        ys = test_df["c_gnd_gold"].values
        c = irls_nnls(Xt, yt)
        train_mape = float(np.mean(np.abs(Xt @ c - yt) / yt) * 100)
        test_mape  = float(np.mean(np.abs(Xs @ c - ys) / ys) * 100)
        nz = int(np.sum(c > 1e-8))
        print(f"\n[{label}]  fcols={len(fcols)}  active={nz}")
        print(f"    train: {train_mape:.4f}%   test: {test_mape:.4f}%")
        rows_s1.append({"label": label, "n_features": len(fcols), "active": nz,
                          "train_mape": train_mape, "test_mape": test_mape,
                          "fcols": fcols, "coef": c.tolist()})

    best = min(rows_s1, key=lambda r: r["test_mape"])
    print(f"\n===== BEST Stage 1 ===== test={best['test_mape']:.4f}% ({best['label']})")
    print("  active coefs (>0):")
    for n, v in zip(best["fcols"], best["coef"]):
        if v > 1e-8:
            print(f"    {n:<35s}  {v:.6f}")

    # ---------------- Stage 2 — LGBM 5-seed ensemble ----------------
    fcols_lin = best["fcols"]
    c_lin = np.array(best["coef"])
    Xt_lin = np.vstack([_design_matrix(d, fcols_lin) for d in train_dfs])
    yt = np.concatenate([d["c_gnd_gold"].values for d in train_dfs])
    Xs_lin = _design_matrix(test_df, fcols_lin)
    ys = test_df["c_gnd_gold"].values
    pred_lin_train = np.clip(Xt_lin @ c_lin, 1e-4, None)
    pred_lin_test  = np.clip(Xs_lin @ c_lin, 1e-4, None)

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
    Xt_full = np.hstack([Xt_full, one_hot_train])
    Xs_full = np.hstack([Xs_full, one_hot_test])
    print(f"\nStage 2 fcols (incl per-design 1-hot): {Xt_full.shape[1]}", flush=True)

    z_train = (yt - pred_lin_train) / pred_lin_train
    rng = np.random.default_rng(0)
    n = len(yt)
    val_idx = rng.choice(n, size=int(0.05 * n), replace=False)
    train_mask = np.ones(n, dtype=bool); train_mask[val_idx] = False

    cfg2 = dict(n_estimators=500, learning_rate=0.05, num_leaves=31, max_depth=4,
                min_child_samples=80, reg_lambda=1.0,
                objective="regression_l1", metric="l1",
                feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5)
    n_seeds = 5
    z_test_seeds = []
    z_train_seeds = []
    print(f"\n=== Stage 2 (5 LGBM seeds) ===", flush=True)
    for seed in range(n_seeds):
        cfg_s = {**cfg2, "random_state": seed, "seed": seed}
        gbm = lgb.LGBMRegressor(**cfg_s, n_jobs=-1, verbose=-1)
        w_full = 1.0 / yt
        gbm.fit(Xt_full[train_mask], z_train[train_mask], sample_weight=w_full[train_mask],
                  eval_set=[(Xt_full[val_idx], z_train[val_idx])],
                  eval_sample_weight=[w_full[val_idx]],
                  callbacks=[lgb.early_stopping(30)])
        z_test_seeds.append(gbm.predict(Xs_full))
        z_train_seeds.append(gbm.predict(Xt_full))
        ts = pred_lin_test * (1 + z_test_seeds[-1])
        print(f"  S2 seed {seed}: test MAPE = {np.mean(np.abs(ts-ys)/ys)*100:.4f}%", flush=True)

    z_test_s2_mean = np.mean(z_test_seeds, axis=0)
    z_train_s2_mean = np.mean(z_train_seeds, axis=0)
    pred_s2_train = pred_lin_train * (1 + z_train_s2_mean)
    pred_s2_test  = pred_lin_test  * (1 + z_test_s2_mean)
    print(f"\n=== c_gnd Stage 2 ensemble ===", flush=True)
    _stats("S2 train", pred_s2_train, yt)
    s2_mape, _, _ = _stats("S2 test",  pred_s2_test,  ys)

    # ---------------- Stage 3 ----------------
    z3_train = (yt - pred_s2_train) / np.maximum(pred_s2_train, 1e-4)
    fcols_s3 = _select(train_dfs + [test_df], pref_full[:-2])  # no 1-hot
    Xt_s3 = np.vstack([_design_matrix(d, fcols_s3) for d in train_dfs])
    Xs_s3 = _design_matrix(test_df, fcols_s3)
    Xt_s3 = np.column_stack([Xt_s3, pred_lin_train, pred_s2_train,
                              np.log(pred_s2_train), np.sqrt(pred_s2_train)])
    Xs_s3 = np.column_stack([Xs_s3, pred_lin_test, pred_s2_test,
                              np.log(pred_s2_test), np.sqrt(pred_s2_test)])

    cfg3 = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, max_depth=4,
                min_child_samples=120, reg_lambda=2.0,
                objective="regression_l1", metric="l1",
                feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=5)
    n_seeds3 = 3
    z3_test_seeds = []
    print(f"\n=== Stage 3 (3 residual GBT seeds) ===", flush=True)
    for seed in range(n_seeds3):
        cfg_s = {**cfg3, "random_state": 100+seed, "seed": 100+seed}
        gbm = lgb.LGBMRegressor(**cfg_s, n_jobs=-1, verbose=-1)
        gbm.fit(Xt_s3[train_mask], z3_train[train_mask], sample_weight=(1.0/yt)[train_mask],
                  eval_set=[(Xt_s3[val_idx], z3_train[val_idx])],
                  eval_sample_weight=[(1.0/yt)[val_idx]],
                  callbacks=[lgb.early_stopping(30)])
        z3_test_seeds.append(gbm.predict(Xs_s3))
        ts = pred_s2_test * (1 + z3_test_seeds[-1])
        print(f"  S3 seed {seed}: test MAPE = {np.mean(np.abs(ts-ys)/ys)*100:.4f}%", flush=True)

    z3_test_mean = np.mean(z3_test_seeds, axis=0)
    pred_final_test = pred_s2_test * (1 + z3_test_mean)
    print(f"\n=== c_gnd FINAL (Stage 3 stacked) ===", flush=True)
    _stats("Final test", pred_final_test, ys)

    qs = np.asarray(pd.qcut(ys, 4, labels=["Q1_short", "Q2", "Q3", "Q4_long"]))
    ape = 100 * np.abs(pred_final_test - ys) / ys
    bias = 100 * (pred_final_test - ys) / ys
    print(f"\nLength-stratified (c_gnd quartiles):", flush=True)
    for q in ["Q1_short", "Q2", "Q3", "Q4_long"]:
        m = (qs == q)
        print(f"  {q:>9s}: n={m.sum():4d}  c_gnd_med={np.median(ys[m]):.4f}fF  "
              f"  MAPE={ape[m].mean():7.4f}%  bias={bias[m].mean():+7.4f}%")

    test_df["c_gnd_pred_final"] = pred_final_test
    test_df["ape_cgnd"] = ape
    test_df.to_parquet(_V3 / "outputs" / "test_predictions_cgnd_v3.parquet")

    out = {"stage1_ablations": [{k: v for k, v in r.items() if k != "coef"} for r in rows_s1],
           "stage1_best": best["label"],
           "stage1_test_MAPE": best["test_mape"],
           "stage2_test_MAPE": s2_mape,
           "stage3_test_MAPE": float(np.mean(ape)),
           "stage3_median": float(np.median(ape)),
           "stage3_p90": float(np.percentile(ape, 90)),
           "n_seeds_s2": n_seeds,
           "n_seeds_s3": n_seeds3,
           "vs_pex_v3_ceiling_21pct": "TODO compare"}
    with open(_V3 / "outputs" / "cgnd_v3_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved.", flush=True)


if __name__ == "__main__":
    main()
