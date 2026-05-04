"""Phase 14 — Canonical train/test split re-fit (PAPER-GRADE).

Train (9): aes, gcd, ibex, ldpc, mc, spi, usbf, vga_enh, wb_conmax
Test  (2): nova, tv80s   ← matches official `configs/config.py`

Stages 1+2+3 hybrid for total_R, evaluating per-design separately
(consistent with pex_v3 reporting style).
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

# Canonical split (per configs/config.py)
DESIGNS_TRAIN = [
    "intel22_aes_cipher_top_f3", "intel22_gcd_f3", "intel22_ibex_core_f3",
    "intel22_ldpc_decoder_802_3an_f3",
    "intel22_mc_top_f3", "intel22_spi_top_f3", "intel22_usbf_top_f3",
    "intel22_vga_enh_top_f3", "intel22_wb_conmax_top_f3",
]
DESIGNS_TEST = ["intel22_nova_f3", "intel22_tv80s_f3"]


def _load_R(d):
    """For R fitting: needs feat_v4 + pins + v6 + R_gold (from compare_spef parsing)."""
    df = pd.read_parquet(_V3 / "cache" / f"feat_v4_{d}.parquet")
    pins = pd.read_parquet(_V3 / "cache" / f"pins_{d}.parquet")
    v6 = pd.read_parquet(_V3 / "cache" / f"feat_v6_{d}.parquet")
    df = df.merge(pins, on="net_name", how="left").merge(v6, on="net_name", how="left").fillna(0.0)
    df = df.dropna(subset=["R_gold"])
    df = df[df["R_gold"] > 0.1].reset_index(drop=True).copy()
    df["design"] = d
    return df


def _load_cgnd(d):
    df = pd.read_parquet(_V3 / "cache" / f"feat_v4_{d}.parquet")
    pins = pd.read_parquet(_V3 / "cache" / f"pins_{d}.parquet")
    v6 = pd.read_parquet(_V3 / "cache" / f"feat_v6_{d}.parquet")
    cgnd = pd.read_parquet(_V3 / "cache" / f"cgnd_{d}.parquet")
    df = df.merge(pins, on="net_name", how="left")
    df = df.merge(v6, on="net_name", how="left")
    df = df.merge(cgnd, on="net_name", how="left").fillna(0.0)
    df = df.dropna(subset=["c_gnd_gold"])
    df = df[df["c_gnd_gold"] > 1e-3].reset_index(drop=True).copy()
    df["design"] = d
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


def stats_per_design(label, pred, y, design_arr, n_boot=1000):
    """Report per-design MAPE + combined."""
    designs = sorted(set(design_arr))
    results = {}
    print(f"\n{label}:", flush=True)
    for des in designs:
        m = np.array([d == des for d in design_arr])
        ape = 100 * np.abs(pred[m] - y[m]) / y[m]
        bias = 100 * (pred[m] - y[m]) / y[m]
        rng = np.random.default_rng(0)
        boots = [ape[rng.integers(0, len(ape), len(ape))].mean() for _ in range(n_boot)]
        ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
        results[des] = {
            "n": int(m.sum()),
            "mape_mean": float(ape.mean()),
            "mape_median": float(np.median(ape)),
            "mape_p90": float(np.percentile(ape, 90)),
            "bias": float(bias.mean()),
            "ci_95": list(ci),
        }
        print(f"  {des:<40s} n={m.sum():5d}  MAPE={ape.mean():6.3f}%  med={np.median(ape):6.3f}%  P90={np.percentile(ape,90):6.2f}%  bias={bias.mean():+6.3f}%  CI=[{ci[0]:.3f}, {ci[1]:.3f}]")
    # combined
    ape = 100 * np.abs(pred - y) / y
    bias = 100 * (pred - y) / y
    rng = np.random.default_rng(0)
    boots = [ape[rng.integers(0, len(ape), len(ape))].mean() for _ in range(n_boot)]
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    results["combined"] = {
        "n": int(len(y)),
        "mape_mean": float(ape.mean()),
        "mape_median": float(np.median(ape)),
        "mape_p90": float(np.percentile(ape, 90)),
        "bias": float(bias.mean()),
        "ci_95": list(ci),
    }
    print(f"  {'COMBINED':<40s} n={len(y):5d}  MAPE={ape.mean():6.3f}%  med={np.median(ape):6.3f}%  P90={np.percentile(ape,90):6.2f}%  bias={bias.mean():+6.3f}%  CI=[{ci[0]:.3f}, {ci[1]:.3f}]")
    return results


def fit_R(verbose=True):
    print("\n" + "="*72, flush=True)
    print("=== total_R fit (canonical split) ===", flush=True)
    print("="*72, flush=True)
    train_dfs = [_load_R(d) for d in DESIGNS_TRAIN]
    test_dfs  = [_load_R(d) for d in DESIGNS_TEST]
    test_design_arr = np.concatenate([np.full(len(d), d["design"].iloc[0]) for d in test_dfs])
    test_combined = pd.concat(test_dfs, ignore_index=True)
    print(f"  train: {sum(len(d) for d in train_dfs):,} nets across {len(train_dfs)} designs", flush=True)
    for d in train_dfs:
        print(f"    {d['design'].iloc[0]:<40s} n={len(d):,}")
    print(f"  test:  {len(test_combined):,} nets across {len(test_dfs)} designs", flush=True)
    for d in test_dfs:
        print(f"    {d['design'].iloc[0]:<40s} n={len(d):,}")

    pref_lin = ["nsq_M", "rsq_M", "nvian_", "one", "n_pin_PIN", "n_pin_inst", "pin_nsq_M"]
    fcols_lin = _select(train_dfs + test_dfs, pref_lin)
    Xt = np.vstack([_design_matrix(d, fcols_lin) for d in train_dfs])
    yt = np.concatenate([d["R_gold"].values for d in train_dfs])
    Xs = _design_matrix(test_combined, fcols_lin)
    ys = test_combined["R_gold"].values

    c_lin = irls_nnls(Xt, yt)
    pred_lin_train = np.maximum(Xt @ c_lin, 1e-3)
    pred_lin_test  = np.maximum(Xs @ c_lin, 1e-3)
    print(f"\nStage 1 NNLS (fcols={len(fcols_lin)}):")
    s1_results = stats_per_design("Stage 1 (linear)", pred_lin_test, ys, test_design_arr)

    # Stage 2: 5-seed LGBM ensemble on relative residual + per-design 1-hot
    pref_full = ["nsq_M", "rsq_M", "nvian_", "one", "n_pin_PIN", "n_pin_inst", "pin_nsq_M",
                  "v6_", "n_segments", "n_zero_l_wire", "n_pins", "n_pins_total", "n_pins_matched"]
    fcols_full = _select(train_dfs + test_dfs, pref_full)
    Xt_full = np.vstack([_design_matrix(d, fcols_full) for d in train_dfs])
    Xs_full = _design_matrix(test_combined, fcols_full)
    n_des = len(DESIGNS_TRAIN)
    one_hot_train = np.zeros((Xt_full.shape[0], n_des))
    cum = 0
    for di, df in enumerate(train_dfs):
        one_hot_train[cum:cum+len(df), di] = 1.0
        cum += len(df)
    one_hot_test = np.full((Xs_full.shape[0], n_des), 1.0 / n_des)
    Xt_full = np.hstack([Xt_full, one_hot_train])
    Xs_full = np.hstack([Xs_full, one_hot_test])

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
    print(f"\n=== Stage 2 (5 LGBM seeds, R) ===", flush=True)
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
        print(f"  S2 seed {seed}: combined test MAPE = {np.mean(np.abs(ts-ys)/ys)*100:.4f}%", flush=True)

    pred_s2_train = pred_lin_train * (1 + np.mean(z_train_seeds, axis=0))
    pred_s2_test  = pred_lin_test  * (1 + np.mean(z_test_seeds, axis=0))
    s2_results = stats_per_design("Stage 2 (linear + 5-LGBM ensemble)", pred_s2_test, ys, test_design_arr)

    # Stage 3 stacking
    z3_train = (yt - pred_s2_train) / np.maximum(pred_s2_train, 1e-3)
    fcols_s3 = _select(train_dfs + test_dfs, pref_full[:-2])
    Xt_s3 = np.vstack([_design_matrix(d, fcols_s3) for d in train_dfs])
    Xs_s3 = _design_matrix(test_combined, fcols_s3)
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
    print(f"\n=== Stage 3 stacking (3 seeds, R) ===", flush=True)
    for seed in range(n_seeds3):
        cfg_s = {**cfg3, "random_state": 100+seed, "seed": 100+seed}
        gbm = lgb.LGBMRegressor(**cfg_s, n_jobs=-1, verbose=-1)
        gbm.fit(Xt_s3[train_mask], z3_train[train_mask], sample_weight=(1.0/yt)[train_mask],
                  eval_set=[(Xt_s3[val_idx], z3_train[val_idx])],
                  eval_sample_weight=[(1.0/yt)[val_idx]],
                  callbacks=[lgb.early_stopping(30)])
        z3_test_seeds.append(gbm.predict(Xs_s3))

    pred_final_test = pred_s2_test * (1 + np.mean(z3_test_seeds, axis=0))
    s3_results = stats_per_design("Stage 3 (stacked, FINAL)", pred_final_test, ys, test_design_arr)

    return {"stage1": s1_results, "stage2": s2_results, "stage3": s3_results,
            "n_train_nets": int(len(yt))}


def main():
    out = {"split": "canonical (configs/config.py)",
           "DESIGNS_TRAIN": DESIGNS_TRAIN,
           "DESIGNS_TEST":  DESIGNS_TEST}
    out["R"] = fit_R()

    out_path = _V3 / "outputs" / "canonical_split_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n=== SUMMARY (R, canonical split) ===", flush=True)
    print(f"  Stage 1 NNLS:")
    for d in DESIGNS_TEST + ["combined"]:
        if d in out["R"]["stage1"]:
            print(f"    {d:<40s} {out['R']['stage1'][d]['mape_mean']:.3f}%")
    print(f"  Stage 2 hybrid (linear + LGBM):")
    for d in DESIGNS_TEST + ["combined"]:
        if d in out["R"]["stage2"]:
            print(f"    {d:<40s} {out['R']['stage2'][d]['mape_mean']:.3f}%")
    print(f"  Stage 3 stacked (FINAL):")
    for d in DESIGNS_TEST + ["combined"]:
        if d in out["R"]["stage3"]:
            print(f"    {d:<40s} {out['R']['stage3'][d]['mape_mean']:.3f}%")
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
