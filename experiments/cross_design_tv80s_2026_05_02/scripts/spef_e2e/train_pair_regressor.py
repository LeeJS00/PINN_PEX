"""Train per-pair coupling regressor (ParaGraph-style edge model).

For each (target, aggressor) pair within geometric cutoff, predict
c_pair_fF directly. Trained on whatever train design pair_features
parquets are available; validated on nova; evaluated on tv80s.

The regressor is used to override the geometric heuristic in
distribute_cpl_to_pairs — instead of weighting by 1/d² × overlap × ε,
we weight by predicted c_pair, then RESCALE so Σc_pair_pred = c_cpl_total
(matches our total cap prediction).

This decouples per-pair allocation quality from total prediction quality.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent.parent
sys.path.insert(0, str(_WS))

from configs import cfg


PAIR_FEAT_DIR = cfg.CACHE_DIR / "pair_features"
TRAIN_DESIGNS = list(cfg.TRAIN_DESIGNS)
VAL_DESIGNS = ["intel22_nova_f3"]
TEST_DESIGNS = ["intel22_tv80s_f3"]


def load_pair_df(designs: list[str]) -> pd.DataFrame:
    parts = []
    for d in designs:
        p = PAIR_FEAT_DIR / f"{d}.parquet"
        if not p.exists():
            print(f"  skip {d} (no pair_features)")
            continue
        df = pd.read_parquet(p)
        parts.append(df)
        print(f"  loaded {d}: {len(df)} pairs")
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


PAIR_FEAT_COLS = [
    "n_pairs", "min_dist", "mean_dist", "p25_dist", "p75_dist",
    "lat_overlap_total", "bs_overlap_total",
    "agg_n_cuboids", "agg_metal_area",
    "same_layer_pairs", "diff_layer_pairs",
    "target_n_cuboids", "target_metal_area",
    "target_eps_mean", "agg_eps_mean",
    "sum_inv_d", "sum_inv_d2",
    "target_layer", "agg_layer",
]


def main():
    print("=== Loading pair features ===")
    print("Train:")
    train = load_pair_df(TRAIN_DESIGNS)
    print(f"\nVal:")
    val = load_pair_df(VAL_DESIGNS)
    print(f"\nTest:")
    test = load_pair_df(TEST_DESIGNS)

    if train.empty or test.empty:
        print("Insufficient data"); return

    fcols = [c for c in PAIR_FEAT_COLS if c in train.columns]
    print(f"\nFeatures: {len(fcols)}")
    print(f"Train: {len(train):,} pairs")
    print(f"Val:   {len(val):,} pairs" if not val.empty else "Val: empty")
    print(f"Test:  {len(test):,} pairs")

    # Target: log(c_pair_fF + eps)
    eps = 1e-4
    y_tr = np.log(np.clip(train["c_pair_fF"].to_numpy(), eps, None))
    y_te_lin = test["c_pair_fF"].to_numpy()

    X_tr = train[fcols].to_numpy(np.float32)
    X_te = test[fcols].to_numpy(np.float32)
    if not val.empty:
        y_va = np.log(np.clip(val["c_pair_fF"].to_numpy(), eps, None))
        X_va = val[fcols].to_numpy(np.float32)
    else:
        # Use 10% of train as held-out
        n = len(X_tr)
        rng = np.random.default_rng(0)
        perm = rng.permutation(n)
        cut = n // 10
        v_idx = perm[:cut]
        t_idx = perm[cut:]
        X_va, y_va = X_tr[v_idx], y_tr[v_idx]
        X_tr, y_tr = X_tr[t_idx], y_tr[t_idx]
        print(f"  no val — using 10% holdout: train={len(y_tr):,} val={len(y_va):,}")

    out_dir = cfg.OUTPUT_DIR / "spef_e2e" / "pair_regressor"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "fcols.json", "w") as f:
        json.dump(fcols, f)

    import lightgbm as lgb
    preds_v, preds_t = [], []
    for s in range(5):
        ts = lgb.Dataset(X_tr, y_tr)
        vs = lgb.Dataset(X_va, y_va, reference=ts)
        booster = lgb.train(
            dict(objective="regression", metric="rmse",
                 learning_rate=0.05, num_leaves=128, min_data_in_leaf=50,
                 feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
                 max_bin=511, seed=s, verbose=-1, n_jobs=8),
            ts, num_boost_round=2000, valid_sets=[vs],
            callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        pv = np.exp(booster.predict(X_va, num_iteration=booster.best_iteration))
        pt = np.exp(booster.predict(X_te, num_iteration=booster.best_iteration))
        preds_v.append(pv); preds_t.append(pt)
        with open(out_dir / f"seed{s}.pkl", "wb") as f:
            pickle.dump(booster, f)

    pv_mean = np.mean(preds_v, axis=0)
    pt_mean = np.mean(preds_t, axis=0)

    # Per-pair MAPE on test
    nz = y_te_lin > 1e-6
    ape = 100 * np.abs(pt_mean - y_te_lin) / np.maximum(y_te_lin, 1e-3)
    print(f"\n=== Test per-pair MAPE (n={nz.sum()} non-zero golden pairs) ===")
    print(f"  mean={ape[nz].mean():.3f}%  median={np.median(ape[nz]):.3f}%  p90={np.percentile(ape[nz], 90):.2f}%")

    # Stratified
    print("\n=== Stratified by golden c_pair (fF) ===")
    edges = [0, 0.001, 0.005, 0.01, 0.05, 0.1, np.inf]
    labels = ["<0.001", "0.001-0.005", "0.005-0.01", "0.01-0.05", "0.05-0.1", ">=0.1"]
    idx = np.clip(np.digitize(y_te_lin, edges) - 1, 0, len(labels) - 1)
    for i, lb in enumerate(labels):
        m = (idx == i) & nz
        if m.sum() > 0:
            print(f"  {lb:>14s}: n={m.sum():>6d}  mape_mean={ape[m].mean():.2f}%  median={np.median(ape[m]):.2f}%")

    # Save predictions
    out_csv = out_dir / "test_predictions.csv"
    out_df = pd.DataFrame({
        "design_name": test["design_name"].values,
        "target_net": test["target_net"].values,
        "aggressor_net": test["aggressor_net"].values,
        "c_pair_pred": pt_mean,
        "c_pair_true": y_te_lin,
    })
    out_df.to_csv(out_csv, index=False)
    print(f"\nsaved {out_csv}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal time: {time.time() - t0:.1f}s")
