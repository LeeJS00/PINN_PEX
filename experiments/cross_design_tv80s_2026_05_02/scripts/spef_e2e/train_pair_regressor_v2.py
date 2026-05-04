"""Train pair regressor v2: 5 LGBM (already done) + 5 CatBoost.

Reuses pair_features parquets from 9 train designs (~3.5M pairs).
Saves CatBoost models to output/spef_e2e/pair_regressor/ alongside LGBM.
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


def load_pair_df(designs):
    parts = []
    for d in designs:
        p = PAIR_FEAT_DIR / f"{d}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        parts.append(df)
        print(f"  loaded {d}: {len(df)} pairs")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main():
    print("Loading pair features...")
    train = load_pair_df(TRAIN_DESIGNS)
    val = load_pair_df(VAL_DESIGNS)
    test = load_pair_df(TEST_DESIGNS)

    fcols = [c for c in PAIR_FEAT_COLS if c in train.columns]
    print(f"\nFeatures: {len(fcols)}")
    print(f"Train: {len(train):,} pairs, Val: {len(val):,}, Test: {len(test):,}")

    eps = 1e-4
    y_tr = np.log(np.clip(train["c_pair_fF"].to_numpy(), eps, None))
    y_te_lin = test["c_pair_fF"].to_numpy()
    X_tr = train[fcols].to_numpy(np.float32)
    X_te = test[fcols].to_numpy(np.float32)

    if not val.empty:
        y_va = np.log(np.clip(val["c_pair_fF"].to_numpy(), eps, None))
        X_va = val[fcols].to_numpy(np.float32)
    else:
        n = len(X_tr); rng = np.random.default_rng(0)
        perm = rng.permutation(n); cut = n // 10
        v_idx = perm[:cut]; t_idx = perm[cut:]
        X_va, y_va = X_tr[v_idx], y_tr[v_idx]
        X_tr, y_tr = X_tr[t_idx], y_tr[t_idx]

    out_dir = cfg.OUTPUT_DIR / "spef_e2e" / "pair_regressor"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== CatBoost (5 seeds) ===", flush=True)
    from catboost import CatBoostRegressor
    for s in range(5):
        mdl = CatBoostRegressor(
            iterations=2000, learning_rate=0.05, depth=8,
            l2_leaf_reg=3.0, loss_function="RMSE", eval_metric="RMSE",
            random_seed=s, early_stopping_rounds=150, verbose=0,
            thread_count=8)
        mdl.fit(X_tr, y_tr, eval_set=(X_va, y_va))
        pt = np.exp(mdl.predict(X_te))
        nz = y_te_lin > 1e-6
        ape = 100 * np.abs(pt - y_te_lin) / np.maximum(y_te_lin, 1e-3)
        print(f"  CatBoost seed{s}: per-pair raw test_mape (nz)={ape[nz].mean():.3f}%")
        mdl.save_model(str(out_dir / f"cat_seed{s}.cbm"))

    # Per-pair val + test predictions for stratum fitting later
    out_pred = out_dir / "preds_per_pair"
    out_pred.mkdir(parents=True, exist_ok=True)

    # Save LGBM val + test predictions
    print("\n=== Generating per-model val + test pair preds ===", flush=True)
    import lightgbm as lgb
    for f in sorted(out_dir.glob("seed*.pkl")):
        with open(f, "rb") as fh:
            booster = pickle.load(fh)
        pv = np.exp(booster.predict(X_va, num_iteration=booster.best_iteration))
        pt = np.exp(booster.predict(X_te, num_iteration=booster.best_iteration))
        tag = f"lgbm_{f.stem}"
        pd.DataFrame({
            "target_net": val["target_net"].values if not val.empty else range(len(X_va)),
            "aggressor_net": val["aggressor_net"].values if not val.empty else range(len(X_va)),
            "y_true": np.exp(y_va), "y_pred": pv,
        }).to_csv(out_pred / f"{tag}__val.csv", index=False)
        pd.DataFrame({
            "target_net": test["target_net"].values,
            "aggressor_net": test["aggressor_net"].values,
            "y_true": y_te_lin, "y_pred": pt,
        }).to_csv(out_pred / f"{tag}__test.csv", index=False)

    for f in sorted(out_dir.glob("cat_seed*.cbm")):
        mdl = CatBoostRegressor(); mdl.load_model(str(f))
        pv = np.exp(mdl.predict(X_va))
        pt = np.exp(mdl.predict(X_te))
        tag = f"cat_{f.stem}"
        pd.DataFrame({
            "target_net": val["target_net"].values if not val.empty else range(len(X_va)),
            "aggressor_net": val["aggressor_net"].values if not val.empty else range(len(X_va)),
            "y_true": np.exp(y_va), "y_pred": pv,
        }).to_csv(out_pred / f"{tag}__val.csv", index=False)
        pd.DataFrame({
            "target_net": test["target_net"].values,
            "aggressor_net": test["aggressor_net"].values,
            "y_true": y_te_lin, "y_pred": pt,
        }).to_csv(out_pred / f"{tag}__test.csv", index=False)

    print(f"\nsaved val+test predictions to {out_pred}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal time: {time.time() - t0:.1f}s")
