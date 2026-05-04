"""Train pair regressor with L1 (MAE) objective on log target.

L1 in log-space ≈ geometric MAPE — more aligned with our per-pair MAPE metric
than the previous RMSE objective.
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
    print(f"Features: {len(fcols)}")
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

    out_dir = cfg.OUTPUT_DIR / "spef_e2e" / "pair_regressor_l1"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "fcols.json", "w") as f:
        json.dump(fcols, f)

    import lightgbm as lgb
    print("\n=== LightGBM L1 (5 seeds) ===")
    for s in range(5):
        ts = lgb.Dataset(X_tr, y_tr)
        vs = lgb.Dataset(X_va, y_va, reference=ts)
        booster = lgb.train(
            dict(objective="regression_l1", metric="mae",  # L1 / MAE
                 learning_rate=0.05, num_leaves=128, min_data_in_leaf=50,
                 feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
                 max_bin=511, seed=s, verbose=-1, n_jobs=8),
            ts, num_boost_round=2000, valid_sets=[vs],
            callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        pt = np.exp(booster.predict(X_te, num_iteration=booster.best_iteration))
        nz = y_te_lin > 1e-6
        ape = 100 * np.abs(pt - y_te_lin) / np.maximum(y_te_lin, 1e-3)
        print(f"  L1 seed{s}: per-pair MAPE (nz)={ape[nz].mean():.3f}% best_iter={booster.best_iteration}")
        with open(out_dir / f"seed{s}.pkl", "wb") as f:
            pickle.dump(booster, f)

    # Per-pair val + test predictions for stratum fitting
    out_pred = out_dir / "preds_per_pair"
    out_pred.mkdir(parents=True, exist_ok=True)

    for f_pkl in sorted(out_dir.glob("seed*.pkl")):
        with open(f_pkl, "rb") as fh:
            booster = pickle.load(fh)
        pv = np.exp(booster.predict(X_va, num_iteration=booster.best_iteration))
        pt = np.exp(booster.predict(X_te, num_iteration=booster.best_iteration))
        tag = f"l1_{f_pkl.stem}"
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
    print(f"saved {out_pred}")

    # Ensemble per-pair MAPE
    preds_t = []
    for f_pkl in sorted(out_dir.glob("seed*.pkl")):
        with open(f_pkl, "rb") as fh:
            booster = pickle.load(fh)
        preds_t.append(np.exp(booster.predict(X_te, num_iteration=booster.best_iteration)))
    ens = np.mean(preds_t, axis=0)
    nz = y_te_lin > 1e-6
    ape = 100 * np.abs(ens - y_te_lin) / np.maximum(y_te_lin, 1e-3)
    print(f"\n=== L1 ensemble (5 seeds) per-pair raw MAPE on test (nz) ===")
    print(f"  mean={ape[nz].mean():.3f}%  median={np.median(ape[nz]):.3f}%  p90={np.percentile(ape[nz], 90):.2f}%")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal time: {time.time() - t0:.1f}s")
