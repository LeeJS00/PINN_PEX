"""Train per-pair LightGBM regressor on (target, aggressor) pair features.

Target: log(c_pair_fF + 1e-4)

At inference time:
  - Read tv80s pair features
  - Predict per-pair c_pair
  - Sum per target_net → predicted c_cpl_total

Combine with c_gnd predictor (use existing best ensemble's predicted c_gnd):
  c_gnd_pred = best_ensemble_total_pred - sum_of_SPEF_c_cpl_total
  Wait — at test time we don't have SPEF. Use this approach instead:
    1. Get current ensemble prediction for total cap (e.g. val-tuned 8.05%)
    2. Get pair-regressor prediction for c_cpl
    3. New prediction = α * ensemble + (1-α) * (gnd_estimate + pair_cpl)
  where gnd_estimate = ensemble - cpl_estimate_from_aggregate_features

Simpler: just blend pair-regressor's c_cpl_pred + existing ensemble's total prediction
in a calibrated way.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent
sys.path.insert(0, str(_WS))

from configs import cfg


FEATURE_COLS = [
    "target_layer", "agg_layer", "n_pairs",
    "min_dist", "mean_dist", "p25_dist", "p75_dist",
    "lat_overlap_total", "bs_overlap_total",
    "agg_n_cuboids", "agg_metal_area",
    "same_layer_pairs", "diff_layer_pairs",
    "target_n_cuboids", "target_metal_area",
    "target_eps_mean", "agg_eps_mean",
    "sum_inv_d", "sum_inv_d2",
]


def load_design_pairs(designs):
    dfs = []
    for d in designs:
        p = cfg.CACHE_DIR / "pair_features" / f"{d}.parquet"
        if not p.exists():
            print(f"  missing: {p}")
            continue
        df = pd.read_parquet(p)
        dfs.append(df)
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def main():
    train_pool = [d for d in cfg.TRAIN_DESIGNS if (cfg.CACHE_DIR / "pair_features" / f"{d}.parquet").exists()]
    val_pool = [d for d in cfg.VAL_DESIGNS if (cfg.CACHE_DIR / "pair_features" / f"{d}.parquet").exists()]
    test_pool = ["intel22_tv80s_f3"]
    print(f"train: {train_pool}\nval: {val_pool}")

    train = load_design_pairs(train_pool)
    val = load_design_pairs(val_pool)
    test = load_design_pairs(test_pool)
    if train is None or test is None:
        print("missing data"); return
    if val is None:
        val = test  # sanity
    print(f"train pairs: {len(train):,}, val pairs: {len(val):,}, test pairs: {len(test):,}")

    X_tr = train[FEATURE_COLS].to_numpy(np.float32)
    X_va = val[FEATURE_COLS].to_numpy(np.float32)
    X_te = test[FEATURE_COLS].to_numpy(np.float32)

    eps = 1e-4
    y_tr = np.log(train["c_pair_fF"].to_numpy(np.float64) + eps)
    y_va = np.log(val["c_pair_fF"].to_numpy(np.float64) + eps)

    import lightgbm as lgb
    train_set = lgb.Dataset(X_tr, y_tr)
    val_set   = lgb.Dataset(X_va, y_va, reference=train_set)
    booster = lgb.train(
        dict(objective="regression", metric="rmse",
             learning_rate=0.03, num_leaves=255, min_data_in_leaf=50,
             feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
             max_bin=511, seed=0, verbose=-1, n_jobs=8),
        train_set, num_boost_round=4000,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])

    pred_te = np.exp(booster.predict(X_te, num_iteration=booster.best_iteration)) - eps
    pred_te = np.clip(pred_te, 0.0, None)

    # Per-net sum
    test["c_pair_pred"] = pred_te
    sum_per_target = test.groupby(["design_name","target_net"])["c_pair_pred"].sum().reset_index()
    sum_per_target.columns = ["design_name", "net_name", "c_cpl_pred_pair"]

    out_path = cfg.REPORTS_DIR / "pair_regressor_cpl_test.csv"
    sum_per_target.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}: {len(sum_per_target)} target nets")

    # Compare against truth (if v3 features parquet has the labels)
    feat_path = cfg.CACHE_DIR / "features_v3" / "intel22_tv80s_f3.parquet"
    if feat_path.exists():
        feat = pd.read_parquet(feat_path)
        merged = feat.merge(sum_per_target, on=["design_name","net_name"], how="left")
        merged["c_cpl_pred_pair"] = merged["c_cpl_pred_pair"].fillna(0.0)
        ape_cpl = 100 * np.abs(merged["c_cpl_pred_pair"] - merged["c_cpl_total_fF"]) / np.maximum(merged["c_cpl_total_fF"], 1e-3)
        print(f"\nC_CPL prediction MAPE: mean={ape_cpl.mean():.3f}% med={ape_cpl.median():.3f}%")
        # Also evaluate (gnd_oracle + cpl_pair_predicted) MAPE on total_cap
        total_estimate = merged["c_gnd_fF"] + merged["c_cpl_pred_pair"]
        ape_tot = 100 * np.abs(total_estimate - merged["total_cap_fF"]) / np.maximum(merged["total_cap_fF"], 1e-3)
        print(f"(c_gnd_oracle + cpl_pair_pred) total MAPE: mean={ape_tot.mean():.3f}%  (this is an upper bound only, c_gnd is not predicted here)")


if __name__ == "__main__":
    main()
