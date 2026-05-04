"""
Train a specialty LGBM model only on LARGE nets (target_cap >= 1 fF).

The intuition: regular GBDT under-predicts large nets (12-13% MAPE on cap >=1fF
buckets, vs 7% on 0.1-0.5 fF). A specialised model for large nets, blended
with the general model only for those same large nets, may improve the tail.

Output: predictions for tv80s, blended with the existing general predictions.
"""
from __future__ import annotations

import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent
sys.path.insert(0, str(_WS))

from configs import cfg
from src.data_loader import _select_feature_cols, report_mape


def main():
    cache = cfg.CACHE_DIR / "features_v2"
    avail = {p.stem for p in cache.glob("*.parquet")}
    train_pool = [d for d in cfg.TRAIN_DESIGNS if d in avail]
    val_pool = ["intel22_nova_f3"] if "intel22_nova_f3" in avail else ["intel22_ibex_core_f3"]
    if val_pool[0] in train_pool:
        train_pool.remove(val_pool[0])
    print(f"train: {train_pool}\nval: {val_pool}")

    train = pd.concat([
        pd.read_parquet(cache / f"{d}.parquet").assign(design_name=d) for d in train_pool
    ], ignore_index=True)
    val   = pd.concat([
        pd.read_parquet(cache / f"{d}.parquet").assign(design_name=d) for d in val_pool
    ], ignore_index=True)
    test  = pd.read_parquet(cache / "intel22_tv80s_f3.parquet").assign(design_name="intel22_tv80s_f3")

    fcols = _select_feature_cols(train)

    # Filter LARGE nets (>= 1 fF)
    cap_threshold = 1.0
    train_large = train[train["total_cap_fF"] >= cap_threshold].reset_index(drop=True)
    val_large   = val  [val  ["total_cap_fF"] >= cap_threshold].reset_index(drop=True)
    print(f"large train: {len(train_large)}, large val: {len(val_large)}, test (all): {len(test)}")

    X_tr = train_large[fcols].to_numpy(np.float32)
    X_va = val_large  [fcols].to_numpy(np.float32)
    X_te = test       [fcols].to_numpy(np.float32)

    y_tr = np.log(train_large["total_cap_fF"].clip(lower=1e-4).to_numpy())
    y_va = np.log(val_large  ["total_cap_fF"].clip(lower=1e-4).to_numpy())

    import lightgbm as lgb
    out_dir = cfg.OUTPUT_DIR / "specialist_large"
    out_dir.mkdir(parents=True, exist_ok=True)

    for s in [0, 1, 2]:
        ts = lgb.Dataset(X_tr, y_tr); vs = lgb.Dataset(X_va, y_va, reference=ts)
        booster = lgb.train(
            dict(objective="regression", metric="rmse",
                 learning_rate=0.025, num_leaves=255, min_data_in_leaf=10,
                 feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
                 max_bin=511, seed=s, verbose=-1, n_jobs=8),
            ts, num_boost_round=4000, valid_sets=[vs],
            callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        pv = np.exp(booster.predict(X_va, num_iteration=booster.best_iteration))
        pt = np.exp(booster.predict(X_te, num_iteration=booster.best_iteration))
        report_mape(val_large["total_cap_fF"].to_numpy(), pv, f"large s{s} val")
        report_mape(test["total_cap_fF"].to_numpy(), pt, f"large s{s} test (all-net APE)")

        pd.DataFrame({"design_name": test["design_name"].values,
                      "net_name":    test["net_name"].values,
                      "y_true":      test["total_cap_fF"].values,
                      "y_pred":      pt}).to_csv(out_dir / f"seed{s}__test.csv", index=False)
        # Note: val csv uses LARGE nets only — not directly stackable with full val
        pd.DataFrame({"design_name": val_large["design_name"].values,
                      "net_name":    val_large["net_name"].values,
                      "y_true":      val_large["total_cap_fF"].values,
                      "y_pred":      pv}).to_csv(out_dir / f"seed{s}__val_large.csv", index=False)


if __name__ == "__main__":
    main()
