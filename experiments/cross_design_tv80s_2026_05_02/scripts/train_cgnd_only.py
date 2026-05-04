"""Train LGBM models that predict log(c_gnd_fF) directly.

Output 5 seeds, used as the c_gnd component when combining with pair regressor.
"""
from __future__ import annotations
import sys, os
from pathlib import Path
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent
sys.path.insert(0, str(_WS))

from configs import cfg
from src.data_loader import _select_feature_cols, report_mape


def _load(d, cache):
    df = pd.read_parquet(cache / f"{d}.parquet")
    df["design_name"] = d
    return df


def main():
    cache = cfg.CACHE_DIR / "features_v3"
    train_pool = list(cfg.TRAIN_DESIGNS)
    val_pool = ["intel22_nova_f3"]
    test_pool = ["intel22_tv80s_f3"]

    train = pd.concat([_load(d, cache) for d in train_pool], ignore_index=True)
    val = pd.concat([_load(d, cache) for d in val_pool], ignore_index=True)
    test = pd.concat([_load(d, cache) for d in test_pool], ignore_index=True)

    fcols = _select_feature_cols(train)
    print(f"features: {len(fcols)}, train: {len(train):,}, val: {len(val):,}, test: {len(test):,}")

    X_tr = train[fcols].to_numpy(np.float32)
    X_va = val[fcols].to_numpy(np.float32)
    X_te = test[fcols].to_numpy(np.float32)

    y_tr_lin = train["c_gnd_fF"].to_numpy(np.float64)
    y_va_lin = val["c_gnd_fF"].to_numpy(np.float64)
    y_te_lin = test["c_gnd_fF"].to_numpy(np.float64)
    eps = 1e-4

    out_dir = cfg.OUTPUT_DIR / "cgnd_only"
    out_dir.mkdir(parents=True, exist_ok=True)

    import lightgbm as lgb
    for s in [0, 1, 2, 3, 4]:
        ts = lgb.Dataset(X_tr, np.log(y_tr_lin.clip(min=eps)))
        vs = lgb.Dataset(X_va, np.log(y_va_lin.clip(min=eps)), reference=ts)
        booster = lgb.train(
            dict(objective="regression", metric="rmse",
                 learning_rate=0.03, num_leaves=255, min_data_in_leaf=20,
                 feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
                 max_bin=511, seed=s, verbose=-1, n_jobs=8),
            ts, num_boost_round=4000, valid_sets=[vs],
            callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        pv = np.exp(booster.predict(X_va, num_iteration=booster.best_iteration))
        pt = np.exp(booster.predict(X_te, num_iteration=booster.best_iteration))
        print(f"\n--- seed{s} ---")
        report_mape(y_va_lin, pv, "c_gnd val nova")
        report_mape(y_te_lin, pt, "c_gnd test tv80s")
        pd.DataFrame({"design_name": test["design_name"].values,
                      "net_name": test["net_name"].values,
                      "y_true_c_gnd": y_te_lin,
                      "y_pred_c_gnd": pt}).to_csv(out_dir / f"seed{s}__test.csv", index=False)
        pd.DataFrame({"design_name": val["design_name"].values,
                      "net_name": val["net_name"].values,
                      "y_true_c_gnd": y_va_lin,
                      "y_pred_c_gnd": pv}).to_csv(out_dir / f"seed{s}__val.csv", index=False)


if __name__ == "__main__":
    main()
