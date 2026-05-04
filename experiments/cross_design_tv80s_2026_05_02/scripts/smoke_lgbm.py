"""Quick LightGBM smoke test on whatever parquet files exist."""
from __future__ import annotations
import os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent
sys.path.insert(0, str(_WS))

from configs import cfg
from src.data_loader import _select_feature_cols, mape_per_net, report_mape


def main():
    cache = cfg.CACHE_DIR / "features"
    avail = {p.stem for p in cache.glob("*.parquet")}
    print("available parquets:", sorted(avail))

    # Use whatever train designs are available, ibex as val, tv80s as test
    train_pool = [d for d in cfg.TRAIN_DESIGNS if d in avail and d != "intel22_ibex_core_f3"]
    val_pool = ["intel22_ibex_core_f3"] if "intel22_ibex_core_f3" in avail else []
    if not val_pool:
        val_pool = [train_pool.pop(0)] if train_pool else []
    test_pool = ["intel22_tv80s_f3"] if "intel22_tv80s_f3" in avail else []
    if not test_pool:
        print("no tv80s parquet"); return

    print(f"train: {train_pool}")
    print(f"val:   {val_pool}")
    print(f"test:  {test_pool}")

    train = pd.concat([pd.read_parquet(cache / f"{d}.parquet").assign(design_name=d) for d in train_pool], ignore_index=True)
    val   = pd.concat([pd.read_parquet(cache / f"{d}.parquet").assign(design_name=d) for d in val_pool],   ignore_index=True)
    test  = pd.concat([pd.read_parquet(cache / f"{d}.parquet").assign(design_name=d) for d in test_pool],  ignore_index=True)

    fcols = _select_feature_cols(train)
    print(f"feature dim: {len(fcols)}, train rows: {len(train):,}, val: {len(val):,}, test: {len(test):,}")

    X_tr = train[fcols].to_numpy(dtype=np.float32)
    X_va = val[fcols].to_numpy(dtype=np.float32)
    X_te = test[fcols].to_numpy(dtype=np.float32)
    y_tr = np.log(np.maximum(train["total_cap_fF"].to_numpy(np.float64), 1e-4))
    y_va = np.log(np.maximum(val["total_cap_fF"].to_numpy(np.float64),   1e-4))

    train_set = lgb.Dataset(X_tr, y_tr)
    val_set   = lgb.Dataset(X_va, y_va, reference=train_set)
    booster = lgb.train(
        dict(objective="regression", metric="rmse", learning_rate=0.05,
             num_leaves=127, min_data_in_leaf=20, feature_fraction=0.9,
             bagging_fraction=0.9, bagging_freq=5, max_bin=255,
             verbose=-1, seed=0, n_jobs=8),
        train_set, num_boost_round=2000, valid_sets=[val_set],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )
    yhat_va = np.exp(booster.predict(X_va, num_iteration=booster.best_iteration))
    yhat_te = np.exp(booster.predict(X_te, num_iteration=booster.best_iteration))
    print()
    report_mape(val["total_cap_fF"].to_numpy(),  yhat_va, "val_lgbm")
    report_mape(test["total_cap_fF"].to_numpy(), yhat_te, "test_lgbm tv80s")


if __name__ == "__main__":
    main()
