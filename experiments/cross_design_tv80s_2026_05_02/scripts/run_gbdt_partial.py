"""Run lgbm/xgb/cat with whatever cached parquets are available.

Treats `intel22_ibex_core_f3` as val until nova arrives. Saves preds and
summary to output/preds_partial.
"""
from __future__ import annotations
import json, os, pickle, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent
sys.path.insert(0, str(_WS))

from configs import cfg
from src.data_loader import _select_feature_cols, report_mape


def _load(d: str, cache: Path) -> pd.DataFrame:
    df = pd.read_parquet(cache / f"{d}.parquet")
    df["design_name"] = d
    return df


def assemble_split(cache: Path, val_fallback: str = "intel22_ibex_core_f3"):
    avail = {p.stem for p in cache.glob("*.parquet")}
    print("available parquets:", sorted(avail))

    train_pool = [d for d in cfg.TRAIN_DESIGNS if d in avail and d != val_fallback]
    val_design = "intel22_nova_f3" if "intel22_nova_f3" in avail else val_fallback
    if val_design in train_pool:
        train_pool.remove(val_design)
    test_design = "intel22_tv80s_f3"
    if test_design not in avail:
        raise SystemExit("test design tv80s not yet built")
    print(f"train designs: {train_pool}")
    print(f"val design   : {val_design}")
    print(f"test design  : {test_design}")
    train = pd.concat([_load(d, cache) for d in train_pool], ignore_index=True)
    val   = _load(val_design, cache)
    test  = _load(test_design, cache)
    return train, val, test


def main():
    seeds = [0, 1, 2, 3, 4]
    cache = cfg.CACHE_DIR / "features"
    train, val, test = assemble_split(cache)

    fcols = _select_feature_cols(train)
    target = "total_cap_fF"
    print(f"features: {len(fcols)}, train rows: {len(train):,}, val: {len(val):,}, test: {len(test):,}")

    X_tr = train[fcols].to_numpy(np.float32)
    X_va = val[fcols].to_numpy(np.float32)
    X_te = test[fcols].to_numpy(np.float32)
    y_tr_lin = train[target].to_numpy(np.float64)
    y_va_lin = val[target].to_numpy(np.float64)
    y_te_lin = test[target].to_numpy(np.float64)
    y_tr = np.log(np.maximum(y_tr_lin, 1e-4))
    y_va = np.log(np.maximum(y_va_lin, 1e-4))

    out_root = cfg.OUTPUT_DIR / "preds_partial"
    out_root.mkdir(parents=True, exist_ok=True)
    summary = []
    preds_test_all = {}

    # ----- LightGBM -----
    import lightgbm as lgb
    print("\n=== LightGBM ===")
    for s in seeds:
        t0 = time.time()
        train_set = lgb.Dataset(X_tr, y_tr)
        val_set   = lgb.Dataset(X_va, y_va, reference=train_set)
        params = dict(objective="regression", metric="rmse",
                      learning_rate=0.03, num_leaves=255, min_data_in_leaf=20,
                      feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
                      max_bin=511, seed=s, verbose=-1, n_jobs=12)
        booster = lgb.train(params, train_set, num_boost_round=4000, valid_sets=[val_set],
                            callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        pv = np.exp(booster.predict(X_va, num_iteration=booster.best_iteration))
        pt = np.exp(booster.predict(X_te, num_iteration=booster.best_iteration))
        vm = report_mape(y_va_lin, pv, f"lgbm-val s{s}")
        tm = report_mape(y_te_lin, pt, f"lgbm-test s{s}")
        preds_test_all[("lgbm", s)] = pt
        summary.append({"model":"lgbm","seed":s, **{f"val_{k}":v for k,v in vm.items()}, **{f"test_{k}":v for k,v in tm.items()}, "wall_sec": time.time()-t0})

    # ----- XGBoost (CPU hist for safety) -----
    import xgboost as xgb
    print("\n=== XGBoost ===")
    for s in seeds:
        t0 = time.time()
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval   = xgb.DMatrix(X_va, label=y_va)
        dtest  = xgb.DMatrix(X_te)
        params = dict(objective="reg:squarederror", eval_metric="rmse",
                      eta=0.03, max_depth=10, min_child_weight=5,
                      subsample=0.85, colsample_bytree=0.85,
                      tree_method="hist", seed=s, verbosity=0, nthread=12)
        booster = xgb.train(params, dtrain, num_boost_round=4000,
                            evals=[(dval, "val")], early_stopping_rounds=150, verbose_eval=0)
        pv = np.exp(booster.predict(dval,  iteration_range=(0, booster.best_iteration+1)))
        pt = np.exp(booster.predict(dtest, iteration_range=(0, booster.best_iteration+1)))
        vm = report_mape(y_va_lin, pv, f"xgb-val s{s}")
        tm = report_mape(y_te_lin, pt, f"xgb-test s{s}")
        preds_test_all[("xgb", s)] = pt
        summary.append({"model":"xgb","seed":s, **{f"val_{k}":v for k,v in vm.items()}, **{f"test_{k}":v for k,v in tm.items()}, "wall_sec": time.time()-t0})

    # ----- CatBoost (CPU here for safety, GPU 2 if available) -----
    from catboost import CatBoostRegressor
    print("\n=== CatBoost ===")
    for s in seeds:
        t0 = time.time()
        try:
            model = CatBoostRegressor(iterations=4000, learning_rate=0.03, depth=10,
                                       l2_leaf_reg=4.0, loss_function="RMSE",
                                       random_seed=s, verbose=0, task_type="CPU", thread_count=12)
            model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=150, use_best_model=True)
        except Exception as e:
            print(f"  catboost s{s} failed: {e}")
            continue
        pv = np.exp(model.predict(X_va))
        pt = np.exp(model.predict(X_te))
        vm = report_mape(y_va_lin, pv, f"cat-val s{s}")
        tm = report_mape(y_te_lin, pt, f"cat-test s{s}")
        preds_test_all[("cat", s)] = pt
        summary.append({"model":"cat","seed":s, **{f"val_{k}":v for k,v in vm.items()}, **{f"test_{k}":v for k,v in tm.items()}, "wall_sec": time.time()-t0})

    df_sum = pd.DataFrame(summary)
    df_sum.to_csv(out_root / "summary.csv", index=False)
    print("\nWritten", out_root / "summary.csv")
    print(df_sum.groupby("model")[["test_mape_mean","test_mape_median"]].mean().round(3))

    # Save preds
    for (m, s), p in preds_test_all.items():
        pd.DataFrame({
            "design_name": test["design_name"].values,
            "net_name": test["net_name"].values,
            "y_true": y_te_lin,
            "y_pred": p,
        }).to_csv(out_root / f"{m}_seed{s}_test.csv", index=False)


if __name__ == "__main__":
    main()
