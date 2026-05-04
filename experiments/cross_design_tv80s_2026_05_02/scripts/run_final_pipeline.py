"""
Final pipeline — runs once all v2 parquets are ready.

  - Loads 9 train + nova val + tv80s test
  - Adds derived features
  - Trains 5 seeds × {LGBM, XGBoost, CatBoost} × {direct, residual} = 30 GBDT models
  - Saves preds and a summary CSV

Designed to chain with stack_eval.py + the existing MLP outputs.
"""
from __future__ import annotations

import argparse
import gc
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent
sys.path.insert(0, str(_WS))

from configs import cfg
from src.data_loader import _select_feature_cols, report_mape
from src.derived_feats import add_derived


def _load(d: str, cache: Path) -> pd.DataFrame:
    df = pd.read_parquet(cache / f"{d}.parquet")
    df["design_name"] = d
    return df


def assemble(cache: Path):
    avail = {p.stem for p in cache.glob("*.parquet")}
    if "intel22_tv80s_f3" not in avail:
        raise SystemExit("test design tv80s not built")
    train_pool = [d for d in cfg.TRAIN_DESIGNS if d in avail]
    val_pool = [d for d in cfg.VAL_DESIGNS if d in avail]
    if not val_pool:
        if "intel22_ibex_core_f3" in avail:
            val_pool = ["intel22_ibex_core_f3"]
        else:
            val_pool = [train_pool[-1]]
        for v in val_pool:
            if v in train_pool:
                train_pool.remove(v)
    print(f"avail: {sorted(avail)}")
    print(f"train: {train_pool}\nval:   {val_pool}")
    train = pd.concat([_load(d, cache) for d in train_pool], ignore_index=True)
    val   = pd.concat([_load(d, cache) for d in val_pool],   ignore_index=True)
    test  = _load("intel22_tv80s_f3", cache)
    return train, val, test


def fit_lgbm(X_tr, y_tr, X_va, y_va, seed):
    import lightgbm as lgb
    ts = lgb.Dataset(X_tr, y_tr); vs = lgb.Dataset(X_va, y_va, reference=ts)
    booster = lgb.train(
        dict(objective="regression", metric="rmse",
             learning_rate=0.025, num_leaves=511, min_data_in_leaf=15,
             feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
             max_bin=511, seed=seed, verbose=-1, n_jobs=8),
        ts, num_boost_round=6000, valid_sets=[vs],
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)]
    )
    return booster


def fit_xgb(X_tr, y_tr, X_va, y_va, seed):
    import xgboost as xgb
    dt = xgb.DMatrix(X_tr, label=y_tr); dv = xgb.DMatrix(X_va, label=y_va)
    params = dict(objective="reg:squarederror", eval_metric="rmse",
                  eta=0.025, max_depth=12, min_child_weight=4,
                  subsample=0.85, colsample_bytree=0.85,
                  tree_method="hist", seed=seed, verbosity=0, nthread=8)
    return xgb.train(params, dt, num_boost_round=6000, evals=[(dv, "val")],
                     early_stopping_rounds=200, verbose_eval=0)


def fit_cat(X_tr, y_tr, X_va, y_va, seed):
    from catboost import CatBoostRegressor
    booster = CatBoostRegressor(iterations=6000, learning_rate=0.025, depth=10,
                                l2_leaf_reg=4.0, loss_function="RMSE",
                                random_seed=seed, verbose=0, task_type="CPU", thread_count=8)
    booster.fit(X_tr, y_tr, eval_set=(X_va, y_va),
                early_stopping_rounds=200, use_best_model=True)
    return booster


def predict_test(booster, X, model_name):
    if model_name == "lgbm":
        return booster.predict(X, num_iteration=booster.best_iteration)
    if model_name == "xgb":
        import xgboost as xgb
        return booster.predict(xgb.DMatrix(X), iteration_range=(0, booster.best_iteration+1))
    return booster.predict(X)


def run_one(strat, mname, seed, X_tr, X_va, X_te, train, val, test, fcols, out_dir):
    eps = 1e-4
    y_tr_lin = train["total_cap_fF"].to_numpy(np.float64)
    y_va_lin = val  ["total_cap_fF"].to_numpy(np.float64)
    y_te_lin = test ["total_cap_fF"].to_numpy(np.float64)
    if strat == "direct":
        y_tr = np.log(y_tr_lin.clip(min=eps))
        y_va = np.log(y_va_lin.clip(min=eps))
        ce = None
    else:  # residual
        ct = train["compact_total_fF"].to_numpy(np.float64).clip(min=eps)
        cv = val  ["compact_total_fF"].to_numpy(np.float64).clip(min=eps)
        ce = test ["compact_total_fF"].to_numpy(np.float64).clip(min=eps)
        y_tr = np.log(y_tr_lin.clip(min=eps)) - np.log(ct)
        y_va = np.log(y_va_lin.clip(min=eps)) - np.log(cv)

    if mname == "lgbm":
        booster = fit_lgbm(X_tr, y_tr, X_va, y_va, seed)
    elif mname == "xgb":
        booster = fit_xgb(X_tr, y_tr, X_va, y_va, seed)
    else:
        booster = fit_cat(X_tr, y_tr, X_va, y_va, seed)

    pv = predict_test(booster, X_va, mname)
    pt = predict_test(booster, X_te, mname)
    if strat == "residual":
        yhat_va = np.exp(pv) * (val["compact_total_fF"].clip(lower=eps).to_numpy())
        yhat_te = np.exp(pt) * ce
    else:
        yhat_va = np.exp(pv); yhat_te = np.exp(pt)

    tag = f"{strat}-{mname}-s{seed}"
    vm = report_mape(y_va_lin, yhat_va, f"{tag} val")
    tm = report_mape(y_te_lin, yhat_te, f"{tag} test")
    sub = out_dir / f"{strat}_{mname}"
    sub.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "design_name": test["design_name"].values,
        "net_name":    test["net_name"].values,
        "y_true":      y_te_lin,
        "y_pred":      yhat_te,
    }).to_csv(sub / f"seed{seed}__test.csv", index=False)
    pd.DataFrame({
        "design_name": val["design_name"].values,
        "net_name":    val["net_name"].values,
        "y_true":      y_va_lin,
        "y_pred":      yhat_va,
    }).to_csv(sub / f"seed{seed}__val.csv", index=False)
    return {"strategy": strat, "model": mname, "seed": seed,
            **{f"val_{k}":v  for k,v in vm.items()},
            **{f"test_{k}":v for k,v in tm.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="features_v2")
    ap.add_argument("--out",   default="final_gbdt")
    ap.add_argument("--seeds", nargs="+", type=int, default=cfg.SEEDS)
    ap.add_argument("--models", nargs="+", default=["lgbm", "xgb", "cat"])
    ap.add_argument("--strategies", nargs="+", default=["direct", "residual"])
    ap.add_argument("--add-derived", action="store_true",
                    help="add interaction/derived features")
    args = ap.parse_args()

    cache = cfg.CACHE_DIR / args.cache
    out_dir = cfg.OUTPUT_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    train, val, test = assemble(cache)
    if args.add_derived:
        train = add_derived(train)
        val   = add_derived(val)
        test  = add_derived(test)

    fcols = _select_feature_cols(train)
    print(f"features: {len(fcols)}, train: {len(train):,}, val: {len(val):,}, test: {len(test):,}")

    X_tr = train[fcols].to_numpy(np.float32)
    X_va = val  [fcols].to_numpy(np.float32)
    X_te = test [fcols].to_numpy(np.float32)

    summary = []
    t_start = time.time()
    for strat in args.strategies:
        for mname in args.models:
            for seed in args.seeds:
                t0 = time.time()
                print(f"\n=== {strat}-{mname}-s{seed} ===  (elapsed {time.time()-t_start:.0f}s)")
                try:
                    rec = run_one(strat, mname, seed, X_tr, X_va, X_te, train, val, test, fcols, out_dir)
                    rec["wall_sec"] = time.time() - t0
                    summary.append(rec)
                except Exception:
                    import traceback; traceback.print_exc()
                gc.collect()

    df = pd.DataFrame(summary)
    df.to_csv(out_dir / "summary.csv", index=False)
    if not df.empty:
        print("\nGroup-by strategy + model:")
        print(df.groupby(["strategy", "model"])[["test_mape_mean","test_mape_median","test_mape_p90"]].mean().round(3))
    print(f"\nTotal wall: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
