"""Full GBDT 5-seed run on v2 features.

Uses 9 train designs (or whatever's available), nova as val (or fallback ibex),
tv80s as test. Trains LightGBM + XGBoost + CatBoost on log(total_cap+1e-4)
and on residual-from-compact log(true / max(compact, 1e-4)). Saves preds.
"""
from __future__ import annotations

import argparse
import gc
import json
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


def _load(d: str, cache: Path) -> pd.DataFrame:
    df = pd.read_parquet(cache / f"{d}.parquet")
    df["design_name"] = d
    return df


def assemble_split(cache: Path):
    avail = {p.stem for p in cache.glob("*.parquet")}
    if "intel22_tv80s_f3" not in avail:
        raise SystemExit("test design tv80s not built")
    train_pool = [d for d in cfg.TRAIN_DESIGNS if d in avail]
    val_pool = [d for d in cfg.VAL_DESIGNS if d in avail]
    if not val_pool:
        # Use largest available training design as val if nova missing
        if "intel22_ldpc_decoder_802_3an_f3" in avail:
            val_pool = ["intel22_ldpc_decoder_802_3an_f3"]
        elif "intel22_ibex_core_f3" in avail:
            val_pool = ["intel22_ibex_core_f3"]
        else:
            val_pool = [train_pool[0]]
        for v in val_pool:
            if v in train_pool:
                train_pool.remove(v)
    print(f"avail: {sorted(avail)}")
    print(f"train: {train_pool}")
    print(f"val:   {val_pool}")
    train = pd.concat([_load(d, cache) for d in train_pool], ignore_index=True)
    val   = pd.concat([_load(d, cache) for d in val_pool],   ignore_index=True)
    test  = _load("intel22_tv80s_f3", cache)
    return train, val, test


def run(target: str, models, seeds, cache_dir: Path, out_dir: Path,
        residual: bool, weight_by_log_cap: bool):
    train, val, test = assemble_split(cache_dir)
    fcols = _select_feature_cols(train)
    print(f"features: {len(fcols)}, train: {len(train):,}, val: {len(val):,}, test: {len(test):,}")

    X_tr = train[fcols].to_numpy(np.float32)
    X_va = val[fcols].to_numpy(np.float32)
    X_te = test[fcols].to_numpy(np.float32)
    y_tr_lin = train[target].to_numpy(np.float64)
    y_va_lin = val[target].to_numpy(np.float64)
    y_te_lin = test[target].to_numpy(np.float64)
    eps = 1e-4

    if residual:
        # predict log(true) - log(compact) i.e. log(true / compact)
        compact_tr = train["compact_total_fF"].to_numpy(np.float64).clip(min=eps)
        compact_va = val  ["compact_total_fF"].to_numpy(np.float64).clip(min=eps)
        compact_te = test ["compact_total_fF"].to_numpy(np.float64).clip(min=eps)
        y_tr = np.log(np.maximum(y_tr_lin, eps)) - np.log(compact_tr)
        y_va = np.log(np.maximum(y_va_lin, eps)) - np.log(compact_va)
    else:
        y_tr = np.log(np.maximum(y_tr_lin, eps))
        y_va = np.log(np.maximum(y_va_lin, eps))

    sw = None
    if weight_by_log_cap:
        # Weight rows so small-cap nets matter more (since MAPE penalises them)
        # Use 1.0 for all rows but boost outliers
        sw = np.ones_like(y_tr_lin, dtype=np.float64)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for model_name in models:
        model_dir = out_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        for s in seeds:
            t0 = time.time()
            tag = f"{model_name}-{'res' if residual else 'lin'}-s{s}"
            print(f"\n=== {tag} ===")
            try:
                if model_name == "lgbm":
                    import lightgbm as lgb
                    train_set = lgb.Dataset(X_tr, y_tr, weight=sw)
                    val_set   = lgb.Dataset(X_va, y_va, reference=train_set)
                    params = dict(objective="regression", metric="rmse",
                                  learning_rate=0.03, num_leaves=255, min_data_in_leaf=20,
                                  feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
                                  max_bin=511, seed=s, verbose=-1, n_jobs=8)
                    booster = lgb.train(params, train_set, num_boost_round=4000,
                                        valid_sets=[val_set],
                                        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
                    pv = booster.predict(X_va, num_iteration=booster.best_iteration)
                    pt = booster.predict(X_te, num_iteration=booster.best_iteration)
                elif model_name == "xgb":
                    import xgboost as xgb
                    dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=sw)
                    dval   = xgb.DMatrix(X_va, label=y_va)
                    dtest  = xgb.DMatrix(X_te)
                    params = dict(objective="reg:squarederror", eval_metric="rmse",
                                  eta=0.03, max_depth=10, min_child_weight=5,
                                  subsample=0.85, colsample_bytree=0.85,
                                  tree_method="hist", seed=s, verbosity=0, nthread=8)
                    booster = xgb.train(params, dtrain, num_boost_round=4000,
                                        evals=[(dval, "val")],
                                        early_stopping_rounds=150, verbose_eval=0)
                    pv = booster.predict(dval,  iteration_range=(0, booster.best_iteration+1))
                    pt = booster.predict(dtest, iteration_range=(0, booster.best_iteration+1))
                elif model_name == "cat":
                    from catboost import CatBoostRegressor
                    booster = CatBoostRegressor(iterations=4000, learning_rate=0.03, depth=10,
                                                l2_leaf_reg=4.0, loss_function="RMSE",
                                                random_seed=s, verbose=0, task_type="CPU", thread_count=8)
                    booster.fit(X_tr, y_tr, eval_set=(X_va, y_va),
                                early_stopping_rounds=150, use_best_model=True, sample_weight=sw)
                    pv = booster.predict(X_va)
                    pt = booster.predict(X_te)
                else:
                    print(f"unknown {model_name}"); continue

                if residual:
                    yhat_va = np.exp(pv) * compact_va
                    yhat_te = np.exp(pt) * compact_te
                else:
                    yhat_va = np.exp(pv)
                    yhat_te = np.exp(pt)

                vm = report_mape(y_va_lin, yhat_va, f"{tag} val")
                tm = report_mape(y_te_lin, yhat_te, f"{tag} test")
                summary.append({"model":model_name, "residual": residual, "seed": s,
                                **{f"val_{k}":v for k,v in vm.items()},
                                **{f"test_{k}":v for k,v in tm.items()},
                                "wall_sec": time.time()-t0})

                tag_full = f"{'res' if residual else 'lin'}_seed{s}"
                pd.DataFrame({
                    "design_name": test["design_name"].values,
                    "net_name":    test["net_name"].values,
                    "y_true":      y_te_lin,
                    "y_pred":      yhat_te,
                }).to_csv(model_dir / f"{tag_full}__test.csv", index=False)
                pd.DataFrame({
                    "design_name": val["design_name"].values,
                    "net_name":    val["net_name"].values,
                    "y_true":      y_va_lin,
                    "y_pred":      yhat_va,
                }).to_csv(model_dir / f"{tag_full}__val.csv", index=False)
                with open(model_dir / f"{tag_full}.pkl", "wb") as f:
                    pickle.dump(booster, f)
                print(f"  wall = {time.time()-t0:.1f}s")
            except Exception:
                import traceback; traceback.print_exc()
            gc.collect()

    df_sum = pd.DataFrame(summary)
    df_sum.to_csv(out_dir / f"summary_{target}{'_res' if residual else '_lin'}.csv", index=False)
    print("\nWritten summary")
    if not df_sum.empty:
        print(df_sum.groupby("model")[["test_mape_mean","test_mape_median","test_mape_p90"]].mean().round(3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=cfg.PRIMARY_TARGET)
    ap.add_argument("--models", nargs="+", default=["lgbm", "xgb", "cat"])
    ap.add_argument("--seeds",  nargs="+", type=int, default=cfg.SEEDS)
    ap.add_argument("--cache",  default="features_v2")
    ap.add_argument("--out",    default="gbdt_v2")
    ap.add_argument("--residual", action="store_true",
                    help="predict log(true/compact) and combine")
    ap.add_argument("--linear",   action="store_true",
                    help="predict log(true) directly (default)")
    ap.add_argument("--weighted", action="store_true",
                    help="row weights")
    args = ap.parse_args()

    cache_dir = cfg.CACHE_DIR / args.cache
    out_dir = cfg.OUTPUT_DIR / args.out
    do_lin = args.linear or (not args.residual)
    do_res = args.residual

    if do_lin:
        run(args.target, args.models, args.seeds, cache_dir, out_dir,
            residual=False, weight_by_log_cap=args.weighted)
    if do_res:
        run(args.target, args.models, args.seeds, cache_dir, out_dir,
            residual=True, weight_by_log_cap=args.weighted)


if __name__ == "__main__":
    main()
