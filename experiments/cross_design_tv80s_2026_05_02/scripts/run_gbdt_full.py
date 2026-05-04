"""
Full GBDT run on v2 features.

Strategy:
  - Direct: predict log(total_cap_fF).
  - Split:  predict log(c_gnd_fF) and log(c_cpl_total_fF) separately, sum.
            (Reduces label noise when one component dominates.)
  - Residual-from-compact: predict log(true / compact_total).

For each strategy run lgbm + xgb + cat × 5 seeds. Save preds + summary.
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
        if "intel22_ldpc_decoder_802_3an_f3" in avail:
            val_pool = ["intel22_ldpc_decoder_802_3an_f3"]
        else:
            val_pool = ["intel22_ibex_core_f3"] if "intel22_ibex_core_f3" in avail else [train_pool[0]]
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


def train_lgbm(X_tr, y_tr, X_va, y_va, seed):
    import lightgbm as lgb
    ts = lgb.Dataset(X_tr, y_tr); vs = lgb.Dataset(X_va, y_va, reference=ts)
    params = dict(objective="regression", metric="rmse",
                  learning_rate=0.03, num_leaves=255, min_data_in_leaf=20,
                  feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
                  max_bin=511, seed=seed, verbose=-1, n_jobs=8)
    booster = lgb.train(params, ts, num_boost_round=4000, valid_sets=[vs],
                        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
    pv = booster.predict(X_va, num_iteration=booster.best_iteration)
    return booster, pv


def train_xgb(X_tr, y_tr, X_va, y_va, seed):
    import xgboost as xgb
    dt = xgb.DMatrix(X_tr, label=y_tr); dv = xgb.DMatrix(X_va, label=y_va)
    params = dict(objective="reg:squarederror", eval_metric="rmse",
                  eta=0.03, max_depth=10, min_child_weight=5,
                  subsample=0.85, colsample_bytree=0.85,
                  tree_method="hist", seed=seed, verbosity=0, nthread=8)
    booster = xgb.train(params, dt, num_boost_round=4000, evals=[(dv, "val")],
                        early_stopping_rounds=150, verbose_eval=0)
    pv = booster.predict(dv, iteration_range=(0, booster.best_iteration+1))
    return booster, pv


def train_cat(X_tr, y_tr, X_va, y_va, seed):
    from catboost import CatBoostRegressor
    booster = CatBoostRegressor(iterations=4000, learning_rate=0.03, depth=10,
                                l2_leaf_reg=4.0, loss_function="RMSE",
                                random_seed=seed, verbose=0, task_type="CPU", thread_count=8)
    booster.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=150, use_best_model=True)
    pv = booster.predict(X_va)
    return booster, pv


def predict(booster, X, model_name: str) -> np.ndarray:
    if model_name == "lgbm":
        return booster.predict(X, num_iteration=booster.best_iteration)
    if model_name == "xgb":
        import xgboost as xgb
        return booster.predict(xgb.DMatrix(X), iteration_range=(0, booster.best_iteration+1))
    return booster.predict(X)


def run_strategy(strategy: str, train, val, test, fcols,
                 models, seeds, out_dir: Path):
    """strategy in {"direct", "split", "residual"}."""
    eps = 1e-4
    X_tr = train[fcols].to_numpy(np.float32)
    X_va = val  [fcols].to_numpy(np.float32)
    X_te = test [fcols].to_numpy(np.float32)

    y_tr_lin = train["total_cap_fF"].to_numpy(np.float64)
    y_va_lin = val  ["total_cap_fF"].to_numpy(np.float64)
    y_te_lin = test ["total_cap_fF"].to_numpy(np.float64)

    summary = []

    for model_name in models:
        model_dir = out_dir / strategy / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        for s in seeds:
            t0 = time.time()
            tag = f"{strategy}-{model_name}-s{s}"
            print(f"\n=== {tag} ===")
            try:
                if strategy == "direct":
                    y_tr = np.log(y_tr_lin.clip(min=eps))
                    y_va = np.log(y_va_lin.clip(min=eps))
                    if model_name == "lgbm":
                        b, pv = train_lgbm(X_tr, y_tr, X_va, y_va, s)
                    elif model_name == "xgb":
                        b, pv = train_xgb (X_tr, y_tr, X_va, y_va, s)
                    else:
                        b, pv = train_cat (X_tr, y_tr, X_va, y_va, s)
                    yhat_va = np.exp(pv)
                    pt = predict(b, X_te, model_name)
                    yhat_te = np.exp(pt)
                elif strategy == "split":
                    # gnd
                    yg_tr_lin = train["c_gnd_fF"].to_numpy(np.float64)
                    yg_va_lin = val  ["c_gnd_fF"].to_numpy(np.float64)
                    yc_tr_lin = train["c_cpl_total_fF"].to_numpy(np.float64)
                    yc_va_lin = val  ["c_cpl_total_fF"].to_numpy(np.float64)
                    yg_tr = np.log(yg_tr_lin.clip(min=eps))
                    yg_va = np.log(yg_va_lin.clip(min=eps))
                    yc_tr = np.log(yc_tr_lin.clip(min=eps))
                    yc_va = np.log(yc_va_lin.clip(min=eps))
                    if model_name == "lgbm":
                        bg, pv_g = train_lgbm(X_tr, yg_tr, X_va, yg_va, s)
                        bc, pv_c = train_lgbm(X_tr, yc_tr, X_va, yc_va, s)
                    elif model_name == "xgb":
                        bg, pv_g = train_xgb(X_tr, yg_tr, X_va, yg_va, s)
                        bc, pv_c = train_xgb(X_tr, yc_tr, X_va, yc_va, s)
                    else:
                        bg, pv_g = train_cat(X_tr, yg_tr, X_va, yg_va, s)
                        bc, pv_c = train_cat(X_tr, yc_tr, X_va, yc_va, s)
                    yhat_va = np.exp(pv_g) + np.exp(pv_c)
                    pt_g = predict(bg, X_te, model_name)
                    pt_c = predict(bc, X_te, model_name)
                    yhat_te = np.exp(pt_g) + np.exp(pt_c)
                    b = (bg, bc)
                elif strategy == "residual":
                    ct = train["compact_total_fF"].to_numpy(np.float64).clip(min=eps)
                    cv = val  ["compact_total_fF"].to_numpy(np.float64).clip(min=eps)
                    ce = test ["compact_total_fF"].to_numpy(np.float64).clip(min=eps)
                    y_tr = np.log(y_tr_lin.clip(min=eps)) - np.log(ct)
                    y_va = np.log(y_va_lin.clip(min=eps)) - np.log(cv)
                    if model_name == "lgbm":
                        b, pv = train_lgbm(X_tr, y_tr, X_va, y_va, s)
                    elif model_name == "xgb":
                        b, pv = train_xgb (X_tr, y_tr, X_va, y_va, s)
                    else:
                        b, pv = train_cat (X_tr, y_tr, X_va, y_va, s)
                    yhat_va = np.exp(pv) * cv
                    pt = predict(b, X_te, model_name)
                    yhat_te = np.exp(pt) * ce
                else:
                    print("unknown strategy"); continue

                vm = report_mape(y_va_lin, yhat_va, f"{tag} val")
                tm = report_mape(y_te_lin, yhat_te, f"{tag} test")
                summary.append({
                    "strategy": strategy, "model": model_name, "seed": s,
                    **{f"val_{k}":v for k,v in vm.items()},
                    **{f"test_{k}":v for k,v in tm.items()},
                    "wall_sec": time.time() - t0,
                })
                tag_full = f"seed{s}"
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
            except Exception:
                import traceback; traceback.print_exc()
            gc.collect()

    df_sum = pd.DataFrame(summary)
    return df_sum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="features_v2")
    ap.add_argument("--out",   default="gbdt_v2")
    ap.add_argument("--strategies", nargs="+",
                    default=["direct", "split", "residual"])
    ap.add_argument("--models", nargs="+", default=["lgbm", "xgb", "cat"])
    ap.add_argument("--seeds",  nargs="+", type=int, default=cfg.SEEDS)
    args = ap.parse_args()

    cache_dir = cfg.CACHE_DIR / args.cache
    out_dir   = cfg.OUTPUT_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    train, val, test = assemble_split(cache_dir)
    fcols = _select_feature_cols(train)
    print(f"features: {len(fcols)}, train: {len(train):,}, val: {len(val):,}, test: {len(test):,}")

    all_summary = []
    for strat in args.strategies:
        df = run_strategy(strat, train, val, test, fcols, args.models, args.seeds, out_dir)
        if not df.empty:
            all_summary.append(df)

    if all_summary:
        full = pd.concat(all_summary, ignore_index=True)
        full.to_csv(out_dir / "summary_full.csv", index=False)
        print("\nGroup-by strategy + model:")
        print(full.groupby(["strategy", "model"])[["test_mape_mean","test_mape_median","test_mape_p90"]].mean().round(3))
    print("\nDone.")


if __name__ == "__main__":
    main()
