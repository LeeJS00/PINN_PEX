"""
LightGBM with custom MAPE-style objective.

The default `regression` objective minimises MSE on log(y), which doesn't
directly target MAPE. We use a smoothed APE objective on the linear y:
    L = | y_pred - y_true | / y_true
    grad ∝ sign(y_pred - y_true) / y_true
    hess ∝ 1e-3 / y_true^2  (small constant for stability)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent
sys.path.insert(0, str(_WS))

from configs import cfg
from src.data_loader import _select_feature_cols, report_mape


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

    train = pd.concat([
        pd.read_parquet(cache / f"{d}.parquet").assign(design_name=d)
        for d in train_pool
    ], ignore_index=True)
    val   = pd.concat([
        pd.read_parquet(cache / f"{d}.parquet").assign(design_name=d)
        for d in val_pool
    ], ignore_index=True)
    test  = pd.read_parquet(cache / "intel22_tv80s_f3.parquet").assign(design_name="intel22_tv80s_f3")
    return train, val, test, train_pool, val_pool


def mape_objective(eps: float = 1e-3):
    """Returns a (grad, hess) callable for LightGBM where labels are linear y."""
    def obj(y_pred_log, dtrain):
        y_true = dtrain.get_label()
        y_pred = np.exp(y_pred_log)
        denom = np.maximum(y_true, eps)
        residual = y_pred - y_true
        grad = np.sign(residual) * y_pred / denom
        # Smooth hess via |residual| terms
        hess = y_pred / denom + 1e-4
        return grad, hess
    return obj


def mape_eval(eps: float = 1e-3):
    def fn(y_pred_log, dtrain):
        y_true = dtrain.get_label()
        y_pred = np.exp(y_pred_log)
        denom = np.maximum(y_true, eps)
        ape = np.abs(y_pred - y_true) / denom
        return "mape", float(np.mean(ape)), False
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="features_v2")
    ap.add_argument("--seeds", nargs="+", type=int, default=cfg.SEEDS)
    ap.add_argument("--out", default="lgbm_mape")
    args = ap.parse_args()

    cache_dir = cfg.CACHE_DIR / args.cache
    train, val, test, _, _ = assemble_split(cache_dir)
    fcols = _select_feature_cols(train)
    print(f"features: {len(fcols)}, train: {len(train):,}, val: {len(val):,}, test: {len(test):,}")

    X_tr = train[fcols].to_numpy(np.float32)
    X_va = val[fcols].to_numpy(np.float32)
    X_te = test[fcols].to_numpy(np.float32)
    y_tr_lin = train["total_cap_fF"].to_numpy(np.float64)
    y_va_lin = val  ["total_cap_fF"].to_numpy(np.float64)
    y_te_lin = test ["total_cap_fF"].to_numpy(np.float64)

    out_dir = cfg.OUTPUT_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    import lightgbm as lgb
    for s in args.seeds:
        ts = lgb.Dataset(X_tr, y_tr_lin)
        vs = lgb.Dataset(X_va, y_va_lin, reference=ts)
        params = dict(objective=mape_objective(), metric=None,
                      learning_rate=0.03, num_leaves=255, min_data_in_leaf=30,
                      feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
                      max_bin=511, seed=s, verbose=-1, n_jobs=8)
        booster = lgb.train(
            params, ts, num_boost_round=4000,
            valid_sets=[vs], feval=mape_eval(),
            callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)],
        )
        pv = np.exp(booster.predict(X_va, num_iteration=booster.best_iteration))
        pt = np.exp(booster.predict(X_te, num_iteration=booster.best_iteration))
        report_mape(y_va_lin, pv, f"mape-lgbm s{s} val")
        report_mape(y_te_lin, pt, f"mape-lgbm s{s} test")
        pd.DataFrame({
            "design_name": test["design_name"].values,
            "net_name":    test["net_name"].values,
            "y_true":      y_te_lin,
            "y_pred":      pt,
        }).to_csv(out_dir / f"seed{s}__test.csv", index=False)
        pd.DataFrame({
            "design_name": val["design_name"].values,
            "net_name":    val["net_name"].values,
            "y_true":      y_va_lin,
            "y_pred":      pv,
        }).to_csv(out_dir / f"seed{s}__val.csv", index=False)


if __name__ == "__main__":
    main()
