"""
Train gradient-boosted decision tree baselines on cross-design features.

Three models, all targeting `log(total_cap_fF + 1e-3)`:
    - LightGBM   (fast, GPU-friendly)
    - XGBoost    (gpu-hist)
    - CatBoost   (gpu)

Each writes:
    output/models/<name>/seed<S>.pkl              the booster
    output/preds/<name>/seed<S>__test.csv         per-net pred on tv80s
    output/preds/<name>/seed<S>__val.csv          per-net pred on nova

Per-target log-transform helps with the ~3 orders of magnitude span on tv80s.
"""
from __future__ import annotations

import argparse
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
from src.data_loader import load_split, report_mape, mape_per_net


def _log_transform(y: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(y, 1e-4))


def _exp_transform(y: np.ndarray) -> np.ndarray:
    return np.exp(y)


def train_lgbm(X_train, y_train, X_val, y_val, seed: int):
    import lightgbm as lgb
    params = dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.03,
        num_leaves=255,
        min_data_in_leaf=20,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        max_bin=511,
        seed=seed,
        verbose=-1,
        n_jobs=8,
    )
    train_set = lgb.Dataset(X_train, y_train)
    val_set   = lgb.Dataset(X_val, y_val, reference=train_set)
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=4000,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)],
    )
    return booster


def train_xgb(X_train, y_train, X_val, y_val, seed: int):
    import xgboost as xgb
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val,   label=y_val)
    params = dict(
        objective="reg:squarederror",
        eval_metric="rmse",
        eta=0.03,
        max_depth=10,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        device="cuda",
        seed=seed,
        verbosity=0,
    )
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=4000,
        evals=[(dval, "val")],
        early_stopping_rounds=150,
        verbose_eval=0,
    )
    return booster


def train_catboost(X_train, y_train, X_val, y_val, seed: int):
    from catboost import CatBoostRegressor
    model = CatBoostRegressor(
        iterations=4000,
        learning_rate=0.03,
        depth=10,
        l2_leaf_reg=4.0,
        loss_function="RMSE",
        random_seed=seed,
        verbose=0,
        task_type="GPU",
        devices="2",          # default to GPU 2; override via CUDA_VISIBLE_DEVICES
        bagging_temperature=0.5,
        bootstrap_type="Bayesian",
    )
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=150,
        use_best_model=True,
    )
    return model


def predict(model, X) -> np.ndarray:
    if hasattr(model, "best_iteration"):
        # lightgbm
        return model.predict(X, num_iteration=model.best_iteration)
    if hasattr(model, "best_iteration_") or hasattr(model, "predict") and "xgboost" in type(model).__module__:
        import xgboost as xgb
        if isinstance(model, xgb.Booster):
            return model.predict(xgb.DMatrix(X), iteration_range=(0, model.best_iteration + 1))
    # catboost
    return model.predict(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=cfg.PRIMARY_TARGET)
    ap.add_argument("--seeds",  nargs="+", type=int, default=cfg.SEEDS)
    ap.add_argument("--models", nargs="+", default=["lgbm", "xgb", "cat"])
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print(f"target = {args.target}")
    print(f"seeds  = {args.seeds}")
    print(f"models = {args.models}")

    splits, feature_cols = load_split()
    train, val, test = splits["train"], splits["val"], splits["test"]

    X_train = train[feature_cols].to_numpy(dtype=np.float32)
    X_val   = val[feature_cols].to_numpy(dtype=np.float32)
    X_test  = test[feature_cols].to_numpy(dtype=np.float32)

    y_train = _log_transform(train[args.target].to_numpy(dtype=np.float64))
    y_val   = _log_transform(val[args.target].to_numpy(dtype=np.float64))

    out_root = cfg.OUTPUT_DIR
    (out_root / "models").mkdir(parents=True, exist_ok=True)
    (out_root / "preds").mkdir(parents=True, exist_ok=True)

    summary = []
    for model_name in args.models:
        for seed in args.seeds:
            t0 = time.time()
            print(f"\n=== {model_name} seed={seed} ===")
            try:
                if model_name == "lgbm":
                    booster = train_lgbm(X_train, y_train, X_val, y_val, seed)
                    pred_test = booster.predict(X_test, num_iteration=booster.best_iteration)
                    pred_val  = booster.predict(X_val,   num_iteration=booster.best_iteration)
                elif model_name == "xgb":
                    import xgboost as xgb
                    booster = train_xgb(X_train, y_train, X_val, y_val, seed)
                    pred_test = booster.predict(xgb.DMatrix(X_test), iteration_range=(0, booster.best_iteration + 1))
                    pred_val  = booster.predict(xgb.DMatrix(X_val),  iteration_range=(0, booster.best_iteration + 1))
                elif model_name == "cat":
                    booster = train_catboost(X_train, y_train, X_val, y_val, seed)
                    pred_test = booster.predict(X_test)
                    pred_val  = booster.predict(X_val)
                else:
                    print(f"  unknown model {model_name}")
                    continue

                # un-log
                yhat_test = _exp_transform(pred_test)
                yhat_val  = _exp_transform(pred_val)

                test_metrics = report_mape(test[args.target].to_numpy(), yhat_test, f"{model_name}-test")
                val_metrics  = report_mape(val[args.target].to_numpy(),  yhat_val,  f"{model_name}-val")

                # save preds
                preds_dir = out_root / "preds" / model_name
                preds_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({
                    "design_name": test["design_name"].values,
                    "net_name":    test["net_name"].values,
                    f"y_true_{args.target}": test[args.target].values,
                    f"y_pred_{args.target}": yhat_test,
                }).to_csv(preds_dir / f"seed{seed}__test.csv", index=False)
                pd.DataFrame({
                    "design_name": val["design_name"].values,
                    "net_name":    val["net_name"].values,
                    f"y_true_{args.target}": val[args.target].values,
                    f"y_pred_{args.target}": yhat_val,
                }).to_csv(preds_dir / f"seed{seed}__val.csv", index=False)

                # save model
                models_dir = out_root / "models" / model_name
                models_dir.mkdir(parents=True, exist_ok=True)
                with open(models_dir / f"seed{seed}.pkl", "wb") as f:
                    pickle.dump(booster, f)

                summary.append({
                    "model": model_name, "seed": seed,
                    **{f"val_{k}":  v for k, v in val_metrics.items()},
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                    "wall_sec": time.time() - t0,
                })
                print(f"  wall = {time.time()-t0:.1f}s")
            except Exception:
                import traceback; traceback.print_exc()

    pd.DataFrame(summary).to_csv(out_root / f"summary_gbdt_{args.target}.csv", index=False)
    print(f"\nWrote {out_root / f'summary_gbdt_{args.target}.csv'}")


if __name__ == "__main__":
    main()
