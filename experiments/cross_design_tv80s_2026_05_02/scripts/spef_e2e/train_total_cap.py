"""Train + save LGBM and CatBoost models for total_cap prediction.

Models are saved as pickles for use by predict_spef_e2e.py at inference time.
Trained on 9 train designs with nova as validation, evaluated on tv80s.

Output: output/spef_e2e/total_cap/{lgbm,cat}_seed{S}.pkl + summary.json
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent.parent
sys.path.insert(0, str(_WS))

from configs import cfg
from src.data_loader import _select_feature_cols


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

    eps = 1e-4
    y_tr = np.log(train["total_cap_fF"].to_numpy().clip(min=eps))
    y_va = np.log(val["total_cap_fF"].to_numpy().clip(min=eps))
    y_te_lin = test["total_cap_fF"].to_numpy()

    X_tr = train[fcols].to_numpy(np.float32)
    X_va = val[fcols].to_numpy(np.float32)
    X_te = test[fcols].to_numpy(np.float32)

    out_dir = cfg.OUTPUT_DIR / "spef_e2e" / "total_cap"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save feature column names for inference
    with open(out_dir / "fcols.json", "w") as f:
        json.dump(fcols, f)

    summary = []

    # === LightGBM ensemble ===
    import lightgbm as lgb
    print("\n=== LightGBM ===")
    for s in range(5):
        ts = lgb.Dataset(X_tr, y_tr)
        vs = lgb.Dataset(X_va, y_va, reference=ts)
        booster = lgb.train(
            dict(objective="regression", metric="rmse",
                 learning_rate=0.03, num_leaves=255, min_data_in_leaf=20,
                 feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
                 max_bin=511, seed=s, verbose=-1, n_jobs=8),
            ts, num_boost_round=4000, valid_sets=[vs],
            callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        pt = np.exp(booster.predict(X_te, num_iteration=booster.best_iteration))
        ape = 100 * np.abs(pt - y_te_lin) / np.maximum(y_te_lin, 1e-3)
        print(f"  LGBM seed{s}: test_mape={ape.mean():.3f}% best_iter={booster.best_iteration}")
        with open(out_dir / f"lgbm_seed{s}.pkl", "wb") as f:
            pickle.dump(booster, f)
        summary.append({"model": "lgbm", "seed": s, "test_mape": float(ape.mean())})

    # === CatBoost ensemble ===
    print("\n=== CatBoost ===")
    try:
        from catboost import CatBoostRegressor
        for s in range(5):
            mdl = CatBoostRegressor(
                iterations=2000, learning_rate=0.05, depth=8,
                l2_leaf_reg=3.0, loss_function="RMSE",
                eval_metric="RMSE", random_seed=s,
                early_stopping_rounds=150, verbose=0)
            mdl.fit(X_tr, y_tr, eval_set=(X_va, y_va))
            pt = np.exp(mdl.predict(X_te))
            ape = 100 * np.abs(pt - y_te_lin) / np.maximum(y_te_lin, 1e-3)
            print(f"  CatBoost seed{s}: test_mape={ape.mean():.3f}% best_iter={mdl.tree_count_}")
            mdl.save_model(str(out_dir / f"cat_seed{s}.cbm"))
            summary.append({"model": "cat", "seed": s, "test_mape": float(ape.mean())})
    except Exception as e:
        print(f"  CatBoost failed: {e}")

    # Mean ensemble
    print("\n=== Ensemble (uniform mean of all 10) ===")
    all_preds = []
    for f in sorted(out_dir.glob("lgbm_seed*.pkl")):
        with open(f, "rb") as fh:
            booster = pickle.load(fh)
        all_preds.append(np.exp(booster.predict(X_te, num_iteration=booster.best_iteration)))
    try:
        from catboost import CatBoostRegressor
        for f in sorted(out_dir.glob("cat_seed*.cbm")):
            mdl = CatBoostRegressor()
            mdl.load_model(str(f))
            all_preds.append(np.exp(mdl.predict(X_te)))
    except Exception:
        pass
    if all_preds:
        ens = np.mean(all_preds, axis=0)
        ape = 100 * np.abs(ens - y_te_lin) / np.maximum(y_te_lin, 1e-3)
        print(f"  Ensemble (n={len(all_preds)}): test_mape_mean={ape.mean():.3f}% median={np.median(ape):.3f}% p90={np.percentile(ape, 90):.2f}%")
        rng = np.random.default_rng(0)
        boots = [ape[rng.integers(0, len(ape), len(ape))].mean() for _ in range(2000)]
        print(f"  CI=[{np.percentile(boots, 2.5):.3f}, {np.percentile(boots, 97.5):.3f}]")
        # save ensemble predictions
        pd.DataFrame({"design_name": test["design_name"].values,
                      "net_name": test["net_name"].values,
                      "y_true": y_te_lin,
                      "y_pred": ens}).to_csv(out_dir / "ensemble_test.csv", index=False)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved {out_dir}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal time: {time.time() - t0:.1f}s")
