"""Train a small LGBM that predicts c_gnd/total ratio per net.

Uses v3 features. Output: 5 seeds, ratios are mean across seeds.
Trained on 9 train designs, validated on nova, applied at inference time.
"""
from __future__ import annotations

import sys
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

    # target: log-ratio  log(c_gnd / (total - c_gnd + ε)) = log(c_gnd / c_cpl_total)
    # We model ratio in [0, 1]; equivalent to logit
    eps = 1e-4
    def to_logit_ratio(c_g, c_t):
        c_t = np.clip(c_t, eps, None)
        c_g = np.clip(c_g, eps, c_t - eps)
        r = c_g / c_t
        r = np.clip(r, 1e-3, 1 - 1e-3)
        return np.log(r / (1 - r))

    y_tr = to_logit_ratio(train["c_gnd_fF"].to_numpy(), train["total_cap_fF"].to_numpy())
    y_va = to_logit_ratio(val["c_gnd_fF"].to_numpy(), val["total_cap_fF"].to_numpy())
    y_te = to_logit_ratio(test["c_gnd_fF"].to_numpy(), test["total_cap_fF"].to_numpy())

    X_tr = train[fcols].to_numpy(np.float32)
    X_va = val[fcols].to_numpy(np.float32)
    X_te = test[fcols].to_numpy(np.float32)

    out_dir = cfg.OUTPUT_DIR / "spef_e2e" / "gnd_ratio"
    out_dir.mkdir(parents=True, exist_ok=True)

    import lightgbm as lgb
    preds_v = []
    preds_t = []
    for s in range(5):
        ts = lgb.Dataset(X_tr, y_tr)
        vs = lgb.Dataset(X_va, y_va, reference=ts)
        booster = lgb.train(
            dict(objective="regression", metric="rmse",
                 learning_rate=0.05, num_leaves=128, min_data_in_leaf=20,
                 feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
                 max_bin=511, seed=s, verbose=-1, n_jobs=8),
            ts, num_boost_round=2000, valid_sets=[vs],
            callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        pv = booster.predict(X_va, num_iteration=booster.best_iteration)
        pt = booster.predict(X_te, num_iteration=booster.best_iteration)
        preds_v.append(pv); preds_t.append(pt)
        # Save model
        with open(out_dir / f"seed{s}.pkl", "wb") as f:
            import pickle; pickle.dump(booster, f)

    # Ensemble (mean of logits)
    pv_mean = np.mean(preds_v, axis=0)
    pt_mean = np.mean(preds_t, axis=0)

    # Convert back to ratio
    def logit_to_ratio(z):
        return 1.0 / (1.0 + np.exp(-z))

    rv = logit_to_ratio(pv_mean)
    rt = logit_to_ratio(pt_mean)
    rv_true = logit_to_ratio(y_va)
    rt_true = logit_to_ratio(y_te)

    # Report
    print(f"\nVal ratio: pred mean={rv.mean():.3f} median={np.median(rv):.3f}, true mean={rv_true.mean():.3f}")
    print(f"     RMSE on ratio={np.sqrt(((rv - rv_true)**2).mean()):.4f}")
    print(f"Test ratio: pred mean={rt.mean():.3f} median={np.median(rt):.3f}, true mean={rt_true.mean():.3f}")
    print(f"     RMSE on ratio={np.sqrt(((rt - rt_true)**2).mean()):.4f}")

    # Save predictions
    out_csv = cfg.OUTPUT_DIR / "spef_e2e" / "gnd_ratio_preds.csv"
    pd.DataFrame({
        "design_name": test["design_name"].values,
        "net_name": test["net_name"].values,
        "y_true_ratio": rt_true,
        "y_pred_ratio": rt,
    }).to_csv(out_csv, index=False)
    print(f"\nsaved {out_csv}")


if __name__ == "__main__":
    main()
