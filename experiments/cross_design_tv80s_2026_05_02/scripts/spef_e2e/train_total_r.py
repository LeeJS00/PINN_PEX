"""Train LGBM to predict total wire resistance per net (cross-design).

Uses v3 features. Output: 5 seeds.
Trained on 9 train designs, validated on nova, applied at inference time.

Target: log(total_res_ohm + 1).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent.parent
# PINNPEX root first (lower priority), workspace second (higher priority)
sys.path.insert(0, str(_WS.parent.parent))
sys.path.insert(0, str(_WS))

from configs import cfg
from src.data_loader import _select_feature_cols

# compare_spef lives at PINNPEX root, but workspace's src takes priority via
# sys.path; load directly to avoid shadowing.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "compare_spef_pinpex",
    str(_WS.parent.parent / "src" / "evaluation" / "compare_spef.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
parse_spef_with_coordinates = _mod.parse_spef_with_coordinates


def load_R_labels(design):
    """Load total_res per net from golden SPEF."""
    spef_path = cfg.SPEF_DIR / f"{design}_starrc.spef"
    if not spef_path.exists():
        return {}
    nets = parse_spef_with_coordinates(spef_path)
    return {n: info["total_res"] for n, info in nets.items()}


def _load(d, cache):
    df = pd.read_parquet(cache / f"{d}.parquet")
    df["design_name"] = d
    R = load_R_labels(d)
    df["total_res_label"] = df["net_name"].map(R).fillna(0.0)
    return df


def main():
    cache = cfg.CACHE_DIR / "features_v3"
    train_pool = list(cfg.TRAIN_DESIGNS)
    val_pool = ["intel22_nova_f3"]
    test_pool = ["intel22_tv80s_f3"]

    print(f"Loading {len(train_pool)} train + 1 val + 1 test designs (with golden R labels)...")
    train = pd.concat([_load(d, cache) for d in train_pool], ignore_index=True)
    val = pd.concat([_load(d, cache) for d in val_pool], ignore_index=True)
    test = pd.concat([_load(d, cache) for d in test_pool], ignore_index=True)

    fcols = [c for c in _select_feature_cols(train) if c != "total_res_label"]
    print(f"features: {len(fcols)}, train: {len(train):,}, val: {len(val):,}, test: {len(test):,}")
    print(f"R label coverage: train={(train['total_res_label']>0).sum()}/{len(train)} "
          f"val={(val['total_res_label']>0).sum()}/{len(val)} "
          f"test={(test['total_res_label']>0).sum()}/{len(test)}")

    eps = 0.1
    y_tr = np.log(train["total_res_label"].to_numpy().clip(min=eps))
    y_va = np.log(val["total_res_label"].to_numpy().clip(min=eps))
    y_te = np.log(test["total_res_label"].to_numpy().clip(min=eps))

    X_tr = train[fcols].to_numpy(np.float32)
    X_va = val[fcols].to_numpy(np.float32)
    X_te = test[fcols].to_numpy(np.float32)

    out_dir = cfg.OUTPUT_DIR / "spef_e2e" / "total_r"
    out_dir.mkdir(parents=True, exist_ok=True)

    import lightgbm as lgb
    preds_v, preds_t = [], []
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
        pv = np.exp(booster.predict(X_va, num_iteration=booster.best_iteration))
        pt = np.exp(booster.predict(X_te, num_iteration=booster.best_iteration))
        preds_v.append(pv); preds_t.append(pt)
        with open(out_dir / f"seed{s}.pkl", "wb") as f:
            import pickle; pickle.dump(booster, f)

    pv_mean = np.mean(preds_v, axis=0)
    pt_mean = np.mean(preds_t, axis=0)

    yt_lin = test["total_res_label"].to_numpy()
    ape = 100 * np.abs(pt_mean - yt_lin) / np.maximum(yt_lin, 1e-3)
    nz = yt_lin > 1e-6
    print(f"\nTest R MAPE: n={nz.sum()} mean={ape[nz].mean():.2f}% median={np.median(ape[nz]):.2f}% p90={np.percentile(ape[nz], 90):.2f}%")

    out_csv = cfg.OUTPUT_DIR / "spef_e2e" / "total_r_preds.csv"
    pd.DataFrame({
        "design_name": test["design_name"].values,
        "net_name": test["net_name"].values,
        "y_true_R": yt_lin,
        "y_pred_R": pt_mean,
    }).to_csv(out_csv, index=False)
    print(f"saved {out_csv}")


if __name__ == "__main__":
    main()
