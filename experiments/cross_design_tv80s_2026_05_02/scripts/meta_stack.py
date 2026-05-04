"""Meta-stacking: train a small MLP/GBDT on individual model val preds + key features.

Out-of-fold style: use val preds (nova) to fit a meta-model that predicts
log(true). Apply to test (tv80s) preds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent
sys.path.insert(0, str(_WS))

from configs import cfg
from src.data_loader import _select_feature_cols


def collect(roots, kind="test"):
    out = {}
    for root in roots:
        rp = Path(root)
        if not rp.exists():
            continue
        for csv in sorted(rp.rglob(f"*__{kind}.csv")):
            tag = f"{rp.name}::{csv.parent.name}::{csv.stem.replace(f'__{kind}','')}"
            try:
                df = pd.read_csv(csv)
                if {"y_true","y_pred"}.issubset(df.columns):
                    df = df.set_index(["design_name","net_name"]) if "design_name" in df.columns else df
                    out[tag] = df
            except Exception:
                continue
    return out


def main():
    roots = [str(cfg.OUTPUT_DIR / "final_pipe"),
             str(cfg.OUTPUT_DIR / "final_pipe_nova"),
             str(cfg.OUTPUT_DIR / "resmlp_v2"),
             str(cfg.OUTPUT_DIR / "mlp_hand_v2")]
    test_csvs = collect(roots, "test")
    val_csvs = collect(roots, "val")
    common = sorted(set(test_csvs) & set(val_csvs))
    if not common:
        print("no common")
        return
    print(f"{len(common)} models")

    # Filter to models whose val_csv has the SAME shape as the majority
    val_sizes = {k: len(val_csvs[k]) for k in common}
    most_common_size = max(set(val_sizes.values()), key=lambda s: list(val_sizes.values()).count(s))
    common = [k for k in common if val_sizes[k] == most_common_size]
    print(f"Filtered to {len(common)} models with val size {most_common_size}")

    yv = list(val_csvs[common[0]]["y_true"].to_numpy())
    yt = list(test_csvs[common[0]]["y_true"].to_numpy())
    yv = np.array(yv); yt = np.array(yt)
    Pv = np.stack([np.log(np.maximum(val_csvs[k]["y_pred"].to_numpy(), 1e-4)) for k in common], axis=1)
    Pt = np.stack([np.log(np.maximum(test_csvs[k]["y_pred"].to_numpy(), 1e-4)) for k in common], axis=1)

    # Add a few key features
    cache = cfg.CACHE_DIR / "features_v2"
    val_df = pd.concat([
        pd.read_parquet(cache / f"{d}.parquet").assign(design_name=d) for d in cfg.VAL_DESIGNS
    ], ignore_index=True)
    test_df = pd.read_parquet(cache / "intel22_tv80s_f3.parquet").assign(design_name="intel22_tv80s_f3")

    extra_cols = ["tgt_total_metal_area_um2","tgt_n_cuboids","tgt_n_tiles",
                  "tgt_bbox_xy_um2","tgt_bbox_z_um",
                  "agg_total_count","agg_top1_area","agg_top3_area",
                  "cpl_n_pairs","cpl_total_lateral_overlap_um2",
                  "compact_total_fF","tgt_eps_mean","pwr_total_metal_area_um2"]

    if val_df.shape[0] != Pv.shape[0]:
        # Index merge by net_name (val_csvs are indexed by (design,net))
        # Simplify: use Pv's order via val_csvs keys
        first_val = list(val_csvs.values())[0]
        val_df = val_df.set_index(["design_name","net_name"]).reindex(first_val.index).reset_index()

    test_df = test_df.set_index(["design_name","net_name"])
    first_test = list(test_csvs.values())[0]
    test_df = test_df.reindex(first_test.index).reset_index()

    Xv_extra = np.log1p(val_df[extra_cols].to_numpy(np.float32))
    Xt_extra = np.log1p(test_df[extra_cols].to_numpy(np.float32))
    Xv_extra = np.nan_to_num(Xv_extra)
    Xt_extra = np.nan_to_num(Xt_extra)

    Xv = np.concatenate([Pv, Xv_extra], axis=1)
    Xt = np.concatenate([Pt, Xt_extra], axis=1)
    yv_log = np.log(np.maximum(yv, 1e-4))

    # Try Ridge stacking
    from sklearn.linear_model import Ridge, ElasticNet
    print(f"Xv: {Xv.shape}, Xt: {Xt.shape}")

    for alpha in [0.1, 1.0, 10.0]:
        ridge = Ridge(alpha=alpha)
        ridge.fit(Xv, yv_log)
        pred_log = ridge.predict(Xt)
        yhat = np.exp(pred_log)
        ape = 100.0 * np.abs(yhat - yt) / np.maximum(yt, 1e-3)
        print(f"  Ridge alpha={alpha}: mean MAPE={ape.mean():.3f}%, median={np.median(ape):.3f}%")

    # Try LightGBM stacking
    import lightgbm as lgb
    ts = lgb.Dataset(Xv, yv_log)
    val_size = int(len(yv_log) * 0.2)
    perm = np.random.RandomState(0).permutation(len(yv_log))
    tr_idx = perm[val_size:]; va_idx = perm[:val_size]
    ts2 = lgb.Dataset(Xv[tr_idx], yv_log[tr_idx])
    vs2 = lgb.Dataset(Xv[va_idx], yv_log[va_idx], reference=ts2)

    booster = lgb.train(
        dict(objective="regression", metric="rmse", learning_rate=0.03,
             num_leaves=63, min_data_in_leaf=30, max_bin=255,
             feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
             verbose=-1, seed=0, n_jobs=8),
        ts2, num_boost_round=2000, valid_sets=[vs2],
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
    pred_log = booster.predict(Xt, num_iteration=booster.best_iteration)
    yhat = np.exp(pred_log)
    ape = 100.0 * np.abs(yhat - yt) / np.maximum(yt, 1e-3)
    print(f"  LGBM stacker: mean MAPE={ape.mean():.3f}%, median={np.median(ape):.3f}%")


if __name__ == "__main__":
    main()
