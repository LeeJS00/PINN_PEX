"""01b_train_specialist.py — L11 large-net specialist Tweedie XGBoost.

Trains a 5-seed specialist on the top-quintile of training data by
`gold_total > 3 fF` (covers ~8% of train rows on ASAP7, ~6% on intel22).

Specialist is deeper (depth=9, n_est=750) per pex-domain-reviewer
recommendation (vs canonical depth=8, n_est=500) — extra capacity for the
long-tail subset, early_stopping_rounds=100 to limit overfit.

At inference (pex_cold.py), the canonical 5-seed prediction goes through
a feature-based switch:
    if total_wire_length_um > 15.35 → specialist
    else                            → canonical

The switch feature was chosen for: (a) high AUC vs gold_total>3fF (0.9966),
(b) deterministic at cold inference (no proxy noise like fanout).

Threshold 15.35 μm = 30th percentile of large-net wire-length distribution.
Recall 70%, precision 97% on training. Acceptable as conservative gate —
small nets go to canonical (safer), only confident large goes to specialist.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from pdk_paths import get_pdk  # noqa: E402

_ap = argparse.ArgumentParser()
_ap.add_argument("--pdk", default="asap7", choices=["intel22", "asap7"])
_ap.add_argument("--threshold_fF", type=float, default=3.0,
                 help="Train on rows with gold_total > this (fF)")
_ap.add_argument("--out_dir", default=None,
                 help="Override output dir (default = PDK models_dir)")
_ap.add_argument("--depth", type=int, default=9,
                 help="XGB max_depth (default 9; canonical specialist)")
_ap.add_argument("--n_est", type=int, default=750,
                 help="XGB n_estimators (default 750; canonical specialist)")
_ap.add_argument("--no_h3", action="store_true",
                 help="Drop 26-D V4 H3 features; train V3-only 41-D specialist.")
_args = _ap.parse_args()
_PDK = get_pdk(_args.pdk)

MODELS = Path(_args.out_dir) if _args.out_dir else _PDK.models_dir
MODELS.mkdir(parents=True, exist_ok=True)
print(f">>> output dir: {MODELS}", flush=True)
V3_FEATURES = _PDK.v3_features
V4_NEW_FEATS = _PDK.v4_new_feats

# Feature columns must match 01_train_save_models.py
BASE_FEATURE_COLS = [
    "total_wire_length_um", "total_metal_area_um2", "n_cuboids",
    "bbox_xy_um2", "bbox_z_um", "aspect_ratio",
    "layer_hist_M1", "layer_hist_M2", "layer_hist_M3", "layer_hist_M4",
    "layer_hist_M5", "layer_hist_M6", "layer_hist_M7", "layer_hist_M8",
    "layer_hist_M9_plus",
    "n_aggressor_nets",
    "broadside_overlap_total_um2", "broadside_overlap_p95_um2",
    "lateral_overlap_total_um2", "lateral_overlap_p95_um2",
    "spacing_min_um", "spacing_p25_um", "spacing_p50_um", "spacing_p95_um",
    "n_edges_lt_1um", "n_edges_1_to_3um", "n_edges_3_to_4um",
    "vss_n_cuboids", "vss_total_metal_area_um2",
    "vss_shield_M1_M3", "vss_shield_M4_M5", "vss_shield_M6_plus",
    "fanout",
    "eps_min", "eps_max", "eps_mean", "n_layers_present",
    "density_M1_M3", "density_M4_M5", "density_M6_plus",
    "compact_gnd_estimate_fF", "compact_cpl_estimate_total_fF",
]
H3_FEATURE_COLS = [
    "target_n_cuboids_check",
    "agg_count_above_target_z", "agg_count_below_target_z", "agg_n_distinct",
    "top1_score", "top1_overlap_um2", "top1_min_xy_dist_um",
    "top1_mean_dz_um", "top1_agg_size_um2", "top1_layer_diff_flag",
    "top2_score", "top2_overlap_um2", "top2_min_xy_dist_um",
    "top2_mean_dz_um", "top2_agg_size_um2", "top2_layer_diff_flag",
    "top3_score", "top3_overlap_um2", "top3_min_xy_dist_um",
    "top3_mean_dz_um", "top3_agg_size_um2", "top3_layer_diff_flag",
    "topk_score_concentration",
    "agg_count_within_1um_xyz", "agg_count_within_3um_xyz",
    "agg_count_within_5um_xyz",
]
FEAT_ORDER = (BASE_FEATURE_COLS if _args.no_h3
              else BASE_FEATURE_COLS + H3_FEATURE_COLS)

CONFIG = {"depth": _args.depth, "n_est": _args.n_est,
          "lr": 0.05, "vp": 1.5, "early_stop": 100}
SEEDS = [42, 0, 1, 2, 3]
L6_FANOUT_NOISE_STD = float(os.environ.get("TREEPEX_L6_FANOUT_NOISE", "0.2"))
SWITCH_FEATURE = "total_wire_length_um"
SWITCH_THRESHOLD = 15.35  # μm (large-net classifier threshold, recall 70 / prec 97 %)


def train_save(X_tr, y_tr, X_va, y_va, *, seed: int, channel: str) -> str:
    import xgboost as xgb
    model = xgb.XGBRegressor(
        n_estimators=CONFIG["n_est"], max_depth=CONFIG["depth"],
        learning_rate=CONFIG["lr"], random_state=seed,
        tree_method="hist", objective="reg:tweedie",
        tweedie_variance_power=CONFIG["vp"],
        subsample=0.8, colsample_bytree=0.8,
        verbosity=0, early_stopping_rounds=CONFIG["early_stop"],
    )
    t0 = time.time()
    model.fit(X_tr, np.clip(y_tr, 0, None),
              eval_set=[(X_va, np.clip(y_va, 0, None))], verbose=False)
    out_path = MODELS / f"tweedie_specialist_{channel}_seed{seed}.json"
    model.save_model(str(out_path))
    print(f"  [{channel} seed={seed}] saved {out_path.name}  "
          f"train_wall={time.time()-t0:.0f}s  best_iter={model.best_iteration}", flush=True)
    return str(out_path)


def main():
    print(f">>> L11 specialist trainer pdk={_args.pdk}  threshold={_args.threshold_fF}fF")
    base = pd.read_csv(V3_FEATURES)
    new = pd.read_csv(V4_NEW_FEATS)
    df = base.merge(new, on=["design_name", "net_name"], how="left")
    df = df.dropna(subset=H3_FEATURE_COLS).reset_index(drop=True)
    print(f">>> joined: {len(df):,} feats={len(FEAT_ORDER)}")

    if L6_FANOUT_NOISE_STD > 0:
        rng = np.random.default_rng(seed=42)
        noise = rng.normal(1.0, L6_FANOUT_NOISE_STD, size=len(df))
        noise = np.clip(noise, 0.5, 2.0)
        df["fanout"] = np.maximum(df["fanout"] * noise, 1.0)
        print(f">>> L6 fanout noise σ={L6_FANOUT_NOISE_STD}")

    train = df[df["split"] == "train"].reset_index(drop=True)
    valid = df[df["split"] == "valid"].reset_index(drop=True)

    # Filter to large-cap subset (gold-based, train only — no leak)
    train_large = train[train["total_cap_fF"] > _args.threshold_fF].reset_index(drop=True)
    valid_large = valid[valid["total_cap_fF"] > _args.threshold_fF].reset_index(drop=True)
    print(f">>> large-net subset (gold>{_args.threshold_fF}fF): "
          f"train={len(train_large):,}/{len(train):,} ({len(train_large)/len(train)*100:.1f}%) "
          f"valid={len(valid_large):,}/{len(valid):,} ({len(valid_large)/len(valid)*100:.1f}%)")

    # Sanity: switch-feature distribution on large subset
    sw_pct = (train_large[SWITCH_FEATURE] >= SWITCH_THRESHOLD).mean() * 100
    print(f">>> switch feature recall in large subset: "
          f"({SWITCH_FEATURE} >= {SWITCH_THRESHOLD}) = {sw_pct:.1f}%")

    X_tr = train_large[FEAT_ORDER].astype(np.float32).values
    X_va = valid_large[FEAT_ORDER].astype(np.float32).values
    y_tr_g = train_large["c_gnd_fF"].values
    y_tr_c = train_large["c_cpl_total_fF"].values
    y_va_g = valid_large["c_gnd_fF"].values
    y_va_c = valid_large["c_cpl_total_fF"].values

    for seed in SEEDS:
        train_save(X_tr, y_tr_g, X_va, y_va_g, seed=seed, channel="gnd")
        train_save(X_tr, y_tr_c, X_va, y_va_c, seed=seed, channel="cpl")

    # Write switch meta — pex_cold.py reads to know how to route
    import json
    meta = {
        "switch_feature": SWITCH_FEATURE,
        "switch_threshold": SWITCH_THRESHOLD,
        "train_threshold_fF": _args.threshold_fF,
        "n_train_subset": int(len(train_large)),
        "config": CONFIG,
        "seeds": SEEDS,
    }
    meta_path = MODELS / "specialist_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f">>> wrote {meta_path}")
    print(">>> done — specialist 5-seed saved + meta written")


if __name__ == "__main__":
    main()
