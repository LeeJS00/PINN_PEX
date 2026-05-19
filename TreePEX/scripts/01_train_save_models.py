"""01_train_save_models.py — one-time trainer that saves S4 Tweedie weights.

Trains 5-seed Tweedie XGBoost (Small_combined config: depth=8, n_est=500, lr=0.05,
67 features = 41 base + 26 H3 top-K aggressor) on train+valid, saves
gnd/cpl model.json per seed to TreePEX/models/.

Ran OFFLINE during TreePEX setup (mimics "carry only model weights" from
clean-clone perspective).
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
_ap.add_argument("--pdk", default="intel22", choices=["intel22", "asap7"])
_ap.add_argument("--out_dir", default=None,
                 help="Override output dir (default = PDK models_dir). "
                      "Use for ablation retrains to avoid overwriting canonical weights.")
_ap.add_argument("--no_h3", action="store_true",
                 help="Drop 26-D V4 H3 features; train V3-only 41-D model.")
_ap.add_argument("--drop_features", default=None,
                 help="Path to text file listing one feature name per line to drop. "
                      "Use for fine-grained feature pruning (F3/F4 sprint).")
_args = _ap.parse_args()
_PDK = get_pdk(_args.pdk)

MODELS = Path(_args.out_dir) if _args.out_dir else _PDK.models_dir
MODELS.mkdir(parents=True, exist_ok=True)
print(f">>> output dir: {MODELS}", flush=True)

V3_FEATURES = _PDK.v3_features
V4_NEW_FEATS = _PDK.v4_new_feats

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
if _args.drop_features:
    _drop = set(Path(_args.drop_features).read_text().strip().split("\n"))
    FEAT_ORDER = [f for f in FEAT_ORDER if f not in _drop]
    print(f">>> --drop_features: dropped {len(_drop)} features, retained {len(FEAT_ORDER)}-D", flush=True)

CONFIG = {"depth": 8, "n_est": 500, "lr": 0.05, "vp": 1.5, "early_stop": 100}
SEEDS = [42, 0, 1, 2, 3]


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
    out_path = MODELS / f"tweedie_{channel}_seed{seed}.json"
    model.save_model(str(out_path))
    print(f"  [{channel} seed={seed}] saved {out_path.name}  train_wall={time.time()-t0:.0f}s",
          flush=True)
    return str(out_path)


def main():
    print(f">>> Loading features ...")
    base = pd.read_csv(V3_FEATURES)
    new = pd.read_csv(V4_NEW_FEATS)
    df = base.merge(new, on=["design_name", "net_name"], how="left")
    # When --no_h3 is set we still drop H3-NaN rows to keep the same train set
    # so V3-only vs V3+H3 is a clean apples-to-apples comparison.
    df = df.dropna(subset=H3_FEATURE_COLS).reset_index(drop=True)
    print(f">>> joined: {len(df):,}  feats={len(FEAT_ORDER)}  no_h3={_args.no_h3}")

    # L1 cold-aware training was tested 2026-05-15 and rejected:
    # tv80s cold MAPE 10.66% (no L1) → 11.64% with L1 (+0.98pp regression).
    # Proxy-fanout in training removes useful signal (golden SPEF fanout is
    # informative); model couldn't simultaneously learn "fanout is noisy here
    # but accurate elsewhere". L2 (stronger proxy at inference time) keeps
    # the gain without the L1 cost. Code preserved as reference; flag-gated.
    USE_L1_COLD_AWARE = False
    if USE_L1_COLD_AWARE and (MODELS / "fanout_proxy_meta.json").exists() and \
       (MODELS / "fanout_proxy_xgb_tweedie.json").exists():
        import json
        import xgboost as xgb
        meta = json.loads((MODELS / "fanout_proxy_meta.json").read_text())
        proxy = xgb.XGBRegressor()
        proxy.load_model(str(MODELS / meta["model_file"]))
        Xp = df[meta["feats"]].fillna(0.0).values
        df["fanout"] = np.maximum(proxy.predict(Xp.astype(np.float32)), 1.0)

    # L6 noise-aware training (2026-05-15): multiplicative noise on fanout
    # during training so model doesn't over-fit to exact fanout values.
    # Cold inference uses proxy fanout (15-20% OOS MAPE) — training-time
    # noise injection regularizes the model to be robust to this perturbation
    # while keeping the golden signal mostly intact (vs full L1 replacement).
    # σ matches proxy OOS std; clamp to keep fanout positive.
    L6_FANOUT_NOISE_STD = float(os.environ.get("TREEPEX_L6_FANOUT_NOISE", "0.2"))
    if L6_FANOUT_NOISE_STD > 0:
        rng = np.random.default_rng(seed=42)
        noise = rng.normal(1.0, L6_FANOUT_NOISE_STD, size=len(df))
        noise = np.clip(noise, 0.5, 2.0)
        old_fanout = df["fanout"].copy()
        df["fanout"] = np.maximum(df["fanout"] * noise, 1.0)
        print(f">>> L6 noise-aware: fanout σ={L6_FANOUT_NOISE_STD} mult-noise "
              f"(mean {old_fanout.mean():.1f} → {df['fanout'].mean():.1f}, "
              f"unchanged in expectation)", flush=True)

    train = df[df["split"] == "train"].reset_index(drop=True)
    valid = df[df["split"] == "valid"].reset_index(drop=True)
    print(f">>> train={len(train):,}  valid={len(valid):,}")

    X_tr = train[FEAT_ORDER].astype(np.float32).values
    X_va = valid[FEAT_ORDER].astype(np.float32).values
    y_tr_g = train["c_gnd_fF"].values
    y_tr_c = train["c_cpl_total_fF"].values
    y_va_g = valid["c_gnd_fF"].values
    y_va_c = valid["c_cpl_total_fF"].values

    for seed in SEEDS:
        train_save(X_tr, y_tr_g, X_va, y_va_g, seed=seed, channel="gnd")
        train_save(X_tr, y_tr_c, X_va, y_va_c, seed=seed, channel="cpl")

    # Save feature order for inference
    feat_meta = MODELS / "FEATURE_ORDER.txt"
    feat_meta.write_text("\n".join(FEAT_ORDER))
    print(f"\n>>> Feature order written to {feat_meta}")
    print(f">>> Total weight files in {MODELS}:")
    for f in sorted(MODELS.glob("*")):
        print(f"  {f.name}  ({f.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
