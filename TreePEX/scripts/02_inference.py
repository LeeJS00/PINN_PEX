"""02_inference.py — STAGE 1 of TreePEX tool: features → 5-seed ensemble predictions.

Loads 10 saved XGBoost models (gnd/cpl × 5 seeds) and runs inference on the
input feature CSV (filtered to one design or all-test).

Output: outputs/predictions/<design>_pred.csv
        with columns (design_name, net_name, pred_gnd, pred_cpl, c_gnd_fF, c_cpl_total_fF)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs" / "predictions"
OUTPUTS.mkdir(parents=True, exist_ok=True)

V3_FEATURES = "/data/PINNPEX/data/processed_v3/intel22/features/all_designs.csv"
V4_NEW_FEATS = "/home/jslee/projects/PINNPEX/archive/pex_v4/results/new_features_with_ids.csv"

SEEDS = [42, 0, 1, 2, 3]


def load_models():
    import xgboost as xgb
    feat_order = (MODELS / "FEATURE_ORDER.txt").read_text().strip().split("\n")
    g_models = []
    c_models = []
    for seed in SEEDS:
        mg = xgb.XGBRegressor()
        mg.load_model(str(MODELS / f"tweedie_gnd_seed{seed}.json"))
        g_models.append(mg)
        mc = xgb.XGBRegressor()
        mc.load_model(str(MODELS / f"tweedie_cpl_seed{seed}.json"))
        c_models.append(mc)
    return feat_order, g_models, c_models


def predict_ensemble(models, X):
    """5-seed mean of clipped predictions."""
    preds = np.stack([m.predict(X).clip(0.0) for m in models], axis=0)
    return preds.mean(axis=0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--design", type=str, required=True,
                   help="e.g., intel22_tv80s_f3")
    return p.parse_args()


def main():
    args = parse_args()
    print(f">>> TreePEX STAGE 1 [inference] on design={args.design}")
    feat_order, g_models, c_models = load_models()
    print(f">>> loaded {len(g_models)} gnd models + {len(c_models)} cpl models, "
          f"{len(feat_order)} features")

    print(">>> loading per-net features ...")
    base = pd.read_csv(V3_FEATURES)
    new = pd.read_csv(V4_NEW_FEATS)
    df = base.merge(new, on=["design_name", "net_name"], how="left")
    df = df.dropna(subset=feat_order).reset_index(drop=True)
    df = df[df["design_name"] == args.design].reset_index(drop=True)
    print(f">>> nets for {args.design}: {len(df):,}")
    if len(df) == 0:
        print("[error] no nets matched; aborting"); return 1

    X = df[feat_order].astype(np.float32).values
    t0 = time.time()
    pred_g = predict_ensemble(g_models, X)
    pred_c = predict_ensemble(c_models, X)
    t_inf = time.time() - t0
    print(f">>> inference wall: {t_inf:.3f} s for {len(df):,} nets")

    out = df[["design_name", "net_name", "c_gnd_fF", "c_cpl_total_fF"]].copy()
    out["pred_gnd"] = pred_g
    out["pred_cpl"] = pred_c
    out["pred_total"] = pred_g + pred_c
    out["gold_total"] = out["c_gnd_fF"] + out["c_cpl_total_fF"]

    out_path = OUTPUTS / f"{args.design}_pred.csv"
    out.to_csv(out_path, index=False)
    print(f">>> wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
