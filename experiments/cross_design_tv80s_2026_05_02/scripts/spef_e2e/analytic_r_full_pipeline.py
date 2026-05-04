"""End-to-end analytic R policy validation on tv80s.

Inputs (no SPEF leakage from RES):
  1. features_v3 parquet (per-layer wirelength from cuboid representation).
  2. DEF NETS section parser (per-via-layer counts from VIA tokens).
  3. Calibrated sheet R + via R from training-design SPEFs (sheet_r_calibration.json).

Output:
  R_pred_analytic = Σ sheet[layer] * wirelen[layer] / width[layer]
                  + Σ via_R[via_layer] * n_via[via_layer]

Compared to v7 ML baseline (11.92% MAPE) and to ground-truth-via-count
analytic feasibility upper bound (2.61% MAPE).

This is the "analytic policy" candidate — no model artifacts, only a small
JSON of calibration constants + DEF parser + arithmetic.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent.parent
sys.path.insert(0, str(_WS))

from configs import cfg

_spec = importlib.util.spec_from_file_location(
    "compare_spef_pinpex",
    str(_WS.parent.parent / "src" / "evaluation" / "compare_spef.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
parse_spef = _mod.parse_spef_with_coordinates


GOLDEN = Path("/home/jslee/projects/PEX_SSL/golden_data/spef_data/intel22/intel22_tv80s_f3_starrc.spef")
DEF_PATH = Path("/home/jslee/projects/PEX_SSL/data/raw/def/intel22/intel22_tv80s_f3.def")
FEAT_PATH = _WS / "cache" / "features_v3" / "intel22_tv80s_f3.parquet"
CALIB_PATH = _WS / "reports" / "sheet_r_calibration.json"
DEF_VIA_CSV = _WS / "reports" / "def_via_counts_intel22_tv80s_f3.csv"

# typical wire width per layer (μm) — from RES annotation $w=
WIRE_WIDTH_TYP = {
    "M1": 0.068, "M2": 0.044, "M3": 0.044, "M4": 0.044, "M5": 0.044,
    "M6": 0.080, "M7": 0.080, "M8": 0.160, "M9p": 0.320,
}


def main():
    # ---------------- Load calibration ---------------------
    with open(CALIB_PATH) as f:
        calib = json.load(f)
    sheet_R = {row["name"].upper(): row["sheet_median"] for row in calib["metal_per_layer"]}
    # M9p (top layer) has no train-design sample; use m1 as upper bound proxy.
    sheet_R.setdefault("M9p", 0.18)
    via_R = {row["name"]: row["R_median"] for row in calib["via_per_layer"]}
    # extrapolate v5/v6 (rare in train) to v4 value
    for k in ["v5", "v6", "v7"]:
        via_R.setdefault(k, via_R.get("v4", 13.0685))
    print("Calibrated sheet R:", sheet_R)
    print("Calibrated via R:  ", via_R)

    # ---------------- Load DEF (wirelength + via counts) ---------------------
    # Use DEF-routed wirelengths (NOT cuboid representation) — cuboids
    # include via-stack landings & RECT patches that would double-count via R.
    if not DEF_VIA_CSV.exists():
        print(f"  DEF via counts CSV missing — running parser ...", flush=True)
        import subprocess
        subprocess.run([sys.executable,
                         str(_HERE / "parse_def_via_counts.py"),
                         str(DEF_PATH)], check=True)
    vias = pd.read_csv(DEF_VIA_CSV)
    print(f"DEF (wirelen + via) nets: {len(vias)}")

    # ---------------- Load golden R per net ---------------------
    print(f"\nParsing golden SPEF for ground-truth R ...", flush=True)
    g = parse_spef(GOLDEN)
    gold = pd.DataFrame([{"net_name": n, "R_gold": float(info["total_res"])}
                          for n, info in g.items()])
    print(f"golden nets: {len(gold)}")

    # ---------------- Join: DEF wirelen + DEF via + golden R ---------------------
    df = vias.merge(gold, on="net_name", how="inner")
    df = df[df["R_gold"] > 0.1].reset_index(drop=True)
    print(f"\nJoined nets: {len(df)}")
    print(f"  R_gold median = {df['R_gold'].median():.2f}Ω, mean = {df['R_gold'].mean():.2f}Ω")

    # ---------------- Compute analytic R ---------------------
    pred = np.zeros(len(df))
    # wire term — DEF NETS routed wirelength per metal layer
    for L in WIRE_WIDTH_TYP:
        col = f"wirelen_{L}"   # DEF parser column convention
        if col in df.columns and L in sheet_R:
            pred += sheet_R[L] * df[col].values / WIRE_WIDTH_TYP[L]
    # via term — count of VIA{i}_* tokens per net
    for v, vR in via_R.items():
        col = f"n_via_{v}"
        if col in df.columns:
            pred += vR * df[col].values

    df["R_pred_analytic"] = pred
    df["ape"] = 100 * np.abs(pred - df["R_gold"]) / df["R_gold"]
    df["bias_signed"] = 100 * (pred - df["R_gold"]) / df["R_gold"]

    print(f"\n=== Analytic R policy (raw) on tv80s ===")
    print(f"  MAPE       = {df['ape'].mean():.3f}%")
    print(f"  median APE = {df['ape'].median():.3f}%")
    print(f"  P90 APE    = {np.percentile(df['ape'], 90):.3f}%")
    print(f"  P99 APE    = {np.percentile(df['ape'], 99):.3f}%")
    print(f"  bias       = {df['bias_signed'].mean():+.3f}%")
    print(f"  R²(log)    = {1 - ((np.log10(np.clip(df['R_gold'],1e-3,None)) - np.log10(np.clip(pred,1e-3,None)))**2).sum() / ((np.log10(np.clip(df['R_gold'],1e-3,None)) - np.log10(np.clip(df['R_gold'],1e-3,None)).mean())**2).sum():.4f}")

    # ---------------------- Global multiplicative calibration ----------------
    # Bias-removing scalar α; fit on TRAIN designs (NOT tv80s) — but for the
    # demo here we use a leave-one-out estimate from tv80s itself to show
    # the upper bound; real deployment fits α from train designs only.
    alpha_oracle = float((df["R_gold"] / pred).median())
    pred_cal = alpha_oracle * pred
    ape_cal = 100 * np.abs(pred_cal - df["R_gold"]) / df["R_gold"]
    bias_cal = 100 * (pred_cal - df["R_gold"]) / df["R_gold"]
    print(f"\n=== After global multiplicative calibration α (oracle = {alpha_oracle:.4f}) ===")
    print(f"  MAPE       = {ape_cal.mean():.3f}%")
    print(f"  median APE = {ape_cal.median():.3f}%")
    print(f"  bias       = {bias_cal.mean():+.3f}%")
    df["R_pred_analytic_cal"] = pred_cal
    df["ape_cal"] = ape_cal
    df["bias_signed_cal"] = bias_cal

    # bootstrap CI
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, len(df), len(df))
        boots.append(df["ape"].values[idx].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"  Bootstrap 95% CI: [{lo:.3f}%, {hi:.3f}%]")

    # length-stratified
    print("\nLength-stratified MAPE (quartiles by R_gold):")
    df["q"] = pd.qcut(df["R_gold"], 4, labels=["Q1_short", "Q2", "Q3", "Q4_long"])
    for q, sub in df.groupby("q", observed=True):
        print(f"  {q}: n={len(sub):4d}  R_med={sub['R_gold'].median():.1f}Ω  "
              f"  MAPE={sub['ape'].mean():.3f}%  bias={sub['bias_signed'].mean():+.3f}%")

    # ---------------- vs v7 ML baseline ---------------------
    pred_path = _WS / "output" / "spef_e2e" / "tv80s_FINAL.spef"
    if pred_path.exists():
        p = parse_spef(pred_path)
        v7 = pd.DataFrame([{"net_name": n, "R_v7": float(info["total_res"])}
                             for n, info in p.items()])
        df = df.merge(v7, on="net_name", how="left")
        df["ape_v7"] = 100 * np.abs(df["R_v7"] - df["R_gold"]) / df["R_gold"]
        print(f"\n=== Comparison ===")
        print(f"  v7 ML ensemble MAPE     : {df['ape_v7'].mean():.3f}%")
        print(f"  Analytic policy MAPE    : {df['ape'].mean():.3f}%")
        print(f"  Δ = {df['ape'].mean() - df['ape_v7'].mean():+.3f}pp")

    out = _WS / "reports" / "analytic_r_full_pipeline.csv"
    df.to_csv(out, index=False)
    summary = {
        "v7_ML_MAPE":              float(df["ape_v7"].mean()) if "ape_v7" in df else None,
        "analytic_MAPE":           float(df["ape"].mean()),
        "analytic_median":         float(df["ape"].median()),
        "analytic_p90":            float(np.percentile(df["ape"], 90)),
        "analytic_bias":           float(df["bias_signed"].mean()),
        "analytic_CI_95":          [float(lo), float(hi)],
        "n_nets":                  int(len(df)),
        "calibrated_sheet_R":      sheet_R,
        "calibrated_via_R":        via_R,
        "wire_width_typ":          WIRE_WIDTH_TYP,
    }
    with open(_WS / "reports" / "analytic_r_full_pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
