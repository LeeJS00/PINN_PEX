"""Test the *analytic* R policy on tv80s.

Three reconstructions, increasing in fidelity:

  (A) Sum-of-golden-RES sanity:   R_net = Σ R_{seg in *RES}    vs  *D_NET total_res
       Confirms the SPEF *RES section IS the full topology — lumped total = sum.

  (B) Per-layer-wirelength analytic (no vias):
       R_net = Σ_layer  sheet_calib[layer] * wirelen[layer] / width_typ[layer]
       Pure feature-space analytic. NO ML, NO calibration scale.
       Compares MAPE to v7 (11.92%) and to our default_sheet (39%).

  (C) Per-layer-wirelength + per-via analytic (uses ground-truth via count
       extracted from golden RES):
       R_net = wire_part(B) + Σ_via_lvl  via_calib[via_lvl] * n_via_lvl
       This is the BEST analytic policy can do given accurate via counts.
       Tells us how good a pure analytic policy with via-features could be.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent.parent
sys.path.insert(0, str(_WS))

from configs import cfg

# load PINNPEX SPEF parser (for total_res from *D_NET line)
_spec = importlib.util.spec_from_file_location(
    "compare_spef_pinpex",
    str(_WS.parent.parent / "src" / "evaluation" / "compare_spef.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
parse_spef = _mod.parse_spef_with_coordinates


GOLDEN = Path("/home/jslee/projects/PEX_SSL/golden_data/spef_data/intel22/intel22_tv80s_f3_starrc.spef")
CALIB_PATH = _WS / "reports" / "sheet_r_calibration.json"

LAYER_INDEX_TO_NAME = {
    1: "c4", 2: "m8", 3: "m7", 4: "m6", 5: "m5", 6: "m4",
    7: "m3", 8: "m2", 9: "m1",
    10: "v1", 11: "v2", 12: "v3", 13: "v4", 14: "v5", 15: "v6", 16: "v7",
}

# Inverse map (name → SPEF lvl index)
NAME_TO_LVL = {v: k for k, v in LAYER_INDEX_TO_NAME.items()}

# Typical width per metal layer in μm. Calibrated from golden RES annotations
# ($w=...) — minimum-width nets dominate the distribution. m1 is wider track.
WIRE_WIDTH_TYP = {
    "m1": 0.068, "m2": 0.044, "m3": 0.044, "m4": 0.044, "m5": 0.044,
    "m6": 0.080, "m7": 0.080, "m8": 0.160,
}

LINE_RE = re.compile(
    r"^\s*\d+\s+\S+\s+\S+\s+(?P<R>[\d.eE+-]+)\s*//\s*(?P<ann>.+)$"
)
KV_RE = re.compile(r"\$(\w+)=([-+]?[\d.eE]+(?:[A-Za-z][\w]*)?|\d+x\d+|[\w_]+)")


def stream_d_nets_with_res(spef_path: Path):
    """Yield (net_name, total_res_label, list_of_(R, kv)) — full per-net RES table."""
    cur_net = None
    cur_total = None
    cur_res = []
    in_res = False
    with open(spef_path, "r", errors="replace") as f:
        for line in f:
            s = line.rstrip()
            if s.startswith("*D_NET"):
                if cur_net is not None:
                    yield cur_net, cur_total, cur_res
                parts = s.split()
                cur_net = parts[1]
                cur_total = float(parts[2]) if len(parts) > 2 else float("nan")
                cur_res = []
                in_res = False
            elif s.startswith("*RES"):
                in_res = True
            elif s.startswith("*END"):
                in_res = False
            elif in_res and s and not s.startswith("*"):
                m = LINE_RE.match(s)
                if not m:
                    continue
                R = float(m.group("R"))
                ann = m.group("ann")
                kv = dict(KV_RE.findall(ann))
                cur_res.append((R, kv))
    if cur_net is not None:
        yield cur_net, cur_total, cur_res


def main():
    # ---------------------------------------------------------------------
    # Load calibration
    # ---------------------------------------------------------------------
    with open(CALIB_PATH) as f:
        calib = json.load(f)
    sheet_R = {row["name"]: row["sheet_median"] for row in calib["metal_per_layer"]}
    via_R_per_layer = {row["name"]: row["R_median"] for row in calib["via_per_layer"]}
    print("Calibrated sheet R (Ω/sq):", sheet_R)
    print("Calibrated via R (Ω):     ", via_R_per_layer)

    # ---------------------------------------------------------------------
    # Stream golden SPEF — collect per-net (total_res, sum_RES, wirelen[layer], n_vias[via_layer])
    # ---------------------------------------------------------------------
    print(f"\nStreaming golden SPEF: {GOLDEN}", flush=True)
    rows = []
    for net, dnet_val, res_list in stream_d_nets_with_res(GOLDEN):
        # NOTE: SPEF *D_NET <name> <X> -> X is total CAPACITANCE not resistance.
        # Ground-truth lumped R = sum of all *RES R values (this IS what
        # compare_spef.py computes as 'total_res').
        wirelen = defaultdict(float)
        wire_R_true = 0.0  # ground-truth sum of metal R per net (for diagnosis)
        n_vias_layer = defaultdict(int)
        sum_R = 0.0
        for R, kv in res_list:
            sum_R += R
            lvl = int(kv.get("lvl", -1))
            name = LAYER_INDEX_TO_NAME.get(lvl)
            if name is None:
                continue
            if "l" in kv and "w" in kv:
                L = float(kv["l"])
                if L > 0:
                    wirelen[name] += L
                    wire_R_true += R
            elif "vc" in kv:
                n_vias_layer[name] += 1
        rows.append({
            "net": net,
            "total_cap_dnet":  float(dnet_val),
            "total_res":       float(sum_R),       # GROUND-TRUTH lumped R
            "wire_R_true":     float(wire_R_true),
            **{f"wirelen_{k}": v for k, v in wirelen.items()},
            **{f"n_via_{k}": v for k, v in n_vias_layer.items()},
        })
    df = pd.DataFrame(rows).fillna(0.0)
    df = df[df["total_res"] > 0.1].reset_index(drop=True)
    print(f"  parsed {len(df)} nets   (median R = {df['total_res'].median():.2f}Ω, mean = {df['total_res'].mean():.2f}Ω)")

    # ---------------------------------------------------------------------
    # (A) Sanity — does Σ RES make sense? (total_cap_dnet for reference)
    # ---------------------------------------------------------------------
    print(f"\n(A) Sanity — total_res = Σ RES by definition; total_cap_dnet ranges "
          f"{df['total_cap_dnet'].min():.3f} ~ {df['total_cap_dnet'].max():.1f} fF")

    # ---------------------------------------------------------------------
    # (B) Pure metal-only analytic (no vias) using calibrated sheet R
    # ---------------------------------------------------------------------
    pred_B = np.zeros(len(df))
    for layer, sR in sheet_R.items():
        col = f"wirelen_{layer}"
        if col in df.columns and layer in WIRE_WIDTH_TYP:
            pred_B += sR * df[col].values / WIRE_WIDTH_TYP[layer]
    df["R_analytic_metal"] = pred_B
    df["ape_B"] = 100 * np.abs(pred_B - df["total_res"]) / df["total_res"]
    df["bias_B_signed"] = 100 * (pred_B - df["total_res"]) / df["total_res"]
    print(f"\n(B) Metal-only analytic (calibrated sheet R, no vias):")
    print(f"     MAPE={df['ape_B'].mean():.3f}%  median={df['ape_B'].median():.3f}%  "
          f"bias={df['bias_B_signed'].mean():+.3f}%")
    # how close is pred_B to the true metal R (excludes vias)?
    df["ape_metal_only_vs_truth"] = 100 * np.abs(pred_B - df["wire_R_true"]) / np.maximum(df["wire_R_true"], 1e-3)
    print(f"     vs ground-truth metal R (excl. vias): MAPE={df['ape_metal_only_vs_truth'].mean():.3f}%")

    # ---------------------------------------------------------------------
    # (C) Metal-only + per-via R using GROUND-TRUTH via counts from RES
    # ---------------------------------------------------------------------
    pred_C = pred_B.copy()
    for via_lvl, vR in via_R_per_layer.items():
        col = f"n_via_{via_lvl}"
        if col in df.columns:
            pred_C += vR * df[col].values
    df["R_analytic_full"] = pred_C
    df["ape_C"] = 100 * np.abs(pred_C - df["total_res"]) / df["total_res"]
    df["bias_C_signed"] = 100 * (pred_C - df["total_res"]) / df["total_res"]
    print(f"\n(C) Metal+via analytic (ground-truth via counts, calibrated R):")
    print(f"     MAPE={df['ape_C'].mean():.3f}%  median={df['ape_C'].median():.3f}%  "
          f"bias={df['bias_C_signed'].mean():+.3f}%")
    print(f"     P90={np.percentile(df['ape_C'], 90):.3f}%  P99={np.percentile(df['ape_C'], 99):.3f}%")

    # length-stratified MAPE for (C)
    print(f"\nLength-stratified MAPE (quartiles by total_res):")
    df["q"] = pd.qcut(df["total_res"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    for q, sub in df.groupby("q", observed=True):
        print(f"  {q}: n={len(sub):4d}  R_med={sub['total_res'].median():.1f}Ω  "
              f"  ape_B={sub['ape_B'].mean():.3f}%  ape_C={sub['ape_C'].mean():.3f}%  "
              f"bias_C={sub['bias_C_signed'].mean():+.3f}%")

    # ---------------------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------------------
    out_csv = _WS / "reports" / "analytic_r_feasibility.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    summary = {
        "v7_baseline_MAPE": 11.925,
        "B_metal_only_analytic_MAPE": float(df["ape_B"].mean()),
        "B_metal_vs_true_metal_only": float(df["ape_metal_only_vs_truth"].mean()),
        "C_metal_plus_via_analytic_MAPE": float(df["ape_C"].mean()),
        "C_median_MAPE": float(df["ape_C"].median()),
        "C_bias": float(df["bias_C_signed"].mean()),
        "calibrated_sheet_R": sheet_R,
        "calibrated_via_R": via_R_per_layer,
    }
    with open(_WS / "reports" / "analytic_r_feasibility_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nVerdict:")
    if summary["C_metal_plus_via_analytic_MAPE"] < 4.0:
        print(f"  ✅ Analytic policy with via-count features could hit <4% MAPE.")
    else:
        print(f"  ⚠ Analytic policy ceiling is {summary['C_metal_plus_via_analytic_MAPE']:.2f}% — gap to 4% is {summary['C_metal_plus_via_analytic_MAPE'] - 4:.2f}pp.")


if __name__ == "__main__":
    main()
