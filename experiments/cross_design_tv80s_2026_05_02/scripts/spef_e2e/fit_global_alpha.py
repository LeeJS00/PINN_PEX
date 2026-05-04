"""Fit a single global multiplicative calibration α on training designs and
report the analytic R policy MAPE on the test design (tv80s).

α is the only "fitted" parameter. Everything else (sheet R per layer, via R
per via_layer, wire widths) comes from physical calibration.

Output: alpha_global.json + per-design diagnostics.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
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

CALIB_PATH = _WS / "reports" / "sheet_r_calibration.json"

WIRE_WIDTH_TYP = {
    "M1": 0.068, "M2": 0.044, "M3": 0.044, "M4": 0.044, "M5": 0.044,
    "M6": 0.080, "M7": 0.080, "M8": 0.160, "M9p": 0.320,
}

DEF_DIR = Path("/home/jslee/projects/PEX_SSL/data/raw/def/intel22")
SPEF_DIR = Path("/home/jslee/projects/PEX_SSL/golden_data/spef_data/intel22")

TRAIN_DESIGNS = [
    "intel22_aes_cipher_top_f3",
    "intel22_gcd_f3",
    "intel22_ibex_core_f3",
    "intel22_mc_top_f3",
    "intel22_spi_top_f3",
    "intel22_usbf_top_f3",
    "intel22_vga_enh_top_f3",
    "intel22_wb_conmax_top_f3",
    "intel22_nova_f3",
]
TEST_DESIGN = "intel22_tv80s_f3"


def parse_def_for(design: str) -> pd.DataFrame:
    """Run parse_def_via_counts.py for a design and return DataFrame."""
    csv_path = _WS / "reports" / f"def_via_counts_{design}.csv"
    if not csv_path.exists():
        def_path = DEF_DIR / f"{design}.def"
        print(f"  parsing DEF for {design} ...", flush=True)
        subprocess.run([sys.executable,
                         str(_HERE / "parse_def_via_counts.py"),
                         str(def_path)], check=True,
                        capture_output=True)
    return pd.read_csv(csv_path)


def get_golden_R(design: str) -> pd.DataFrame:
    spef = SPEF_DIR / f"{design}_starrc.spef"
    g = parse_spef(spef)
    return pd.DataFrame([{"net_name": n, "R_gold": float(info["total_res"])}
                         for n, info in g.items()])


def compute_pred(df_def: pd.DataFrame, sheet_R, via_R) -> np.ndarray:
    pred = np.zeros(len(df_def))
    for L in WIRE_WIDTH_TYP:
        col = f"wirelen_{L}"
        if col in df_def.columns and L in sheet_R:
            pred += sheet_R[L] * df_def[col].values / WIRE_WIDTH_TYP[L]
    for v, vR in via_R.items():
        col = f"n_via_{v}"
        if col in df_def.columns:
            pred += vR * df_def[col].values
    return pred


def main():
    with open(CALIB_PATH) as f:
        calib = json.load(f)
    sheet_R = {row["name"].upper(): row["sheet_median"] for row in calib["metal_per_layer"]}
    sheet_R.setdefault("M9p", 0.18)
    via_R = {row["name"]: row["R_median"] for row in calib["via_per_layer"]}
    for k in ["v5", "v6", "v7"]:
        via_R.setdefault(k, via_R.get("v4", 13.07))

    print("Calibrated sheet R:", sheet_R)
    print("Calibrated via R:  ", via_R)

    # ---------------- TRAIN: estimate α per design + global ----------------
    rows_per_design = []
    all_pred = []
    all_gold = []
    for d in TRAIN_DESIGNS:
        df_def = parse_def_for(d)
        df_g = get_golden_R(d)
        joined = df_def.merge(df_g, on="net_name").query("R_gold > 0.1")
        if len(joined) == 0:
            print(f"  [skip] {d}: no joined nets")
            continue
        pred = compute_pred(joined, sheet_R, via_R)
        # per-design α candidates: median ratio, mean ratio, RMSE-fit, MAPE-min
        ratio = joined["R_gold"].values / np.maximum(pred, 1e-3)
        alpha_med   = float(np.median(ratio))
        # MAPE-min α: minimize |α p - y| / y → solve α* = argmin
        # closed-form for L2: α = Σ (p y)/y² ... but for MAPE no closed form.
        # use weighted-median weight = 1/y as MAPE-style estimate.
        sort_idx = np.argsort(ratio)
        sorted_ratio = ratio[sort_idx]
        weights = (1.0 / np.maximum(joined["R_gold"].values, 1e-3))[sort_idx]
        cum = np.cumsum(weights) / weights.sum()
        alpha_wmedian = float(sorted_ratio[np.searchsorted(cum, 0.5)])
        ape_with_alpha = lambda a: float(np.mean(np.abs(a*pred - joined['R_gold'].values) / joined['R_gold'].values) * 100)
        rows_per_design.append({
            "design": d, "n_nets": len(joined),
            "alpha_med": alpha_med,
            "alpha_wmedian": alpha_wmedian,
            "raw_MAPE": ape_with_alpha(1.0),
            "med_MAPE": ape_with_alpha(alpha_med),
            "wmed_MAPE": ape_with_alpha(alpha_wmedian),
        })
        all_pred.append(pred)
        all_gold.append(joined["R_gold"].values)

    df_train = pd.DataFrame(rows_per_design)
    print("\n=== Per-train-design α candidates ===")
    print(df_train.to_string(index=False))

    # Global α: minimize sum of MAPE across all training nets
    pred_all = np.concatenate(all_pred)
    gold_all = np.concatenate(all_gold)

    # Try both alpha types
    alpha_global_med = float(np.median(gold_all / np.maximum(pred_all, 1e-3)))
    # MAPE-minimizing α via grid search
    grid = np.linspace(0.8, 1.5, 501)
    apes = [float(np.mean(np.abs(a*pred_all - gold_all)/gold_all)*100) for a in grid]
    alpha_global_mape = float(grid[int(np.argmin(apes))])
    print(f"\nGlobal α (median of ratio across {len(gold_all)} train nets): {alpha_global_med:.4f}")
    print(f"Global α (MAPE-min via grid):                                 {alpha_global_mape:.4f}")
    print(f"Min MAPE on train (with α=alpha_global_mape):                 {min(apes):.3f}%")

    # ---------------- TEST: apply α_global to tv80s ----------------
    df_def_t = parse_def_for(TEST_DESIGN)
    df_g_t   = get_golden_R(TEST_DESIGN)
    joined_t = df_def_t.merge(df_g_t, on="net_name").query("R_gold > 0.1")
    pred_t = compute_pred(joined_t, sheet_R, via_R)
    gold_t = joined_t["R_gold"].values

    def _stats(label, pred):
        ape = 100 * np.abs(pred - gold_t) / gold_t
        bias = 100 * (pred - gold_t) / gold_t
        rng = np.random.default_rng(0)
        boots = [ape[rng.integers(0, len(ape), len(ape))].mean() for _ in range(2000)]
        ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
        print(f"  {label:>32s}: MAPE={ape.mean():6.3f}%  med={np.median(ape):6.3f}%  "
              f"P90={np.percentile(ape, 90):6.2f}%  bias={bias.mean():+6.3f}%  CI=[{ci[0]:.2f}, {ci[1]:.2f}]")

    print(f"\n=== TEST: tv80s with train-fit α (n={len(joined_t)}) ===")
    _stats("raw analytic (α=1)",        pred_t)
    _stats("train-α_med (no offset)",   alpha_global_med * pred_t)
    _stats("train-α_mape-min",          alpha_global_mape * pred_t)
    # for reference: oracle α from tv80s itself
    alpha_oracle = float(np.median(gold_t / np.maximum(pred_t, 1e-3)))
    _stats(f"ORACLE α={alpha_oracle:.3f} (cheat)", alpha_oracle * pred_t)

    # save
    out = {
        "calibrated_sheet_R": sheet_R,
        "calibrated_via_R":   via_R,
        "wire_width_typ":     WIRE_WIDTH_TYP,
        "alpha_global_median":   alpha_global_med,
        "alpha_global_mape_min": alpha_global_mape,
        "per_train_design":      rows_per_design,
        "test_design":           TEST_DESIGN,
    }
    with open(_WS / "reports" / "alpha_global.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: reports/alpha_global.json")


if __name__ == "__main__":
    main()
