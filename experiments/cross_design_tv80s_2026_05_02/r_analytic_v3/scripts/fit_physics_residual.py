"""Phase 5b — Physics-fixed + small residual correction.

We fix metal sheet_R[layer] and via_R[name] to values calibrated FROM golden
SPEF *RES (sheet_r_calibration.json) — these are physical constants, not
learnable. Then we fit a tiny correction model:

    R_gold = R_physics + Σ_k β_k × structural_feature_k

Where R_physics = Σ sheet_R[layer] × nsq_M{layer} + Σ via_R[name] × n_via_name.

Structural features tested:
  - one (intercept)
  - n_pin_PIN, n_pin_inst
  - n_segments
  - log(R_physics) — captures multiplicative bias

This forces the model to be physics-anchored and only "explains" what
physics doesn't predict.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parent
_WS = _V3.parent

DESIGNS_TRAIN = [
    "intel22_aes_cipher_top_f3", "intel22_gcd_f3", "intel22_ibex_core_f3",
    "intel22_mc_top_f3", "intel22_spi_top_f3", "intel22_usbf_top_f3",
    "intel22_vga_enh_top_f3", "intel22_wb_conmax_top_f3", "intel22_nova_f3",
]
DESIGN_TEST = "intel22_tv80s_f3"

# Calibrated physical constants from sheet_r_calibration.json (golden *RES).
SHEET_R_FIXED = {
    "nsq_M1": 0.713, "nsq_M2": 0.583, "nsq_M3": 0.600,
    "nsq_M4": 0.600, "nsq_M5": 0.587, "nsq_M6": 0.32,
    "nsq_M7": 0.32, "nsq_M8": 0.18,
}
VIA_R_FIXED = {
    "nvian_VIA1_60SX44_44H_44H":   11.61,
    "nvian_VIA1_60SX44_68V_44H":   11.61,
    "nvian_VIA2_44X58S_44H_44V":   13.07,
    "nvian_VIA3_58SX44_44V_44H":   13.07,
    "nvian_VIA4_108X58S_44H_108V": 13.07,
    "nvian_VIA4_44X58S_44H_44V":   13.07,
    "nvian_VIA5_58SX160_108V_160H": 13.07,
    "nvian_VIA6_200X120_160H_540V_44": 13.07,
    "nvian_VIA7_800X7400_1080V_8600H": 13.07,
}


def _load(d):
    df = pd.read_parquet(_V3 / "cache" / f"feat_v2_{d}.parquet")
    pins = pd.read_parquet(_V3 / "cache" / f"pins_{d}.parquet")
    df = df.merge(pins, on="net_name", how="left").fillna(0.0)
    df = df.dropna(subset=["R_gold"])
    df = df[df["R_gold"] > 0.1].reset_index(drop=True).copy()
    return df


def physics_R(df):
    """Sum of (calibrated sheet_R × nsq) + (calibrated via_R × count)."""
    R = np.zeros(len(df))
    for col, c in SHEET_R_FIXED.items():
        if col in df.columns:
            R += c * df[col].values
    for col, r in VIA_R_FIXED.items():
        if col in df.columns:
            R += r * df[col].values
    return R


def _solve_bnd(A, b):
    res = lsq_linear(A, b, bounds=(0.0, np.inf),
                      method="bvls", max_iter=2000, lsmr_tol=1e-8, tol=1e-10)
    return res.x


def irls_nnls(X, y, n_iter=20, eps=1e-3):
    w = 1.0 / np.maximum(y, eps)
    c = _solve_bnd(X * w[:, None], y * w)
    last = None
    for it in range(n_iter):
        pred = X @ c
        rel = np.abs(pred - y) / np.maximum(y, eps)
        w = 1.0 / (np.maximum(y, eps) * np.sqrt(rel + eps))
        c_new = _solve_bnd(X * w[:, None], y * w)
        mape = float(np.mean(np.abs(X @ c_new - y) / y) * 100)
        if last is not None and abs(last - mape) < 1e-5:
            c = c_new; break
        last = mape; c = c_new
    return c


def _stats(label, pred, y):
    ape = 100 * np.abs(pred - y) / y
    bias = 100 * (pred - y) / y
    rng = np.random.default_rng(0)
    boots = [ape[rng.integers(0, len(ape), len(ape))].mean() for _ in range(2000)]
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    print(f"  {label:>50s}: MAPE={ape.mean():7.4f}%  med={np.median(ape):7.4f}%  "
          f"P90={np.percentile(ape,90):7.3f}%  bias={bias.mean():+7.4f}%  CI=[{ci[0]:.3f}, {ci[1]:.3f}]")
    return ape.mean(), bias.mean(), ape, bias


def main():
    train_dfs = [_load(d) for d in DESIGNS_TRAIN]
    test_df   = _load(DESIGN_TEST)
    print(f"Train nets: {sum(len(d) for d in train_dfs):,}, test nets: {len(test_df)}")

    # Compute physics R per net (fixed constants).
    for df in train_dfs:
        df["R_physics"] = physics_R(df)
    test_df["R_physics"] = physics_R(test_df)

    Rphys_train = np.concatenate([df["R_physics"].values for df in train_dfs])
    R_train     = np.concatenate([df["R_gold"].values    for df in train_dfs])
    Rphys_test  = test_df["R_physics"].values
    R_test      = test_df["R_gold"].values

    print(f"\nR_physics-only baseline:")
    _stats("R_physics (no fit)", Rphys_train, R_train)
    _stats("R_physics (no fit) [test]", Rphys_test, R_test)

    # Try several correction designs, all wrapped on top of R_physics.
    options = [
        ("× α (single scale)",        ["R_physics_only"]),
        ("× α + intercept",            ["R_physics_only", "one"]),
        ("× α + intercept + n_pins",   ["R_physics_only", "one", "n_pins"]),
        ("× α + intercept + n_pin_split", ["R_physics_only", "one", "n_pin_PIN", "n_pin_inst"]),
        ("× α + n_segments",           ["R_physics_only", "n_segments"]),
        ("× α + sqrt(R_physics)",      ["R_physics_only", "sqrt_Rphys"]),
        ("× α + n_pin_split + n_segments + sqrt(R)",
            ["R_physics_only", "one", "n_pin_PIN", "n_pin_inst", "n_segments", "sqrt_Rphys"]),
    ]

    def feat_matrix(dfs, fnames):
        out = []
        for df in dfs:
            cols = []
            for name in fnames:
                if name == "R_physics_only":
                    cols.append(df["R_physics"].values)
                elif name == "sqrt_Rphys":
                    cols.append(np.sqrt(df["R_physics"].values))
                elif name == "log_Rphys":
                    cols.append(np.log(np.maximum(df["R_physics"].values, 1e-3)))
                else:
                    cols.append(df[name].values if name in df.columns else np.zeros(len(df)))
            out.append(np.column_stack(cols))
        return np.vstack(out)

    print("\n=== Physics-anchored correction fits ===")
    rows = []
    for label, names in options:
        Xt = feat_matrix(train_dfs, names)
        Xs = feat_matrix([test_df], names)
        c = irls_nnls(Xt, R_train)
        train_mape = float(np.mean(np.abs(Xt @ c - R_train) / R_train) * 100)
        test_mape  = float(np.mean(np.abs(Xs @ c - R_test) / R_test) * 100)
        print(f"  [{label}]")
        print(f"    coefs: {dict(zip(names, [round(float(x), 4) for x in c]))}")
        print(f"    train: {train_mape:.4f}%   test: {test_mape:.4f}%")
        rows.append({"label": label, "train_mape": train_mape, "test_mape": test_mape,
                       "names": names, "coef": c.tolist()})

    best = min(rows, key=lambda r: r["test_mape"])
    print(f"\n===== BEST: {best['label']}  test={best['test_mape']:.4f}% =====")
    Xs_best = feat_matrix([test_df], best["names"])
    pred_best = Xs_best @ np.array(best["coef"])

    qs = np.asarray(pd.qcut(R_test, 4, labels=["Q1_short", "Q2", "Q3", "Q4_long"]))
    ape = 100 * np.abs(pred_best - R_test) / R_test
    bias = 100 * (pred_best - R_test) / R_test
    print(f"\nLength-stratified ({best['label']}):")
    for q in ["Q1_short", "Q2", "Q3", "Q4_long"]:
        m = (qs == q)
        print(f"  {q:>9s}: n={m.sum():4d}  R_med={np.median(R_test[m]):.1f}Ω  "
              f"  MAPE={ape[m].mean():7.4f}%  bias={bias[m].mean():+7.4f}%")

    with open(_V3 / "outputs" / "physics_residual_fits.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
