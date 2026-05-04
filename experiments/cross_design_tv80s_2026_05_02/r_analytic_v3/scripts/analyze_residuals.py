"""Phase 4 — Diagnose where the remaining 3.6% MAPE comes from.

Look at:
  1. Worst-error nets — are they outliers or systematic?
  2. Per-design residual mean (suggests design-level miscalibration).
  3. Residual vs net structure (n_segments, n_pins, total_wirelen).
  4. Sign-of-residual vs n_via_<NAME>: which via type pushes pred up/down?
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parent

DESIGNS_TRAIN = [
    "intel22_aes_cipher_top_f3", "intel22_gcd_f3", "intel22_ibex_core_f3",
    "intel22_mc_top_f3", "intel22_spi_top_f3", "intel22_usbf_top_f3",
    "intel22_vga_enh_top_f3", "intel22_wb_conmax_top_f3", "intel22_nova_f3",
]


def _load(design):
    df = pd.read_parquet(_V3 / "cache" / f"feat_{design}.parquet")
    df = df.dropna(subset=["R_gold"])
    df = df[df["R_gold"] > 0.1].reset_index(drop=True).copy()
    return df


def main():
    coefs = json.load(open(_V3 / "outputs" / "coefs_irls_log.json"))
    fcols = coefs["feature_columns"]
    c_irls = np.array([coefs["coefficients_irls"][c] for c in fcols])

    rows_per_design = []
    for d in DESIGNS_TRAIN + ["intel22_tv80s_f3"]:
        df = _load(d)
        X = np.zeros((len(df), len(fcols)))
        for j, c in enumerate(fcols):
            if c in df.columns:
                X[:, j] = df[c].values
        pred = X @ c_irls
        df["pred"] = pred
        df["res_signed"] = (pred - df["R_gold"]) / df["R_gold"] * 100
        df["ape"] = (pred - df["R_gold"]).abs() / df["R_gold"] * 100
        df["design"] = d
        rows_per_design.append(df)

    full = pd.concat(rows_per_design, ignore_index=True)

    # ---------------- Per-design summary ----------------
    print("=== Per-design residual summary ===")
    summary = full.groupby("design").agg(
        n=("R_gold", "count"),
        ape_mean=("ape", "mean"),
        ape_median=("ape", "median"),
        ape_p90=("ape", lambda s: float(np.percentile(s, 90))),
        bias=("res_signed", "mean"),
    ).reset_index()
    print(summary.to_string(index=False))

    test = full[full["design"] == "intel22_tv80s_f3"].copy()

    # ---------------- Worst nets ----------------
    print("\n=== Top-30 worst nets (tv80s) ===")
    worst = test.nlargest(30, "ape")
    cols = ["net_name", "R_gold", "pred", "ape", "res_signed", "n_segments"]
    extras = [c for c in test.columns if c.startswith("nsq_M") and test[c].sum() > 0]
    print(worst[cols + extras].to_string(index=False))

    # ---------------- Bin worst by sign ----------------
    print("\n=== Sign of residual (tv80s) ===")
    for thr in [1, 3, 5, 10, 20, 50]:
        m = test["ape"] > thr
        if m.sum():
            print(f"  ape > {thr:>3}%: n={m.sum():4d} ({100*m.mean():5.2f}%)  "
                  f"mean signed={test.loc[m, 'res_signed'].mean():+6.2f}%  "
                  f"mean R_gold={test.loc[m, 'R_gold'].mean():.1f}Ω")

    # ---------------- Residual vs n_segments ----------------
    print("\n=== Residual buckets by n_segments (tv80s) ===")
    test["seg_bin"] = pd.cut(test["n_segments"], bins=[0,2,4,8,16,32,1e6],
                              labels=["1-2","3-4","5-8","9-16","17-32",">32"])
    grp = test.groupby("seg_bin", observed=True).agg(
        n=("R_gold", "count"),
        R_med=("R_gold", "median"),
        ape=("ape", "mean"),
        bias=("res_signed", "mean"),
    )
    print(grp.to_string())

    # ---------------- Correlation: residual vs each via_type ----------------
    print("\n=== Correlation of residual sign with each via type (tv80s) ===")
    via_cols = [c for c in test.columns if c.startswith("nvian_")]
    cors = []
    for c in via_cols:
        if test[c].sum() == 0:
            continue
        cor = np.corrcoef(test[c], test["res_signed"])[0, 1]
        cors.append((c, float(cor), int(test[c].sum())))
    for c, r, n in sorted(cors, key=lambda x: abs(x[1]), reverse=True):
        print(f"  {c:<35s}  corr={r:+.4f}  total_count={n}")

    # ---------------- save residual table for further use ----------------
    test.to_parquet(_V3 / "outputs" / "tv80s_residuals.parquet")
    print(f"\nSaved: {_V3 / 'outputs' / 'tv80s_residuals.parquet'}")


if __name__ == "__main__":
    main()
