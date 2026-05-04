#!/usr/bin/env python3
"""
08_b1_stratified_report.py — A2 audit's 24h next-action.

Build per-design / per-quartile / per-channel breakdown of B1 XGBoost
performance on v3 valid split. Answers two paper-critical questions:

1. Where is the 4.66% headline concentrated? (per-quartile by golden cap)
2. Per-channel reality vs cancellation-driven total: gnd / cpl / total separately.

Output: pex_v3/output/baselines/B1_xgboost_real/stratified_report.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "pex_v3"))

import numpy as np
import pandas as pd

from src.evaluation.metrics import cap_mape, cap_mape_summary


def main():
    # Use seed 0 predictions (representative; all seeds tightly clustered per per_method.csv)
    pred_csv = (
        _PROJECT_ROOT
        / "pex_v3" / "output" / "baselines" / "B1_xgboost_real"
        / "seed0" / "eval_predictions.csv"
    )
    df = pd.read_csv(pred_csv)
    print(f">>> loaded {len(df):,} rows from {pred_csv.name}")
    print(f"    columns: {list(df.columns)}")

    # ---------------- Per-channel MAPE on full set ----------------
    print()
    print("=== Channel-level MAPE (B1 seed 0, valid split) ===")
    for ch in ["gnd", "cpl", "total"]:
        pred = df[f"pred_{ch}_fF"].to_numpy()
        gold = df[f"golden_{ch}_fF"].to_numpy()
        s = cap_mape_summary(pred, gold)
        print(f"  {ch:5s}: median={s['median_mape']*100:6.3f}%  "
              f"mean={s['mean_mape']*100:6.3f}%  "
              f"P95={s['p95_mape']*100:6.2f}%  "
              f"n_valid={s['n_valid']}/{s['n_valid']+s['n_zero_target']}")

    # ---------------- Per-design breakdown ----------------
    rows = []
    for design, sub in df.groupby("design_name"):
        for ch in ["gnd", "cpl", "total"]:
            pred = sub[f"pred_{ch}_fF"].to_numpy()
            gold = sub[f"golden_{ch}_fF"].to_numpy()
            s = cap_mape_summary(pred, gold)
            chip_ratio = float(pred.sum() / gold.sum()) if gold.sum() > 0 else float("nan")
            rows.append({
                "design": design,
                "channel": ch,
                "n_nets": len(sub),
                "median_mape": s.get("median_mape"),
                "mean_mape": s.get("mean_mape"),
                "p95_mape": s.get("p95_mape"),
                "chip_ratio": chip_ratio,
            })

    by_design = pd.DataFrame(rows)
    by_design.to_csv(
        _PROJECT_ROOT / "pex_v3" / "output" / "baselines" / "B1_xgboost_real"
        / "stratified_per_design.csv",
        index=False,
    )

    print()
    print("=== Per-design × per-channel (median MAPE %) ===")
    pivot = by_design.pivot(index="design", columns="channel", values="median_mape")
    pivot = pivot * 100
    print(pivot.to_string(float_format=lambda v: f"{v:6.2f}"))

    # ---------------- Per-quartile of golden_total ----------------
    print()
    print("=== Per-quartile (by golden_total_fF) ===")
    quartile_bins_fF = (0.0, 0.05, 0.5, 5.0, np.inf)
    quartile_labels = ["Q1 (<0.05fF)", "Q2 (0.05-0.5)", "Q3 (0.5-5)", "Q4 (>5fF)"]
    df = df.copy()
    df["q_bin"] = pd.cut(
        df["golden_total_fF"],
        bins=list(quartile_bins_fF),
        labels=quartile_labels,
        right=False,
        include_lowest=True,
    )

    quartile_rows = []
    for q, sub in df.groupby("q_bin", observed=True):
        for ch in ["gnd", "cpl", "total"]:
            pred = sub[f"pred_{ch}_fF"].to_numpy()
            gold = sub[f"golden_{ch}_fF"].to_numpy()
            s = cap_mape_summary(pred, gold)
            quartile_rows.append({
                "quartile": str(q),
                "channel": ch,
                "n_nets": len(sub),
                "median_mape": s.get("median_mape"),
                "mean_mape": s.get("mean_mape"),
                "p95_mape": s.get("p95_mape"),
            })
    by_q = pd.DataFrame(quartile_rows)
    by_q.to_csv(
        _PROJECT_ROOT / "pex_v3" / "output" / "baselines" / "B1_xgboost_real"
        / "stratified_per_quartile.csv",
        index=False,
    )

    pivot_q = by_q.pivot(index="quartile", columns="channel", values="median_mape") * 100
    print(pivot_q.to_string(float_format=lambda v: f"{v:6.2f}" if pd.notnull(v) else "n/a"))

    print()
    print("=== Counts per quartile ===")
    print(df["q_bin"].value_counts().sort_index().to_string())

    print()
    print("=== Files written ===")
    print(f"  {_PROJECT_ROOT}/pex_v3/output/baselines/B1_xgboost_real/stratified_per_design.csv")
    print(f"  {_PROJECT_ROOT}/pex_v3/output/baselines/B1_xgboost_real/stratified_per_quartile.csv")


if __name__ == "__main__":
    main()
