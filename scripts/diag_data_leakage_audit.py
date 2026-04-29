#!/usr/bin/env python3
"""
Audit train/validation design overlap and summarize per-group MAPE.

Usage:
    source tool.env && python3 scripts/diag_data_leakage_audit.py

This script is CSV-only. It compares the predefined train/validation manifests,
documents the design-overlap leakage situation, optionally merges in
diagnose_hurdle.csv for validation-net error summaries, writes a compact group
table, and prints recommended conclusions.
"""

import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/jslee/projects/PINNPEX")


REPO_ROOT = Path("/home/jslee/projects/PINNPEX")
TRAIN_MANIFEST = REPO_ROOT / "output_intel22/active_learning/cache/predefined_train_subset.csv"
VAL_MANIFEST = REPO_ROOT / "output_intel22/active_learning/cache/predefined_valid_subset.csv"
HURDLE_CSV = REPO_ROOT / "output_intel22/active_learning/v3_netlevel/diagnose_hurdle.csv"
OUTPUT_DIR = REPO_ROOT / "output_intel22/diag"
OUTPUT_CSV = OUTPUT_DIR / "leakage_audit.csv"

AL_PREDEFINED_DESIGNS = [
    "intel22_gcd_f3",
    "intel22_spi_top_f3",
    "intel22_aes_cipher_top_f3",
]
RECOMMENDED_HOLDOUT = [
    "intel22_vga_enh_top_f3",
    "intel22_wb_conmax_top_f3",
]
KNOWN_NO_SPEF_OOD = ["nova_f3", "tv80s_f3"]


def print_header(title):
    print(f"\n=== {title} ===")


def warn(msg):
    warnings.warn(msg, RuntimeWarning)


def safe_mean(series):
    series = pd.Series(series).dropna()
    return float(series.mean()) if len(series) else math.nan


def safe_median(series):
    series = pd.Series(series).dropna()
    return float(series.median()) if len(series) else math.nan


def make_group_row(group_name, net_names, hurdle_df):
    net_names = sorted({str(x) for x in net_names if pd.notna(x)})
    row = {"group": group_name, "n_nets": int(len(net_names)), "mean_mape": math.nan, "median_ratio": math.nan}
    if hurdle_df is None or len(net_names) == 0:
        return row

    subset = hurdle_df[hurdle_df["net_name"].isin(net_names)]
    row["n_nets"] = int(subset["net_name"].nunique())
    if "rel_err" in subset.columns:
        row["mean_mape"] = safe_mean(subset["rel_err"])
    if "ratio" in subset.columns:
        row["median_ratio"] = safe_median(subset["ratio"])
    return row


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print_header("Inputs")
    print(f"Train manifest : {TRAIN_MANIFEST}")
    print(f"Val manifest   : {VAL_MANIFEST}")
    print(f"Diagnose CSV   : {HURDLE_CSV}")

    if not TRAIN_MANIFEST.exists():
        warn(f"Missing train manifest: {TRAIN_MANIFEST}")
    if not VAL_MANIFEST.exists():
        warn(f"Missing val manifest: {VAL_MANIFEST}")

    if not TRAIN_MANIFEST.exists() or not VAL_MANIFEST.exists():
        pd.DataFrame(columns=["group", "n_nets", "mean_mape", "median_ratio"]).to_csv(OUTPUT_CSV, index=False)
        print(f"Wrote empty output to {OUTPUT_CSV}")
        return

    train_df = pd.read_csv(TRAIN_MANIFEST)
    val_df = pd.read_csv(VAL_MANIFEST)

    required = {"design_name", "net_name"}
    for name, df in [("train", train_df), ("val", val_df)]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name} manifest missing required columns: {sorted(missing)}")

    train_designs = set(train_df["design_name"].dropna().astype(str))
    val_designs = set(val_df["design_name"].dropna().astype(str))
    overlap_designs = train_designs & val_designs
    val_only_designs = val_designs - train_designs
    train_only_designs = train_designs - val_designs

    train_nets = set(train_df["net_name"].dropna().astype(str))
    val_nets = set(val_df["net_name"].dropna().astype(str))
    overlap_nets = train_nets & val_nets

    hurdle_df = None
    if HURDLE_CSV.exists():
        hurdle_df = pd.read_csv(HURDLE_CSV)
        needed = {"net_name", "rel_err", "ratio"}
        missing = needed - set(hurdle_df.columns)
        if missing:
            warn(f"Diagnose CSV missing columns {sorted(missing)}; skipping MAPE analysis")
            hurdle_df = None
    else:
        warn(f"Missing diagnose_hurdle.csv: {HURDLE_CSV}. Skipping MAPE analysis.")

    val_al_designs = set(AL_PREDEFINED_DESIGNS) & val_designs
    val_non_al_designs = val_designs - set(AL_PREDEFINED_DESIGNS)
    val_holdout_designs = set(RECOMMENDED_HOLDOUT) & val_designs
    val_remaining_designs = val_designs - set(RECOMMENDED_HOLDOUT)

    val_overlap_df = val_df[val_df["design_name"].isin(overlap_designs)]
    val_non_overlap_df = val_df[val_df["design_name"].isin(val_only_designs)]
    val_al_df = val_df[val_df["design_name"].isin(val_al_designs)]
    val_non_al_df = val_df[val_df["design_name"].isin(val_non_al_designs)]
    val_holdout_df = val_df[val_df["design_name"].isin(val_holdout_designs)]
    val_remaining_df = val_df[val_df["design_name"].isin(val_remaining_designs)]

    rows = [
        make_group_row("val_all", val_df["net_name"], hurdle_df),
        make_group_row("val_designs_seen_in_train", val_overlap_df["net_name"], hurdle_df),
        make_group_row("val_designs_unseen_in_train", val_non_overlap_df["net_name"], hurdle_df),
        make_group_row("val_net_names_seen_in_train", val_df[val_df["net_name"].isin(overlap_nets)]["net_name"], hurdle_df),
        make_group_row("val_net_names_unseen_in_train", val_df[~val_df["net_name"].isin(overlap_nets)]["net_name"], hurdle_df),
        make_group_row("val_al_designs", val_al_df["net_name"], hurdle_df),
        make_group_row("val_non_al_designs", val_non_al_df["net_name"], hurdle_df),
        make_group_row("recommended_holdout_designs", val_holdout_df["net_name"], hurdle_df),
        make_group_row("remaining_val_designs", val_remaining_df["net_name"], hurdle_df),
    ]

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_CSV, index=False)

    print_header("Leakage Facts")
    print(f"train: {len(train_df)} tiles, {train_df['design_name'].nunique()} designs, {train_df['net_name'].nunique()} nets")
    print(f"val:   {len(val_df)} tiles, {val_df['design_name'].nunique()} designs, {val_df['net_name'].nunique()} nets")
    print(f"Val-only designs: {sorted(val_only_designs) if val_only_designs else 'NONE'}")
    print(f"Train-only designs: {sorted(train_only_designs) if train_only_designs else 'NONE'}")
    print(f"Overlap designs ({len(overlap_designs)}/{len(val_designs) if val_designs else 0} val designs): {sorted(overlap_designs)}")
    print(f"Overlap net names: {len(overlap_nets)} / {len(val_nets)} val nets")

    print_header("Conclusions")
    print("Current val MAPE (33%) measures in-distribution performance only")
    print("True OOD performance unknown: nova_f3, tv80s_f3 have no SPEF")
    print("Recommended fix: hold out vga_enh_top_f3 and wb_conmax_top_f3 from training")

    print_header("Per-Group MAPE")
    print(out_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print_header("AL vs Non-AL Design Breakdown")
    breakdown = pd.DataFrame(
        [
            {
                "bucket": "AL_predefined_designs",
                "designs": ", ".join(sorted(val_al_designs)) if val_al_designs else "(none in val)",
                "n_designs": len(val_al_designs),
                "n_tiles": int(len(val_al_df)),
                "n_nets": int(val_al_df["net_name"].nunique()),
            },
            {
                "bucket": "non_AL_designs",
                "designs": ", ".join(sorted(val_non_al_designs)) if val_non_al_designs else "(none)",
                "n_designs": len(val_non_al_designs),
                "n_tiles": int(len(val_non_al_df)),
                "n_nets": int(val_non_al_df["net_name"].nunique()),
            },
            {
                "bucket": "recommended_holdout_designs",
                "designs": ", ".join(sorted(val_holdout_designs)) if val_holdout_designs else "(none in val)",
                "n_designs": len(val_holdout_designs),
                "n_tiles": int(len(val_holdout_df)),
                "n_nets": int(val_holdout_df["net_name"].nunique()),
            },
            {
                "bucket": "known_OOD_without_SPEF",
                "designs": ", ".join(KNOWN_NO_SPEF_OOD),
                "n_designs": len(KNOWN_NO_SPEF_OOD),
                "n_tiles": 0,
                "n_nets": 0,
            },
        ]
    )
    print(breakdown.to_string(index=False))

    print_header("Output")
    print(f"Wrote {len(out_df)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
