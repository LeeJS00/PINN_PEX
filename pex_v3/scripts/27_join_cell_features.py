#!/usr/bin/env python3
"""
27_join_cell_features.py — Join sister cell-internal features into v3 features.

Sister (`r_analytic_v3/cache/feat_v6_*.parquet`) has per-net cell-internal
features for all 11 designs:
    n_pins_obs_matched
    v6_cell_size_w_sum, v6_cell_size_h_sum, v6_cell_area_sum
    v6_n_pins_signal, v6_n_pins_input, v6_n_pins_output
    v6_obs_signal_nsq_M1/M2, v6_obs_signal_area_M1/M2
    v6_obs_n_via_v0_pin, v6_obs_n_via_v1_pin

These capture cell-internal substrate cap contribution (the missing
~15-30 squares/net that DEF/LEF wire features can't see). Hypothesis:
adding these to self_features will break the C_gnd 19-22% ceiling.

Output: extended features CSV with ~13 new cols, drop-in replacement
for v3 features.csv.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--features-csv", type=Path,
        default=Path("/data/PINNPEX/data/processed_v3/intel22/features/all_designs.csv"),
    )
    p.add_argument(
        "--cell-feature-dir", type=Path,
        default=Path("/home/jslee/projects/PINNPEX/experiments/cross_design_tv80s_2026_05_02/r_analytic_v3/cache"),
    )
    p.add_argument(
        "--out-csv", type=Path,
        default=Path("/data/PINNPEX/data/processed_v3/intel22/features/all_designs_with_cell.csv"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f">>> Loading v3 features: {args.features_csv}")
    base = pd.read_csv(args.features_csv)
    print(f"  shape: {base.shape}")
    print(f"  designs: {base.design_name.nunique()}")

    # Load all sister feat_v6_* parquets
    print(f">>> Loading sister feat_v6 parquets from {args.cell_feature_dir}")
    cell_dfs = []
    for parquet_path in sorted(args.cell_feature_dir.glob("feat_v6_intel22_*.parquet")):
        design = parquet_path.stem.replace("feat_v6_", "")
        df = pd.read_parquet(parquet_path)
        df["design_name"] = design
        cell_dfs.append(df)
        print(f"  {design}: {len(df):,} nets, {len(df.columns)-2} cell features")
    cell_all = pd.concat(cell_dfs, ignore_index=True)
    print(f">>> total cell features: {cell_all.shape}")

    # Join (left join to keep all v3 nets)
    merged = base.merge(
        cell_all,
        on=["design_name", "net_name"],
        how="left",
    )
    print(f">>> after merge: {merged.shape}")

    n_matched = merged["v6_cell_area_sum"].notna().sum() if "v6_cell_area_sum" in merged.columns else 0
    n_unmatched = len(merged) - n_matched
    print(f"  matched cell features: {n_matched:,} / {len(merged):,}")
    print(f"  unmatched (filled with 0): {n_unmatched:,}")

    # Fill NaN cell features with 0 (interpretation: net has no cell pins -- typical for floating nets)
    cell_cols = [c for c in cell_all.columns if c not in ("design_name", "net_name")]
    merged[cell_cols] = merged[cell_cols].fillna(0.0)

    # Write
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)
    print(f">>> wrote {args.out_csv}: {len(merged):,} rows × {len(merged.columns)} cols")

    print()
    print(">>> Cell features added:")
    for c in cell_cols:
        print(f"    {c}")


if __name__ == "__main__":
    main()
