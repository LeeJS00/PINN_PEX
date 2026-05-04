#!/usr/bin/env python3
"""31_join_pincap_features.py — join Liberty pin caps into v3 features."""
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
        "--pincap-parquet", type=Path,
        default=Path("/data/PINNPEX/data/processed_v3/intel22/pin_caps_per_net/pin_caps_all_designs.parquet"),
    )
    p.add_argument(
        "--out-csv", type=Path,
        default=Path("/data/PINNPEX/data/processed_v3/intel22/features/all_designs_with_pincap.csv"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = pd.read_csv(args.features_csv)
    pincap = pd.read_parquet(args.pincap_parquet)
    print(f">>> base: {base.shape}  pincap: {pincap.shape}")
    merged = base.merge(pincap, on=["design_name", "net_name"], how="left")
    pincap_cols = [c for c in pincap.columns if c not in ("design_name", "net_name")]
    n_matched = merged[pincap_cols[0]].notna().sum()
    merged[pincap_cols] = merged[pincap_cols].fillna(0.0)
    print(f">>> merged: {merged.shape}  matched: {n_matched:,} / {len(merged):,}")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)
    print(f">>> wrote {args.out_csv}")
    print()
    print("Pin cap features added:")
    for c in pincap_cols:
        print(f"  {c}: median {merged[c].median():.3f}, P95 {merged[c].quantile(0.95):.3f}")


if __name__ == "__main__":
    main()
