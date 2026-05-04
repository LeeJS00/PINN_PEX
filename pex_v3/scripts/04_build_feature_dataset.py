#!/usr/bin/env python3
"""
04_build_feature_dataset.py — orchestrate per-design feature extraction.

For each design in TRAIN_DEFS + TEST_DEFS, scans DEF + golden SPEF, computes
NetFeatureVector per net, joins with v3 manifest split + targets, writes
parquet at:

    cfg.PROCESSED_DIR_V3 / features / <design>.parquet

After all designs, concatenates into:

    cfg.PROCESSED_DIR_V3 / features / all_designs.parquet

Cost: ~5-10 min per design × 11 designs ≈ 60 min total.
Idempotent: skips designs whose parquet already exists.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "pex_v3"))

import pandas as pd  # noqa: E402

from configs import config_v3 as cfg  # noqa: E402
from src.baselines.feature_dataset import write_feature_dataset_for_design  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Build feature dataset for B1/B4")
    p.add_argument("--only", type=str, default=None,
                   help="Only build for this design stem (e.g. intel22_gcd_f3)")
    p.add_argument("--max-designs", type=int, default=None)
    p.add_argument("--cutoff-um", type=float, default=4.0)
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--ext", choices=["parquet", "csv"], default="csv",
                   help="Output extension; csv has no parquet dep")
    return p.parse_args()


def main():
    args = parse_args()
    manifest = pd.read_csv(cfg.MANIFEST_PATH_V3)

    feature_dir = cfg.PROCESSED_DIR_V3 / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    train_defs = list(cfg.TRAIN_DEFS)
    test_defs = list(cfg.TEST_DEFS)
    all_defs = train_defs + test_defs
    if args.only:
        all_defs = [p for p in all_defs if p.stem == args.only]
    if args.max_designs is not None:
        all_defs = all_defs[: args.max_designs]

    print(f">>> Building features for {len(all_defs)} designs")

    written = []
    for def_path in all_defs:
        design = def_path.stem
        out_path = feature_dir / f"{design}.{args.ext}"
        if args.skip_existing and out_path.exists():
            print(f"⏭️  {design}: already built, skipping")
            written.append(out_path)
            continue

        spef_path = cfg.SPEF_DIR / f"{design}_starrc.spef"
        if not spef_path.exists():
            print(f"⚠️  SPEF missing for {design}; skipping")
            continue
        if not def_path.exists():
            print(f"⚠️  DEF missing for {design}; skipping")
            continue

        manifest_subset = manifest[manifest["design_name"] == design]
        if len(manifest_subset) == 0:
            print(f"⚠️  manifest empty for {design}; skipping")
            continue

        print(f">>> {design}")
        t0 = time.time()
        n_rows = write_feature_dataset_for_design(
            def_path=def_path,
            spef_path=spef_path,
            manifest_subset=manifest_subset,
            out_path=out_path,
            cutoff_um=args.cutoff_um,
        )
        elapsed = time.time() - t0
        print(f"    ✅ {n_rows:,} rows  in  {elapsed:.1f}s")
        written.append(out_path)

    # Concatenate all into a single file
    if len(written) > 1:
        print(f">>> Concatenating into all_designs.{args.ext} ...")
        dfs = [
            pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
            for p in written
        ]
        big = pd.concat(dfs, ignore_index=True)
        all_path = feature_dir / f"all_designs.{args.ext}"
        if args.ext == "parquet":
            big.to_parquet(all_path, index=False)
        else:
            big.to_csv(all_path, index=False)
        print(f"    ✅ wrote {len(big):,} rows to {all_path}")

    print("✅ 04_build_feature_dataset.py complete.")


if __name__ == "__main__":
    main()
