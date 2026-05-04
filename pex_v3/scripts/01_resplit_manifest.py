#!/usr/bin/env python3
"""
01_resplit_manifest.py — Phase 0 H1 fix.

Reads the legacy manifest (read-only), recomputes the `split` column using
the deterministic (design_name, net_name) hash, writes a v3 manifest at
`cfg.MANIFEST_PATH_V3`. The legacy manifest is NEVER overwritten.

Outputs:
    /data/PINNPEX/data/processed_v3/intel22/dataset_manifest_v3.csv
    pex_v3/output/01_resplit_manifest_summary.json

Validates:
    - All H1 invariants pass (no net mixing across splits, test designs pure).
    - Schema version stamped.

Cost: ~10 min (no data files touched, manifest CSV only).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "pex_v3"))

import pandas as pd  # noqa: E402

from configs import config_v3 as cfg  # noqa: E402
from src.data.manifest import (  # noqa: E402
    build_v3_manifest,
    write_v3_manifest,
    manifest_summary,
)
from src.data.leak_check import run_all_checks  # noqa: E402


def main():
    legacy_path = cfg.LEGACY_MANIFEST_PATH
    v3_path = cfg.MANIFEST_PATH_V3
    test_stems = {p.stem for p in cfg.TEST_DEFS}

    print(f">>> Reading legacy manifest: {legacy_path}")
    if not legacy_path.exists():
        raise SystemExit(f"Legacy manifest not found at {legacy_path}")

    print(f">>> H1 hash seed: {cfg.H1_HASH_SEED}")
    print(f">>> Valid ratio:  {cfg.VALID_RATIO_V3}")
    print(f">>> Test designs: {sorted(test_stems)}")

    df_v3 = build_v3_manifest(
        legacy_manifest_path=legacy_path,
        test_design_stems=test_stems,
        valid_ratio=cfg.VALID_RATIO_V3,
        hash_seed=cfg.H1_HASH_SEED,
        schema_version=cfg.SCHEMA_VERSION,
    )

    print(">>> Validating H1 invariants ...")
    run_all_checks(df_v3, test_stems, expected_schema=cfg.SCHEMA_VERSION)
    print("    ✅ all invariants pass")

    print(f">>> Writing v3 manifest: {v3_path}")
    write_v3_manifest(df_v3, v3_path)
    print(f"    ✅ wrote {len(df_v3):,} rows")

    # ---- Summary report ------------------------------------------------
    summary = manifest_summary(df_v3)
    summary["legacy_manifest_path"] = str(legacy_path)
    summary["v3_manifest_path"] = str(v3_path)
    summary["h1_hash_seed"] = cfg.H1_HASH_SEED
    summary["valid_ratio_v3"] = cfg.VALID_RATIO_V3
    summary["schema_version"] = cfg.SCHEMA_VERSION

    out_dir = _PROJECT_ROOT / "pex_v3" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "01_resplit_manifest_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f">>> Summary: {summary_path}")

    print(">>> Distribution:")
    print(f"    train:  {summary['nets_by_split'].get('train', 0):>7} nets, "
          f"{summary['tiles_by_split'].get('train', 0):>9,} tiles")
    print(f"    valid:  {summary['nets_by_split'].get('valid', 0):>7} nets, "
          f"{summary['tiles_by_split'].get('valid', 0):>9,} tiles")
    print(f"    test:   {summary['nets_by_split'].get('test', 0):>7} nets, "
          f"{summary['tiles_by_split'].get('test', 0):>9,} tiles")

    # ---- Compare to legacy: how many nets shifted? ---------------------
    legacy = pd.read_csv(legacy_path)
    if "split" in legacy.columns:
        legacy_split = (
            legacy.groupby(["design_name", "net_name"])["split"]
            .agg(lambda s: ",".join(sorted(set(s))))
            .reset_index(name="legacy_split")
        )
        v3_split = (
            df_v3.groupby(["design_name", "net_name"])["split"]
            .first()
            .reset_index(name="v3_split")
        )
        merged = pd.merge(
            legacy_split, v3_split, on=["design_name", "net_name"], how="inner"
        )
        legacy_mixed = (legacy_split["legacy_split"].str.contains(",")).sum()
        shifted = (merged["legacy_split"] != merged["v3_split"]).sum()
        print(f">>> Comparison vs legacy:")
        print(f"    legacy nets that span multiple splits: {legacy_mixed:,}  "
              f"(this was the H1 leak)")
        print(f"    nets reassigned by H1 hash:            {shifted:,}")

    print("✅ 01_resplit_manifest.py complete.")


if __name__ == "__main__":
    main()
