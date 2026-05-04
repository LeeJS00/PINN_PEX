#!/usr/bin/env python3
"""
02_rebuild_dataset_h3.py — Phase 0 H3 fix (context margin 2 → 6 μm rebuild).

Orchestrates per-design tile generation with `PEX_CONTEXT_MARGIN=5.0` env
var, then aggregates per-design maps into a v3 manifest using H1 hash split.

Cost: 2-4 GPU-day (CPU-bound, parallel workers per design)
Disk: ~1.2 TB written to /data/PINNPEX/data/processed_v3/intel22/

Idempotency: per-design directories that already contain pickled tiles +
a `<design>_map.csv` are skipped on re-run, so this script is resumable.

Boundary: relies on a 1-line env var read in legacy `scripts/build_dataset.py`
(see pex_v3/docs/CROSS_BOUNDARY_h3_context_margin.md). All v3 outputs go
to /data/PINNPEX/data/processed_v3/...; legacy data path is never touched.

Usage:
    # dry-run (default — prints plan, doesn't write)
    python3 pex_v3/scripts/02_rebuild_dataset_h3.py

    # actual rebuild (gated)
    python3 pex_v3/scripts/02_rebuild_dataset_h3.py --confirm

    # rebuild a single design (for testing / partial rerun)
    python3 pex_v3/scripts/02_rebuild_dataset_h3.py --confirm \
        --only intel22_gcd_f3

    # adjust worker count
    python3 pex_v3/scripts/02_rebuild_dataset_h3.py --confirm --num_workers 32
"""
from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "pex_v3"))

import pandas as pd  # noqa: E402

from configs import config_v3 as cfg  # noqa: E402
from src.data.manifest import (  # noqa: E402
    write_v3_manifest,
    manifest_summary,
    net_split_bucket,
)
from src.data.leak_check import run_all_checks  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(
        description="Phase 0 H3 dataset rebuild (14×14μm window)"
    )
    p.add_argument(
        "--confirm",
        action="store_true",
        help="Required to actually run. Without it, dry-run only.",
    )
    p.add_argument(
        "--only",
        type=str,
        default=None,
        help="Limit to a single design stem (e.g. intel22_gcd_f3). "
             "Useful for testing with one small design before full run.",
    )
    p.add_argument(
        "--max_designs",
        type=int,
        default=None,
        help="Limit to first N designs from TRAIN_DEFS+TEST_DEFS.",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=64,
        help="Workers passed to legacy build_dataset.py (default 64)",
    )
    p.add_argument(
        "--required_disk_gb",
        type=float,
        default=1400.0,
        help="Minimum free disk required at v3 path (default 1400 GB)",
    )
    return p.parse_args()


def check_disk(target_dir: Path, required_gb: float) -> bool:
    """Check that target's filesystem has at least `required_gb` GB free."""
    target_dir.mkdir(parents=True, exist_ok=True)
    stat = shutil.disk_usage(target_dir)
    free_gb = stat.free / (1024 ** 3)
    print(f">>> Disk free at {target_dir}: {free_gb:.1f} GB "
          f"(need ≥ {required_gb:.0f} GB)")
    return free_gb >= required_gb


def design_already_built(design_dir: Path, map_csv: Path) -> bool:
    """Idempotency check: design_dir has pickled tiles AND map csv exists."""
    if not map_csv.exists():
        return False
    # crude check: at least one .pkl.gz present in design_dir
    if not design_dir.exists():
        return False
    has_tiles = any(design_dir.glob("*.pkl.gz"))
    return has_tiles


def build_one_design(
    def_path: Path,
    out_dir: Path,
    pt_out_dir: Path,
    num_workers: int,
    context_margin: float,
) -> int:
    """Subprocess-launch legacy build_dataset.py for one design.

    Sets PEX_CONTEXT_MARGIN env var so legacy reads our v3 value at line 528.
    Returns subprocess exit code.
    """
    env = os.environ.copy()
    env["PEX_CONTEXT_MARGIN"] = str(context_margin)

    cmd = [
        "python3",
        str(_PROJECT_ROOT / "scripts" / "build_dataset.py"),
        "--def_path", str(def_path),
        "--out_dir", str(out_dir),
        "--pt_out_dir", str(pt_out_dir),
        "--num_workers", str(num_workers),
    ]

    print(f">>> [build] {def_path.stem}")
    print(f"    cmd: {' '.join(cmd)}")
    print(f"    env: PEX_CONTEXT_MARGIN={context_margin}")
    print(f"    cwd: {_PROJECT_ROOT}")
    t0 = time.time()
    result = subprocess.run(cmd, env=env, cwd=str(_PROJECT_ROOT))
    elapsed = time.time() - t0
    print(f"    [done] {def_path.stem} in {elapsed/60:.1f} min "
          f"(rc={result.returncode})")
    return result.returncode


def aggregate_v3_manifest(
    root_out_dir: Path,
    train_def_paths: list[Path],
    test_def_paths: list[Path],
    valid_ratio: float,
    hash_seed: str,
    schema_version: str,
) -> pd.DataFrame:
    """Aggregate per-design map CSVs into v3 manifest with H1 hash split.

    Each per-design map csv is at `{root_out_dir}/{design_stem}_map.csv`
    after build_dataset.py completes for that design.

    Returns the assembled v3 manifest DataFrame (caller writes it).
    """
    test_stems = {p.stem for p in test_def_paths}
    all_def_paths = list(train_def_paths) + list(test_def_paths)

    rows = []
    for def_path in all_def_paths:
        design = def_path.stem
        map_csv = root_out_dir / f"{design}_map.csv"
        if not map_csv.exists():
            print(f"⚠️  missing map csv for {design} at {map_csv}; skipping")
            continue
        df = pd.read_csv(map_csv)
        df["design_name"] = design
        df["rel_path"] = df["sample_filename"].apply(
            lambda s: f"{design}/{s}"
        )

        if design in test_stems:
            df["split"] = "test"
        else:
            df["split"] = df.apply(
                lambda r: net_split_bucket(
                    design_name=design,
                    net_name=str(r["net_name"]),
                    valid_ratio=valid_ratio,
                    hash_seed=hash_seed,
                ),
                axis=1,
            )
        rows.append(df)

    if not rows:
        raise RuntimeError(
            "No per-design maps found — every design build seems to have "
            "failed. Inspect subprocess output before re-running."
        )

    manifest = pd.concat(rows, ignore_index=True)
    manifest["schema_version"] = schema_version
    return manifest


def main():
    args = parse_args()

    target = cfg.PROCESSED_DIR_V3
    pt_target = cfg.PT_DIR_V3
    target.mkdir(parents=True, exist_ok=True)
    pt_target.mkdir(parents=True, exist_ok=True)

    # Resolve design list
    train_defs = list(cfg.TRAIN_DEFS)
    test_defs = list(cfg.TEST_DEFS)
    all_defs = train_defs + test_defs

    if args.only:
        all_defs = [p for p in all_defs if p.stem == args.only]
        if not all_defs:
            raise SystemExit(f"--only {args.only}: no matching design found")
    if args.max_designs is not None:
        all_defs = all_defs[: args.max_designs]

    print(f">>> Target dir:   {target}")
    print(f">>> PT dir:       {pt_target}")
    print(f">>> Context margin: {cfg.CONTEXT_MARGIN_V3} μm "
          f"(legacy 2.0 μm → H3 fix)")
    print(f">>> Stored window: "
          f"{cfg.WINDOW_SIZE[0] + 2 * cfg.CONTEXT_MARGIN_V3:.1f} × "
          f"{cfg.WINDOW_SIZE[1] + 2 * cfg.CONTEXT_MARGIN_V3:.1f} × "
          f"{cfg.WINDOW_SIZE[2]:.1f} μm")
    print(f">>> Designs:      {len(all_defs)} "
          f"({len(train_defs)} train + {len(test_defs)} test in plan)")
    for p in all_defs:
        marker = "🎯 train" if p.stem not in {q.stem for q in test_defs} else "🧪 test"
        print(f"    {marker}  {p.stem}  ←  {p}")

    # Disk check (relative to /data/ filesystem)
    if not check_disk(target.parent.parent, args.required_disk_gb):
        print("❌ Insufficient disk free; aborting.")
        sys.exit(1)

    if not args.confirm:
        print()
        print("⚠️  Dry-run only. Pass --confirm to actually rebuild.")
        print(f"   Estimated cost: 2-4 GPU-day total, ~1.2 TB disk write.")
        print(f"   Per-design: ~10-30 min depending on design size.")
        return

    # ---- Per-design build ---------------------------------------------
    summary = {"started": time.time(), "designs": []}
    failed = []
    skipped = []
    built = []
    for def_path in all_defs:
        design = def_path.stem
        design_dir = target / design
        pt_design_dir = pt_target / design
        map_csv = target / f"{design}_map.csv"

        if design_already_built(design_dir, map_csv):
            print(f">>> [skip] {design}: already built (idempotent)")
            skipped.append(design)
            continue

        if not def_path.exists():
            print(f"⚠️  DEF missing: {def_path}; skipping")
            failed.append((design, "def_missing"))
            continue

        rc = build_one_design(
            def_path=def_path,
            out_dir=design_dir,
            pt_out_dir=pt_design_dir,
            num_workers=args.num_workers,
            context_margin=cfg.CONTEXT_MARGIN_V3,
        )
        if rc != 0:
            print(f"❌ build_dataset.py failed for {design} (rc={rc})")
            failed.append((design, f"rc={rc}"))
        else:
            built.append(design)

    # ---- Aggregate v3 manifest ----------------------------------------
    print()
    print(">>> Aggregating per-design maps into v3 manifest ...")
    manifest = aggregate_v3_manifest(
        root_out_dir=target,
        train_def_paths=train_defs,
        test_def_paths=test_defs,
        valid_ratio=cfg.VALID_RATIO_V3,
        hash_seed=cfg.H1_HASH_SEED,
        schema_version=cfg.SCHEMA_VERSION,
    )

    print(">>> Validating H1 invariants on v3 manifest ...")
    test_stems = {p.stem for p in test_defs}
    run_all_checks(manifest, test_stems, expected_schema=cfg.SCHEMA_VERSION)
    print("    ✅ all H1 invariants pass")

    print(f">>> Writing v3 manifest: {cfg.MANIFEST_PATH_V3}")
    write_v3_manifest(manifest, cfg.MANIFEST_PATH_V3)
    print(f"    ✅ wrote {len(manifest):,} rows")

    # ---- Final summary -------------------------------------------------
    summary_data = manifest_summary(manifest)
    summary_data["context_margin_um"] = cfg.CONTEXT_MARGIN_V3
    summary_data["stored_window_xy_um"] = (
        cfg.WINDOW_SIZE[0] + 2 * cfg.CONTEXT_MARGIN_V3
    )
    summary_data["v3_manifest_path"] = str(cfg.MANIFEST_PATH_V3)
    summary_data["built_designs"] = built
    summary_data["skipped_designs"] = skipped
    summary_data["failed_designs"] = failed

    out_dir = _PROJECT_ROOT / "pex_v3" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "02_rebuild_dataset_h3_summary.json"
    import json
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f">>> Summary: {summary_path}")

    print(f">>> Distribution:")
    print(f"    train: {summary_data['nets_by_split'].get('train', 0):>7} nets, "
          f"{summary_data['tiles_by_split'].get('train', 0):>10,} tiles")
    print(f"    valid: {summary_data['nets_by_split'].get('valid', 0):>7} nets, "
          f"{summary_data['tiles_by_split'].get('valid', 0):>10,} tiles")
    print(f"    test:  {summary_data['nets_by_split'].get('test', 0):>7} nets, "
          f"{summary_data['tiles_by_split'].get('test', 0):>10,} tiles")

    if failed:
        print()
        print("⚠️  Some designs failed:")
        for d, reason in failed:
            print(f"    {d}: {reason}")
        sys.exit(1)
    print("✅ 02_rebuild_dataset_h3.py complete.")


if __name__ == "__main__":
    main()
