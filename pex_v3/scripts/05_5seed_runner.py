#!/usr/bin/env python3
"""
05_5seed_runner.py — model-agnostic 5-seed orchestrator.

Drives any "trainable + evaluable" baseline through 5 seeds, writes per-seed
provenance + metrics, then aggregates into per-method, MWU, bootstrap CI
tables suitable for the paper.

Per `benchmarking-statistician.md`: this is the ONLY way an improvement claim
makes it into PHASE_STATUS.md or the paper. n=1 results are rejected.

Usage:
    python3 pex_v3/scripts/05_5seed_runner.py \\
        --method-spec pex_v3/src/baselines/xgboost_baseline.py:run_one_seed \\
        --output-dir pex_v3/output/baselines/B1_xgboost \\
        --seeds 0 1 2 3 4 \\
        --train-manifest /data/PINNPEX/data/processed_v3/intel22/dataset_manifest_v3.csv \\
        --golden-spef-dir /home/jslee/projects/PEX_SSL/golden_data/spef_data/intel22/

The `--method-spec` is `path:function_name`. The named function must have
the signature:

    def run_one_seed(
        seed: int,
        train_manifest_path: Path,
        golden_spef_dir: Path,
        output_dir: Path,           # per-seed output dir
        config_snapshot: dict,
    ) -> dict:                       # returns the metrics_row dict (will be written as CSV)

The function is responsible for: setting seeds via `set_all_seeds`, training
+ evaluating, writing `metrics_row.csv` into `output_dir`. The orchestrator
takes care of provenance logging + aggregation.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Callable

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "pex_v3"))

import pandas as pd  # noqa: E402

from configs import config_v3 as cfg  # noqa: E402
from src.utils.seeds import set_all_seeds  # noqa: E402
from src.utils.manifest_hash import write_provenance  # noqa: E402
from src.evaluation.seed_aggregator import (  # noqa: E402
    collect_per_run_csvs,
    write_aggregation,
)


def parse_args():
    p = argparse.ArgumentParser(description="5-seed orchestrator")
    p.add_argument(
        "--method-spec",
        required=True,
        type=str,
        help="path:function_name — e.g. pex_v3/src/baselines/xgboost_baseline.py:run_one_seed",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Root output directory; will contain seed{N}/ subdirs + aggregates",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="List of integer seeds (default: 0 1 2 3 4)",
    )
    p.add_argument(
        "--train-manifest",
        type=Path,
        default=cfg.MANIFEST_PATH_V3,
        help="v3 manifest path",
    )
    p.add_argument(
        "--golden-spef-dir",
        type=Path,
        default=cfg.SPEF_DIR,
        help="Golden SPEF directory",
    )
    p.add_argument(
        "--metric-col",
        type=str,
        default="cap_mape_median",
        help="Which column to aggregate / MWU / bootstrap (default: cap_mape_median)",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip seeds whose metrics_row.csv already exists",
    )
    return p.parse_args()


def load_method(spec: str) -> Callable:
    """Resolve `path:function_name` to a callable."""
    if ":" not in spec:
        raise SystemExit(f"--method-spec must be path:function_name, got {spec!r}")
    path_str, fn_name = spec.split(":", 1)
    path = Path(path_str).resolve()
    if not path.exists():
        raise SystemExit(f"method file not found: {path}")
    mod_name = path.stem
    spec_obj = importlib.util.spec_from_file_location(mod_name, str(path))
    mod = importlib.util.module_from_spec(spec_obj)
    spec_obj.loader.exec_module(mod)
    if not hasattr(mod, fn_name):
        raise SystemExit(f"{path} has no function {fn_name!r}")
    return getattr(mod, fn_name)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f">>> Method:    {args.method_spec}")
    print(f">>> Seeds:     {args.seeds}")
    print(f">>> Output:    {args.output_dir}")
    print(f">>> Manifest:  {args.train_manifest}")

    method_fn = load_method(args.method_spec)

    config_snapshot = cfg.v3_snapshot()
    config_snapshot["method_spec"] = args.method_spec

    # ---- per-seed loop -------------------------------------------------
    per_run_rows = []
    for seed in args.seeds:
        seed_dir = args.output_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        metrics_csv = seed_dir / "metrics_row.csv"

        if args.skip_existing and metrics_csv.exists():
            print(f">>> seed {seed} — already run, skipping (--skip-existing)")
            df = pd.read_csv(metrics_csv)
            per_run_rows.append(df.iloc[0].to_dict())
            continue

        print(f">>> seed {seed} — starting")
        set_all_seeds(seed, deterministic=True)
        write_provenance(
            run_dir=seed_dir,
            manifest_path=args.train_manifest,
            config_snapshot=config_snapshot,
            seed=seed,
            project_root=_PROJECT_ROOT,
        )

        t0 = time.time()
        result = method_fn(
            seed=seed,
            train_manifest_path=args.train_manifest,
            golden_spef_dir=args.golden_spef_dir,
            output_dir=seed_dir,
            config_snapshot=config_snapshot,
        )
        elapsed = time.time() - t0

        if result is None:
            raise RuntimeError(f"method returned None for seed {seed}")
        # Normalize into a dict (may be a MetricsRow dataclass)
        if hasattr(result, "__dict__"):
            row = vars(result)
        elif isinstance(result, dict):
            row = dict(result)
        else:
            raise RuntimeError(
                f"method must return dict or dataclass, got {type(result)}"
            )
        row.setdefault("seed", seed)
        row.setdefault("method", args.method_spec.split(":")[1])
        row["__seed_dir"] = str(seed_dir)
        row["__elapsed_sec"] = float(elapsed)

        # Write the row to seed_dir/metrics_row.csv
        pd.DataFrame([row]).to_csv(metrics_csv, index=False)
        per_run_rows.append(row)
        print(f"    seed {seed} done in {elapsed:.1f} s")

    # ---- aggregation ---------------------------------------------------
    if not per_run_rows:
        print("⚠️  no seeds ran. Nothing to aggregate.")
        return

    per_run_df = pd.DataFrame(per_run_rows)
    print(">>> Aggregating ...")
    paths = write_aggregation(per_run_df, args.output_dir, metric_col=args.metric_col)

    print(f">>> Wrote:")
    for k, p in paths.items():
        print(f"    {k}: {p}")

    print("✅ 05_5seed_runner.py complete.")


if __name__ == "__main__":
    main()
