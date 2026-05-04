#!/usr/bin/env python3
"""run_one_seed.py — single-seed driver for exp_007 Sakurai-Tamaru gnd allocator.

Identical pipeline structure to exp_006/run_one_seed.py:
    1. Build SPEF using the Sakurai-Tamaru engine.
    2. XGB calibrate (per-net rescale) using B1 seed-N CSV.
    3. Sister-R per-net rescale.
    4. Compare vs golden, persist per-channel + per-net stats.
"""
from __future__ import annotations

import argparse
import json
import sys
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[4]
sys.path.insert(0, str(_ROOT))

from configs import config_v3 as cfg  # noqa: E402
from src.preprocessing.layer_parser import LayerInfoParser  # noqa: E402
from src.preprocessing.lef_parser import LefParser  # noqa: E402

sys.path.insert(0, str(_THIS.parent))
from engine import write_sakurai_spef_parallel  # noqa: E402


DESIGN = "intel22_tv80s_f3"
TOPO_DIR = Path("/data/PINNPEX/data/processed_v3/intel22") / DESIGN / "topology"
GOLDEN = Path("/home/jslee/projects/PEX_SSL/golden_data/spef_data/intel22") / f"{DESIGN}_starrc.spef"
R_PARQUET = Path("/home/jslee/projects/PINNPEX/experiments/cross_design_tv80s_2026_05_02/r_analytic_v3/outputs/test_predictions_v6_s3.parquet")
R_PRED_COL = "R_pred_v6_s3"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--max-dist-um", type=float, default=5.0)
    p.add_argument("--top-k-aggressors", type=int, default=20)
    p.add_argument("--skip-engine", action="store_true")
    return p.parse_args()


def _xgb_csv_for_seed(seed: int) -> Path:
    return Path(f"/home/jslee/projects/PINNPEX/pex_v3/output/baselines/B1_xgboost_real/seed{seed}/eval_predictions_test.csv")


def _run_xgb_calibrate(in_spef: Path, xgb_csv: Path, out_spef: Path) -> None:
    cmd = [
        "python3",
        str(_ROOT / "pex_v3/scripts/16_xgb_calibrate_spef.py"),
        "--in-spef", str(in_spef),
        "--xgb-csv", str(xgb_csv),
        "--design", DESIGN,
        "--out-spef", str(out_spef),
    ]
    subprocess.run(cmd, check=True)


def _run_sister_r(in_spef: Path, out_spef: Path) -> None:
    cmd = [
        "python3",
        str(_ROOT / "pex_v3/scripts/23_r_per_net_calibrate_spef.py"),
        "--in-spef", str(in_spef),
        "--out-spef", str(out_spef),
        "--r-pred-parquet", str(R_PARQUET),
        "--r-pred-col", R_PRED_COL,
    ]
    subprocess.run(cmd, check=True)


def _run_compare(pred: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python3",
        str(_ROOT / "src/evaluation/compare_spef.py"),
        "--golden", str(GOLDEN),
        "--pred", str(pred),
        "--out_dir", str(out_dir),
    ]
    subprocess.run(cmd, check=True)


def _per_channel_metrics(compare_csv: Path, xgb_csv: Path) -> dict:
    df = pd.read_csv(compare_csv)
    xgb = pd.read_csv(xgb_csv)
    matched_nets = set(xgb[xgb["design_name"] == DESIGN]["net_name"].astype(str))

    df = df.copy()
    df["net"] = df["net"].astype(str)
    df["mape_tot"] = (df["p_tot"] - df["g_tot"]).abs() / df["g_tot"].clip(lower=1e-9) * 100.0
    df["mape_gnd"] = (df["p_gnd"] - df["g_gnd"]).abs() / df["g_gnd"].clip(lower=1e-9) * 100.0
    df["mape_cpl"] = (df["p_cpl"] - df["g_cpl"]).abs() / df["g_cpl"].clip(lower=1e-9) * 100.0

    df["matched"] = df["net"].isin(matched_nets)
    matched_df = df[df["matched"]]
    unmatched_df = df[~df["matched"]]

    g = df["g_tot"].to_numpy()
    p = df["p_tot"].to_numpy()
    ss_res = float(((p - g) ** 2).sum())
    ss_tot = float(((g - g.mean()) ** 2).sum())
    r2_c = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "n_nets": int(len(df)),
        "n_matched": int(matched_df.shape[0]),
        "n_unmatched": int(unmatched_df.shape[0]),
        "tot_mape_mean": float(df["mape_tot"].mean()),
        "tot_mape_median": float(df["mape_tot"].median()),
        "tot_mape_p95": float(np.percentile(df["mape_tot"], 95)),
        "gnd_mape_matched_mean": float(matched_df["mape_gnd"].mean()),
        "gnd_mape_matched_median": float(matched_df["mape_gnd"].median()),
        "gnd_mape_unmatched_mean": float(unmatched_df["mape_gnd"].mean()) if len(unmatched_df) > 0 else float("nan"),
        "cpl_mape_matched_mean": float(matched_df["mape_cpl"].mean()),
        "cpl_mape_matched_median": float(matched_df["mape_cpl"].median()),
        "cpl_mape_unmatched_mean": float(unmatched_df["mape_cpl"].mean()) if len(unmatched_df) > 0 else float("nan"),
        "r_squared_c": r2_c,
    }


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seed = args.seed

    auto_spef = args.out_dir / f"seed{seed}_sakurai.spef"
    xgb_spef = args.out_dir / f"seed{seed}_xgb.spef"
    hero_spef = args.out_dir / f"seed{seed}_HERO.spef"
    runtime_json = args.out_dir / f"seed{seed}_runtime.json"
    metrics_json = args.out_dir / f"seed{seed}_metrics.json"
    compare_dir = args.out_dir / f"seed{seed}_compare"

    if not args.skip_engine:
        layer_info = LayerInfoParser(cfg.LAYERS_INFO_PATH).parse()
        tech_lef = LefParser(cfg.TECH_LEF_PATH).parse()

        t_total = time.perf_counter()
        stats = write_sakurai_spef_parallel(
            design_name=DESIGN,
            topology_dir=TOPO_DIR,
            layer_info=layer_info,
            tech_lef=tech_lef,
            out_spef_path=auto_spef,
            max_dist_um=args.max_dist_um,
            top_k=args.top_k_aggressors,
            n_workers_pass2=args.workers,
        )
        stats["wall_clock_s"] = time.perf_counter() - t_total
        runtime_json.write_text(json.dumps(stats, indent=2))
        print(f"\n>>> seed{seed} engine stats:")
        for k, v in stats.items():
            print(f"    {k:24s} {v}")

        # Hard-kill check: 100s wall-clock cap
        if stats["wall_clock_s"] > 100.0:
            print(f"\n[HARD-KILL] seed{seed} wall_clock={stats['wall_clock_s']:.2f}s > 100s",
                  flush=True)
            return 2

    xgb_csv = _xgb_csv_for_seed(seed)
    print(f"\n>>> seed{seed} XGB calibrate ({xgb_csv.name})")
    _run_xgb_calibrate(auto_spef, xgb_csv, xgb_spef)

    print(f"\n>>> seed{seed} sister-R per-net rescale")
    _run_sister_r(xgb_spef, hero_spef)

    print(f"\n>>> seed{seed} compare vs golden")
    _run_compare(hero_spef, compare_dir)

    compare_csv = compare_dir / "spef_comparison_report.csv"
    metrics = _per_channel_metrics(compare_csv, xgb_csv)
    metrics["seed"] = seed
    metrics_json.write_text(json.dumps(metrics, indent=2))
    print(f"\n>>> seed{seed} metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k:24s} {v:8.4f}")
        else:
            print(f"    {k:24s} {v}")

    # Hard-kill check: gnd matched > 35%
    if metrics["gnd_mape_matched_mean"] > 35.0:
        print(f"\n[HARD-KILL] seed{seed} gnd_matched_mean={metrics['gnd_mape_matched_mean']:.2f}% > 35%",
              flush=True)
        return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())
