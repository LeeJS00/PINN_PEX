#!/usr/bin/env python3
"""
40_fast_autonomous_spef.py — produce an "autonomous-fast" SPEF without PINN inference.

This is the entrypoint for Option D' (2026-05-03 Codex deliberation):
deterministic DEF/LEF-driven SPEF generator that bypasses the slow legacy
DeepPEX (1M params, 14.4 min for tv80s) and produces a SPEF whose per-net
totals are then rescaled by the existing XGB anchor + sister-R post-process
scripts (`16_xgb_calibrate_spef.py`, `23_r_per_net_calibrate_spef.py`).

Usage
-----
    python3 pex_v3/scripts/40_fast_autonomous_spef.py \
        --design intel22_tv80s_f3 \
        --out-dir output_intel22/active_learning/m6_v10b_baseline_seed0/

The output `intel22_tv80s_f3_autonomous_fast.spef` is a drop-in replacement
for `intel22_tv80s_f3_autonomous.spef` and feeds the same downstream pipeline.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Resolve `src.preprocessing.*` to legacy populated package by putting project
# root first; `src.v3` is a sub-package within the same legacy `src/`.
sys.path.insert(0, str(_PROJECT_ROOT))

from configs import config_v3 as cfg  # noqa: E402
from src.preprocessing.layer_parser import LayerInfoParser  # noqa: E402
from src.preprocessing.lef_parser import LefParser  # noqa: E402
from src.v3.utils.fast_spef_engine import write_fast_autonomous_spef  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Fast autonomous SPEF generator")
    p.add_argument("--design", required=True, help="design name e.g. intel22_tv80s_f3")
    p.add_argument(
        "--processed-dir",
        type=Path,
        default=Path(cfg.PROCESSED_DIR_V3) if hasattr(cfg, "PROCESSED_DIR_V3") else None,
        help="root containing <design>/topology/*.pkl.gz",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/home/jslee/projects/PINNPEX/pex_v3/output/spef_fast"),
    )
    p.add_argument("--max-dist-um", type=float, default=5.0)
    p.add_argument("--top-k-aggressors", type=int, default=20)
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel processes for topology index pass (default 1)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.processed_dir is None:
        raise SystemExit("Could not infer processed_dir from config_v3; pass --processed-dir")

    topo_dir = args.processed_dir / args.design / "topology"
    if not topo_dir.exists():
        raise SystemExit(f"Missing topology dir: {topo_dir}")

    layer_info = LayerInfoParser(cfg.LAYERS_INFO_PATH).parse()
    tech_lef = LefParser(cfg.TECH_LEF_PATH).parse()

    out_path = args.out_dir / f"{args.design}_autonomous_fast.spef"
    print(f">>> Fast autonomous SPEF for {args.design}")
    print(f"    topology dir: {topo_dir}")
    print(f"    output:       {out_path}")

    t_total = time.perf_counter()
    stats = write_fast_autonomous_spef(
        design_name=args.design,
        topology_dir=topo_dir,
        layer_info=layer_info,
        tech_lef=tech_lef,
        out_spef_path=out_path,
        max_dist_um=args.max_dist_um,
        top_k=args.top_k_aggressors,
        n_workers=args.workers,
    )
    stats["wall_clock_s"] = time.perf_counter() - t_total

    # Persist a runtime artefact next to the SPEF.
    runtime_path = out_path.with_suffix(".runtime.json")
    runtime_path.write_text(json.dumps(stats, indent=2))
    print(f"\n>>> Done.")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"    {k:24s} {v:8.2f}")
        else:
            print(f"    {k:24s} {v}")
    print(f"\n>>> Runtime stats → {runtime_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
