"""Per-net feature dump for patch validation.

Runs the canonical `_v3_per_net` and `_v4_process_net` on a configurable
selection of nets, wall-times each call, and emits:
  - per-net runtime (V3, V4)
  - per-net full 41-D V3 feature dict + 26-D V4 feature dict

Used BEFORE Round 1 patches (`--label baseline`) and AFTER (`--label patched`)
so `compare_features.py` can produce the per-feature runtime + MAE + R² table.

Usage:
    # baseline, top-20 worst nets + 100 random sample on nova
    python3 TreePEX/scripts/dump_features.py --design intel22_nova_f3 \\
        --select top:20,sample:100 --label baseline

    # after patches, same selection (re-seed sample with --sample-seed to match)
    python3 TreePEX/scripts/dump_features.py --design intel22_nova_f3 \\
        --select top:20,sample:100 --label patched

See TreePEX/FEATURE_SPEEDUP_PLAN.md §6 for the required acceptance report.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "TreePEX" / "scripts"))

from pex_cold import (
    DESIGNS,
    TECH_LEF_PATH,
    CELL_LEF_PATH,
    LAYERS_INFO_PATH,
    TILE_CACHE_ROOT,
    SpatialGrid,
    N_LAYERS_EPS,
    scan_design,
    init_worker_v3,
    _layer_eps_array,
    _v3_per_net,
    _v4_process_net,
    _bbox_xy,
)
from src.preprocessing.lef_parser import LefParser
from src.preprocessing.cell_parser import CellLibParser
from src.preprocessing.layer_parser import LayerInfoParser


def parse_select(spec: str, all_targets: List[str], geo, seed: int) -> List[str]:
    """spec = 'top:K,sample:N,list:nm1+nm2' — comma-separated selectors."""
    chosen: List[str] = []
    seen = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        kind, _, val = part.partition(":")
        if kind == "all":
            for n in all_targets:
                if n not in seen:
                    chosen.append(n); seen.add(n)
        elif kind == "top":
            k = int(val)
            ranked = sorted(all_targets, key=lambda n: -len(geo["nets"].get(n, [])))
            for n in ranked[:k]:
                if n not in seen:
                    chosen.append(n); seen.add(n)
        elif kind == "sample":
            n_s = int(val)
            rng = np.random.RandomState(seed)
            pool = [n for n in all_targets if n not in seen]
            if n_s >= len(pool):
                pick = pool
            else:
                pick = list(rng.choice(pool, size=n_s, replace=False))
            for n in pick:
                if n not in seen:
                    chosen.append(n); seen.add(n)
        elif kind == "list":
            for n in val.split("+"):
                n = n.strip()
                if n and n not in seen and n in geo["target_set"]:
                    chosen.append(n); seen.add(n)
        else:
            raise SystemExit(f"unknown selector {kind!r}")
    return chosen


def build_tile_map(design: str, target_nets: set) -> Dict[str, List[Path]]:
    tile_dir = TILE_CACHE_ROOT / design
    map_csv = TILE_CACHE_ROOT / f"{design}_map.csv"
    if not tile_dir.exists() or not map_csv.exists():
        return {}
    df = pd.read_csv(map_csv)
    df = df[df["net_name"].isin(target_nets)].reset_index(drop=True)
    grp: Dict[str, List[Path]] = defaultdict(list)
    for r in df.itertuples(index=False):
        grp[r.net_name].append(tile_dir / r.sample_filename)
    return grp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="intel22_nova_f3", choices=list(DESIGNS.keys()))
    ap.add_argument("--select", default="top:20,sample:100",
                    help="comma-separated: top:K, sample:N, all, list:nm1+nm2")
    ap.add_argument("--label", required=True,
                    help="output filename tag (e.g. baseline, patched_v3a)")
    ap.add_argument("--sample-seed", type=int, default=2026)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--skip-v4", action="store_true")
    ap.add_argument("--v3-algo", default="auto",
                    choices=["auto", "per_target", "legacy"],
                    help="V3 closest-pair backend (Round 3).")
    args = ap.parse_args()
    import pex_cold as _px
    _px._V3_PER_TARGET_MODE = args.v3_algo
    print(f"[dump] v3_algo={args.v3_algo}", flush=True)

    out_dir = Path(args.out_dir) if args.out_dir else (
        ROOT / "TreePEX" / "outputs" / "cold_reports" / "feature_dumps")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.design}__{args.label}.json"

    print(f"[dump] design={args.design} select={args.select} "
          f"label={args.label} sample_seed={args.sample_seed}", flush=True)

    t_total = time.perf_counter()
    layer_map = LayerInfoParser(LAYERS_INFO_PATH).parse()
    tech_lef = LefParser(TECH_LEF_PATH).parse()
    cell_lib = CellLibParser(CELL_LEF_PATH).parse()
    geo = scan_design(DESIGNS[args.design], layer_map, tech_lef, cell_lib)

    target_list = sorted(geo["target_set"])
    chosen = parse_select(args.select, target_list, geo, args.sample_seed)
    print(f"  selected {len(chosen)} nets out of {len(target_list)}", flush=True)

    # V3 globals once (single-process)
    eps_by_layer = _layer_eps_array(layer_map, N_LAYERS_EPS)
    grid = SpatialGrid()
    grid.build(geo["all_cuboids"])
    density_per_layer = np.zeros(N_LAYERS_EPS + 2, dtype=np.float64)
    if len(geo["all_cuboids"]) > 0:
        for li in range(1, N_LAYERS_EPS + 1):
            mask = geo["all_cuboids"][:, 6] == li
            density_per_layer[li] = float(
                (geo["all_cuboids"][mask, 3] * geo["all_cuboids"][mask, 4]).sum())
        xmin, xmax, ymin, ymax = _bbox_xy(geo["all_cuboids"])
        density_window = max(1.0, (xmax - xmin) * (ymax - ymin))
    else:
        density_window = 1.0
    init_worker_v3(geo, grid, eps_by_layer, density_per_layer, density_window)

    # V3 per-net feature + runtime
    print(f"[V3] dumping features for {len(chosen)} nets...", flush=True)
    v3_rows = []
    for i, nm in enumerate(chosen):
        t0 = time.perf_counter()
        feats = _v3_per_net(nm)
        wall = time.perf_counter() - t0
        if feats:
            feats["_runtime_s"] = wall
            feats["_n_cuboids_raw"] = int(len(geo["nets"][nm]))
            v3_rows.append(feats)
        if (i + 1) % 25 == 0:
            print(f"  V3 progress {i + 1}/{len(chosen)}", flush=True)

    # V4 per-net feature + runtime
    v4_rows = []
    if not args.skip_v4:
        tile_map = build_tile_map(args.design, set(chosen))
        if not tile_map:
            print(f"[V4] tile cache missing; skipping", flush=True)
        else:
            print(f"[V4] dumping features for {sum(1 for n in chosen if n in tile_map)} nets...",
                  flush=True)
            for i, nm in enumerate(chosen):
                tps = tile_map.get(nm, [])
                if not tps:
                    continue
                t0 = time.perf_counter()
                feats = _v4_process_net((nm, tps))
                wall = time.perf_counter() - t0
                if feats:
                    feats["_runtime_s"] = wall
                    feats["_n_tiles"] = len(tps)
                    v4_rows.append(feats)
                if (i + 1) % 25 == 0:
                    print(f"  V4 progress {i + 1}/{len(chosen)}", flush=True)

    # Write
    out = {
        "design": args.design,
        "label": args.label,
        "select_spec": args.select,
        "sample_seed": args.sample_seed,
        "n_selected": len(chosen),
        "n_target_nets_total": len(target_list),
        "wall_total_s": round(time.perf_counter() - t_total, 3),
        "v3": v3_rows,
        "v4": v4_rows,
    }
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n>>> wrote {out_path} "
          f"(V3 rows={len(v3_rows)} V4 rows={len(v4_rows)})", flush=True)


if __name__ == "__main__":
    main()
