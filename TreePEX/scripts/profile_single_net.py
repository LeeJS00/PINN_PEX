"""Round 0 profiling gate — decompose `_v3_per_net` and `_v4_process_net`
wall-time on nova worst-case nets.

Decides whether V3-A alone (N_t bound) or V3-A + V3-B (N_t + N_c bound)
is needed to hit the Round 1 nova ≤ 1,800 s budget. See
TreePEX/FEATURE_SPEEDUP_PLAN.md §5 Round 0.

Usage:
    python3 TreePEX/scripts/profile_single_net.py
    python3 TreePEX/scripts/profile_single_net.py --design intel22_tv80s_f3
    python3 TreePEX/scripts/profile_single_net.py --top 5
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "TreePEX" / "scripts"))

from pex_cold import (
    CUTOFF_UM,
    MAX_AGGR_PER_NET,
    MAX_TARGET_CUBS_V4,
    SLACK_UM_V4,
    EPS_Z_V4,
    TOP_K,
    CB_X, CB_Y, CB_Z, CB_W, CB_H, CB_EPS,
    DESIGNS,
    TECH_LEF_PATH,
    CELL_LEF_PATH,
    LAYERS_INFO_PATH,
    TILE_CACHE_ROOT,
    SpatialGrid,
    scan_design,
)
from src.preprocessing.lef_parser import LefParser
from src.preprocessing.cell_parser import CellLibParser
from src.preprocessing.layer_parser import LayerInfoParser


def profile_v3_net(net_name: str, target_arr: np.ndarray, geo: dict,
                   grid: SpatialGrid) -> Dict[str, float]:
    """Reproduce `_v3_per_net` with per-stage timing. Out: dict of stage seconds."""
    t = {"net": net_name, "n_t": int(len(target_arr))}

    # Stage 1: scalar / sum features over full target_arr
    t0 = time.perf_counter()
    n = len(target_arr)
    _ = float(np.maximum.reduce([target_arr[:, 3], target_arr[:, 4], target_arr[:, 5]]).sum())
    _ = float((target_arr[:, 3] * target_arr[:, 4]).sum())
    x_min = float((target_arr[:, 0] - target_arr[:, 3] / 2).min())
    x_max = float((target_arr[:, 0] + target_arr[:, 3] / 2).max())
    y_min = float((target_arr[:, 1] - target_arr[:, 4] / 2).min())
    y_max = float((target_arr[:, 1] + target_arr[:, 4] / 2).max())
    layer_idx = np.clip(target_arr[:, 6].astype(np.int64), 1, 9)
    _, _ = np.histogram(layer_idx, bins=np.arange(1, 11)), None
    t["t_scalar_s"] = time.perf_counter() - t0

    # Stage 2: SpatialGrid query
    t0 = time.perf_counter()
    cand_idx = grid.query_bbox(x_min - CUTOFF_UM, x_max + CUTOFF_UM,
                               y_min - CUTOFF_UM, y_max + CUTOFF_UM)
    t["t_grid_query_s"] = time.perf_counter() - t0
    t["n_grid_hits"] = int(len(cand_idx))

    if len(cand_idx) == 0:
        for k in ("t_owner_filter_s", "t_broadcast_s", "t_dict_aggr_s",
                  "t_compact_gnd_loop_s", "t_compact_gnd_vec_s"):
            t[k] = 0.0
        t["n_c"] = 0
        return t

    # Stage 3: owner filter (exclude self-net candidates)
    t0 = time.perf_counter()
    cand_arr = geo["all_cuboids"][cand_idx]
    cand_owners = geo["all_owner"][cand_idx]
    keep = cand_owners != net_name
    cand_arr = cand_arr[keep]
    cand_owners = cand_owners[keep]
    t["t_owner_filter_s"] = time.perf_counter() - t0
    t["n_c"] = int(len(cand_arr))

    # Stage 4: (N_t x N_c) broadcast distance + argmin
    t0 = time.perf_counter()
    tx2 = target_arr[:, 0:1]; ty2 = target_arr[:, 1:2]
    tw2 = target_arr[:, 3:4]; th2 = target_arr[:, 4:5]
    ax = cand_arr[:, 0]; ay = cand_arr[:, 1]
    aw = cand_arr[:, 3]; ah = cand_arr[:, 4]
    dx = np.maximum(np.abs(tx2 - ax) - (tw2 + aw) / 2, 0)
    dy = np.maximum(np.abs(ty2 - ay) - (th2 + ah) / 2, 0)
    d_mat = np.sqrt(dx * dx + dy * dy)
    closest_t = d_mat.argmin(axis=0)
    closest_d = d_mat.min(axis=0)
    in_range = closest_d <= CUTOFF_UM
    t["t_broadcast_s"] = time.perf_counter() - t0
    t["n_in_range"] = int(in_range.sum())
    t["broadcast_pair_count"] = int(len(target_arr) * len(cand_arr))
    t["peak_pair_mb"] = round(len(target_arr) * len(cand_arr) * 8 / 1024**2, 1)

    # Stage 5: dict aggregation (closest aggressor per owner)
    t0 = time.perf_counter()
    if in_range.any():
        sel_owners = cand_owners[in_range]
        sel_dist = closest_d[in_range]
        sel_tidx = closest_t[in_range]
        sel_cuboids = cand_arr[in_range]
        matched_t = target_arr[sel_tidx]
        bs_x = np.maximum(np.minimum(matched_t[:, 0] + matched_t[:, 3] / 2,
                                     sel_cuboids[:, 0] + sel_cuboids[:, 3] / 2)
                          - np.maximum(matched_t[:, 0] - matched_t[:, 3] / 2,
                                       sel_cuboids[:, 0] - sel_cuboids[:, 3] / 2), 0)
        bs_y = np.maximum(np.minimum(matched_t[:, 1] + matched_t[:, 4] / 2,
                                     sel_cuboids[:, 1] + sel_cuboids[:, 4] / 2)
                          - np.maximum(matched_t[:, 1] - matched_t[:, 4] / 2,
                                       sel_cuboids[:, 1] - sel_cuboids[:, 4] / 2), 0)
        broadside = bs_x * bs_y
        lateral = matched_t[:, 5] * np.maximum(bs_x, bs_y)
        aggr_to_closest: Dict[str, dict] = {}
        for k in range(len(sel_owners)):
            a_owner = str(sel_owners[k])
            d_k = float(sel_dist[k])
            prior = aggr_to_closest.get(a_owner)
            if prior is None or d_k < prior["dist"]:
                aggr_to_closest[a_owner] = {
                    "dist": d_k,
                    "broadside": float(broadside[k]),
                    "lateral": float(lateral[k]),
                }
        t["n_unique_aggressors"] = len(aggr_to_closest)
    else:
        t["n_unique_aggressors"] = 0
    t["t_dict_aggr_s"] = time.perf_counter() - t0

    # Stage 6: compact_gnd Python loop (current implementation)
    t0 = time.perf_counter()
    eps_default = 4.0
    compact_gnd = 0.0
    for i in range(n):
        li = int(target_arr[i, 6])
        d_layers = max(1, li)
        d_um = max(0.05, d_layers * 0.1)
        A = float(target_arr[i, 3] * target_arr[i, 4])
        compact_gnd += 8.8541878128e-3 * eps_default * A / d_um
    t["t_compact_gnd_loop_s"] = time.perf_counter() - t0

    # Stage 6b: compact_gnd vectorized (V3-A' proposed fix)
    t0 = time.perf_counter()
    li_v = np.clip(target_arr[:, 6].astype(np.int64), 1, None)
    d_um_v = np.maximum(0.05, li_v * 0.1)
    A_v = target_arr[:, 3] * target_arr[:, 4]
    _ = float((8.8541878128e-3 * eps_default * A_v / d_um_v).sum())
    t["t_compact_gnd_vec_s"] = time.perf_counter() - t0

    return t


def profile_v4_net(net_name: str, tile_paths: List[Path]) -> Dict[str, float]:
    """Decompose `_v4_process_net`: tile load+decompress vs broadcast."""
    t = {"net": net_name, "n_tiles": len(tile_paths)}

    # Stage 1: tile load + gzip decompress + pickle
    t0 = time.perf_counter()
    target_chunks: List[np.ndarray] = []
    agg_groups: Dict[str, List[np.ndarray]] = defaultdict(list)
    bytes_read = 0
    for tp in tile_paths:
        try:
            bytes_read += tp.stat().st_size
            with gzip.open(tp, "rb") as f:
                tile = pickle.load(f)
        except Exception:
            continue
        cubs = tile.get("cuboids")
        names = tile.get("cuboid_net_names")
        if cubs is None or names is None or len(names) == 0:
            continue
        cubs = np.asarray(cubs, dtype=np.float32)
        names_arr = np.asarray([str(n) for n in names])
        t_mask = (names_arr == net_name)
        if t_mask.any():
            target_chunks.append(cubs[t_mask])
        a_mask = ~t_mask
        if a_mask.any():
            agg_cubs = cubs[a_mask]
            agg_names = names_arr[a_mask]
            for an in np.unique(agg_names):
                agg_groups[an].append(agg_cubs[agg_names == an])
    t["t_tile_load_s"] = time.perf_counter() - t0
    t["mb_read"] = round(bytes_read / 1024**2, 2)

    if not target_chunks:
        t["t_concat_s"] = 0.0
        t["t_broadcast_s"] = 0.0
        return t

    # Stage 2: concat
    t0 = time.perf_counter()
    target_cubs = np.concatenate(target_chunks, axis=0)
    agg_groups_np = {k: np.concatenate(v, axis=0) for k, v in agg_groups.items()}
    t["t_concat_s"] = time.perf_counter() - t0
    t["n_target_cubs"] = int(target_cubs.shape[0])
    t["n_agg_groups"] = len(agg_groups_np)
    t["n_total_agg_cubs"] = int(sum(v.shape[0] for v in agg_groups_np.values()))

    # Stage 3: V4 broadcast (subset of _v4_net_features)
    t0 = time.perf_counter()
    if target_cubs.shape[0] > MAX_TARGET_CUBS_V4:
        idx = np.random.RandomState(42).choice(target_cubs.shape[0],
                                               MAX_TARGET_CUBS_V4, replace=False)
        target_cubs = target_cubs[idx]
    tx = target_cubs[:, None, CB_X]; ty = target_cubs[:, None, CB_Y]
    tz = target_cubs[:, None, CB_Z]; tw = target_cubs[:, None, CB_W]
    th = target_cubs[:, None, CB_H]; teps = target_cubs[:, None, CB_EPS]
    for agg_name, agg_cubs in agg_groups_np.items():
        if agg_cubs.shape[0] == 0:
            continue
        ax = agg_cubs[None, :, CB_X]; ay = agg_cubs[None, :, CB_Y]
        az = agg_cubs[None, :, CB_Z]; aw = agg_cubs[None, :, CB_W]
        ah = agg_cubs[None, :, CB_H]; aeps = agg_cubs[None, :, CB_EPS]
        ovx = np.maximum(0.0, np.minimum(tx + tw / 2, ax + aw / 2)
                         - np.maximum(tx - tw / 2, ax - aw / 2))
        ovy = np.maximum(0.0, np.minimum(ty + th / 2, ay + ah / 2)
                         - np.maximum(ty - th / 2, ay - ah / 2))
        _ = ovx * ovy
        dz_mat = np.abs(tz - az)
        eps_avg = 0.5 * (teps + aeps)
        _ = float((eps_avg * _ / np.maximum(dz_mat, EPS_Z_V4)).sum())
    t["t_broadcast_s"] = time.perf_counter() - t0
    return t


def select_worst_v3_nets(geo: dict, top: int) -> List[Tuple[str, np.ndarray]]:
    rows = [(n, arr) for n, arr in geo["nets"].items() if n in geo["target_set"]]
    rows.sort(key=lambda x: -len(x[1]))
    return rows[:top]


def build_tile_map(design: str) -> Dict[str, List[Path]]:
    tile_dir = TILE_CACHE_ROOT / design
    map_csv = TILE_CACHE_ROOT / f"{design}_map.csv"
    if not tile_dir.exists() or not map_csv.exists():
        return {}
    df = pd.read_csv(map_csv)
    grp: Dict[str, List[Path]] = defaultdict(list)
    for r in df.itertuples(index=False):
        grp[r.net_name].append(tile_dir / r.sample_filename)
    return grp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="intel22_nova_f3", choices=list(DESIGNS.keys()))
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"[profile] design={args.design} top={args.top}", flush=True)
    t0 = time.perf_counter()
    layer_map = LayerInfoParser(LAYERS_INFO_PATH).parse()
    tech_lef = LefParser(TECH_LEF_PATH).parse()
    cell_lib = CellLibParser(CELL_LEF_PATH).parse()
    print(f"  PDK parse: {time.perf_counter() - t0:.2f} s", flush=True)

    t0 = time.perf_counter()
    geo = scan_design(DESIGNS[args.design], layer_map, tech_lef, cell_lib)
    print(f"  DEF parse: {time.perf_counter() - t0:.2f} s "
          f"target={len(geo['target_set'])} cuboids={len(geo['all_cuboids'])}",
          flush=True)

    t0 = time.perf_counter()
    grid = SpatialGrid()
    grid.build(geo["all_cuboids"])
    print(f"  SpatialGrid build: {time.perf_counter() - t0:.2f} s "
          f"buckets={len(grid.grid)}", flush=True)

    worst_v3 = select_worst_v3_nets(geo, args.top)
    print(f"\n[V3] worst {args.top} nets by n_cuboids:", flush=True)
    v3_results = []
    for nm, arr in worst_v3:
        r = profile_v3_net(nm, arr, geo, grid)
        v3_results.append(r)
        tot = (r["t_scalar_s"] + r["t_grid_query_s"] + r["t_owner_filter_s"]
               + r["t_broadcast_s"] + r["t_dict_aggr_s"] + r["t_compact_gnd_loop_s"])
        print(f"  {nm[:48]:48s} N_t={r['n_t']:>5d} N_c={r['n_c']:>6d}  "
              f"tot={tot:.3f}s | "
              f"broadcast={r['t_broadcast_s']:.3f}s ({r['t_broadcast_s']/tot*100:.0f}%) "
              f"compact_gnd_loop={r['t_compact_gnd_loop_s']:.3f}s "
              f"(vec={r['t_compact_gnd_vec_s']*1000:.1f}ms) "
              f"dict_aggr={r['t_dict_aggr_s']:.3f}s "
              f"grid={r['t_grid_query_s']:.3f}s "
              f"pair_MB={r['peak_pair_mb']:.1f}", flush=True)

    print(f"\n[V4] tile-cache profile (same {args.top} nets):", flush=True)
    tile_map = build_tile_map(args.design)
    v4_results = []
    if not tile_map:
        print(f"  tile cache missing for {args.design}; skipping V4", flush=True)
    else:
        for nm, _ in worst_v3:
            tps = tile_map.get(nm, [])
            if not tps:
                continue
            r = profile_v4_net(nm, tps)
            v4_results.append(r)
            print(f"  {nm[:48]:48s} tiles={r['n_tiles']:>4d} "
                  f"MB={r.get('mb_read', 0):>6.1f}  "
                  f"load={r['t_tile_load_s']:.3f}s "
                  f"concat={r.get('t_concat_s', 0):.3f}s "
                  f"broadcast={r.get('t_broadcast_s', 0):.3f}s", flush=True)

    out_path = (Path(args.out) if args.out
                else ROOT / "TreePEX" / "outputs" / "cold_reports"
                / f"profile_{args.design}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "design": args.design,
        "n_target_nets": len(geo["target_set"]),
        "n_total_cuboids": int(len(geo["all_cuboids"])),
        "v3": v3_results,
        "v4": v4_results,
    }, indent=2))
    print(f"\n>>> wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
