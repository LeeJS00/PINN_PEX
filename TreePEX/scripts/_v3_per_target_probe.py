"""Per-net diff probe: compare legacy vs per-target on ONE net.

Picks the worst-drift net's edges list (sorted by aggressor name to align)
and prints side-by-side per-aggressor (dist, broadside, lateral) so we
can pinpoint whether the divergence is tie-breaking, missing pairs, or a
bug in the broadside/lateral computation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "TreePEX" / "scripts"))

import pex_cold as px


def main():
    design = "intel22_tv80s_f3"
    target_net = "FE_OFN3_blkout_xor_0"  # placeholder; will pick after init
    target_net = None

    from src.preprocessing.lef_parser import LefParser
    from src.preprocessing.cell_parser import CellLibParser
    from src.preprocessing.layer_parser import LayerInfoParser

    layer_map = LayerInfoParser(px.LAYERS_INFO_PATH).parse()
    tech_lef = LefParser(px.TECH_LEF_PATH).parse()
    cell_lib = CellLibParser(px.CELL_LEF_PATH).parse()
    geo = px.scan_design(px.DESIGNS[design], layer_map, tech_lef, cell_lib)

    eps_by_layer = px._layer_eps_array(layer_map, px.N_LAYERS_EPS)
    grid = px.SpatialGrid()
    grid.build(geo["all_cuboids"])
    density_per_layer = np.zeros(px.N_LAYERS_EPS + 2, dtype=np.float64)
    if len(geo["all_cuboids"]) > 0:
        for li in range(1, px.N_LAYERS_EPS + 1):
            mask = geo["all_cuboids"][:, 6] == li
            density_per_layer[li] = float(
                (geo["all_cuboids"][mask, 3] * geo["all_cuboids"][mask, 4]).sum())
        xmin, xmax, ymin, ymax = px._bbox_xy(geo["all_cuboids"])
        density_window = max(1.0, (xmax - xmin) * (ymax - ymin))
    else:
        density_window = 1.0
    px.init_worker_v3(geo, grid, eps_by_layer, density_per_layer, density_window)

    # Pick the biggest net (most cuboids) -- those are where the diff lives.
    ranked = sorted(geo["target_set"],
                    key=lambda n: -len(geo["nets"].get(n, [])))
    target_net = ranked[0]
    print(f"probe net: {target_net}  "
          f"N_t={len(geo['nets'][target_net])}", flush=True)

    px._V3_PER_TARGET_MODE = "legacy"
    fl = px._v3_per_net(target_net)
    px._V3_PER_TARGET_MODE = "per_target"
    fp = px._v3_per_net(target_net)

    print("\n== scalar feature diffs ==")
    for k in sorted(fl.keys()):
        a = fl.get(k); b = fp.get(k)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if abs((a or 0) - (b or 0)) > 1e-9:
                print(f"  {k:40s}  legacy={a!r}  pertgt={b!r}  Δ={(b or 0)-(a or 0):+.6g}")
            else:
                pass  # equal
        elif a != b:
            print(f"  {k:40s}  legacy={a!r}  pertgt={b!r}")

    # Compare the full edges list by reconstructing via internal helpers.
    # We need the edges list, not just the aggregated scalars. So call the
    # closest-pair stage manually for each path.
    nets = geo["nets"]
    target_arr = nets[target_net]
    if len(target_arr) > px.MAX_TARGET_CUBS_V3:
        rng_t = np.random.RandomState(hash(target_net) & 0xFFFFFFFF)
        sub_idx = rng_t.choice(len(target_arr), px.MAX_TARGET_CUBS_V3, replace=False)
        target_arr_bc = target_arr[sub_idx]
    else:
        target_arr_bc = target_arr

    # Legacy: bbox query + cpu broadcast + dict aggregator
    x_min = float((target_arr[:, 0] - target_arr[:, 3] / 2).min())
    x_max = float((target_arr[:, 0] + target_arr[:, 3] / 2).max())
    y_min = float((target_arr[:, 1] - target_arr[:, 4] / 2).min())
    y_max = float((target_arr[:, 1] + target_arr[:, 4] / 2).max())
    cand_idx = grid.query_bbox(x_min - px.CUTOFF_UM, x_max + px.CUTOFF_UM,
                                y_min - px.CUTOFF_UM, y_max + px.CUTOFF_UM)
    cand_owners = geo["all_owner"][cand_idx]
    keep = cand_owners != target_net
    cand_idx2 = cand_idx[keep]
    cand_arr = geo["all_cuboids"][cand_idx2]
    cand_owners2 = cand_owners[keep]
    print(f"legacy: cand_arr size = {len(cand_arr)}, pair count = {len(target_arr_bc)*len(cand_arr):,}")

    closest_t, closest_d, in_range = px._v3_compute_closest_cpu(
        target_arr_bc, cand_arr, px.CUTOFF_UM)
    sel_cubs = cand_arr[in_range]
    sel_owners = cand_owners2[in_range]
    sel_dist = closest_d[in_range]
    sel_tidx = closest_t[in_range]
    matched_t = target_arr_bc[sel_tidx]
    aggr_legacy = {}
    for k in range(len(sel_owners)):
        a_owner = str(sel_owners[k])
        d_k = float(sel_dist[k])
        prior = aggr_legacy.get(a_owner)
        if prior is None or d_k < prior["dist"]:
            aggr_legacy[a_owner] = {
                "dist": d_k,
                "tgt_layer": int(matched_t[k, 6]),
                "aggr_layer": int(sel_cubs[k, 6]),
            }

    # Per-target
    edges_pertgt = px._v3_aggregate_per_target(target_net, target_arr_bc, px.CUTOFF_UM)
    aggr_pertgt = {e["aggressor_net"]: e for e in edges_pertgt}

    only_l = set(aggr_legacy) - set(aggr_pertgt)
    only_p = set(aggr_pertgt) - set(aggr_legacy)
    common = set(aggr_legacy) & set(aggr_pertgt)
    print(f"\n== aggressor sets ==")
    print(f"  legacy: {len(aggr_legacy)}  per_target: {len(aggr_pertgt)}  "
          f"common: {len(common)}  legacy_only: {len(only_l)}  per_target_only: {len(only_p)}")
    if only_l:
        print(f"  legacy_only sample: {list(only_l)[:5]}")
    if only_p:
        print(f"  per_target_only sample: {list(only_p)[:5]}")

    # Compare distances
    diffs = []
    for a in common:
        dl = aggr_legacy[a]["dist"]
        dp = aggr_pertgt[a]["surface_dist_um"]
        if abs(dl - dp) > 1e-9:
            diffs.append((a, dl, dp, dp - dl))
    print(f"\n== dist mismatches ==")
    print(f"  total common: {len(common)}, mismatched: {len(diffs)}")
    if diffs:
        diffs.sort(key=lambda r: -abs(r[3]))
        print(f"  worst 10:")
        for a, dl, dp, d in diffs[:10]:
            print(f"    {a[:50]:50s}  legacy={dl:.6f}  per_target={dp:.6f}  Δ={d:+.6f}")


if __name__ == "__main__":
    main()
