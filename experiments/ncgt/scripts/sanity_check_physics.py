#!/usr/bin/env python3
"""
Sanity check: physics base accuracy vs StarRC SPEF.

Risk #1 from review: physics_base.gnd_base + cpl_base must be within ~30-50% of
StarRC for the ResCap residual paradigm to work. If physics base is 10× off,
residual range [-0.5, +1.0] cannot compensate → paradigm crisis.

Output:
    Per-net pure-physics-base MAPE on net-total GND, CPL, total.
    If mean MAPE < 50%: residual head can compensate → OK.
    If mean MAPE > 100%: physics formula broken → halt before Phase 2.0.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from configs import config as cfg  # noqa: E402
from experiments.ncgt.src.data.segment_extractor import (  # noqa: E402
    classify_net,
    iter_design_segments,
)
from experiments.ncgt.src.data.edge_builder import build_edges_for_net  # noqa: E402
from experiments.ncgt.src.data.layer_physics import LayerPhysicsTable  # noqa: E402
from experiments.ncgt.src.data.physics_base import (  # noqa: E402
    cpl_base_per_edge,
    compute_segment_geometry,
    edge_overlap_length,
    gnd_base_per_segment,
)
from experiments.ncgt.src.data.spef_to_targets import parse_spef  # noqa: E402


def physics_base_for_net(target_segs, aggressors, layer_table=None, eps=3.0, d=0.2, t=0.144):
    """Compute pure physics base (no NN) for one net.

    If layer_table is provided, uses per-layer ε/d/t (proper); else placeholder.
    """
    # GND
    p_start = torch.tensor([s.p_start for s in target_segs], dtype=torch.float32)
    p_end = torch.tensor([s.p_end for s in target_segs], dtype=torch.float32)
    width = torch.tensor([s.w for s in target_segs], dtype=torch.float32)

    if layer_table is not None:
        layer_idxs = torch.tensor([s.layer_idx for s in target_segs], dtype=torch.long)
        seg_phys = layer_table.build_seg_tensors(layer_idxs)
        thick = seg_phys["t_metal"].clamp(min=0.05)
        area, perim = compute_segment_geometry(p_start, p_end, width, thick)
        cap_gnd = gnd_base_per_segment(
            seg_area_top=area, seg_area_bot=area,
            seg_perimeter=perim, seg_thickness=thick,
            d_top=seg_phys["d_above"], d_bot=seg_phys["d_below"],
            eps_top=seg_phys["eps_above"], eps_bot=seg_phys["eps_below"],
        )
    else:
        thick = torch.tensor([max(s.h, t) for s in target_segs], dtype=torch.float32)
        area, perim = compute_segment_geometry(p_start, p_end, width, thick)
        eps_t = torch.full_like(area, eps)
        d_t = torch.full_like(area, d)
        cap_gnd = gnd_base_per_segment(
            seg_area_top=area, seg_area_bot=area,
            seg_perimeter=perim, seg_thickness=thick,
            d_top=d_t, d_bot=d_t,
            eps_top=eps_t, eps_bot=eps_t,
        )

    # CPL: build edges first, then physics.
    edges = build_edges_for_net(
        targets=list(target_segs),
        aggressors=list(aggressors),
        r_edge_local=4.0, r_edge_mid=8.0, r_aggr=12.0,
        k_mid=8,
    )
    if len(edges) == 0:
        return float(cap_gnd.sum().item()), 0.0

    ti = edges.edge_index[0]
    ai = edges.edge_index[1]
    same_layer = torch.tensor(
        [target_segs[t].layer_idx == aggressors[a].layer_idx for t, a in zip(ti, ai)],
        dtype=torch.bool,
    )
    d_xy = torch.tensor(
        [np.hypot(target_segs[t].x_mid - aggressors[a].x_mid,
                  target_segs[t].y_mid - aggressors[a].y_mid)
         for t, a in zip(ti, ai)],
        dtype=torch.float32,
    )
    d_z = torch.tensor(
        [abs(target_segs[t].z - aggressors[a].z) for t, a in zip(ti, ai)],
        dtype=torch.float32,
    )
    # Proper parallel-projection overlap with parallelism gate.
    t_ps = torch.tensor([target_segs[t].p_start for t in ti], dtype=torch.float32)
    t_pe = torch.tensor([target_segs[t].p_end for t in ti], dtype=torch.float32)
    a_ps = torch.tensor([aggressors[a].p_start for a in ai], dtype=torch.float32)
    a_pe = torch.tensor([aggressors[a].p_end for a in ai], dtype=torch.float32)
    ov_len = edge_overlap_length(t_ps, t_pe, a_ps, a_pe)
    a_w = torch.tensor([aggressors[a].w for a in ai], dtype=torch.float32)
    ov_area = ov_len * a_w

    if layer_table is not None:
        t_layer_idxs = torch.tensor([target_segs[t].layer_idx for t in ti], dtype=torch.long)
        a_layer_idxs = torch.tensor([aggressors[a].layer_idx for a in ai], dtype=torch.long)
        pair_phys = layer_table.build_pair_tensors(t_layer_idxs, a_layer_idxs)
        thick_e = pair_phys["t_pair"].clamp(min=0.05)
        eps_pair = pair_phys["eps_pair"]
    else:
        thick_e = torch.tensor([max(target_segs[t].h, 0.05) for t in ti], dtype=torch.float32)
        eps_pair = torch.full_like(thick_e, eps)
    cap_cpl = cpl_base_per_edge(
        same_layer=same_layer,
        overlap_length=ov_len,
        overlap_area=ov_area,
        lateral_distance=d_xy,
        vertical_distance=d_z,
        metal_thickness=thick_e,
        eps_pair=eps_pair,
    )

    return float(cap_gnd.sum().item()), float(cap_cpl.sum().item())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="intel22_gcd_f3")
    ap.add_argument("--n_nets", type=int, default=20)
    ap.add_argument("--r_aggr", type=float, default=12.0)
    ap.add_argument("--max_aggr", type=int, default=2000)
    args = ap.parse_args()

    from src.preprocessing.layer_parser import LayerInfoParser
    from src.preprocessing.lef_parser import LefParser
    from src.preprocessing.cell_parser import CellLibParser

    layer_info = LayerInfoParser(str(cfg.LAYERS_INFO_PATH)).parse()
    tech_lef = LefParser(str(cfg.TECH_LEF_PATH)).parse()
    cell_lib = CellLibParser(str(cfg.CELL_LEF_PATH)).parse()
    layer_table = LayerPhysicsTable(layer_info)

    def_path = None
    spef_path = None
    for d, s in zip(cfg.TRAIN_DEFS + cfg.TEST_DEFS, cfg.TRAIN_SPEFS + cfg.TEST_SPEFS):
        if Path(d).stem == args.design:
            def_path = Path(d)
            spef_path = Path(s)
            break
    assert def_path is not None and spef_path.exists()

    print(f"[sanity] parsing DEF + SPEF for {args.design}")
    nets_segs = list(iter_design_segments(str(def_path), layer_info, tech_lef, cell_lib))
    spef_nets = parse_spef(spef_path)

    # Build flat aggressor index (vias excluded).
    all_segs = []
    for nm, segs in nets_segs:
        for s in segs:
            if s.seg_type != "VIA":
                all_segs.append((nm, s))
    coords = np.array([(s.x_mid, s.y_mid, s.z) for _, s in all_segs], dtype=np.float32)
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(coords)
    except Exception:
        tree = None

    rows = []
    n_done = 0
    for net_name, target_segs in nets_segs:
        if classify_net(net_name) != "signal" or len(target_segs) < 5:
            continue
        if net_name not in spef_nets:
            continue
        if n_done >= args.n_nets:
            break

        target_mids = np.array([(s.x_mid, s.y_mid, s.z) for s in target_segs], dtype=np.float32)
        aggr_idx_set = set()
        if tree is not None:
            for tm in target_mids:
                for j in tree.query_ball_point(tm, r=args.r_aggr):
                    if all_segs[j][0] != net_name:
                        aggr_idx_set.add(j)
        if not aggr_idx_set:
            continue
        aggr_indices = list(aggr_idx_set)[:args.max_aggr]
        aggressors = [all_segs[j][1] for j in aggr_indices]

        gnd_base, cpl_base = physics_base_for_net(target_segs, aggressors, layer_table=layer_table)

        spef = spef_nets[net_name]
        # Net-level GND ≈ sum of target-only *CAP entries (where neither node is in another net).
        # CPL ≈ sum of *CAP where one node is target and other is different net.
        cpl_gt = 0.0
        for n1, n2, c in spef.cap_entries:
            net1 = n1.split(":")[0]
            net2 = n2.split(":")[0]
            if (net1 == net_name) != (net2 == net_name):  # exactly one side is target
                cpl_gt += c
        gnd_gt = max(0.0, spef.total_cap - cpl_gt)

        gnd_mape = abs(gnd_base - gnd_gt) / max(gnd_gt, 1e-3)
        cpl_mape = abs(cpl_base - cpl_gt) / max(cpl_gt, 1e-3)
        total_pred = gnd_base + cpl_base
        total_gt = gnd_gt + cpl_gt
        total_mape = abs(total_pred - total_gt) / max(total_gt, 1e-3)

        rows.append({
            "net": net_name, "T": len(target_segs), "A": len(aggressors),
            "gnd_pred": gnd_base, "gnd_gt": gnd_gt, "gnd_mape": gnd_mape,
            "cpl_pred": cpl_base, "cpl_gt": cpl_gt, "cpl_mape": cpl_mape,
            "tot_pred": total_pred, "tot_gt": total_gt, "tot_mape": total_mape,
        })
        n_done += 1

    print(f"\n[sanity] {len(rows)} nets analyzed (placeholder ε=3.0, d=0.2, t=0.144)\n")
    print(f"{'net':<25} {'T':>4} {'A':>5} {'gnd_pred':>9} {'gnd_gt':>9} {'mape':>6} "
          f"{'cpl_pred':>9} {'cpl_gt':>9} {'mape':>6} {'tot_mape':>8}")
    for r in rows:
        print(f"{r['net'][:24]:<25} {r['T']:>4} {r['A']:>5} "
              f"{r['gnd_pred']:>9.4f} {r['gnd_gt']:>9.4f} {r['gnd_mape']:>6.2f} "
              f"{r['cpl_pred']:>9.4f} {r['cpl_gt']:>9.4f} {r['cpl_mape']:>6.2f} "
              f"{r['tot_mape']:>8.2f}")

    if rows:
        gnd_mapes = [r["gnd_mape"] for r in rows]
        cpl_mapes = [r["cpl_mape"] for r in rows]
        tot_mapes = [r["tot_mape"] for r in rows]
        print(f"\n[sanity] mean GND MAPE: {np.mean(gnd_mapes):.2%}")
        print(f"[sanity] mean CPL MAPE: {np.mean(cpl_mapes):.2%}")
        print(f"[sanity] mean total MAPE: {np.mean(tot_mapes):.2%}")
        print(f"[sanity] median total MAPE: {np.median(tot_mapes):.2%}")

        verdict_gnd = np.mean(gnd_mapes)
        verdict_cpl = np.mean(cpl_mapes)
        if verdict_gnd < 0.5 and verdict_cpl < 0.5:
            print("\n[sanity] PASS — physics base within 50% MAPE; residual range can compensate.")
        elif verdict_gnd < 1.0 and verdict_cpl < 2.0:
            print("\n[sanity] MARGINAL — physics base 50-200% off; consider widening residual range OR threading layer_info.")
        else:
            print("\n[sanity] FAIL — physics base wildly off (>200% MAPE); paradigm crisis. Review formula.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
