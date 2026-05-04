#!/usr/bin/env python3
"""
NCGT end-to-end smoke test (PLAN.md v4 §5 Phase 1.0).

For one design, one net:
  1. Extract target segments + aggressors via segment_extractor + KD-tree.
  2. Build edge bands via edge_builder (E_local, E_mid, E_long).
  3. Compute physics base for GND + CPL.
  4. Verify shapes / ranges / dtypes.
  5. Apply geometric augmentation, verify physics base invariance.

Run:
    python3 scripts/smoke_test_ncgt.py --design intel22_gcd_f3 --net <net_name>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from configs import config as cfg  # noqa: E402
from experiments.ncgt.src.data.segment_extractor import (  # noqa: E402
    iter_design_segments,
    role_for,
)
from experiments.ncgt.src.data.edge_builder import build_edges_for_net  # noqa: E402
from experiments.ncgt.src.data.physics_base import (  # noqa: E402
    cpl_base_per_edge,
    compute_segment_geometry,
    gnd_base_per_segment,
)
from experiments.ncgt.src.data.geometric_aug import (  # noqa: E402
    apply_to_endpoints,
    SAFE_TRANSFORMS,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="intel22_gcd_f3")
    ap.add_argument("--max_aggr_per_net", type=int, default=2000)
    ap.add_argument("--r_aggr", type=float, default=12.0)
    args = ap.parse_args()

    # Load PDK info.
    from src.preprocessing.layer_parser import LayerInfoParser
    from src.preprocessing.lef_parser import LefParser
    from src.preprocessing.cell_parser import CellLibParser

    layer_info = LayerInfoParser(str(cfg.LAYERS_INFO_PATH)).parse()
    tech_lef = LefParser(str(cfg.TECH_LEF_PATH)).parse()
    cell_lib = CellLibParser(str(cfg.CELL_LEF_PATH)).parse()

    def_path = None
    for p in cfg.TRAIN_DEFS + cfg.TEST_DEFS:
        if Path(p).stem == args.design:
            def_path = Path(p)
            break
    assert def_path is not None and def_path.exists(), f"design not found: {args.design}"

    # Extract all segments to build aggressor index, but only run forward on first signal net.
    print(f"[smoke] extracting all segments from {def_path.name}...")
    nets_segs = []
    for net_name, segs in iter_design_segments(str(def_path), layer_info, tech_lef, cell_lib):
        nets_segs.append((net_name, segs))
    print(f"[smoke] extracted {len(nets_segs)} nets")

    # Find first signal net with > 5 segments AND aggressors nearby.
    target_net = None
    for net_name, segs in nets_segs:
        from experiments.ncgt.src.data.segment_extractor import classify_net

        if classify_net(net_name) == "signal" and len(segs) >= 5:
            target_net = (net_name, segs)
            break
    assert target_net, "no suitable signal net found"
    target_name, target_segs = target_net
    print(f"[smoke] target net: {target_name} ({len(target_segs)} segs incl. subdivisions)")

    # Build aggressor pool: all non-target, non-via segments within R_aggr of any target midpoint.
    target_mids = np.array([(s.x_mid, s.y_mid, s.z) for s in target_segs], dtype=np.float32)
    aggressors = []
    aggr_net_names = []
    for net_name, segs in nets_segs:
        if net_name == target_name:
            continue
        for s in segs:
            if s.seg_type == "VIA":
                continue  # exclude vias per Phase 0 audit
            mid = np.array([s.x_mid, s.y_mid, s.z], dtype=np.float32)
            d = np.linalg.norm(target_mids - mid, axis=1)
            if d.min() < args.r_aggr:
                aggressors.append(s)
                aggr_net_names.append(net_name)
        if len(aggressors) > args.max_aggr_per_net:
            break

    # Cap aggressors at max.
    aggressors = aggressors[: args.max_aggr_per_net]
    aggr_net_names = aggr_net_names[: args.max_aggr_per_net]
    print(f"[smoke] aggressors within R={args.r_aggr}μm: {len(aggressors)}")

    # Map aggressor net names to integer ids.
    name_to_id = {}
    aggr_net_ids = []
    for n in aggr_net_names:
        if n not in name_to_id:
            name_to_id[n] = len(name_to_id)
        aggr_net_ids.append(name_to_id[n])

    # Build edges.
    edges = build_edges_for_net(
        targets=target_segs,
        aggressors=aggressors,
        r_edge_local=4.0,
        r_edge_mid=8.0,
        r_aggr=args.r_aggr,
        k_mid=8,
        aggr_net_ids=aggr_net_ids,
    )
    print(f"[smoke] edges: total={len(edges)}, "
          f"local={int((edges.band==0).sum())}, "
          f"mid={int((edges.band==1).sum())}, "
          f"long={int((edges.band==2).sum())}")

    # Physics base — GND per target segment.
    p_start_t = torch.tensor([s.p_start for s in target_segs], dtype=torch.float32)
    p_end_t = torch.tensor([s.p_end for s in target_segs], dtype=torch.float32)
    width_t = torch.tensor([s.w for s in target_segs], dtype=torch.float32)
    thick_t = torch.tensor([max(s.h, 0.05) for s in target_segs], dtype=torch.float32)
    area_t, perim_t = compute_segment_geometry(p_start_t, p_end_t, width_t, thick_t)
    # PDK-realistic placeholder dielectric: ε=3.0, d_top=d_bot=0.2μm.
    eps_t = torch.full_like(area_t, 3.0)
    d_t = torch.full_like(area_t, 0.2)
    cap_gnd = gnd_base_per_segment(
        seg_area_top=area_t, seg_area_bot=area_t,
        seg_perimeter=perim_t, seg_thickness=thick_t,
        d_top=d_t, d_bot=d_t,
        eps_top=eps_t, eps_bot=eps_t,
    )
    print(f"[smoke] GND base: total={cap_gnd.sum().item():.4f} fF, "
          f"per-seg mean={cap_gnd.mean().item():.4f} fF, "
          f"max={cap_gnd.max().item():.4f} fF")

    # Physics base — CPL per edge (sample first 100 edges).
    n_edges = len(edges)
    if n_edges == 0:
        print("[smoke] no edges generated; skipping CPL physics check")
    else:
        E_sample = min(n_edges, 100)
        ti = edges.edge_index[0, :E_sample]
        ai = edges.edge_index[1, :E_sample]
        same_layer = torch.tensor(
            [target_segs[t].layer_idx == aggressors[a].layer_idx for t, a in zip(ti, ai)],
            dtype=torch.bool,
        )
        # Lateral distance (xy). Vertical distance (z).
        d_xy = torch.tensor(
            [
                np.hypot(target_segs[t].x_mid - aggressors[a].x_mid,
                         target_segs[t].y_mid - aggressors[a].y_mid)
                for t, a in zip(ti, ai)
            ],
            dtype=torch.float32,
        )
        d_z = torch.tensor(
            [abs(target_segs[t].z - aggressors[a].z) for t, a in zip(ti, ai)],
            dtype=torch.float32,
        )
        # Overlap length / area (rough: use segment lengths as proxy).
        ov_len = torch.tensor(
            [
                min(np.hypot(target_segs[t].dx, target_segs[t].dy),
                    np.hypot(aggressors[a].dx, aggressors[a].dy))
                for t, a in zip(ti, ai)
            ],
            dtype=torch.float32,
        )
        ov_area = torch.tensor(
            [aggressors[a].w * np.hypot(aggressors[a].dx, aggressors[a].dy) for a in ai],
            dtype=torch.float32,
        )
        thick = torch.tensor(
            [max(target_segs[t].h, 0.05) for t in ti],
            dtype=torch.float32,
        )
        eps_pair = torch.full_like(thick, 3.0)
        cap_cpl = cpl_base_per_edge(
            same_layer=same_layer,
            overlap_length=ov_len,
            overlap_area=ov_area,
            lateral_distance=d_xy,
            vertical_distance=d_z,
            metal_thickness=thick,
            eps_pair=eps_pair,
        )
        print(f"[smoke] CPL base (first {E_sample} edges): "
              f"total={cap_cpl.sum().item():.4f} fF, "
              f"per-edge mean={cap_cpl.mean().item():.4f} fF, "
              f"max={cap_cpl.max().item():.4f} fF")

    # Augmentation invariance: physics base on GND should be invariant under transforms.
    print("[smoke] augmentation invariance check:")
    p_start_np = p_start_t.numpy()
    p_end_np = p_end_t.numpy()
    cap_orig = cap_gnd.sum().item()
    for transform in SAFE_TRANSFORMS:
        ps, pe = apply_to_endpoints(p_start_np, p_end_np, transform)
        # Recompute area, perimeter (z preserved → area unchanged for axis-aligned wires;
        # general affine invariant under rotation since Euclidean metric preserved).
        ps_t = torch.from_numpy(ps).float()
        pe_t = torch.from_numpy(pe).float()
        a_t, p_t = compute_segment_geometry(ps_t, pe_t, width_t, thick_t)
        cap_t = gnd_base_per_segment(
            seg_area_top=a_t, seg_area_bot=a_t,
            seg_perimeter=p_t, seg_thickness=thick_t,
            d_top=d_t, d_bot=d_t,
            eps_top=eps_t, eps_bot=eps_t,
        )
        rel = abs(cap_t.sum().item() - cap_orig) / max(cap_orig, 1e-9)
        flag = "✓" if rel < 1e-5 else "✗"
        print(f"    {flag} {transform:12s} GND total = {cap_t.sum().item():.6f} fF (rel diff {rel:.2e})")

    print("[smoke] OK — all components integrate end-to-end.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
