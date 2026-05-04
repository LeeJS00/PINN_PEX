#!/usr/bin/env python3
"""
NCGT Phase 0 audit script.

Six audits per PLAN.md v3 §5 Phase 0:
  1. Geometric distributions: segments/net, aggressors/net, edges/net.
  2. Bin distributions per design + per layer.
  3. Heterogeneous type counts per design.
  4. SPEF mapping ambiguity rate (per-edge supervisable fraction).
  5. Augmentation invariance numerical check (deferred — needs physics_base).
  6. Worst-tail memory profiling (deferred — needs model).

This script runs audits 1-4 directly on TRAIN/TEST DEFs without needing the model
or physics base. Audits 5-6 are placeholders to be filled when those modules exist.

Usage:
    python3 audit_phase0.py --audit 1
    python3 audit_phase0.py --audit all --max_designs 3 --max_nets_per_design 500
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Allow imports of project src. Fully-qualified `experiments.ncgt.src.*` paths
# resolve via project root; do NOT add experiments/ncgt to path or its `src`
# package shadows the project's `src/`.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from configs import config as cfg  # noqa: E402
from experiments.ncgt.src.data.segment_extractor import (  # noqa: E402
    Segment,
    classify_net,
    iter_design_segments,
    role_for,
)


CPL_BIN_EDGES = (0.0, 0.01, 0.1, 1.0, 10.0, float("inf"))  # fF
GND_BIN_EDGES = (0.0, 0.01, 0.1, 1.0, 10.0, float("inf"))

R_AGGR_SWEEP = (5.0, 8.0, 12.0, 20.0)  # μm — sweep for audit 1
R_EDGE_BANDS = (4.0, 8.0, 12.0)         # μm — local / mid / long thresholds


def bin_index(value: float, edges=CPL_BIN_EDGES) -> int:
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return i
    return len(edges) - 2


def load_layer_info():
    from src.preprocessing.layer_parser import LayerInfoParser

    return LayerInfoParser(str(cfg.LAYERS_INFO_PATH)).parse()


def load_tech_lef():
    try:
        from src.preprocessing.lef_parser import LefParser

        return LefParser(str(cfg.TECH_LEF_PATH)).parse()
    except Exception as exc:  # pragma: no cover
        print(f"  [warn] LEF parse failed: {exc}; using empty stub.")
        return {"vias": {}}


def load_cell_lib():
    try:
        from src.preprocessing.cell_parser import CellLibParser

        return CellLibParser(str(cfg.CELL_LEF_PATH)).parse()
    except Exception as exc:
        print(f"  [warn] cell LEF parse failed: {exc}; using empty stub.")
        return {}


# ---------------------------------------------------------------------------
# Audit 1+2+3: distributions on a single design.
# ---------------------------------------------------------------------------
def audit_design_distributions(
    def_path: Path,
    layer_info: Dict,
    tech_lef: Dict,
    cell_lib: Dict,
    max_nets: int = -1,
    r_aggr_sweep: Tuple[float, ...] = R_AGGR_SWEEP,
    skip_power: bool = False,
) -> Dict:
    """Compute distributions for one design with R_aggr sweep + power-net split.

    Returns nested dict with per-R_aggr metrics and per-net-class breakdown.
    """
    nets: List[Tuple[str, List[Segment]]] = []
    for i, (net_name, segs) in enumerate(iter_design_segments(str(def_path), layer_info, tech_lef, cell_lib)):
        if max_nets > 0 and i >= max_nets:
            break
        if skip_power and classify_net(net_name) in ("VDD", "VSS"):
            continue
        nets.append((net_name, segs))

    if not nets:
        return {"design": def_path.stem, "n_nets": 0}

    # Build a global KD-tree-ish lookup for aggressor counts (audit 1).
    # Use scipy if available, else brute force on midpoints.
    all_segs: List[Segment] = []
    for _, segs in nets:
        all_segs.extend(segs)
    coords = np.asarray(
        [[s.x_mid, s.y_mid, s.z] for s in all_segs], dtype=np.float32
    )
    layer_idx_arr = np.asarray([s.layer_idx for s in all_segs], dtype=np.int32)
    net_idx_arr = np.zeros(len(all_segs), dtype=np.int32)
    net_to_idx = {}
    for i, s in enumerate(all_segs):
        if s.net_name not in net_to_idx:
            net_to_idx[s.net_name] = len(net_to_idx)
        net_idx_arr[i] = net_to_idx[s.net_name]

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(coords)
        have_kdtree = True
    except Exception:
        tree = None
        have_kdtree = False

    # Per-R_aggr buckets
    sig_aggrs = {r: [] for r in r_aggr_sweep}
    sig_edges_local = {r: [] for r in r_aggr_sweep}
    sig_edges_mid = {r: [] for r in r_aggr_sweep}
    sig_edges_long = {r: [] for r in r_aggr_sweep}
    sig_segs_per_net = []
    sig_natural_per_net = []
    sig_subseg_per_net = []
    pwr_segs_per_net = []
    pwr_aggrs_per_net = {r: [] for r in r_aggr_sweep}
    role_counts = Counter()
    type_counts = Counter()
    layer_counts = Counter()
    net_class_counts = Counter()
    length_samples = []
    via_count_signal = 0
    via_count_aggr = 0

    # Iterate per net for aggressor query.
    seg_offset = 0
    for net_name, segs in nets:
        nclass = classify_net(net_name)
        net_class_counts[nclass] += 1
        nat_count = sum(1 for s in segs if not s.is_subdivision)
        sub_count = sum(1 for s in segs if s.is_subdivision)
        if nclass in ("VDD", "VSS"):
            pwr_segs_per_net.append(len(segs))
        else:
            sig_segs_per_net.append(len(segs))
            sig_natural_per_net.append(nat_count)
            sig_subseg_per_net.append(sub_count)

        # Target seg layer summary (use majority layer_idx).
        layer_idxs = [s.layer_idx for s in segs]
        target_layer = int(np.bincount([li for li in layer_idxs if li >= 0]).argmax()) if layer_idxs else 0

        # Per-segment summaries.
        target_indices = list(range(seg_offset, seg_offset + len(segs)))
        for s in segs:
            type_counts[s.seg_type] += 1
            layer_counts[s.layer] += 1
            length_samples.append(float(np.hypot(s.dx, s.dy)))
            if s.seg_type == "VIA":
                via_count_signal += 1

        # R_aggr sweep — for each radius, count aggressors within that radius.
        target_net_idx = net_idx_arr[target_indices[0]]
        for r in r_aggr_sweep:
            if have_kdtree:
                aggr_indices = set()
                for ti in target_indices:
                    aggr_indices.update(tree.query_ball_point(coords[ti], r=r, p=2.0))
                aggr_indices = {
                    j for j in aggr_indices if net_idx_arr[j] != target_net_idx
                }
            else:
                aggr_indices = set()
                for ti in target_indices:
                    d = np.linalg.norm(coords - coords[ti], axis=1)
                    for j in np.where(d < r)[0]:
                        if net_idx_arr[j] != target_net_idx:
                            aggr_indices.add(int(j))

            if nclass in ("VDD", "VSS"):
                pwr_aggrs_per_net[r].append(len(aggr_indices))
            else:
                sig_aggrs[r].append(len(aggr_indices))

                # Edge band counts (only for largest R, fold inwards)
                if r == max(r_aggr_sweep) and aggr_indices:
                    aggr_coords = coords[list(aggr_indices)]
                    loc = mid = long_ = 0
                    for ti in target_indices:
                        d = np.linalg.norm(aggr_coords - coords[ti], axis=1)
                        loc += int(np.sum(d < R_EDGE_BANDS[0]))
                        mid += int(np.sum((d >= R_EDGE_BANDS[0]) & (d < R_EDGE_BANDS[1])))
                        long_ += int(np.sum((d >= R_EDGE_BANDS[1]) & (d < R_EDGE_BANDS[2])))
                    sig_edges_local[r].append(loc)
                    sig_edges_mid[r].append(mid)
                    sig_edges_long[r].append(long_)

                # Per-role counts at largest R (representative).
                if r == max(r_aggr_sweep):
                    for j in aggr_indices:
                        seg_j = all_segs[j]
                        rr = role_for(seg_j, net_name, target_layer)
                        role_counts[rr] += 1
                        if rr == "via":
                            via_count_aggr += 1

        seg_offset += len(segs)

    def pcts(arr):
        if not arr:
            return {"p50": 0, "p95": 0, "p99": 0, "max": 0, "mean": 0}
        a = np.asarray(arr, dtype=np.float32)
        return {
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)),
            "max": float(a.max()),
            "mean": float(a.mean()),
        }

    r_max = max(r_aggr_sweep)
    return {
        "design": def_path.stem,
        "n_nets": len(nets),
        "n_signal_nets": len(sig_segs_per_net),
        "n_power_nets": len(pwr_segs_per_net),
        "n_segs_total": len(all_segs),
        "signal": {
            "segs_per_net": pcts(sig_segs_per_net),
            "natural_segs_per_net": pcts(sig_natural_per_net),
            "subseg_per_net": pcts(sig_subseg_per_net),
            "aggrs_per_net_by_R": {str(r): pcts(sig_aggrs[r]) for r in r_aggr_sweep},
            "edges_local_per_net": pcts(sig_edges_local[r_max]),
            "edges_mid_per_net": pcts(sig_edges_mid[r_max]),
            "edges_long_per_net": pcts(sig_edges_long[r_max]),
        },
        "power": {
            "segs_per_net": pcts(pwr_segs_per_net),
            "aggrs_per_net_by_R": {str(r): pcts(pwr_aggrs_per_net[r]) for r in r_aggr_sweep},
        },
        "segment_length": pcts(length_samples),
        "role_counts_at_R_max": dict(role_counts),
        "type_counts": dict(type_counts),
        "layer_counts": dict(layer_counts),
        "net_class_counts": dict(net_class_counts),
        "via_count_signal_segs": via_count_signal,
        "via_count_aggr_at_R_max": via_count_aggr,
    }


# ---------------------------------------------------------------------------
# Audit 4: SPEF mapping ambiguity.
# ---------------------------------------------------------------------------
def audit_spef_mapping(def_path: Path, spef_path: Path, layer_info: Dict, tech_lef: Dict, cell_lib: Dict, max_nets: int = -1) -> Dict:
    """Quick check of SPEF-N → segment containment uniqueness."""
    if not spef_path.exists():
        return {"design": def_path.stem, "spef_missing": True}

    # Parse SPEF *N entries: lines like "*N net:N *C x y // $lvl=L"
    import re

    spef_nodes = defaultdict(list)  # net_name -> list of (node_id, x, y, lvl)
    re_n = re.compile(r"^\*N\s+(\S+):(\d+)\s+\*C\s+(-?[\d.]+)\s+(-?[\d.]+).*\$lvl=(\d+)")
    n_lines = 0
    with open(spef_path, "r", errors="ignore") as f:
        for line in f:
            n_lines += 1
            m = re_n.match(line.strip())
            if m:
                nm, nid, x, y, lvl = m.groups()
                spef_nodes[nm].append((int(nid), float(x), float(y), int(lvl)))
            if n_lines > 5_000_000:
                break  # Don't crawl huge SPEFs entirely.

    # Build segment registry per net for the first N nets.
    contained = 0
    ambiguous = 0
    unmapped = 0
    total = 0
    nets_processed = 0
    for net_name, segs in iter_design_segments(str(def_path), layer_info, tech_lef, cell_lib):
        if max_nets > 0 and nets_processed >= max_nets:
            break
        nets_processed += 1
        if net_name not in spef_nodes:
            continue
        # Containment check: a SPEF *N is contained iff a segment's xy bbox covers it.
        # We don't reliably have layer→lvl correspondence here (PDK-specific), so we
        # check xy containment + same approximate z layer index.
        for nid, x, y, lvl in spef_nodes[net_name]:
            total += 1
            hits = []  # list of (seg_idx, perp_dist, is_wire) for tie-break
            for si, s in enumerate(segs):
                if s.seg_type == "VIA":
                    d2 = (x - s.x_mid) ** 2 + (y - s.y_mid) ** 2
                    if d2 <= (s.w + 5e-3) ** 2:
                        hits.append((si, np.sqrt(d2), False))
                    continue
                if s.seg_type == "RECT":
                    xlo = s.x_mid - max(abs(s.dx), s.w) / 2 - 1e-3
                    xhi = s.x_mid + max(abs(s.dx), s.w) / 2 + 1e-3
                    ylo = s.y_mid - max(abs(s.dy), s.w) / 2 - 1e-3
                    yhi = s.y_mid + max(abs(s.dy), s.w) / 2 + 1e-3
                    if xlo <= x <= xhi and ylo <= y <= yhi:
                        d = max(abs(x - s.x_mid), abs(y - s.y_mid))
                        hits.append((si, d, True))
                    continue
                # WIRE: parametric line-on-wire containment.
                px, py = s.p_start[0], s.p_start[1]
                qx, qy = s.p_end[0], s.p_end[1]
                vx, vy = qx - px, qy - py
                length_sq = vx * vx + vy * vy
                if length_sq < 1e-12:
                    d2 = (x - px) ** 2 + (y - py) ** 2
                    if d2 <= (s.w + 5e-3) ** 2:
                        hits.append((si, np.sqrt(d2), True))
                    continue
                t = ((x - px) * vx + (y - py) * vy) / length_sq
                tol_t = 5e-3 / np.sqrt(length_sq)
                if t < -tol_t or t > 1.0 + tol_t:
                    continue
                proj_x = px + t * vx
                proj_y = py + t * vy
                perp = float(np.sqrt((x - proj_x) ** 2 + (y - proj_y) ** 2))
                if perp <= s.w / 2 + 5e-3:
                    hits.append((si, perp, True))

            n_hits = len(hits)
            if n_hits == 1:
                contained += 1
            elif n_hits == 0:
                unmapped += 1
            else:
                # Tie-break: prefer WIRE > VIA, then minimum perpendicular distance.
                ambiguous += 1
                # Tie-break would assign to: min by (not is_wire, perp_dist).
                # Counted separately for "tie-break-recoverable" metric.
                hits.sort(key=lambda h: (not h[2], h[1]))

    # With tie-break, ambiguous resolves to a unique segment via WIRE-preference + min perp dist.
    # So usable per-edge supervision = unique + ambiguous (tie-broken).
    usable = contained + ambiguous
    return {
        "design": def_path.stem,
        "spef_missing": False,
        "nets_processed": nets_processed,
        "spef_nodes_total": total,
        "uniquely_contained": contained,
        "ambiguous": ambiguous,
        "unmapped": unmapped,
        "unique_frac": contained / max(1, total),
        "ambiguous_frac": ambiguous / max(1, total),
        "unmapped_frac": unmapped / max(1, total),
        "usable_with_tiebreak_frac": usable / max(1, total),
    }


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="1", choices=["1", "2", "3", "4", "all"], help="Audit number or 'all'")
    ap.add_argument("--max_designs", type=int, default=-1)
    ap.add_argument("--max_nets_per_design", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "PHASE0_AUDIT.json")
    ap.add_argument("--include_test", action="store_true", help="Include TEST_DEFS as well")
    ap.add_argument("--r_aggr_sweep", type=str, default="5,8,12,20",
                    help="Comma-separated R_aggr values (μm)")
    ap.add_argument("--skip_power", action="store_true",
                    help="Skip VDD/VSS nets (gives cleaner signal-only stats)")
    args = ap.parse_args()
    r_aggr_sweep = tuple(float(x) for x in args.r_aggr_sweep.split(","))

    print(f"[phase0] loading layer info: {cfg.LAYERS_INFO_PATH}")
    layer_info = load_layer_info()
    tech_lef = load_tech_lef()
    cell_lib = load_cell_lib()

    def_paths = list(cfg.TRAIN_DEFS)
    spef_paths = list(cfg.TRAIN_SPEFS)
    if args.include_test:
        def_paths += list(cfg.TEST_DEFS)
        spef_paths += list(cfg.TEST_SPEFS)
    if args.max_designs > 0:
        def_paths = def_paths[: args.max_designs]
        spef_paths = spef_paths[: args.max_designs]

    spef_by_design = {Path(d).stem.replace("_starrc", ""): Path(d) for d in spef_paths}

    results = {"by_design": {}, "summary": {}, "config": vars(args), "r_aggr_sweep": list(r_aggr_sweep)}

    for d in def_paths:
        d = Path(d)
        print(f"\n[phase0] === {d.stem} ===")
        if args.audit in ("1", "2", "3", "all"):
            dist = audit_design_distributions(
                d, layer_info, tech_lef, cell_lib,
                max_nets=args.max_nets_per_design,
                r_aggr_sweep=r_aggr_sweep,
                skip_power=args.skip_power,
            )
            results["by_design"].setdefault(d.stem, {})["distributions"] = dist
            sig = dist.get("signal", {})
            seg_p95 = sig.get("segs_per_net", {}).get("p95", 0)
            aggr_by_R = sig.get("aggrs_per_net_by_R", {})
            print(f"  n_signal_nets={dist.get('n_signal_nets')}, n_power_nets={dist.get('n_power_nets')}")
            print(f"  signal segs/net P95={seg_p95:.1f}")
            for r, p in aggr_by_R.items():
                print(f"  signal aggrs/net P95 @ R={r}μm: {p.get('p95', 0):.0f}")
            print(f"  role_counts (R={max(r_aggr_sweep)}μm): {dist.get('role_counts_at_R_max')}")

        if args.audit in ("4", "all"):
            sp = None
            for stem, sp_path in spef_by_design.items():
                if stem in d.stem or d.stem in stem:
                    sp = sp_path
                    break
            if sp is None:
                print(f"  [audit4] no SPEF match for {d.stem}, skipping")
                continue
            mapping = audit_spef_mapping(d, sp, layer_info, tech_lef, cell_lib, max_nets=min(args.max_nets_per_design, 200))
            results["by_design"].setdefault(d.stem, {})["spef_mapping"] = mapping
            print(f"  spef nodes total={mapping.get('spef_nodes_total')}, "
                  f"unique={mapping.get('unique_frac', 0):.2%}, "
                  f"ambig={mapping.get('ambiguous_frac', 0):.2%}, "
                  f"unmapped={mapping.get('unmapped_frac', 0):.2%}, "
                  f"usable_w_tiebreak={mapping.get('usable_with_tiebreak_frac', 0):.2%}")

    # Aggregate summary.
    agg_role = Counter()
    agg_class = Counter()
    n_signal = 0
    n_power = 0
    sig_seg_p95s = []
    sig_aggr_by_R = {str(r): [] for r in r_aggr_sweep}
    unique_fracs = []
    usable_fracs = []
    unmapped_fracs = []
    for stem, data in results["by_design"].items():
        if "distributions" in data:
            dist = data["distributions"]
            agg_role.update(dist.get("role_counts_at_R_max", {}))
            agg_class.update(dist.get("net_class_counts", {}))
            n_signal += dist.get("n_signal_nets", 0)
            n_power += dist.get("n_power_nets", 0)
            sig = dist.get("signal", {})
            sig_seg_p95s.append(sig.get("segs_per_net", {}).get("p95", 0))
            for r in r_aggr_sweep:
                sig_aggr_by_R[str(r)].append(sig.get("aggrs_per_net_by_R", {}).get(str(r), {}).get("p95", 0))
        if "spef_mapping" in data and not data["spef_mapping"].get("spef_missing"):
            unique_fracs.append(data["spef_mapping"].get("unique_frac", 0))
            usable_fracs.append(data["spef_mapping"].get("usable_with_tiebreak_frac", 0))
            unmapped_fracs.append(data["spef_mapping"].get("unmapped_frac", 0))

    results["summary"] = {
        "n_designs": len(def_paths),
        "n_signal_nets_total": n_signal,
        "n_power_nets_total": n_power,
        "agg_role_counts_at_R_max": dict(agg_role),
        "agg_net_class_counts": dict(agg_class),
        "signal_seg_p95_across_designs": pct_summary(sig_seg_p95s),
        "signal_aggr_p95_by_R": {r: pct_summary(v) for r, v in sig_aggr_by_R.items()},
        "spef_unique_frac": pct_summary(unique_fracs),
        "spef_usable_with_tiebreak_frac": pct_summary(usable_fracs),
        "spef_unmapped_frac": pct_summary(unmapped_fracs),
        "phase0_pass_criteria": {
            "spef_usable_target": 0.30,
            "spef_usable_actual_mean": float(np.mean(usable_fracs)) if usable_fracs else 0.0,
            "passes_supervision_gate": (float(np.mean(usable_fracs)) if usable_fracs else 0.0) >= 0.30,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[phase0] results written to {args.out}")
    print(f"[phase0] summary: {json.dumps(results['summary'], indent=2, default=str)}")
    return 0


def pct_summary(arr):
    if not arr:
        return {"p50": 0, "max": 0, "mean": 0}
    a = np.asarray(arr, dtype=np.float32)
    return {"p50": float(np.percentile(a, 50)), "max": float(a.max()), "mean": float(a.mean())}


if __name__ == "__main__":
    sys.exit(main())
