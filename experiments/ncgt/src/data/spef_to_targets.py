"""
SPEF → NCGT supervision target mapping (Plan v3 §3.3).

Builds two-level supervision per net:
  - net-total CPL[a]: sum of *CAP entries between target net and aggressor net `a`.
  - per-edge CPL[seg_t, seg_a]: only when both SPEF endpoints map unambiguously
    to one of OUR segments (line-on-wire containment + WIRE-preferred tie-break).
    Uses `is_supervised[e]` mask in loss.

Audit Phase 4 confirmed ~16% strict-unique + ~?? ambiguous-with-tiebreak fraction.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from experiments.ncgt.src.data.segment_extractor import Segment


_RE_NODE = re.compile(r"^\*N\s+(\S+):(\d+)\s+\*C\s+(-?[\d.]+)\s+(-?[\d.]+).*?\$lvl=(\d+)")
_RE_DNET = re.compile(r"^\*D_NET\s+(\S+)\s+(-?[\d.]+)")
_RE_CAP_LINE = re.compile(r"^\d+\s+(\S+)(?::\d+)?\s+(\S+)(?::\d+)?\s+(-?[\d.eE+-]+)")


@dataclass
class NetSpef:
    net_name: str
    total_cap: float
    nodes: List[Tuple[int, float, float, int]]  # (node_id, x, y, lvl)
    cap_entries: List[Tuple[str, str, float]]   # (node1_full, node2_full, cap_fF)


def parse_spef(spef_path: Path) -> Dict[str, NetSpef]:
    """Stream-parse SPEF, returns {net_name: NetSpef}.

    Notes:
        - Coordinates in micrometers (after StarRC unit normalization).
        - cap_entries reference node names as 'net' (port) or 'net:N' (internal).
    """
    nets: Dict[str, NetSpef] = {}
    current: Optional[NetSpef] = None
    in_cap = False

    with open(spef_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = _RE_DNET.match(line)
            if m:
                if current is not None:
                    nets[current.net_name] = current
                current = NetSpef(
                    net_name=m.group(1),
                    total_cap=float(m.group(2)),
                    nodes=[],
                    cap_entries=[],
                )
                in_cap = False
                continue
            if line.startswith("*CAP"):
                in_cap = True
                continue
            if line.startswith("*RES") or line.startswith("*END") or line.startswith("*CONN"):
                in_cap = False
                continue
            if current is None:
                continue
            mn = _RE_NODE.match(line)
            if mn:
                _net, nid, x, y, lvl = mn.groups()
                current.nodes.append((int(nid), float(x), float(y), int(lvl)))
                continue
            if in_cap:
                mc = _RE_CAP_LINE.match(line)
                if mc:
                    n1, n2, cap = mc.groups()
                    try:
                        current.cap_entries.append((n1, n2, float(cap)))
                    except ValueError:
                        pass

    if current is not None:
        nets[current.net_name] = current
    return nets


def map_spef_node_to_segment(
    x: float,
    y: float,
    lvl: int,
    segments: List[Segment],
    perp_tol: float = 5e-3,
    endpoint_tol: float = 5e-3,
) -> Optional[int]:
    """Line-on-wire containment + WIRE-preferred tie-break.

    Returns the segment index (within `segments`) that best contains the SPEF
    node, or None if no segment contains it (within tolerance).
    """
    best_idx = None
    best_score = (False, 1e9)  # (is_wire, perp_distance) — sort key

    for i, s in enumerate(segments):
        if s.seg_type == "VIA":
            d2 = (x - s.x_mid) ** 2 + (y - s.y_mid) ** 2
            if d2 <= (s.w + endpoint_tol) ** 2:
                d = float(np.sqrt(d2))
                key = (False, d)
                if (key < best_score) if best_idx is not None else True:
                    best_score = key
                    best_idx = i
            continue
        if s.seg_type == "RECT":
            xlo = s.x_mid - max(abs(s.dx), s.w) / 2 - perp_tol
            xhi = s.x_mid + max(abs(s.dx), s.w) / 2 + perp_tol
            ylo = s.y_mid - max(abs(s.dy), s.w) / 2 - perp_tol
            yhi = s.y_mid + max(abs(s.dy), s.w) / 2 + perp_tol
            if xlo <= x <= xhi and ylo <= y <= yhi:
                d = max(abs(x - s.x_mid), abs(y - s.y_mid))
                key = (True, d)
                if best_idx is None or key < best_score:
                    best_score = key
                    best_idx = i
            continue
        # WIRE: parametric line containment.
        px, py = s.p_start[0], s.p_start[1]
        qx, qy = s.p_end[0], s.p_end[1]
        vx, vy = qx - px, qy - py
        length_sq = vx * vx + vy * vy
        if length_sq < 1e-12:
            d2 = (x - px) ** 2 + (y - py) ** 2
            if d2 <= (s.w + endpoint_tol) ** 2:
                d = float(np.sqrt(d2))
                key = (True, d)
                if best_idx is None or key < best_score:
                    best_score = key
                    best_idx = i
            continue
        t = ((x - px) * vx + (y - py) * vy) / length_sq
        tol_t = endpoint_tol / float(np.sqrt(length_sq))
        if t < -tol_t or t > 1.0 + tol_t:
            continue
        proj_x = px + t * vx
        proj_y = py + t * vy
        perp = float(np.sqrt((x - proj_x) ** 2 + (y - proj_y) ** 2))
        if perp <= s.w / 2 + perp_tol:
            key = (True, perp)
            if best_idx is None or key < best_score:
                best_score = key
                best_idx = i

    return best_idx


def build_edge_supervision(
    target_segments: List[Segment],
    aggressor_segments: List[Segment],
    target_net_spef: NetSpef,
    aggr_net_to_spef: Dict[str, NetSpef],
    edge_index: np.ndarray,                  # (2, E) [target_idx, aggr_idx]
    aggr_net_names: List[str],               # length = len(aggressor_segments)
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Construct per-edge GT cap + supervision mask + net-total CPL targets.

    Returns:
        gt_edge: (E,) float32 per-edge target (fF). Zero where unsupervised.
        is_supervised: (E,) bool — True where strict mapping succeeds.
        cpl_net_total: {aggr_net_name: total_cpl_fF}
    """
    n_aggr_segs = len(aggressor_segments)

    # 1) Map each SPEF *N node of target net AND of every aggressor net to a segment.
    target_node_to_seg: Dict[int, int] = {}
    for nid, x, y, lvl in target_net_spef.nodes:
        seg_idx = map_spef_node_to_segment(x, y, lvl, target_segments)
        if seg_idx is not None:
            target_node_to_seg[nid] = seg_idx

    aggr_node_to_seg: Dict[Tuple[str, int], int] = {}
    for aggr_name, aggr_spef in aggr_net_to_spef.items():
        # Filter aggressor_segments to those of this aggr_net.
        aggr_seg_indices = [i for i, n in enumerate(aggr_net_names) if n == aggr_name]
        if not aggr_seg_indices:
            continue
        local_segs = [aggressor_segments[i] for i in aggr_seg_indices]
        for nid, x, y, lvl in aggr_spef.nodes:
            local_idx = map_spef_node_to_segment(x, y, lvl, local_segs)
            if local_idx is not None:
                aggr_node_to_seg[(aggr_name, nid)] = aggr_seg_indices[local_idx]

    # 2) Walk *CAP entries, accumulate per-edge bin and per-net total.
    edge_target = defaultdict(float)
    cpl_net_total: Dict[str, float] = defaultdict(float)

    def parse_node_ref(nref: str) -> Tuple[str, Optional[int]]:
        if ":" in nref:
            n, nid = nref.split(":", 1)
            try:
                return n, int(nid)
            except ValueError:
                return n, None
        return nref, None

    target_name = target_net_spef.net_name
    for n1, n2, cap in target_net_spef.cap_entries:
        net1, nid1 = parse_node_ref(n1)
        net2, nid2 = parse_node_ref(n2)

        # Determine which side is target / aggressor.
        if net1 == target_name and net2 != target_name:
            tgt_nid = nid1
            aggr_name, aggr_nid = net2, nid2
        elif net2 == target_name and net1 != target_name:
            tgt_nid = nid2
            aggr_name, aggr_nid = net1, nid1
        else:
            # both target or neither — skip (not a coupling).
            continue

        # Net-total accumulation always.
        cpl_net_total[aggr_name] += cap

        # Per-edge supervision when both ends map.
        if tgt_nid is None or aggr_nid is None:
            continue
        if tgt_nid not in target_node_to_seg:
            continue
        if (aggr_name, aggr_nid) not in aggr_node_to_seg:
            continue
        ts = target_node_to_seg[tgt_nid]
        as_ = aggr_node_to_seg[(aggr_name, aggr_nid)]
        edge_target[(ts, as_)] += cap

    # 3) Project onto edge_index.
    n_edges = edge_index.shape[1]
    gt_edge = np.zeros(n_edges, dtype=np.float32)
    is_sup = np.zeros(n_edges, dtype=bool)
    for e in range(n_edges):
        ts = int(edge_index[0, e])
        as_ = int(edge_index[1, e])
        if (ts, as_) in edge_target:
            gt_edge[e] = float(edge_target[(ts, as_)])
            is_sup[e] = True

    return gt_edge, is_sup, dict(cpl_net_total)
