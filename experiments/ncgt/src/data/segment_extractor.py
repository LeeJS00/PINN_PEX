"""
NCGT segment extractor.

Converts DefStreamParser output into NCGT-native conductor segment primitives:
- Natural segment: continuous metal piece from one DEF WIRE entry (already broken at jog/via).
- Virtual subsegment: long segments split at L_subdiv to align with SPEF *RES segmentation.
- Heterogeneous net class: signal / VDD / VSS / clock_signal.
- Heterogeneous role: target / signal_aggr_same_layer / signal_aggr_cross_layer / power_VDD / power_VSS / via / pin / branch_node.

Reuses existing src.preprocessing.def_parser.DefStreamParser. Read-only import.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np


L_SUBDIV_DEFAULT = 4.0  # μm; matches R_attn so coupling locality is representable.

POWER_VDD_PATTERNS = (re.compile(r"^vdd", re.IGNORECASE), re.compile(r"^vcc", re.IGNORECASE))
POWER_VSS_PATTERNS = (re.compile(r"^vss", re.IGNORECASE), re.compile(r"^gnd", re.IGNORECASE))
CLOCK_PATTERNS = (re.compile(r"clk", re.IGNORECASE), re.compile(r"^clock"), re.compile(r"^ck"))


def classify_net(net_name: str) -> str:
    """Returns one of: 'VDD', 'VSS', 'clock', 'signal'."""
    for pat in POWER_VDD_PATTERNS:
        if pat.search(net_name):
            return "VDD"
    for pat in POWER_VSS_PATTERNS:
        if pat.search(net_name):
            return "VSS"
    for pat in CLOCK_PATTERNS:
        if pat.search(net_name):
            return "clock"
    return "signal"


@dataclass
class Segment:
    """One NCGT primitive — natural or virtual subsegment of a conductor."""
    seg_id: int                  # unique within (design, net)
    parent_seg_id: int           # natural segment id (== seg_id if natural)
    is_subdivision: bool         # 1 if virtual subsegment
    seg_type: str                # 'WIRE' | 'VIA' | 'PIN' | 'RECT'
    layer: str
    layer_idx: int               # integer layer index (sortable z-order)
    x_mid: float
    y_mid: float
    z: float                     # absolute z-position (μm)
    dx: float                    # xy extent
    dy: float
    w: float                     # cross-section width
    h: float                     # layer thickness
    net_name: str
    net_class: str               # 'signal' | 'VDD' | 'VSS' | 'clock'
    semantic_type: float         # 0=wire, 0.5=pin, 1.0=via
    # Endpoints (for parallel-overlap edge construction)
    p_start: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    p_end: Tuple[float, float, float] = (0.0, 0.0, 0.0)


def _layer_to_idx(layer: str, layer_info: Dict) -> int:
    """Map layer name to integer z-order. Falls back to numeric suffix."""
    if layer in layer_info:
        z = layer_info[layer].get("z", 0.0)
        return int(round(z * 100))  # μm to centimicron, monotonic
    m = re.search(r"(\d+)", layer)
    return int(m.group(1)) if m else 0


def _layer_z_h(layer: str, layer_info: Dict) -> Tuple[float, float]:
    """Returns (z_center, thickness) in μm."""
    info = layer_info.get(layer, {})
    z = float(info.get("z", 0.0))
    h = float(info.get("thickness", info.get("t", 0.1)))
    return z, h


def _split_long_segment(seg: Segment, l_subdiv: float, next_id_start: int) -> List[Segment]:
    """Split a WIRE longer than l_subdiv into virtual subsegments of ~l_subdiv each."""
    length_xy = float(np.hypot(seg.dx, seg.dy))
    if seg.seg_type != "WIRE" or length_xy <= l_subdiv:
        return [seg]

    n_sub = int(np.ceil(length_xy / l_subdiv))
    pieces: List[Segment] = []
    px, py, pz = seg.p_start
    qx, qy, qz = seg.p_end
    for k in range(n_sub):
        t0 = k / n_sub
        t1 = (k + 1) / n_sub
        sx = px + t0 * (qx - px)
        sy = py + t0 * (qy - py)
        ex = px + t1 * (qx - px)
        ey = py + t1 * (qy - py)
        sub = Segment(
            seg_id=next_id_start + k,
            parent_seg_id=seg.seg_id,
            is_subdivision=True,
            seg_type=seg.seg_type,
            layer=seg.layer,
            layer_idx=seg.layer_idx,
            x_mid=0.5 * (sx + ex),
            y_mid=0.5 * (sy + ey),
            z=seg.z,
            dx=ex - sx,
            dy=ey - sy,
            w=seg.w,
            h=seg.h,
            net_name=seg.net_name,
            net_class=seg.net_class,
            semantic_type=seg.semantic_type,
            p_start=(sx, sy, pz),
            p_end=(ex, ey, qz),
        )
        pieces.append(sub)
    return pieces


def extract_segments_for_net(
    net_name: str,
    def_segments: List[Dict],
    layer_info: Dict,
    l_subdiv: float = L_SUBDIV_DEFAULT,
) -> List[Segment]:
    """Convert def_parser output for one net into NCGT Segment list with virtual subsegments."""
    net_class = classify_net(net_name)
    out: List[Segment] = []
    next_id = 0

    for raw in def_segments:
        rtype = raw.get("type")
        layer = raw.get("layer") or raw.get("bot_layer") or "m1"
        z, h = _layer_z_h(layer, layer_info)
        layer_idx = _layer_to_idx(layer, layer_info)

        if rtype == "WIRE":
            sx, sy = float(raw["start"][0]), float(raw["start"][1])
            ex, ey = float(raw["end"][0]), float(raw["end"][1])
            sz = float(raw["start"][2]) if len(raw["start"]) > 2 else z
            ez = float(raw["end"][2]) if len(raw["end"]) > 2 else z
            w = float(raw.get("width", 0.05))
            seg = Segment(
                seg_id=next_id,
                parent_seg_id=next_id,
                is_subdivision=False,
                seg_type="WIRE",
                layer=layer,
                layer_idx=layer_idx,
                x_mid=0.5 * (sx + ex),
                y_mid=0.5 * (sy + ey),
                z=z,
                dx=ex - sx,
                dy=ey - sy,
                w=w,
                h=h,
                net_name=net_name,
                net_class=net_class,
                semantic_type=0.0,
                p_start=(sx, sy, sz),
                p_end=(ex, ey, ez),
            )
            next_id += 1
            pieces = _split_long_segment(seg, l_subdiv, next_id)
            if pieces[0] is not seg:  # was split
                next_id += len(pieces)
            out.extend(pieces)

        elif rtype == "RECT":
            x1, y1, x2, y2 = raw["rect"]
            ref = raw.get("ref_point", (0, 0, 0))
            rz = float(ref[2]) if len(ref) > 2 else z
            seg = Segment(
                seg_id=next_id,
                parent_seg_id=next_id,
                is_subdivision=False,
                seg_type="RECT",
                layer=layer,
                layer_idx=layer_idx,
                x_mid=0.5 * (x1 + x2),
                y_mid=0.5 * (y1 + y2),
                z=z,
                dx=x2 - x1,
                dy=y2 - y1,
                w=min(abs(x2 - x1), abs(y2 - y1)),
                h=h,
                net_name=net_name,
                net_class=net_class,
                semantic_type=0.5 if "pin" in (raw.get("net_name") or "").lower() else 0.0,
                p_start=(x1, y1, rz),
                p_end=(x2, y2, rz),
            )
            out.append(seg)
            next_id += 1

        elif rtype == "VIA":
            px, py = float(raw["pos"][0]), float(raw["pos"][1])
            pz = float(raw["pos"][2]) if len(raw["pos"]) > 2 else z
            bot_z, _ = _layer_z_h(raw.get("bot_layer", layer), layer_info)
            top_z, _ = _layer_z_h(raw.get("top_layer", layer), layer_info)
            seg = Segment(
                seg_id=next_id,
                parent_seg_id=next_id,
                is_subdivision=False,
                seg_type="VIA",
                layer=raw.get("bot_layer", layer),
                layer_idx=layer_idx,
                x_mid=px,
                y_mid=py,
                z=0.5 * (bot_z + top_z),
                dx=0.0,
                dy=0.0,
                w=float(raw.get("width", 0.05)),
                h=abs(top_z - bot_z),
                net_name=net_name,
                net_class=net_class,
                semantic_type=1.0,
                p_start=(px, py, bot_z),
                p_end=(px, py, top_z),
            )
            out.append(seg)
            next_id += 1

    return out


def role_for(seg: Segment, target_net: str, target_layer_idx: Optional[int] = None) -> str:
    """Heterogeneous role assignment for an aggressor wrt a target net.

    Returns one of: 'target', 'signal_aggr_same_layer', 'signal_aggr_cross_layer',
                    'power_VDD', 'power_VSS', 'via', 'pin', 'branch_node'.
    """
    if seg.net_name == target_net:
        return "target"
    if seg.seg_type == "VIA":
        return "via"
    if seg.semantic_type == 0.5:
        return "pin"
    if seg.net_class == "VDD":
        return "power_VDD"
    if seg.net_class == "VSS":
        return "power_VSS"
    if target_layer_idx is not None and seg.layer_idx == target_layer_idx:
        return "signal_aggr_same_layer"
    return "signal_aggr_cross_layer"


def iter_design_segments(
    def_path: str,
    layer_info: Dict,
    tech_lef: Optional[Dict] = None,
    cell_lib: Optional[Dict] = None,
    l_subdiv: float = L_SUBDIV_DEFAULT,
    skip_errors: bool = True,
) -> Iterator[Tuple[str, List[Segment]]]:
    """Stream (net_name, segments) for all nets in a design.

    Imports DefStreamParser lazily so this module is importable without the parser env.
    """
    from src.preprocessing.def_parser import DefStreamParser

    parser = DefStreamParser(def_path, layer_info, tech_lef=tech_lef or {"vias": {}}, cell_lib=cell_lib or {})
    for net_name, _cuboids, def_segs in parser.parse():
        if not net_name:
            continue
        segs = extract_segments_for_net(net_name, def_segs, layer_info, l_subdiv=l_subdiv)
        if segs:
            yield net_name, segs
