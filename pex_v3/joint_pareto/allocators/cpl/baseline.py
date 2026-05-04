"""baseline.py — current Pareto-frontier c_cpl allocator (Path-2 v3).

Mirrors `pex_v3/src/utils/fast_spef_engine.py:compute_aggressor_weights`.
Future variants are diff'd against this.

Contract: given target net's segments + a global KD-tree + a per-net
c_cpl_total target (in fF), return dict mapping aggressor-net-name → cap.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Iterable
import numpy as np


def _layer_neighbours(layer: str) -> set[str]:
    if not layer or len(layer) < 2 or not layer.startswith("m"):
        return {layer}
    try:
        idx = int(layer[1:])
    except ValueError:
        return {layer}
    out = {f"m{idx}"}
    if idx - 1 > 0:
        out.add(f"m{idx-1}")
    out.add(f"m{idx+1}")
    return out


def allocate_cpl(
    target_segments,
    records,
    tree,
    target_net_name: str,
    c_cpl_total: float,
    layer_info: dict | None = None,
    max_dist_um: float = 5.0,
    top_k: int = 20,
    eps: float = 1e-6,
) -> dict:
    """Geometric overlap × 1 / dist² weighting, normalized to c_cpl_total.

    Same logic as fast_spef_engine.compute_aggressor_weights but returns
    absolute caps instead of relative weights.
    """
    if tree is None or not target_segments or c_cpl_total <= 0:
        return {}
    raw: dict[str, float] = defaultdict(float)
    for seg in target_segments:
        idxs = tree.query_ball_point((seg.x_mid, seg.y_mid), r=max_dist_um)
        for i in idxs:
            other = records[i]
            if other.net_name == target_net_name:
                continue
            if other.layer not in _layer_neighbours(seg.layer):
                continue
            dx = seg.x_mid - other.x_mid
            dy = seg.y_mid - other.y_mid
            d2 = dx * dx + dy * dy + eps
            weight = (seg.length * other.length) / d2
            raw[other.net_name] += weight
    if not raw:
        return {}
    if len(raw) > top_k:
        items = sorted(raw.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    else:
        items = list(raw.items())
    total_w = sum(w for _, w in items)
    if total_w <= 0:
        return {}
    return {n: c_cpl_total * w / total_w for n, w in items}
