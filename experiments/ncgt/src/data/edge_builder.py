"""
NCGT edge construction (Plan v3 §2.3).

Builds three edge bands per net:
- E_local: all pairs within R_edge_local (default 4 μm)
- E_mid: kNN per target subsegment, k=8, within (R_edge_local, R_edge_mid] (default (4, 12])
- E_long: per (target_seg, aggr_net) pair: top-1 by parallel-overlap, within (R_edge_mid, R_aggr]

Distance metric: 3D L2 between segment midpoints (cheap proxy; refined by
parallel-overlap heuristic for E_long).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from experiments.ncgt.src.data.segment_extractor import Segment


@dataclass
class EdgeSet:
    """Edges for one net forward pass.

    edge_index: shape (2, E) int — (target_idx, aggr_idx) into target/aggr arrays.
    band: (E,) int — 0=local, 1=mid, 2=long.
    """
    edge_index: np.ndarray
    band: np.ndarray
    midpoint_distance: np.ndarray  # (E,) μm — sanity / debugging

    def __len__(self) -> int:
        return int(self.edge_index.shape[1])


def _midpoint(s: Segment) -> Tuple[float, float, float]:
    return s.x_mid, s.y_mid, s.z


def _parallel_overlap_xy(a: Segment, b: Segment) -> float:
    """Returns overlap length when projected onto a's xy direction.

    For broadside (cross-layer) and non-parallel pairs returns 0; caller decides.
    """
    ax = a.p_end[0] - a.p_start[0]
    ay = a.p_end[1] - a.p_start[1]
    a_len = (ax * ax + ay * ay) ** 0.5
    if a_len < 1e-6:
        return 0.0
    ux, uy = ax / a_len, ay / a_len

    bsx = b.p_start[0] - a.p_start[0]
    bsy = b.p_start[1] - a.p_start[1]
    bex = b.p_end[0] - a.p_start[0]
    bey = b.p_end[1] - a.p_start[1]
    t_bs = bsx * ux + bsy * uy
    t_be = bex * ux + bey * uy
    t_lo = min(t_bs, t_be)
    t_hi = max(t_bs, t_be)
    overlap = min(a_len, t_hi) - max(0.0, t_lo)
    return max(0.0, overlap)


def build_edges_for_net(
    targets: Sequence[Segment],
    aggressors: Sequence[Segment],
    r_edge_local: float = 4.0,
    r_edge_mid: float = 12.0,
    r_aggr: float = 20.0,
    k_mid: int = 8,
    aggr_net_ids: Sequence[int] = (),
) -> EdgeSet:
    """Build edge set for one net forward pass.

    Args:
        targets: list of target net's segments (incl. virtual subsegments).
        aggressors: list of aggressor segments within R_aggr ball of any target.
        aggr_net_ids: parallel array of aggressor net ids (for E_long top-1 per net).
            If empty, treat all aggressors as one logical net for E_long.

    Returns:
        EdgeSet with edge_index, band, midpoint_distance.
    """
    if not targets or not aggressors:
        return EdgeSet(
            edge_index=np.zeros((2, 0), dtype=np.int64),
            band=np.zeros((0,), dtype=np.int8),
            midpoint_distance=np.zeros((0,), dtype=np.float32),
        )

    t_mid = np.array([_midpoint(s) for s in targets], dtype=np.float32)
    a_mid = np.array([_midpoint(s) for s in aggressors], dtype=np.float32)

    # Pairwise distance — O(T·A). For T~70, A~5K (per Phase 0 audit) this is 350K ops.
    diff = t_mid[:, None, :] - a_mid[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)  # (T, A)

    edge_t: List[int] = []
    edge_a: List[int] = []
    edge_band: List[int] = []
    edge_dist: List[float] = []

    # E_local: all pairs within r_edge_local.
    mask_local = dist < r_edge_local
    ti_loc, ai_loc = np.where(mask_local)
    for ti, ai in zip(ti_loc, ai_loc):
        edge_t.append(int(ti))
        edge_a.append(int(ai))
        edge_band.append(0)
        edge_dist.append(float(dist[ti, ai]))

    # E_mid: per target, kNN within (r_edge_local, r_edge_mid].
    for ti in range(len(targets)):
        d_row = dist[ti]
        in_band = (d_row >= r_edge_local) & (d_row < r_edge_mid)
        candidates = np.where(in_band)[0]
        if len(candidates) == 0:
            continue
        if len(candidates) > k_mid:
            order = np.argsort(d_row[candidates])[:k_mid]
            candidates = candidates[order]
        for ai in candidates:
            edge_t.append(int(ti))
            edge_a.append(int(ai))
            edge_band.append(1)
            edge_dist.append(float(d_row[ai]))

    # E_long: per (target, aggr_net), top-1 by parallel-overlap, within (r_edge_mid, r_aggr].
    if not aggr_net_ids:
        aggr_net_ids = [0] * len(aggressors)
    aggr_net_ids_arr = np.asarray(aggr_net_ids, dtype=np.int64)

    for ti in range(len(targets)):
        d_row = dist[ti]
        in_band = (d_row >= r_edge_mid) & (d_row < r_aggr)
        cand = np.where(in_band)[0]
        if len(cand) == 0:
            continue
        # Group by aggressor net id; pick top-1 by parallel-overlap.
        for net_id in np.unique(aggr_net_ids_arr[cand]):
            net_mask = aggr_net_ids_arr[cand] == net_id
            net_cand = cand[net_mask]
            if len(net_cand) == 0:
                continue
            # Score by parallel overlap; fallback to closest.
            scores = []
            for ai in net_cand:
                ov = _parallel_overlap_xy(targets[ti], aggressors[ai])
                scores.append(ov - 1e-6 * d_row[ai])  # break ties by closest
            best = net_cand[int(np.argmax(scores))]
            edge_t.append(int(ti))
            edge_a.append(int(best))
            edge_band.append(2)
            edge_dist.append(float(d_row[best]))

    if not edge_t:
        return EdgeSet(
            edge_index=np.zeros((2, 0), dtype=np.int64),
            band=np.zeros((0,), dtype=np.int8),
            midpoint_distance=np.zeros((0,), dtype=np.float32),
        )

    return EdgeSet(
        edge_index=np.asarray([edge_t, edge_a], dtype=np.int64),
        band=np.asarray(edge_band, dtype=np.int8),
        midpoint_distance=np.asarray(edge_dist, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Self-test.
# ---------------------------------------------------------------------------
def _smoke_test() -> None:
    """Trivial 2-target × 3-aggressor synthetic test."""
    def mk(seg_id, x, y, z, dx, dy, w, role_net="aggr"):
        return Segment(
            seg_id=seg_id, parent_seg_id=seg_id, is_subdivision=False,
            seg_type="WIRE", layer="m4", layer_idx=4,
            x_mid=x, y_mid=y, z=z, dx=dx, dy=dy, w=w, h=0.1,
            net_name=role_net, net_class="signal", semantic_type=0.0,
            p_start=(x - dx/2, y - dy/2, z),
            p_end=(x + dx/2, y + dy/2, z),
        )

    targets = [mk(0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.05, "tgt"),
               mk(1, 5.0, 0.0, 1.0, 2.0, 0.0, 0.05, "tgt")]
    aggressors = [
        mk(10, 0.5, 0.5, 1.0, 2.0, 0.0, 0.05, "a"),    # local for tgt0
        mk(11, 8.0, 0.0, 1.0, 2.0, 0.0, 0.05, "b"),    # mid for tgt1
        mk(12, 18.0, 0.0, 1.0, 2.0, 0.0, 0.05, "c"),   # long for both
    ]
    aggr_net_ids = [0, 1, 2]

    es = build_edges_for_net(targets, aggressors, aggr_net_ids=aggr_net_ids)
    print(f"edges: {len(es)}")
    print(f"  bands: {dict(zip(*np.unique(es.band, return_counts=True)))}")
    assert len(es) > 0
    print("[edge_builder smoke] OK")


if __name__ == "__main__":
    _smoke_test()
