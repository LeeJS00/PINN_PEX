"""Per-aggressor pairwise feature extraction (ParaGraph-style).

For each (target, aggressor) pair within geometric cutoff, extract:
  - target_layer, aggressor_layer (low-card layers M1..M9p)
  - min_distance_um (closest cuboid surface-to-surface)
  - mean_distance_um, p25_distance_um, p75_distance_um
  - lateral_overlap_total_um2 (same-layer overlap area)
  - broadside_overlap_total_um2 (different-layer overlap area)
  - target_metal_area_um2 (subset of target's cuboids in this pair)
  - aggressor_metal_area_um2
  - n_target_cuboids, n_aggressor_cuboids
  - eps_target_mean, eps_aggressor_mean
  - design_name (one-hot or string)
  - target_total_metal_area, aggressor_total_metal_area  (whole-net features)

For training: pair label is c_pair_fF from SPEF coupling section.
For inference: predict c_pair, sum per target.
"""
from __future__ import annotations

import gzip
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Layer mapping copied from feat_extract_v2.py
LAYER_Z_RANGES = [
    (0.0,  0.62),
    (0.62, 0.78),
    (0.78, 0.92),
    (0.92, 1.07),
    (1.07, 1.22),
    (1.22, 1.40),
    (1.40, 2.20),
    (2.20, 5.00),
    (5.00, 999.0),
]


def z_to_layer(z_arr: np.ndarray) -> np.ndarray:
    out = np.ones_like(z_arr, dtype=np.int32)
    for i, (lo, hi) in enumerate(LAYER_Z_RANGES):
        out[(z_arr >= lo) & (z_arr < hi)] = i + 1
    return out


def _load_tile(path: Path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def extract_pairs_for_net(
    pkl_paths: List[Path],
    cutoff_um: float = 4.0,
    target_net_name: str = None,
) -> Tuple[Dict[str, dict], dict]:
    """Walk all tiles for a target net, build per-aggressor pair feature dicts.

    Returns:
        pair_features: {aggressor_net_name: {feature_name: value, ...}}
        target_summary: {feature_name: value, ...} (whole-net)
    """
    if not pkl_paths:
        return {}, {}

    # Aggregate per-pair stats
    pair_stats: Dict[str, dict] = {}
    target_metal_total = 0.0
    target_n_cuboids_total = 0
    target_eps_sum = 0.0
    target_eps_count = 0
    target_layer_counts = np.zeros(9, dtype=np.int32)

    for p in pkl_paths:
        try:
            rec = _load_tile(p)
        except Exception:
            continue
        c = rec["cuboids"]      # (N, 10): x,y,z,w,h,d,sem,logic,eps,nettype
        ag = rec["abs_geometries"]  # (N, 6)
        names = rec["cuboid_net_names"]

        target_mask = c[:, 7] == 1.0
        if not target_mask.any():
            continue

        # Target cuboids (this tile)
        tg = ag[target_mask]
        if target_net_name is None:
            # Infer from rec
            target_net_name = rec.get("net_name", "?")

        target_metal_total += float((ag[target_mask, 3] * ag[target_mask, 4]).sum())
        target_n_cuboids_total += int(target_mask.sum())
        target_eps_sum += float(c[target_mask, 8].sum())
        target_eps_count += int(target_mask.sum())
        tlayers = z_to_layer(ag[target_mask, 2])
        for li in range(1, 10):
            target_layer_counts[li - 1] += int((tlayers == li).sum())

        # Aggressor cuboids (same tile, signal only — exclude power)
        agg_mask = (c[:, 7] == 0.0) & (c[:, 9] < 0.6)
        if not agg_mask.any():
            continue

        aggr = ag[agg_mask]
        agg_indices = np.where(agg_mask)[0]
        agg_layers = z_to_layer(aggr[:, 2])
        agg_eps = c[agg_mask, 8]

        # Pair-wise distances: (T, A)
        txm = tg[:, 0:1]; tym = tg[:, 1:2]; tzm = tg[:, 2:3]
        tw = tg[:, 3:4]; th = tg[:, 4:5]; td = tg[:, 5:6]
        axm = aggr[:, 0]; aym = aggr[:, 1]; azm = aggr[:, 2]
        aw = aggr[:, 3]; ah = aggr[:, 4]; ad = aggr[:, 5]
        dx = np.maximum(0.0, np.abs(axm - txm) - (tw + aw) / 2.0)
        dy = np.maximum(0.0, np.abs(aym - tym) - (th + ah) / 2.0)
        dz = np.maximum(0.0, np.abs(azm - tzm) - (td + ad) / 2.0)
        d = np.sqrt(dx**2 + dy**2 + dz**2)
        # Overlap quantities
        same_layer = np.abs(azm - tzm) < 0.06
        z_overlap = np.maximum(0.0, np.minimum(tzm + td/2, azm + ad/2) - np.maximum(tzm - td/2, azm - ad/2))
        sx_o = np.maximum(0, np.minimum(txm + tw/2, axm + aw/2) - np.maximum(txm - tw/2, axm - aw/2))
        sy_o = np.maximum(0, np.minimum(tym + th/2, aym + ah/2) - np.maximum(tym - th/2, aym - ah/2))
        lat_overlap = z_overlap * np.minimum(sx_o, sy_o)
        bs_overlap = sx_o * sy_o

        # For each aggressor index, find target cuboids within cutoff
        for j, gj in enumerate(agg_indices):
            agg_name = names[gj]
            if agg_name in (target_net_name, "UNKNOWN_PIN"):
                continue
            d_col = d[:, j]
            keep = d_col <= cutoff_um
            if not keep.any():
                continue
            # Aggregate
            stat = pair_stats.setdefault(agg_name, dict(
                n_pairs=0, dists=[], lat=0.0, bs=0.0,
                agg_metal_area=0.0, agg_n_cuboids=0,
                agg_eps_sum=0.0, agg_eps_count=0,
                agg_layer_max=0,
                same_layer_pairs=0, diff_layer_pairs=0,
            ))
            kept = keep
            stat["n_pairs"] += int(kept.sum())
            stat["dists"].extend(d_col[kept].tolist())
            stat["lat"] += float(np.where(same_layer[:, j:j+1], lat_overlap[:, j:j+1], 0.0)[kept].sum())
            stat["bs"]  += float(np.where(~same_layer[:, j:j+1], bs_overlap[:, j:j+1], 0.0)[kept].sum())
            stat["same_layer_pairs"] += int(same_layer[kept, j].sum())
            stat["diff_layer_pairs"] += int((~same_layer[kept, j]).sum())
            # Aggressor properties (might be repeated across tiles for same agg — use max)
            stat["agg_metal_area"] = max(stat["agg_metal_area"],
                                         float(aw[j] * ah[j]))
            stat["agg_n_cuboids"] += 1
            stat["agg_eps_sum"] += float(agg_eps[j])
            stat["agg_eps_count"] += 1
            stat["agg_layer_max"] = max(stat["agg_layer_max"], int(agg_layers[j]))

    # Build feature dicts
    pair_features = {}
    target_eps_mean = target_eps_sum / max(target_eps_count, 1)
    target_dom_layer = int(np.argmax(target_layer_counts) + 1) if target_layer_counts.sum() > 0 else 1

    for agg_name, stat in pair_stats.items():
        if stat["n_pairs"] == 0:
            continue
        dists = np.array(stat["dists"])
        eps_a = stat["agg_eps_sum"] / max(stat["agg_eps_count"], 1)
        feats = dict(
            target_layer=target_dom_layer,
            agg_layer=stat["agg_layer_max"],
            n_pairs=stat["n_pairs"],
            min_dist=float(dists.min()),
            mean_dist=float(dists.mean()),
            p25_dist=float(np.percentile(dists, 25)),
            p75_dist=float(np.percentile(dists, 75)),
            lat_overlap_total=stat["lat"],
            bs_overlap_total=stat["bs"],
            agg_n_cuboids=stat["agg_n_cuboids"],
            agg_metal_area=stat["agg_metal_area"],
            same_layer_pairs=stat["same_layer_pairs"],
            diff_layer_pairs=stat["diff_layer_pairs"],
            target_n_cuboids=target_n_cuboids_total,
            target_metal_area=target_metal_total,
            target_eps_mean=target_eps_mean,
            agg_eps_mean=eps_a,
            sum_inv_d=float((1.0 / np.maximum(dists, 0.05)).sum()),
            sum_inv_d2=float((1.0 / np.maximum(dists, 0.05)**2).sum()),
        )
        pair_features[agg_name] = feats

    target_summary = dict(
        target_metal_area=target_metal_total,
        target_n_cuboids=target_n_cuboids_total,
        target_eps_mean=target_eps_mean,
        target_dom_layer=target_dom_layer,
    )
    return pair_features, target_summary
