"""
Per-net feature extraction from cuboid pkls (no DEF re-parse).

Cuboid pkl schema (from /data/PINNPEX/data/processed_v3/intel22/<design>/<file>.pkl.gz):
    cuboids:           (N, 10) float32   model-input tensor:
                          [0,1] x_rel, y_rel
                          [2]   z_abs
                          [3,4,5] w, h, d
                          [6]   semantic (1.0 wire, 0.5 pin)
                          [7]   logic   (1.0 target, 0.0 aggressor)
                          [8]   epsilon
                          [9]   net_type (0 signal, 0.33 clk, 0.67 vdd, 1.0 vss)
    abs_geometries:    (N, 6) float32   absolute (x, y, z, w, h, d)
    cuboid_net_names:  list[str]        net name per cuboid
    net_name:          target net name
    origin:            (3,) tile origin

For each (design, net):
    1. Load every pkl that targets the net.
    2. Pool target cuboids deduplicated by (x, y, z, w, h, d) hash.
    3. Pool aggressor cuboids (no dedupe — aggregate stats only).
    4. Compute per-net feature vector.

Why hand-engineered + analytic intermediates?
    Goal is MAPE < 4% via cross-design generalization. GBDTs need physically
    meaningful features. Sakurai-Tamaru-style compact estimates serve as
    physics priors that GBDTs can residual-correct.

EPS0 in fF/μm (vacuum permittivity scaled to fF·μm⁻¹).
"""
from __future__ import annotations

import gzip
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


EPS0_FF_UM = 8.854e-3   # ε₀ in fF/μm  (8.854e-18 F/m → 8.854e-3 fF/μm)
N_LAYERS = 9            # M1..M9 (extras folded into M9_plus)


# ---------------------------------------------------------------------------
# Layer index mapping from z_abs (absolute z in μm).
# We learn the layer breakpoints empirically on the first design (the values
# are in the layers.info file, but reading a stack of breakpoints from the
# data is consistent across designs and doesn't require parsing).
# ---------------------------------------------------------------------------


_LAYER_Z_BREAKS_UM: Optional[np.ndarray] = None


def _build_layer_z_breaks(z_abs: np.ndarray) -> np.ndarray:
    """Group z values, return midpoints between adjacent layer means."""
    z = np.unique(np.round(z_abs, decimals=4))
    if len(z) <= 1:
        return np.array([z[0] if len(z) else 0.0])
    return z


def init_layer_breaks_from_design(design_dir: Path, sample_n_files: int = 200):
    """Scan a few pkls from one design to discover unique z values for layer mapping."""
    global _LAYER_Z_BREAKS_UM
    if _LAYER_Z_BREAKS_UM is not None:
        return
    files = sorted(os.listdir(design_dir))[:sample_n_files]
    zs = []
    for f in files:
        try:
            with gzip.open(design_dir / f, "rb") as fh:
                p = pickle.load(fh)
            zs.append(p["abs_geometries"][:, 2])
        except Exception:
            continue
    if not zs:
        _LAYER_Z_BREAKS_UM = np.array([0.0])
        return
    z = np.unique(np.round(np.concatenate(zs), decimals=3))
    _LAYER_Z_BREAKS_UM = z


def _z_to_layer(z_arr: np.ndarray) -> np.ndarray:
    """Map z absolute (μm) to integer layer index 1..N (M1..MN+).
    Returns int array of same shape; 0 = below stack, len = above stack.
    """
    global _LAYER_Z_BREAKS_UM
    if _LAYER_Z_BREAKS_UM is None or len(_LAYER_Z_BREAKS_UM) <= 1:
        return np.ones_like(z_arr, dtype=np.int32)
    breaks = _LAYER_Z_BREAKS_UM
    idx = np.searchsorted(breaks, z_arr, side="left").astype(np.int32)
    # searchsorted returns position where z would be inserted; clamp to [0, N-1]
    return np.clip(idx, 0, len(breaks) - 1) + 1   # 1-based for "M1"


# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------


FEATURE_NAMES = [
    # geometry — target only
    "tgt_n_cuboids",
    "tgt_total_metal_area_um2",
    "tgt_total_volume_um3",
    "tgt_wire_length_um",
    "tgt_bbox_xy_um2",
    "tgt_bbox_z_um",
    "tgt_aspect_ratio",
    "tgt_z_min", "tgt_z_max",
    "tgt_n_tiles",
    # layer histogram (target)
    "tgt_layer_M1", "tgt_layer_M2", "tgt_layer_M3", "tgt_layer_M4",
    "tgt_layer_M5", "tgt_layer_M6", "tgt_layer_M7", "tgt_layer_M8", "tgt_layer_M9p",
    # epsilon
    "tgt_eps_min", "tgt_eps_max", "tgt_eps_mean", "tgt_eps_std",
    # aggressor density
    "agg_total_count", "agg_unique_nets",
    "agg_total_metal_area_um2",
    "agg_density_per_tile",
    # aggressor layer histogram
    "agg_layer_M1", "agg_layer_M2", "agg_layer_M3", "agg_layer_M4",
    "agg_layer_M5", "agg_layer_M6", "agg_layer_M7", "agg_layer_M8", "agg_layer_M9p",
    # geometric coupling stats — pairs of target ↔ aggressor cuboid in same tile
    "cpl_n_pairs",
    "cpl_min_dist_um",
    "cpl_p25_dist_um", "cpl_p50_dist_um", "cpl_p95_dist_um",
    "cpl_mean_dist_um",
    "cpl_total_lateral_overlap_um2",
    "cpl_total_broadside_overlap_um2",
    "cpl_n_below_1um", "cpl_n_below_2um", "cpl_n_below_3um",
    # vss / vdd power-net shielding (net_type 0.67 vdd, 1.0 vss)
    "pwr_n_cuboids",
    "pwr_total_metal_area_um2",
    "pwr_shield_M1_M3", "pwr_shield_M4_M5", "pwr_shield_M6p",
    # net-type indicators of the target itself
    "tgt_is_signal", "tgt_is_clock", "tgt_is_vdd", "tgt_is_vss",
    # analytic compact-model intermediates
    "compact_gnd_fF",
    "compact_cpl_fF",
]


@dataclass
class TargetCuboids:
    abs_geom: np.ndarray   # (N, 6)
    eps:      np.ndarray   # (N,)
    layer:    np.ndarray   # (N,)
    is_pin:   np.ndarray   # (N,)


def _hash_geom(arr: np.ndarray) -> np.ndarray:
    """Produce a 1-D uint64 hash for each row of (N,6) float32 absolute geometry."""
    rounded = np.round(arr, 4).astype(np.float32)
    rb = rounded.tobytes()
    rec_len = arr.shape[1] * 4
    out = np.empty(len(arr), dtype=np.int64)
    for i in range(len(arr)):
        out[i] = hash(rb[i * rec_len:(i + 1) * rec_len])
    return out


def _dedupe_target_cuboids(records: List[dict]) -> TargetCuboids:
    """Dedupe target cuboids across tiles by absolute geometry hash."""
    geoms = []
    eps_vals = []
    is_pin = []
    for rec in records:
        c = rec["cuboids"]
        ag = rec["abs_geometries"]
        # target rows: logic_flag (col 7) == 1.0
        mask = c[:, 7] == 1.0
        geoms.append(ag[mask])
        eps_vals.append(c[mask, 8])
        is_pin.append(c[mask, 6] == 0.5)
    if not geoms or all(g.shape[0] == 0 for g in geoms):
        return TargetCuboids(
            abs_geom=np.zeros((0, 6), dtype=np.float32),
            eps=np.zeros(0, dtype=np.float32),
            layer=np.zeros(0, dtype=np.int32),
            is_pin=np.zeros(0, dtype=bool),
        )
    g_all = np.concatenate(geoms, axis=0)
    e_all = np.concatenate(eps_vals, axis=0)
    p_all = np.concatenate(is_pin, axis=0)
    h = _hash_geom(g_all)
    _, idx = np.unique(h, return_index=True)
    idx = np.sort(idx)
    g = g_all[idx]
    e = e_all[idx]
    p = p_all[idx]
    layer = _z_to_layer(g[:, 2])
    return TargetCuboids(abs_geom=g, eps=e, layer=layer, is_pin=p)


# ---------------------------------------------------------------------------
# Coupling primitives — simple O(T·A) per tile (T,A typically ≤ a few hundred)
# ---------------------------------------------------------------------------


def _tile_coupling_stats(cuboids: np.ndarray, ag: np.ndarray, cutoff_um: float = 4.0) -> dict:
    """Compute geometric coupling between target & aggressor cuboids in one tile.

    Args:
        cuboids: (M, 10) tile tensor.
        ag:      (M, 6)  absolute geometry per row.
    """
    target_mask = cuboids[:, 7] == 1.0
    agg_mask = (cuboids[:, 7] == 0.0)
    # Drop power-net rows from coupling — they're handled separately
    is_pwr = (cuboids[:, 9] >= 0.6)
    agg_signal_mask = agg_mask & (~is_pwr)

    if not target_mask.any() or not agg_signal_mask.any():
        return {
            "n_pairs": 0,
            "min_dist": cutoff_um,
            "dists": np.zeros(0, dtype=np.float32),
            "lateral_overlap_total": 0.0,
            "broadside_overlap_total": 0.0,
            "n_below_1": 0,
            "n_below_2": 0,
            "n_below_3": 0,
        }

    tg = ag[target_mask]
    aggr = ag[agg_signal_mask]
    # Compute per-pair surface distance in xy, layer match for broadside, lateral overlap
    # Vectorize over (T,A) — pairs only computed if surface_dist <= cutoff
    txm = tg[:, 0:1]; tym = tg[:, 1:2]; tzm = tg[:, 2:3]
    tw = tg[:, 3:4]; th = tg[:, 4:5]; td = tg[:, 5:6]

    axm = aggr[:, 0]; aym = aggr[:, 1]; azm = aggr[:, 2]
    aw = aggr[:, 3]; ah = aggr[:, 4]; ad = aggr[:, 5]

    # Surface distance in xy (signed) → max(0, |Δx| - (w_t+w_a)/2)
    dx = np.maximum(0.0, np.abs(axm - txm) - (tw + aw) / 2.0)
    dy = np.maximum(0.0, np.abs(aym - tym) - (th + ah) / 2.0)
    dz = np.maximum(0.0, np.abs(azm - tzm) - (td + ad) / 2.0)
    dist = np.sqrt(dx * dx + dy * dy + dz * dz)

    keep = dist <= cutoff_um
    if not keep.any():
        return {
            "n_pairs": 0,
            "min_dist": cutoff_um,
            "dists": np.zeros(0, dtype=np.float32),
            "lateral_overlap_total": 0.0,
            "broadside_overlap_total": 0.0,
            "n_below_1": 0,
            "n_below_2": 0,
            "n_below_3": 0,
        }

    dist_kept = dist[keep]
    # Lateral overlap: same layer (|Δz| ~ 0), z extents overlap → side-by-side
    same_layer = np.abs(azm - tzm) < 0.01
    z_overlap = np.minimum(tzm + td / 2, azm + ad / 2) - np.maximum(tzm - td / 2, azm - ad / 2)
    z_overlap = np.maximum(z_overlap, 0.0)
    # Broadside overlap: above/below (|Δz| > extent) and xy bbox intersects
    diff_layer = ~same_layer
    xy_overlap = np.maximum(0, np.minimum(txm + tw / 2, axm + aw / 2) - np.maximum(txm - tw / 2, axm - aw / 2)) * \
                 np.maximum(0, np.minimum(tym + th / 2, aym + ah / 2) - np.maximum(tym - th / 2, aym - ah / 2))

    # lateral overlap area = z_overlap × min(side overlaps)
    lat_overlap_area = z_overlap * np.minimum(
        np.maximum(0, np.minimum(txm + tw / 2, axm + aw / 2) - np.maximum(txm - tw / 2, axm - aw / 2)),
        np.maximum(0, np.minimum(tym + th / 2, aym + ah / 2) - np.maximum(tym - th / 2, aym - ah / 2)),
    )

    lat_overlap_total = float(np.where(same_layer, lat_overlap_area, 0.0)[keep].sum())
    bs_overlap_total = float(np.where(diff_layer, xy_overlap, 0.0)[keep].sum())

    return {
        "n_pairs": int(keep.sum()),
        "min_dist": float(dist_kept.min()) if dist_kept.size else cutoff_um,
        "dists": dist_kept.astype(np.float32).ravel(),
        "lateral_overlap_total": lat_overlap_total,
        "broadside_overlap_total": bs_overlap_total,
        "n_below_1": int((dist_kept < 1.0).sum()),
        "n_below_2": int((dist_kept < 2.0).sum()),
        "n_below_3": int((dist_kept < 3.0).sum()),
    }


# ---------------------------------------------------------------------------
# Main per-net feature extractor
# ---------------------------------------------------------------------------


def extract_features_for_net(
    pkl_paths: List[Path],
    cutoff_um: float = 4.0,
) -> Optional[dict]:
    """Read all tiles for one (design, net) and emit a feature dict."""
    records = []
    for p in pkl_paths:
        try:
            with gzip.open(p, "rb") as fh:
                records.append(pickle.load(fh))
        except Exception:
            continue
    if not records:
        return None

    # Target cuboids (deduped)
    tgt = _dedupe_target_cuboids(records)
    n_tiles = len(records)

    # Empty-target safeguard — net might have zero cuboids if it's a one-pin net etc.
    if tgt.abs_geom.shape[0] == 0:
        return None

    # Geometry stats
    g = tgt.abs_geom
    w = g[:, 3]; h = g[:, 4]; d = g[:, 5]
    metal_area = float((w * h).sum())
    volume = float((w * h * d).sum())
    wire_len = float(np.maximum.reduce([w, h, d]).sum())
    x_min, x_max = float((g[:, 0] - w / 2).min()), float((g[:, 0] + w / 2).max())
    y_min, y_max = float((g[:, 1] - h / 2).min()), float((g[:, 1] + h / 2).max())
    z_min, z_max = float((g[:, 2] - d / 2).min()), float((g[:, 2] + d / 2).max())
    bbox_xy = (x_max - x_min) * (y_max - y_min)
    bbox_z = z_max - z_min
    aspect = (x_max - x_min) / max(y_max - y_min, 1e-6)

    layer_hist = np.zeros(N_LAYERS, dtype=np.float64)
    for li in range(1, N_LAYERS):
        layer_hist[li - 1] = float((tgt.layer == li).sum())
    layer_hist[N_LAYERS - 1] = float((tgt.layer >= N_LAYERS).sum())

    eps_vals = tgt.eps[tgt.eps > 0]
    eps_min  = float(eps_vals.min())  if eps_vals.size else 1.0
    eps_max  = float(eps_vals.max())  if eps_vals.size else 1.0
    eps_mean = float(eps_vals.mean()) if eps_vals.size else 1.0
    eps_std  = float(eps_vals.std())  if eps_vals.size else 0.0

    # Aggressor & power stats — across all tiles, no dedupe
    agg_total = 0
    agg_unique_nets: set = set()
    agg_metal_area = 0.0
    agg_layer_hist = np.zeros(N_LAYERS, dtype=np.float64)
    pwr_total_count = 0
    pwr_metal_area = 0.0
    pwr_shield = np.zeros(3, dtype=np.float64)
    cpl_dists_all: List[np.ndarray] = []
    cpl_n_pairs = 0
    cpl_lat = 0.0
    cpl_bs = 0.0
    cpl_n_below_1 = 0
    cpl_n_below_2 = 0
    cpl_n_below_3 = 0
    cpl_min_dist = float("inf")

    target_net_str = records[0].get("net_name", "")

    for rec in records:
        c = rec["cuboids"]
        ag = rec["abs_geometries"]
        names = rec["cuboid_net_names"]
        target_mask = c[:, 7] == 1.0
        agg_mask = (c[:, 7] == 0.0)

        # power: VDD/VSS aggressors (col 9 ≥ 0.6 captures vdd 0.67, vss 1.0)
        pwr_mask = agg_mask & (c[:, 9] >= 0.6)
        sig_agg_mask = agg_mask & (c[:, 9] < 0.6)

        agg_total += int(sig_agg_mask.sum())
        agg_metal_area += float((ag[sig_agg_mask, 3] * ag[sig_agg_mask, 4]).sum())

        if sig_agg_mask.any():
            sig_layers = _z_to_layer(ag[sig_agg_mask, 2])
            for li in range(1, N_LAYERS):
                agg_layer_hist[li - 1] += float((sig_layers == li).sum())
            agg_layer_hist[N_LAYERS - 1] += float((sig_layers >= N_LAYERS).sum())

        # unique aggressor net names
        for i in range(len(names)):
            if sig_agg_mask[i] and names[i] not in (target_net_str, "UNKNOWN_PIN"):
                agg_unique_nets.add(names[i])

        pwr_total_count += int(pwr_mask.sum())
        pwr_metal_area += float((ag[pwr_mask, 3] * ag[pwr_mask, 4]).sum())
        if pwr_mask.any():
            pwr_layers = _z_to_layer(ag[pwr_mask, 2])
            pwr_areas = ag[pwr_mask, 3] * ag[pwr_mask, 4]
            pwr_shield[0] += float(pwr_areas[(pwr_layers >= 1) & (pwr_layers <= 3)].sum())
            pwr_shield[1] += float(pwr_areas[(pwr_layers >= 4) & (pwr_layers <= 5)].sum())
            pwr_shield[2] += float(pwr_areas[pwr_layers >= 6].sum())

        # coupling stats
        st = _tile_coupling_stats(c, ag, cutoff_um=cutoff_um)
        cpl_n_pairs += st["n_pairs"]
        cpl_min_dist = min(cpl_min_dist, st["min_dist"])
        cpl_dists_all.append(st["dists"])
        cpl_lat += st["lateral_overlap_total"]
        cpl_bs += st["broadside_overlap_total"]
        cpl_n_below_1 += st["n_below_1"]
        cpl_n_below_2 += st["n_below_2"]
        cpl_n_below_3 += st["n_below_3"]

    cpl_dists = np.concatenate(cpl_dists_all) if cpl_dists_all else np.zeros(0, dtype=np.float32)
    if cpl_dists.size:
        p25 = float(np.percentile(cpl_dists, 25))
        p50 = float(np.percentile(cpl_dists, 50))
        p95 = float(np.percentile(cpl_dists, 95))
        mean_d = float(cpl_dists.mean())
    else:
        p25 = p50 = p95 = mean_d = cutoff_um
    if cpl_min_dist == float("inf"):
        cpl_min_dist = cutoff_um

    # Net-type indicator from logic ch9 of any target cuboid
    target_nettype_vals = []
    for rec in records:
        c = rec["cuboids"]
        m = c[:, 7] == 1.0
        if m.any():
            target_nettype_vals.append(c[m, 9])
    if target_nettype_vals:
        nt = float(np.concatenate(target_nettype_vals).mean())
    else:
        nt = 0.0
    is_signal = float(nt < 0.2)
    is_clock  = float(0.2 <= nt < 0.5)
    is_vdd    = float(0.5 <= nt < 0.85)
    is_vss    = float(nt >= 0.85)

    # Analytic intermediates ----------------------------------------------
    # compact_gnd: parallel-plate to ground plane (M0 / substrate at z≈0)
    layer = tgt.layer.astype(np.float32)
    z_centers = g[:, 2]
    d_to_gnd = np.maximum(z_centers - d / 2 - 0.0, 0.05)   # μm
    eps_arr = tgt.eps
    A = w * h
    compact_gnd = float((EPS0_FF_UM * eps_arr * A / d_to_gnd).sum())

    # compact_cpl: sum over all coupling pairs of ε·A/d
    if cpl_dists.size and cpl_lat + cpl_bs > 0:
        d_eff = np.clip(cpl_dists, 0.05, None)
        # Crude: total overlap area divided uniformly across pairs
        avg_area = (cpl_lat + cpl_bs) / max(cpl_n_pairs, 1)
        eps_avg = float(eps_mean)
        compact_cpl = float(EPS0_FF_UM * eps_avg * avg_area * (1.0 / d_eff).sum())
    else:
        compact_cpl = 0.0

    return {
        "tgt_n_cuboids": float(g.shape[0]),
        "tgt_total_metal_area_um2": metal_area,
        "tgt_total_volume_um3": volume,
        "tgt_wire_length_um": wire_len,
        "tgt_bbox_xy_um2": float(bbox_xy),
        "tgt_bbox_z_um": float(bbox_z),
        "tgt_aspect_ratio": float(aspect),
        "tgt_z_min": z_min,
        "tgt_z_max": z_max,
        "tgt_n_tiles": float(n_tiles),
        **{f"tgt_layer_M{i+1}" if i < N_LAYERS - 1 else "tgt_layer_M9p": float(layer_hist[i])
           for i in range(N_LAYERS)},
        "tgt_eps_min": eps_min,
        "tgt_eps_max": eps_max,
        "tgt_eps_mean": eps_mean,
        "tgt_eps_std": eps_std,
        "agg_total_count": float(agg_total),
        "agg_unique_nets": float(len(agg_unique_nets)),
        "agg_total_metal_area_um2": agg_metal_area,
        "agg_density_per_tile": float(agg_total) / max(n_tiles, 1),
        **{f"agg_layer_M{i+1}" if i < N_LAYERS - 1 else "agg_layer_M9p": float(agg_layer_hist[i])
           for i in range(N_LAYERS)},
        "cpl_n_pairs": float(cpl_n_pairs),
        "cpl_min_dist_um": float(cpl_min_dist),
        "cpl_p25_dist_um": p25,
        "cpl_p50_dist_um": p50,
        "cpl_p95_dist_um": p95,
        "cpl_mean_dist_um": mean_d,
        "cpl_total_lateral_overlap_um2": float(cpl_lat),
        "cpl_total_broadside_overlap_um2": float(cpl_bs),
        "cpl_n_below_1um": float(cpl_n_below_1),
        "cpl_n_below_2um": float(cpl_n_below_2),
        "cpl_n_below_3um": float(cpl_n_below_3),
        "pwr_n_cuboids": float(pwr_total_count),
        "pwr_total_metal_area_um2": float(pwr_metal_area),
        "pwr_shield_M1_M3": float(pwr_shield[0]),
        "pwr_shield_M4_M5": float(pwr_shield[1]),
        "pwr_shield_M6p": float(pwr_shield[2]),
        "tgt_is_signal": is_signal,
        "tgt_is_clock":  is_clock,
        "tgt_is_vdd":    is_vdd,
        "tgt_is_vss":    is_vss,
        "compact_gnd_fF": compact_gnd,
        "compact_cpl_fF": compact_cpl,
    }
