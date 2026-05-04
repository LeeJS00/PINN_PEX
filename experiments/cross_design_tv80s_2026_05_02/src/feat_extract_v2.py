"""
v2 feature extractor — fixes layer mapping and adds richer features.

Improvements over v1 (`feat_extract.py`):
  - Fixed z-bucketing into M1..M9 using known PDK breakpoints (no more empty
    even-layer buckets).
  - Per-layer wire length and area (not just count).
  - Aggressor-weighted distance summaries: sum of 1/d and 1/d².
  - Layer-pair coupling counts (M1↔M2, M3↔M4, ...).
  - Pin/wire ratio.
  - Per-tile aggressor concentration metric.
  - Refined compact_gnd using accurate layer→z lookup.

Schema is backward-compatible: keeps the original FEATURE_NAMES from v1, then
appends the v2-only fields. Models can train on the union.
"""
from __future__ import annotations

import gzip
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np

EPS0_FF_UM = 8.854e-3

# Fixed z bucketing for intel22 stack — derived from observed unique z values
# in the cuboid pkls (0.583 = M1 top, 0.728 = M2 top, ..., 7.319 = M9 top).
# A cuboid at z falls into the layer whose z range covers it.
LAYER_Z_RANGES = [
    (0.0,  0.62),   # M1   wires/pins, z≈0.583
    (0.62, 0.78),   # M2 (with via1 ~0.65)
    (0.78, 0.92),   # M3
    (0.92, 1.07),   # M4
    (1.07, 1.22),   # M5
    (1.22, 1.40),   # M6
    (1.40, 2.20),   # M7
    (2.20, 5.00),   # M8
    (5.00, 999.0),  # M9
]
N_LAYERS = len(LAYER_Z_RANGES)


def z_to_layer(z_arr: np.ndarray) -> np.ndarray:
    """Map z absolute (μm) → integer layer 1..9."""
    out = np.ones_like(z_arr, dtype=np.int32)
    for i, (lo, hi) in enumerate(LAYER_Z_RANGES):
        out[(z_arr >= lo) & (z_arr < hi)] = i + 1
    return out


# ---------------------------------------------------------------------------
# v2 feature names
# ---------------------------------------------------------------------------


_PER_LAYER = [f"M{i+1}" if i < N_LAYERS - 1 else "M9p" for i in range(N_LAYERS)]


def _feature_names_v2() -> List[str]:
    base = [
        "tgt_n_cuboids", "tgt_n_pins", "tgt_n_wires", "tgt_pin_to_wire_ratio",
        "tgt_total_metal_area_um2", "tgt_total_volume_um3",
        "tgt_wire_length_um",
        "tgt_bbox_xy_um2", "tgt_bbox_z_um", "tgt_aspect_ratio",
        "tgt_z_min", "tgt_z_max", "tgt_z_mean", "tgt_z_std",
        "tgt_n_tiles",
        # eps
        "tgt_eps_min", "tgt_eps_max", "tgt_eps_mean", "tgt_eps_std",
        # net-type indicators of target
        "tgt_is_signal", "tgt_is_clock", "tgt_is_vdd", "tgt_is_vss",
    ]
    # per-layer counts, length, area (target)
    for L in _PER_LAYER:
        base.append(f"tgt_count_{L}")
    for L in _PER_LAYER:
        base.append(f"tgt_wirelen_{L}")
    for L in _PER_LAYER:
        base.append(f"tgt_area_{L}")
    # aggressor totals & per-layer
    base += [
        "agg_total_count", "agg_unique_nets",
        "agg_total_metal_area_um2", "agg_density_per_tile",
        "agg_top1_area", "agg_top3_area",
    ]
    for L in _PER_LAYER:
        base.append(f"agg_count_{L}")
    for L in _PER_LAYER:
        base.append(f"agg_area_{L}")
    # coupling stats
    base += [
        "cpl_n_pairs",
        "cpl_min_dist_um",
        "cpl_p10_dist_um", "cpl_p25_dist_um", "cpl_p50_dist_um",
        "cpl_p75_dist_um", "cpl_p95_dist_um",
        "cpl_mean_dist_um",
        "cpl_total_lateral_overlap_um2",
        "cpl_total_broadside_overlap_um2",
        "cpl_n_below_0p5um", "cpl_n_below_1um", "cpl_n_below_2um",
        # Distance-weighted coupling sums (proxy for capacitance)
        "cpl_sum_inv_d", "cpl_sum_inv_d2",
        "cpl_lat_weighted_inv_d", "cpl_bs_weighted_inv_d",
    ]
    # power
    base += [
        "pwr_n_cuboids", "pwr_total_metal_area_um2",
    ]
    for L in _PER_LAYER:
        base.append(f"pwr_count_{L}")
    for L in _PER_LAYER:
        base.append(f"pwr_area_{L}")
    # analytic intermediates
    base += [
        "compact_gnd_fF",
        "compact_cpl_fF",
        "compact_total_fF",  # gnd + cpl
    ]
    return base


FEATURE_NAMES_V2 = _feature_names_v2()


# ---------------------------------------------------------------------------
# Geometric primitives (similar to v1, vectorised)
# ---------------------------------------------------------------------------


def _hash_geom(arr: np.ndarray) -> np.ndarray:
    rounded = np.round(arr, 4).astype(np.float32)
    rb = rounded.tobytes()
    rec_len = arr.shape[1] * 4
    out = np.empty(len(arr), dtype=np.int64)
    for i in range(len(arr)):
        out[i] = hash(rb[i * rec_len:(i + 1) * rec_len])
    return out


def _per_layer_sums(layer: np.ndarray, vals: np.ndarray) -> np.ndarray:
    out = np.zeros(N_LAYERS, dtype=np.float64)
    for i in range(1, N_LAYERS + 1):
        m = layer == i
        if m.any():
            out[i - 1] = float(vals[m].sum())
    return out


def _per_layer_counts(layer: np.ndarray) -> np.ndarray:
    out = np.zeros(N_LAYERS, dtype=np.float64)
    for i in range(1, N_LAYERS + 1):
        out[i - 1] = float((layer == i).sum())
    return out


# ---------------------------------------------------------------------------
# Coupling computation per tile
# ---------------------------------------------------------------------------


def _tile_coupling_v2(cuboids: np.ndarray, ag: np.ndarray, cutoff_um: float) -> dict:
    target_mask = cuboids[:, 7] == 1.0
    is_pwr = cuboids[:, 9] >= 0.6
    agg_signal_mask = (cuboids[:, 7] == 0.0) & (~is_pwr)

    if not target_mask.any() or not agg_signal_mask.any():
        return {
            "n_pairs": 0, "dists": np.zeros(0, dtype=np.float32),
            "lat_total": 0.0, "bs_total": 0.0,
            "below_0p5": 0, "below_1": 0, "below_2": 0,
            "min_dist": cutoff_um,
            "lat_w_inv_d": 0.0, "bs_w_inv_d": 0.0,
            "sum_inv_d": 0.0, "sum_inv_d2": 0.0,
        }

    tg = ag[target_mask]      # (T, 6)
    aggr = ag[agg_signal_mask]  # (A, 6)

    txm = tg[:, 0:1]; tym = tg[:, 1:2]; tzm = tg[:, 2:3]
    tw = tg[:, 3:4]; th = tg[:, 4:5]; td = tg[:, 5:6]

    axm = aggr[:, 0]; aym = aggr[:, 1]; azm = aggr[:, 2]
    aw = aggr[:, 3]; ah = aggr[:, 4]; ad = aggr[:, 5]

    dx = np.maximum(0.0, np.abs(axm - txm) - (tw + aw) / 2.0)
    dy = np.maximum(0.0, np.abs(aym - tym) - (th + ah) / 2.0)
    dz = np.maximum(0.0, np.abs(azm - tzm) - (td + ad) / 2.0)
    dist = np.sqrt(dx * dx + dy * dy + dz * dz)

    keep = dist <= cutoff_um
    if not keep.any():
        return {
            "n_pairs": 0, "dists": np.zeros(0, dtype=np.float32),
            "lat_total": 0.0, "bs_total": 0.0,
            "below_0p5": 0, "below_1": 0, "below_2": 0,
            "min_dist": cutoff_um,
            "lat_w_inv_d": 0.0, "bs_w_inv_d": 0.0,
            "sum_inv_d": 0.0, "sum_inv_d2": 0.0,
        }

    dist_safe = np.maximum(dist, 0.05)   # μm — 50 nm floor for 1/d
    same_layer = np.abs(azm - tzm) < 0.06   # within ~ M layer thickness
    diff_layer = ~same_layer

    # Lateral overlap: same layer, z extents overlap
    z_overlap = np.minimum(tzm + td / 2, azm + ad / 2) - np.maximum(tzm - td / 2, azm - ad / 2)
    z_overlap = np.maximum(z_overlap, 0.0)
    side_x_overlap = np.maximum(0, np.minimum(txm + tw / 2, axm + aw / 2) - np.maximum(txm - tw / 2, axm - aw / 2))
    side_y_overlap = np.maximum(0, np.minimum(tym + th / 2, aym + ah / 2) - np.maximum(tym - th / 2, aym - ah / 2))
    lat_overlap = z_overlap * np.minimum(side_x_overlap, side_y_overlap)
    bs_overlap  = side_x_overlap * side_y_overlap

    inv_d = 1.0 / dist_safe
    inv_d2 = inv_d * inv_d

    sum_inv_d  = float(inv_d[keep].sum())
    sum_inv_d2 = float(inv_d2[keep].sum())
    lat_w = float((np.where(same_layer, lat_overlap, 0.0)[keep] * inv_d[keep]).sum())
    bs_w  = float((np.where(diff_layer,  bs_overlap, 0.0)[keep] * inv_d[keep]).sum())

    dist_kept = dist[keep]
    return {
        "n_pairs": int(keep.sum()),
        "dists": dist_kept.astype(np.float32).ravel(),
        "lat_total": float(np.where(same_layer, lat_overlap, 0.0)[keep].sum()),
        "bs_total":  float(np.where(diff_layer, bs_overlap, 0.0)[keep].sum()),
        "below_0p5": int((dist_kept < 0.5).sum()),
        "below_1":   int((dist_kept < 1.0).sum()),
        "below_2":   int((dist_kept < 2.0).sum()),
        "min_dist":  float(dist_kept.min()),
        "lat_w_inv_d": lat_w,
        "bs_w_inv_d":  bs_w,
        "sum_inv_d":  sum_inv_d,
        "sum_inv_d2": sum_inv_d2,
    }


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------


def extract_features_for_net_v2(pkl_paths: List[Path], cutoff_um: float = 4.0) -> Optional[dict]:
    records = []
    for p in pkl_paths:
        try:
            with gzip.open(p, "rb") as fh:
                records.append(pickle.load(fh))
        except Exception:
            continue
    if not records:
        return None

    target_net_str = records[0].get("net_name", "")
    n_tiles = len(records)

    # ---- Target dedupe ----
    geoms = []; eps_vals = []; is_pin = []; nettypes = []
    for rec in records:
        c = rec["cuboids"]
        ag = rec["abs_geometries"]
        m = c[:, 7] == 1.0
        geoms.append(ag[m])
        eps_vals.append(c[m, 8])
        is_pin.append(c[m, 6] == 0.5)
        nettypes.append(c[m, 9])
    if not geoms or all(g.shape[0] == 0 for g in geoms):
        return None
    g_all = np.concatenate(geoms, axis=0)
    e_all = np.concatenate(eps_vals, axis=0)
    p_all = np.concatenate(is_pin, axis=0)
    n_all = np.concatenate(nettypes, axis=0)
    h = _hash_geom(g_all)
    _, idx = np.unique(h, return_index=True)
    idx = np.sort(idx)
    g = g_all[idx]; e = e_all[idx]; p = p_all[idx]; n = n_all[idx]
    layer = z_to_layer(g[:, 2])

    # ---- Target geometry stats ----
    w = g[:, 3]; h_ = g[:, 4]; d = g[:, 5]; z = g[:, 2]
    metal_area = float((w * h_).sum())
    volume = float((w * h_ * d).sum())
    wire_len_per = np.maximum.reduce([w, h_, d])
    wire_len = float(wire_len_per.sum())
    x_min, x_max = float((g[:, 0] - w / 2).min()), float((g[:, 0] + w / 2).max())
    y_min, y_max = float((g[:, 1] - h_ / 2).min()), float((g[:, 1] + h_ / 2).max())
    z_min, z_max = float((z - d / 2).min()), float((z + d / 2).max())
    bbox_xy = (x_max - x_min) * (y_max - y_min)
    bbox_z = z_max - z_min
    aspect = (x_max - x_min) / max(y_max - y_min, 1e-6)

    n_pins = float(p.sum())
    n_wires = float((~p).sum())
    pw_ratio = n_pins / max(n_wires, 1.0)

    eps_pos = e[e > 0]
    eps_min  = float(eps_pos.min())  if eps_pos.size else 1.0
    eps_max  = float(eps_pos.max())  if eps_pos.size else 1.0
    eps_mean = float(eps_pos.mean()) if eps_pos.size else 1.0
    eps_std  = float(eps_pos.std())  if eps_pos.size else 0.0

    nt_mean = float(n.mean()) if n.size else 0.0
    is_signal = float(nt_mean < 0.2)
    is_clock  = float(0.2 <= nt_mean < 0.5)
    is_vdd    = float(0.5 <= nt_mean < 0.85)
    is_vss    = float(nt_mean >= 0.85)

    tgt_count_layer = _per_layer_counts(layer)
    tgt_area_layer  = _per_layer_sums(layer, w * h_)
    tgt_wlen_layer  = _per_layer_sums(layer, wire_len_per)

    # ---- Aggressor + power per-tile aggregation ----
    agg_total = 0
    agg_unique_nets: set = set()
    agg_metal = 0.0
    agg_count_layer = np.zeros(N_LAYERS, dtype=np.float64)
    agg_area_layer  = np.zeros(N_LAYERS, dtype=np.float64)
    pwr_total = 0
    pwr_metal = 0.0
    pwr_count_layer = np.zeros(N_LAYERS, dtype=np.float64)
    pwr_area_layer  = np.zeros(N_LAYERS, dtype=np.float64)

    cpl_dists_all: List[np.ndarray] = []
    cpl_n_pairs = 0
    cpl_lat = 0.0
    cpl_bs  = 0.0
    cpl_below_0p5 = 0
    cpl_below_1 = 0
    cpl_below_2 = 0
    cpl_min = float("inf")
    sum_inv_d = 0.0
    sum_inv_d2 = 0.0
    lat_w_inv_d = 0.0
    bs_w_inv_d  = 0.0

    # per-aggressor area accumulator (for top-k)
    aggressor_area_by_net: dict = {}

    for rec in records:
        c = rec["cuboids"]
        ag = rec["abs_geometries"]
        names = rec["cuboid_net_names"]
        agg_mask = (c[:, 7] == 0.0)
        is_pwr = c[:, 9] >= 0.6
        sig_agg = agg_mask & (~is_pwr)
        pwr_agg = agg_mask & is_pwr

        if sig_agg.any():
            sig_layers = z_to_layer(ag[sig_agg, 2])
            sig_areas = ag[sig_agg, 3] * ag[sig_agg, 4]
            agg_total += int(sig_agg.sum())
            agg_metal += float(sig_areas.sum())
            for i in range(1, N_LAYERS + 1):
                m = sig_layers == i
                agg_count_layer[i - 1] += float(m.sum())
                agg_area_layer[i - 1]  += float(sig_areas[m].sum())
            # per-aggressor accumulation
            indices = np.where(sig_agg)[0]
            for k_idx, i_idx in enumerate(indices):
                nm = names[i_idx]
                if nm in (target_net_str, "UNKNOWN_PIN"):
                    continue
                aggressor_area_by_net[nm] = aggressor_area_by_net.get(nm, 0.0) + float(sig_areas[k_idx])

        if pwr_agg.any():
            pwr_layers = z_to_layer(ag[pwr_agg, 2])
            pwr_areas = ag[pwr_agg, 3] * ag[pwr_agg, 4]
            pwr_total += int(pwr_agg.sum())
            pwr_metal += float(pwr_areas.sum())
            for i in range(1, N_LAYERS + 1):
                m = pwr_layers == i
                pwr_count_layer[i - 1] += float(m.sum())
                pwr_area_layer[i - 1]  += float(pwr_areas[m].sum())

        st = _tile_coupling_v2(c, ag, cutoff_um)
        cpl_n_pairs += st["n_pairs"]
        cpl_lat += st["lat_total"]
        cpl_bs  += st["bs_total"]
        cpl_below_0p5 += st["below_0p5"]
        cpl_below_1   += st["below_1"]
        cpl_below_2   += st["below_2"]
        cpl_min = min(cpl_min, st["min_dist"])
        cpl_dists_all.append(st["dists"])
        sum_inv_d += st["sum_inv_d"]
        sum_inv_d2 += st["sum_inv_d2"]
        lat_w_inv_d += st["lat_w_inv_d"]
        bs_w_inv_d  += st["bs_w_inv_d"]

    cpl_dists = np.concatenate(cpl_dists_all) if cpl_dists_all else np.zeros(0, dtype=np.float32)
    if cpl_dists.size:
        p10 = float(np.percentile(cpl_dists, 10))
        p25 = float(np.percentile(cpl_dists, 25))
        p50 = float(np.percentile(cpl_dists, 50))
        p75 = float(np.percentile(cpl_dists, 75))
        p95 = float(np.percentile(cpl_dists, 95))
        meand = float(cpl_dists.mean())
    else:
        p10 = p25 = p50 = p75 = p95 = meand = cutoff_um
    if cpl_min == float("inf"):
        cpl_min = cutoff_um

    # Top-1 / Top-3 aggressor area (proxy for dominant coupling neighbour)
    if aggressor_area_by_net:
        sorted_areas = sorted(aggressor_area_by_net.values(), reverse=True)
        top1 = float(sorted_areas[0])
        top3 = float(sum(sorted_areas[:3]))
    else:
        top1 = 0.0; top3 = 0.0
    agg_unique_nets = float(len(aggressor_area_by_net))

    # Compact gnd: parallel-plate to ground (z=0), per cuboid
    d_gnd = np.maximum(z - d / 2 - 0.0, 0.05)
    compact_gnd = float((EPS0_FF_UM * e * (w * h_) / d_gnd).sum())
    # Compact cpl: ε * sum(overlap_area / d) approximation
    compact_cpl = float(EPS0_FF_UM * eps_mean * (lat_w_inv_d + bs_w_inv_d))
    compact_total = compact_gnd + compact_cpl

    out = {
        "tgt_n_cuboids": float(g.shape[0]),
        "tgt_n_pins": n_pins,
        "tgt_n_wires": n_wires,
        "tgt_pin_to_wire_ratio": float(pw_ratio),
        "tgt_total_metal_area_um2": metal_area,
        "tgt_total_volume_um3": volume,
        "tgt_wire_length_um": wire_len,
        "tgt_bbox_xy_um2": float(bbox_xy),
        "tgt_bbox_z_um": float(bbox_z),
        "tgt_aspect_ratio": float(aspect),
        "tgt_z_min": z_min, "tgt_z_max": z_max,
        "tgt_z_mean": float(z.mean()), "tgt_z_std": float(z.std()),
        "tgt_n_tiles": float(n_tiles),
        "tgt_eps_min": eps_min, "tgt_eps_max": eps_max,
        "tgt_eps_mean": eps_mean, "tgt_eps_std": eps_std,
        "tgt_is_signal": is_signal, "tgt_is_clock": is_clock,
        "tgt_is_vdd": is_vdd, "tgt_is_vss": is_vss,
    }
    for i, L in enumerate(_PER_LAYER):
        out[f"tgt_count_{L}"] = float(tgt_count_layer[i])
    for i, L in enumerate(_PER_LAYER):
        out[f"tgt_wirelen_{L}"] = float(tgt_wlen_layer[i])
    for i, L in enumerate(_PER_LAYER):
        out[f"tgt_area_{L}"] = float(tgt_area_layer[i])

    out.update({
        "agg_total_count": float(agg_total),
        "agg_unique_nets": agg_unique_nets,
        "agg_total_metal_area_um2": agg_metal,
        "agg_density_per_tile": float(agg_total) / max(n_tiles, 1),
        "agg_top1_area": top1,
        "agg_top3_area": top3,
    })
    for i, L in enumerate(_PER_LAYER):
        out[f"agg_count_{L}"] = float(agg_count_layer[i])
    for i, L in enumerate(_PER_LAYER):
        out[f"agg_area_{L}"] = float(agg_area_layer[i])

    out.update({
        "cpl_n_pairs": float(cpl_n_pairs),
        "cpl_min_dist_um": float(cpl_min),
        "cpl_p10_dist_um": p10, "cpl_p25_dist_um": p25, "cpl_p50_dist_um": p50,
        "cpl_p75_dist_um": p75, "cpl_p95_dist_um": p95,
        "cpl_mean_dist_um": meand,
        "cpl_total_lateral_overlap_um2": float(cpl_lat),
        "cpl_total_broadside_overlap_um2": float(cpl_bs),
        "cpl_n_below_0p5um": float(cpl_below_0p5),
        "cpl_n_below_1um":   float(cpl_below_1),
        "cpl_n_below_2um":   float(cpl_below_2),
        "cpl_sum_inv_d":     float(sum_inv_d),
        "cpl_sum_inv_d2":    float(sum_inv_d2),
        "cpl_lat_weighted_inv_d": float(lat_w_inv_d),
        "cpl_bs_weighted_inv_d":  float(bs_w_inv_d),
    })
    out.update({
        "pwr_n_cuboids": float(pwr_total),
        "pwr_total_metal_area_um2": float(pwr_metal),
    })
    for i, L in enumerate(_PER_LAYER):
        out[f"pwr_count_{L}"] = float(pwr_count_layer[i])
    for i, L in enumerate(_PER_LAYER):
        out[f"pwr_area_{L}"] = float(pwr_area_layer[i])
    out.update({
        "compact_gnd_fF": compact_gnd,
        "compact_cpl_fF": compact_cpl,
        "compact_total_fF": compact_total,
    })
    return out
