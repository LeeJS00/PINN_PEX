"""
v3 feature extractor — adds multi-radius spatial density to v2.

New features (on top of v2):
  - agg_count_within_{0p3,0p5,1,2,3}_um   total signal aggressor count at multiple radii
  - agg_area_within_{0p3,0p5,1,2,3}_um    total signal aggressor area at multiple radii
  - tgt_n_neighbors_within_{0p5,1,2}_um   targets within distance r of each other
  - target_x_extent, target_y_extent       bbox dimensions separately

These are computed by re-using the per-tile coupling pass — adding extra
distance bins is essentially free because the distance matrix is already
computed.
"""
from __future__ import annotations

import gzip
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.feat_extract_v2 import (
    z_to_layer, _hash_geom, _per_layer_counts, _per_layer_sums,
    LAYER_Z_RANGES, N_LAYERS, EPS0_FF_UM, FEATURE_NAMES_V2,
)

_PER_LAYER = [f"M{i+1}" if i < N_LAYERS - 1 else "M9p" for i in range(N_LAYERS)]

# Multi-radius density bins
_RADII_UM = [0.3, 0.5, 1.0, 2.0, 3.0]
_RADII_TAGS = ["0p3", "0p5", "1", "2", "3"]


def _feature_names_v3() -> List[str]:
    base = list(FEATURE_NAMES_V2)
    # Multi-radius aggressor density
    for tag in _RADII_TAGS:
        base.append(f"v3_agg_count_within_{tag}um")
    for tag in _RADII_TAGS:
        base.append(f"v3_agg_area_within_{tag}um")
    # Power within radii (capacitive shielding contribution)
    for tag in _RADII_TAGS:
        base.append(f"v3_pwr_count_within_{tag}um")
    for tag in _RADII_TAGS:
        base.append(f"v3_pwr_area_within_{tag}um")
    # Bbox separately + length-density
    base += [
        "v3_bbox_x_um", "v3_bbox_y_um",
        "v3_length_density",        # wire_length / bbox_area
        "v3_cuboid_density",        # n_cuboids / bbox_area
        # Per-layer aggressor density at 1um
        "v3_agg_M1_within_1um", "v3_agg_M2_within_1um", "v3_agg_M3_within_1um",
        "v3_agg_M4_within_1um", "v3_agg_M5_within_1um",
        # Coupling cap proxy refined: use lateral overlap × ε / d
        "v3_cap_proxy_lateral",
        "v3_cap_proxy_broadside",
    ]
    return base


FEATURE_NAMES_V3 = _feature_names_v3()


def _tile_coupling_v3(cuboids: np.ndarray, ag: np.ndarray, cutoff_um: float) -> dict:
    """Same as v2 but with multi-radius density bins."""
    target_mask = cuboids[:, 7] == 1.0
    is_pwr = cuboids[:, 9] >= 0.6
    agg_signal_mask = (cuboids[:, 7] == 0.0) & (~is_pwr)
    pwr_mask = (cuboids[:, 7] == 0.0) & is_pwr

    if not target_mask.any():
        empty = np.zeros(0, dtype=np.float32)
        zeros5 = [0]*5
        return dict(
            n_pairs=0, dists=empty, lat_total=0.0, bs_total=0.0,
            below_0p5=0, below_1=0, below_2=0,
            min_dist=cutoff_um, lat_w_inv_d=0.0, bs_w_inv_d=0.0,
            sum_inv_d=0.0, sum_inv_d2=0.0,
            agg_count=zeros5.copy(), agg_area=zeros5.copy(),
            pwr_count=zeros5.copy(), pwr_area=zeros5.copy(),
            agg_layer_within_1um=[0]*5,
            cap_proxy_lat=0.0, cap_proxy_bs=0.0,
        )

    tg = ag[target_mask]
    tw = tg[:, 3:4]; th = tg[:, 4:5]; td = tg[:, 5:6]
    txm = tg[:, 0:1]; tym = tg[:, 1:2]; tzm = tg[:, 2:3]

    def _dist_matrix(other):
        if other.shape[0] == 0:
            return np.zeros((tg.shape[0], 0), dtype=np.float32)
        oxm = other[:, 0]; oym = other[:, 1]; ozm = other[:, 2]
        ow = other[:, 3]; oh = other[:, 4]; od = other[:, 5]
        dx = np.maximum(0.0, np.abs(oxm - txm) - (tw + ow) / 2.0)
        dy = np.maximum(0.0, np.abs(oym - tym) - (th + oh) / 2.0)
        dz = np.maximum(0.0, np.abs(ozm - tzm) - (td + od) / 2.0)
        return np.sqrt(dx * dx + dy * dy + dz * dz)

    aggr = ag[agg_signal_mask]
    pwr = ag[pwr_mask]

    d_agg = _dist_matrix(aggr)
    d_pwr = _dist_matrix(pwr)

    # Multi-radius counts/areas (de-duplicated by aggressor cuboid: row-wise OR mask)
    agg_count_radii = []
    agg_area_radii = []
    pwr_count_radii = []
    pwr_area_radii = []
    if aggr.shape[0] > 0:
        a_areas = aggr[:, 3] * aggr[:, 4]
        for r in _RADII_UM:
            within = (d_agg <= r).any(axis=0)   # aggressors within r of any target
            agg_count_radii.append(int(within.sum()))
            agg_area_radii.append(float(a_areas[within].sum()))
    else:
        agg_count_radii = [0]*5; agg_area_radii = [0.0]*5
    if pwr.shape[0] > 0:
        p_areas = pwr[:, 3] * pwr[:, 4]
        for r in _RADII_UM:
            within = (d_pwr <= r).any(axis=0)
            pwr_count_radii.append(int(within.sum()))
            pwr_area_radii.append(float(p_areas[within].sum()))
    else:
        pwr_count_radii = [0]*5; pwr_area_radii = [0.0]*5

    # Per-layer aggressors within 1 μm (M1..M5)
    agg_layer_within_1um = [0]*5
    if aggr.shape[0] > 0:
        within_1 = (d_agg <= 1.0).any(axis=0)
        layers = z_to_layer(aggr[within_1, 2])
        for li in range(1, 6):
            agg_layer_within_1um[li - 1] = int((layers == li).sum())

    # Coupling pairs
    if d_agg.size:
        keep = d_agg <= cutoff_um
        if keep.any():
            dist_safe = np.maximum(d_agg, 0.05)
            inv_d = 1.0 / dist_safe; inv_d2 = inv_d * inv_d
            same_layer = np.abs(aggr[:, 2:3].T - tzm) < 0.06
            diff_layer = ~same_layer

            z_overlap = np.minimum(tzm + td/2, aggr[:, 2:3].T + aggr[:, 5:6].T/2) - \
                        np.maximum(tzm - td/2, aggr[:, 2:3].T - aggr[:, 5:6].T/2)
            z_overlap = np.maximum(z_overlap, 0.0)
            sx_ov = np.maximum(0, np.minimum(txm + tw/2, aggr[:, 0:1].T + aggr[:, 3:4].T/2) -
                                  np.maximum(txm - tw/2, aggr[:, 0:1].T - aggr[:, 3:4].T/2))
            sy_ov = np.maximum(0, np.minimum(tym + th/2, aggr[:, 1:2].T + aggr[:, 4:5].T/2) -
                                  np.maximum(tym - th/2, aggr[:, 1:2].T - aggr[:, 4:5].T/2))
            lat_overlap = z_overlap * np.minimum(sx_ov, sy_ov)
            bs_overlap = sx_ov * sy_ov

            sum_inv_d = float(inv_d[keep].sum())
            sum_inv_d2 = float(inv_d2[keep].sum())
            lat_w_inv_d = float((np.where(same_layer, lat_overlap, 0.0)[keep] * inv_d[keep]).sum())
            bs_w_inv_d = float((np.where(diff_layer, bs_overlap, 0.0)[keep] * inv_d[keep]).sum())
            cap_proxy_lat = float((np.where(same_layer, lat_overlap, 0.0)[keep] * inv_d[keep]).sum())
            cap_proxy_bs = float((np.where(diff_layer, bs_overlap, 0.0)[keep] * inv_d[keep]).sum())
            dist_kept = d_agg[keep]
            n_pairs = int(keep.sum())
            min_d = float(dist_kept.min())
            below_0p5 = int((dist_kept < 0.5).sum())
            below_1 = int((dist_kept < 1.0).sum())
            below_2 = int((dist_kept < 2.0).sum())
            lat_total = float(np.where(same_layer, lat_overlap, 0.0)[keep].sum())
            bs_total = float(np.where(diff_layer, bs_overlap, 0.0)[keep].sum())
        else:
            sum_inv_d = sum_inv_d2 = lat_w_inv_d = bs_w_inv_d = 0.0
            cap_proxy_lat = cap_proxy_bs = 0.0
            n_pairs = 0; min_d = cutoff_um
            below_0p5 = below_1 = below_2 = 0
            lat_total = bs_total = 0.0
            dist_kept = np.zeros(0, dtype=np.float32)
    else:
        sum_inv_d = sum_inv_d2 = lat_w_inv_d = bs_w_inv_d = 0.0
        cap_proxy_lat = cap_proxy_bs = 0.0
        n_pairs = 0; min_d = cutoff_um
        below_0p5 = below_1 = below_2 = 0
        lat_total = bs_total = 0.0
        dist_kept = np.zeros(0, dtype=np.float32)

    return dict(
        n_pairs=n_pairs, dists=dist_kept.astype(np.float32).ravel(),
        lat_total=lat_total, bs_total=bs_total,
        below_0p5=below_0p5, below_1=below_1, below_2=below_2,
        min_dist=min_d, lat_w_inv_d=lat_w_inv_d, bs_w_inv_d=bs_w_inv_d,
        sum_inv_d=sum_inv_d, sum_inv_d2=sum_inv_d2,
        agg_count=agg_count_radii, agg_area=agg_area_radii,
        pwr_count=pwr_count_radii, pwr_area=pwr_area_radii,
        agg_layer_within_1um=agg_layer_within_1um,
        cap_proxy_lat=cap_proxy_lat, cap_proxy_bs=cap_proxy_bs,
    )


def extract_features_for_net_v3(pkl_paths: List[Path], cutoff_um: float = 4.0) -> Optional[dict]:
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

    geoms = []; eps_vals = []; is_pin = []; nettypes = []
    for rec in records:
        c = rec["cuboids"]; ag = rec["abs_geometries"]
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

    w = g[:, 3]; h_ = g[:, 4]; d = g[:, 5]; z = g[:, 2]
    metal_area = float((w * h_).sum())
    volume = float((w * h_ * d).sum())
    wire_len_per = np.maximum.reduce([w, h_, d])
    wire_len = float(wire_len_per.sum())
    x_min, x_max = float((g[:, 0] - w/2).min()), float((g[:, 0] + w/2).max())
    y_min, y_max = float((g[:, 1] - h_/2).min()), float((g[:, 1] + h_/2).max())
    z_min, z_max = float((z - d/2).min()), float((z + d/2).max())
    bbox_x = x_max - x_min
    bbox_y = y_max - y_min
    bbox_xy = bbox_x * bbox_y
    bbox_z = z_max - z_min
    aspect = bbox_x / max(bbox_y, 1e-6)

    n_pins = float(p.sum()); n_wires = float((~p).sum())
    pw_ratio = n_pins / max(n_wires, 1.0)

    eps_pos = e[e > 0]
    eps_min = float(eps_pos.min()) if eps_pos.size else 1.0
    eps_max = float(eps_pos.max()) if eps_pos.size else 1.0
    eps_mean = float(eps_pos.mean()) if eps_pos.size else 1.0
    eps_std = float(eps_pos.std()) if eps_pos.size else 0.0

    nt_mean = float(n.mean()) if n.size else 0.0
    is_signal = float(nt_mean < 0.2)
    is_clock  = float(0.2 <= nt_mean < 0.5)
    is_vdd    = float(0.5 <= nt_mean < 0.85)
    is_vss    = float(nt_mean >= 0.85)

    tgt_count_layer = _per_layer_counts(layer)
    tgt_area_layer  = _per_layer_sums(layer, w * h_)
    tgt_wlen_layer  = _per_layer_sums(layer, wire_len_per)

    # Aggregate per-tile coupling stats across tiles
    agg_total_count = 0
    agg_unique_nets: set = set()
    agg_metal = 0.0
    agg_count_layer = np.zeros(N_LAYERS, dtype=np.float64)
    agg_area_layer  = np.zeros(N_LAYERS, dtype=np.float64)
    pwr_total_count = 0
    pwr_metal = 0.0
    pwr_count_layer = np.zeros(N_LAYERS, dtype=np.float64)
    pwr_area_layer  = np.zeros(N_LAYERS, dtype=np.float64)

    cpl_dists_all: List[np.ndarray] = []
    cpl_n_pairs = 0
    cpl_lat = 0.0; cpl_bs = 0.0
    cpl_below_0p5 = 0; cpl_below_1 = 0; cpl_below_2 = 0
    cpl_min = float("inf")
    sum_inv_d = sum_inv_d2 = 0.0
    lat_w_inv_d = bs_w_inv_d = 0.0
    cap_proxy_lat = cap_proxy_bs = 0.0

    # multi-radius accumulators
    agg_count_radii_total = np.zeros(5, dtype=np.float64)
    agg_area_radii_total = np.zeros(5, dtype=np.float64)
    pwr_count_radii_total = np.zeros(5, dtype=np.float64)
    pwr_area_radii_total = np.zeros(5, dtype=np.float64)
    agg_layer_within_1um_total = np.zeros(5, dtype=np.float64)

    aggressor_area_by_net: dict = {}

    for rec in records:
        c = rec["cuboids"]; ag_geo = rec["abs_geometries"]; names = rec["cuboid_net_names"]
        agg_mask = (c[:, 7] == 0.0)
        is_pwr = c[:, 9] >= 0.6
        sig_agg = agg_mask & (~is_pwr); pwr_agg = agg_mask & is_pwr

        if sig_agg.any():
            sig_layers = z_to_layer(ag_geo[sig_agg, 2])
            sig_areas = ag_geo[sig_agg, 3] * ag_geo[sig_agg, 4]
            agg_total_count += int(sig_agg.sum())
            agg_metal += float(sig_areas.sum())
            for li in range(1, N_LAYERS + 1):
                m = sig_layers == li
                agg_count_layer[li-1] += float(m.sum())
                agg_area_layer[li-1] += float(sig_areas[m].sum())
            indices = np.where(sig_agg)[0]
            for k_idx, i_idx in enumerate(indices):
                nm = names[i_idx]
                if nm in (target_net_str, "UNKNOWN_PIN"): continue
                aggressor_area_by_net[nm] = aggressor_area_by_net.get(nm, 0.0) + float(sig_areas[k_idx])

        if pwr_agg.any():
            pwr_layers = z_to_layer(ag_geo[pwr_agg, 2])
            pwr_areas = ag_geo[pwr_agg, 3] * ag_geo[pwr_agg, 4]
            pwr_total_count += int(pwr_agg.sum())
            pwr_metal += float(pwr_areas.sum())
            for li in range(1, N_LAYERS + 1):
                m = pwr_layers == li
                pwr_count_layer[li-1] += float(m.sum())
                pwr_area_layer[li-1] += float(pwr_areas[m].sum())

        st = _tile_coupling_v3(c, ag_geo, cutoff_um)
        cpl_n_pairs += st["n_pairs"]
        cpl_lat += st["lat_total"]; cpl_bs += st["bs_total"]
        cpl_below_0p5 += st["below_0p5"]; cpl_below_1 += st["below_1"]; cpl_below_2 += st["below_2"]
        cpl_min = min(cpl_min, st["min_dist"])
        cpl_dists_all.append(st["dists"])
        sum_inv_d += st["sum_inv_d"]; sum_inv_d2 += st["sum_inv_d2"]
        lat_w_inv_d += st["lat_w_inv_d"]; bs_w_inv_d += st["bs_w_inv_d"]
        cap_proxy_lat += st["cap_proxy_lat"]; cap_proxy_bs += st["cap_proxy_bs"]
        for i in range(5):
            agg_count_radii_total[i] += st["agg_count"][i]
            agg_area_radii_total[i] += st["agg_area"][i]
            pwr_count_radii_total[i] += st["pwr_count"][i]
            pwr_area_radii_total[i] += st["pwr_area"][i]
            agg_layer_within_1um_total[i] += st["agg_layer_within_1um"][i]

    cpl_dists = np.concatenate(cpl_dists_all) if cpl_dists_all else np.zeros(0, dtype=np.float32)
    if cpl_dists.size:
        p10 = float(np.percentile(cpl_dists, 10)); p25 = float(np.percentile(cpl_dists, 25))
        p50 = float(np.percentile(cpl_dists, 50)); p75 = float(np.percentile(cpl_dists, 75))
        p95 = float(np.percentile(cpl_dists, 95)); meand = float(cpl_dists.mean())
    else:
        p10 = p25 = p50 = p75 = p95 = meand = cutoff_um
    if cpl_min == float("inf"): cpl_min = cutoff_um

    if aggressor_area_by_net:
        sorted_areas = sorted(aggressor_area_by_net.values(), reverse=True)
        top1 = float(sorted_areas[0])
        top3 = float(sum(sorted_areas[:3]))
    else:
        top1 = 0.0; top3 = 0.0
    agg_unique_nets_count = float(len(aggressor_area_by_net))

    d_gnd = np.maximum(z - d/2, 0.05)
    compact_gnd = float((EPS0_FF_UM * e * (w * h_) / d_gnd).sum())
    compact_cpl = float(EPS0_FF_UM * eps_mean * (lat_w_inv_d + bs_w_inv_d))
    compact_total = compact_gnd + compact_cpl

    out = {
        "tgt_n_cuboids": float(g.shape[0]),
        "tgt_n_pins": n_pins, "tgt_n_wires": n_wires,
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
    for i, L in enumerate(_PER_LAYER): out[f"tgt_count_{L}"] = float(tgt_count_layer[i])
    for i, L in enumerate(_PER_LAYER): out[f"tgt_wirelen_{L}"] = float(tgt_wlen_layer[i])
    for i, L in enumerate(_PER_LAYER): out[f"tgt_area_{L}"] = float(tgt_area_layer[i])
    out.update({
        "agg_total_count": float(agg_total_count),
        "agg_unique_nets": agg_unique_nets_count,
        "agg_total_metal_area_um2": agg_metal,
        "agg_density_per_tile": float(agg_total_count) / max(n_tiles, 1),
        "agg_top1_area": top1, "agg_top3_area": top3,
    })
    for i, L in enumerate(_PER_LAYER): out[f"agg_count_{L}"] = float(agg_count_layer[i])
    for i, L in enumerate(_PER_LAYER): out[f"agg_area_{L}"] = float(agg_area_layer[i])
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
        "pwr_n_cuboids": float(pwr_total_count),
        "pwr_total_metal_area_um2": float(pwr_metal),
    })
    for i, L in enumerate(_PER_LAYER): out[f"pwr_count_{L}"] = float(pwr_count_layer[i])
    for i, L in enumerate(_PER_LAYER): out[f"pwr_area_{L}"] = float(pwr_area_layer[i])
    out.update({
        "compact_gnd_fF": compact_gnd,
        "compact_cpl_fF": compact_cpl,
        "compact_total_fF": compact_total,
    })

    # v3 multi-radius
    for i, tag in enumerate(_RADII_TAGS):
        out[f"v3_agg_count_within_{tag}um"] = float(agg_count_radii_total[i])
    for i, tag in enumerate(_RADII_TAGS):
        out[f"v3_agg_area_within_{tag}um"]  = float(agg_area_radii_total[i])
    for i, tag in enumerate(_RADII_TAGS):
        out[f"v3_pwr_count_within_{tag}um"] = float(pwr_count_radii_total[i])
    for i, tag in enumerate(_RADII_TAGS):
        out[f"v3_pwr_area_within_{tag}um"]  = float(pwr_area_radii_total[i])

    out["v3_bbox_x_um"] = float(bbox_x)
    out["v3_bbox_y_um"] = float(bbox_y)
    out["v3_length_density"] = float(wire_len) / max(bbox_xy, 1e-6)
    out["v3_cuboid_density"] = float(g.shape[0]) / max(bbox_xy, 1e-6)
    for i in range(5):
        out[f"v3_agg_M{i+1}_within_1um"] = float(agg_layer_within_1um_total[i])
    out["v3_cap_proxy_lateral"]   = cap_proxy_lat
    out["v3_cap_proxy_broadside"] = cap_proxy_bs
    return out
