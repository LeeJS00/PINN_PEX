"""per_pair_residual.py — per-(target, aggressor) coupling capacitance via
analytic-prior + LightGBM log-residual regression.

Replaces v10's uniform `(L_t × L_a / d²)` distribution with a per-pair-specific
analytic prior (Sakurai-Tamaru lateral / parallel-plate vertical) plus a
learned multiplicative residual. Trained on TRAIN designs only.

Key contract:
  - `extract_pair_features(target_segs, aggr_segs, layer_info)` → dict of features
  - `analytic_per_pair_cap(features, layer_info)` → fF (closed-form physics)
  - `train_residual_model(train_pairs_df)` → fitted LightGBM
  - `predict_per_pair(features_df, model, layer_info)` → np.ndarray of fF
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Vacuum permittivity in fF/μm units: ε₀ × 1e-9 (μm) → fF/μm.
EPS0_FF_UM = 8.8541878128e-3

FEATURE_COLS = [
    "target_layer_idx", "agg_layer_idx", "layer_gap", "same_layer_int",
    "n_seg_pairs",
    "min_dist", "mean_dist", "p25_dist", "p75_dist",
    "sum_inv_d", "sum_inv_d2",
    "L_overlap_lateral_um", "A_overlap_vertical_um2",
    "target_total_metal_um", "agg_total_metal_um",
    "target_h_metal", "agg_h_metal",
    "target_eps", "agg_eps",
    "target_n_segs", "agg_n_segs",
    "log_c_analytic_pair",
]


def _parse_metal_layer_idx(name: str) -> int:
    """'m3' → 3. Returns 0 on parse failure."""
    if not name or not name.startswith("m"):
        return 0
    try:
        return int(name[1:])
    except ValueError:
        return 0


def metal_layer_props(layer_info: dict) -> dict[str, dict]:
    """Extract per-metal {layer: {z, thickness, eps, top_z}} from layer_info dict."""
    out = {}
    for k, v in layer_info.items():
        kl = k.lower()
        if kl.startswith("m") and 2 <= len(kl) <= 3:
            try:
                int(kl[1:])
            except ValueError:
                continue
            out[kl] = {
                "z": float(v["z_pos"]),
                "thickness": float(v["thickness"]),
                "eps": float(v["epsilon"]),
                "top_z": float(v["top_z"]),
            }
    return out


def _segs_to_arr(segs: list) -> np.ndarray | None:
    """Convert list of WIRE segment dicts to (N, 6) array
    [layer_idx, x_mid, y_mid, length, width, axis] where axis=0 horizontal, 1 vertical."""
    rows = []
    for s in segs:
        if s.get("type") != "WIRE":
            continue
        lay = _parse_metal_layer_idx(str(s.get("layer", "")))
        if lay == 0:
            continue
        x0, y0 = s["start"]
        x1, y1 = s["end"]
        dx, dy = x1 - x0, y1 - y0
        length = (dx*dx + dy*dy) ** 0.5
        if length < 1e-6:
            continue
        xm, ym = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        axis = 0 if abs(dx) >= abs(dy) else 1
        rows.append([lay, xm, ym, length, float(s.get("width", 0.05)), axis])
    if not rows:
        return None
    return np.asarray(rows, dtype=np.float64)


def _segment_distance_and_overlap(t_seg: np.ndarray, a_seg: np.ndarray):
    """For two segment vectors (each [layer, xm, ym, len, width, axis]), return:
        (d_lat, L_lateral_overlap, A_vert_overlap)
    where:
      d_lat       = lateral edge-to-edge distance (xy)
      L_lateral   = parallel-projection overlap if same axis, else 0
      A_vert      = bounding-box xy overlap area (used only when layer differs)
    Approximate: treats each segment as an axis-aligned bbox of size
    (length × width), centered at midpoint.
    """
    t_lay, txm, tym, tlen, twid, taxis = t_seg
    a_lay, axm, aym, alen, awid, aaxis = a_seg
    # Build half-extents in x, y depending on axis
    txhalf = 0.5 * (tlen if taxis == 0 else twid)
    tyhalf = 0.5 * (twid if taxis == 0 else tlen)
    axhalf = 0.5 * (alen if aaxis == 0 else awid)
    ayhalf = 0.5 * (awid if aaxis == 0 else alen)
    # XY edge-to-edge distance
    dx = max(0.0, abs(txm - axm) - (txhalf + axhalf))
    dy = max(0.0, abs(tym - aym) - (tyhalf + ayhalf))
    d_lat = (dx * dx + dy * dy) ** 0.5
    # Lateral overlap (same-axis parallel)
    if int(taxis) == int(aaxis):
        if taxis == 0:
            # both horizontal — parallel if y separation is non-overlap, overlap = x overlap
            x_overlap = max(0.0, (txhalf + axhalf) - abs(txm - axm))
            L_lat = x_overlap
        else:
            y_overlap = max(0.0, (tyhalf + ayhalf) - abs(tym - aym))
            L_lat = y_overlap
    else:
        L_lat = 0.0
    # Vertical overlap area (bbox xy intersection)
    x_int = max(0.0, (txhalf + axhalf) - abs(txm - axm))
    y_int = max(0.0, (tyhalf + ayhalf) - abs(tym - aym))
    A_vert = x_int * y_int
    return d_lat, L_lat, A_vert


def extract_pair_features_fast(
    target_arr: np.ndarray,
    aggr_arr: np.ndarray,
    metal_props: dict,
    cutoff_um: float = 5.0,
) -> dict:
    """Vectorized per-pair extraction for a SINGLE (target, aggressor) net pair.
    Inputs:
      target_arr: (N_t, 6) WIRE segments of target
      aggr_arr:   (N_a, 6) WIRE segments of aggressor
    Returns: feature dict (or None if no pair within cutoff).
    """
    if target_arr is None or aggr_arr is None or len(target_arr) == 0 or len(aggr_arr) == 0:
        return None

    Nt, Na = len(target_arr), len(aggr_arr)
    # Compute cross distances and overlaps per pair (Nt × Na). Cap with mask.
    # Vectorized bbox math:
    t_layer = target_arr[:, 0:1]; tx = target_arr[:, 1:2]; ty = target_arr[:, 2:3]
    t_len = target_arr[:, 3:4]; t_wid = target_arr[:, 4:5]; t_axis = target_arr[:, 5:6]
    a_layer = aggr_arr[:, 0]; ax = aggr_arr[:, 1]; ay = aggr_arr[:, 2]
    a_len = aggr_arr[:, 3]; a_wid = aggr_arr[:, 4]; a_axis = aggr_arr[:, 5]
    # Half-extents
    txh = 0.5 * np.where(t_axis == 0, t_len, t_wid)
    tyh = 0.5 * np.where(t_axis == 0, t_wid, t_len)
    axh = 0.5 * np.where(a_axis == 0, a_len, a_wid)
    ayh = 0.5 * np.where(a_axis == 0, a_wid, a_len)
    dx = np.maximum(0.0, np.abs(tx - ax) - (txh + axh))
    dy = np.maximum(0.0, np.abs(ty - ay) - (tyh + ayh))
    d_lat = np.sqrt(dx * dx + dy * dy)
    # Mask of pairs within cutoff
    mask = d_lat <= cutoff_um
    if not mask.any():
        return None
    # Same-axis flag (for lateral overlap)
    same_axis = (t_axis == a_axis)
    # Lateral overlap: along the dominant axis
    x_int = np.maximum(0.0, (txh + axh) - np.abs(tx - ax))
    y_int = np.maximum(0.0, (tyh + ayh) - np.abs(ty - ay))
    L_lat_arr = np.where(same_axis,
                         np.where(t_axis == 0, x_int, y_int),
                         0.0)
    A_vert_arr = x_int * y_int

    same_layer_arr = (t_layer == a_layer)
    layer_gap_arr = np.abs(t_layer - a_layer)

    # Apply mask
    d_kept = d_lat[mask]
    L_lat_kept = L_lat_arr[mask]
    A_vert_kept = A_vert_arr[mask]
    same_layer_kept = same_layer_arr[mask]
    layer_gap_kept = layer_gap_arr[mask]

    # Aggregates
    n_seg_pairs = int(mask.sum())
    d_clamped = np.maximum(d_kept, 0.05)
    sum_inv_d = float((1.0 / d_clamped).sum())
    sum_inv_d2 = float((1.0 / d_clamped ** 2).sum())
    min_dist = float(d_kept.min())
    mean_dist = float(d_kept.mean())
    p25_dist = float(np.percentile(d_kept, 25))
    p75_dist = float(np.percentile(d_kept, 75))
    L_overlap_lateral_um = float(np.where(same_layer_kept, L_lat_kept, 0.0).sum())
    A_overlap_vertical_um2 = float(np.where(~same_layer_kept, A_vert_kept, 0.0).sum())

    # Whole-net properties (target / aggressor)
    target_total_metal_um = float(target_arr[:, 3].sum())
    agg_total_metal_um = float(aggr_arr[:, 3].sum())

    # Dominant layer (by metal-area)
    def _dom_layer(arr):
        layers = arr[:, 0].astype(int)
        weights = arr[:, 3] * arr[:, 4]
        if weights.sum() <= 0:
            return int(layers[0])
        return int(np.bincount(layers, weights=weights).argmax())

    target_layer_idx = _dom_layer(target_arr)
    agg_layer_idx = _dom_layer(aggr_arr)
    layer_gap = abs(target_layer_idx - agg_layer_idx)
    same_layer_int = int(target_layer_idx == agg_layer_idx)

    # Layer physics
    t_props = metal_props.get(f"m{target_layer_idx}", None)
    a_props = metal_props.get(f"m{agg_layer_idx}", None)
    target_h_metal = t_props["thickness"] if t_props else 0.087
    agg_h_metal = a_props["thickness"] if a_props else 0.087
    target_eps = t_props["eps"] if t_props else 3.0
    agg_eps = a_props["eps"] if a_props else 3.0

    # ANALYTIC PER-PAIR CAP
    c_analytic = analytic_per_pair_cap_raw(
        target_layer_idx=target_layer_idx, agg_layer_idx=agg_layer_idx,
        target_h_metal=target_h_metal, agg_h_metal=agg_h_metal,
        target_eps=target_eps, agg_eps=agg_eps,
        L_overlap_lateral=L_overlap_lateral_um,
        A_overlap_vertical=A_overlap_vertical_um2,
        min_dist=min_dist, sum_inv_d2=sum_inv_d2,
        metal_props=metal_props,
    )

    return {
        "target_layer_idx": target_layer_idx, "agg_layer_idx": agg_layer_idx,
        "layer_gap": layer_gap, "same_layer_int": same_layer_int,
        "n_seg_pairs": n_seg_pairs,
        "min_dist": min_dist, "mean_dist": mean_dist,
        "p25_dist": p25_dist, "p75_dist": p75_dist,
        "sum_inv_d": sum_inv_d, "sum_inv_d2": sum_inv_d2,
        "L_overlap_lateral_um": L_overlap_lateral_um,
        "A_overlap_vertical_um2": A_overlap_vertical_um2,
        "target_total_metal_um": target_total_metal_um,
        "agg_total_metal_um": agg_total_metal_um,
        "target_h_metal": target_h_metal, "agg_h_metal": agg_h_metal,
        "target_eps": target_eps, "agg_eps": agg_eps,
        "target_n_segs": int(len(target_arr)), "agg_n_segs": int(len(aggr_arr)),
        "c_analytic_pair_fF": c_analytic,
        "log_c_analytic_pair": float(np.log(max(c_analytic, 1e-9))),
    }


def analytic_per_pair_cap_raw(
    target_layer_idx: int, agg_layer_idx: int,
    target_h_metal: float, agg_h_metal: float,
    target_eps: float, agg_eps: float,
    L_overlap_lateral: float, A_overlap_vertical: float,
    min_dist: float, sum_inv_d2: float,
    metal_props: dict,
    fringe_alpha: float = 0.10,
    spacing_floor: float = 0.04,
) -> float:
    """Closed-form per-pair cap estimate (in fF) using Sakurai-Tamaru / parallel-plate.

    Two physics modes — picked by layer relationship:
    1. Same-layer:    C = ε₀ ε_r h L / s   (Sakurai-Tamaru lateral, with fringe)
    2. Adjacent layers (gap=1): C = ε₀ ε_r A_overlap / d_inter   (parallel-plate vertical)
    3. Far layers (gap≥2): geometric fallback proportional to Σ 1/d²

    Returns capacitance in fF.
    """
    if target_layer_idx == agg_layer_idx:
        # Lateral coupling — same layer, parallel wires
        h = max(target_h_metal, 1e-3)
        eps = 0.5 * (target_eps + agg_eps)
        s = max(min_dist, spacing_floor)
        if L_overlap_lateral > 0:
            C = EPS0_FF_UM * eps * h * L_overlap_lateral / s * (1.0 + fringe_alpha)
        else:
            # No direct overlap; use point-coupling proxy from sum_inv_d2
            C = EPS0_FF_UM * eps * h * sum_inv_d2 * 0.05  # μm² × scale
        return max(C, 1e-9)

    # Cross-layer
    layer_gap = abs(target_layer_idx - agg_layer_idx)
    # Inter-metal distance
    t_props = metal_props.get(f"m{target_layer_idx}", None)
    a_props = metal_props.get(f"m{agg_layer_idx}", None)
    if t_props and a_props:
        d_inter = abs(t_props["z"] - a_props["z"]) - 0.5 * (t_props["thickness"] + a_props["thickness"])
        d_inter = max(d_inter, 0.04)
    else:
        d_inter = 0.15  # typical inter-metal spacing
    eps_inter = 0.5 * (target_eps + agg_eps)

    if layer_gap == 1 and A_overlap_vertical > 0:
        C = EPS0_FF_UM * eps_inter * A_overlap_vertical / d_inter
        return max(C, 1e-9)

    # Far layers or no overlap — geometric fallback
    h = 0.5 * (target_h_metal + agg_h_metal)
    if A_overlap_vertical > 0:
        C = EPS0_FF_UM * eps_inter * A_overlap_vertical / d_inter * (0.5 ** layer_gap)
    else:
        C = EPS0_FF_UM * eps_inter * h * sum_inv_d2 * 0.02 * (0.5 ** layer_gap)
    return max(C, 1e-9)


# ---------------- Training / inference ----------------

def train_residual_model(
    train_df: pd.DataFrame,
    feature_cols: list[str] = None,
    objective: str = "regression",  # MSE on log-residual; faster than L1
    n_estimators: int = 200,
    seed: int = 0,
    n_jobs: int = 8,
):
    """Train a LightGBM model to predict log(c_golden / c_analytic) from features.

    Args:
        train_df: must contain feature_cols + 'c_analytic_pair_fF' + 'c_golden_pair_fF'
        feature_cols: defaults to FEATURE_COLS
        objective: lightgbm objective (regression_l1 = MAE; tune later)

    Returns: fitted lgbm.Booster
    """
    import lightgbm as lgb
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    target = np.log(np.maximum(train_df["c_golden_pair_fF"].values, 1e-9)) - \
             np.log(np.maximum(train_df["c_analytic_pair_fF"].values, 1e-9))
    X = train_df[feature_cols].values
    train_set = lgb.Dataset(X, label=target, feature_name=feature_cols)
    params = dict(
        objective=objective,
        learning_rate=0.08,
        num_leaves=31,                # smaller trees → faster
        min_data_in_leaf=500,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=3,
        verbose=-1,
        seed=seed,
        num_threads=n_jobs,
    )
    booster = lgb.train(params, train_set, num_boost_round=n_estimators)
    return booster


def predict_per_pair(features_df: pd.DataFrame, booster, feature_cols=None) -> np.ndarray:
    """Apply analytic prior + booster log-residual.
    Returns predicted c_pair (fF) array.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    X = features_df[feature_cols].values
    log_residual = booster.predict(X)
    c_analytic = features_df["c_analytic_pair_fF"].values
    return np.maximum(c_analytic * np.exp(log_residual), 0.0)
