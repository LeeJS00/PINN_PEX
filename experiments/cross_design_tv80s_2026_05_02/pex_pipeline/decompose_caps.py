"""Cap decomposition: total → c_gnd + c_cpl_total → per-pair coupling.

Stage 1: total_cap_pred → (c_gnd_pred, c_cpl_total_pred)
  Method: compact analytic ratio per net.
    g_ratio = compact_gnd_fF / compact_total_fF
    c_gnd = total × g_ratio
    c_cpl_total = total × (1 − g_ratio)
  Falls back to global mean ratio if compact_* missing.

Stage 2: c_cpl_total → per-aggressor c_pair distribution
  Method: geometric weighted softmax.
    w_pair ∝ (lat_overlap + bs_overlap × bs_factor) × eps_mean / (mean_dist² + 1e-3)
    c_pair = c_cpl_total × w_pair / Σw_pair
  Where bs_factor (broadside damping) defaults to 0.6 (broadside coupling typically weaker than lateral).

If pair_features for the design exist as a parquet, we use those; otherwise we compute on the fly from cuboid pkls (slow; only as a fallback).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Default broadside damping factor (broadside coupling per unit overlap area
# is typically weaker than lateral edge coupling for the same overlap).
BS_FACTOR_DEFAULT = 0.5

# Distance regularization (μm) so close pairs don't blow up.
DIST_REG_UM = 0.05


def split_total_to_gnd_cpl(features_df: pd.DataFrame,
                            total_pred: np.ndarray,
                            global_mean_g_ratio: float = 0.55) -> Tuple[np.ndarray, np.ndarray]:
    """Split predicted total cap into c_gnd and c_cpl_total per net.

    features_df must have one row per net (matching total_pred order) and
    contain compact_gnd_fF + compact_total_fF columns.
    """
    if "compact_gnd_fF" in features_df.columns and "compact_total_fF" in features_df.columns:
        g = features_df["compact_gnd_fF"].to_numpy(np.float64)
        t = features_df["compact_total_fF"].to_numpy(np.float64)
        ratio = g / np.maximum(t, 1e-6)
        # Clamp into a sensible range (0.05, 0.95)
        ratio = np.clip(ratio, 0.05, 0.95)
        # Replace NaN/inf with global mean
        bad = ~np.isfinite(ratio)
        if bad.any():
            ratio[bad] = global_mean_g_ratio
    else:
        ratio = np.full(len(total_pred), global_mean_g_ratio, dtype=np.float64)

    c_gnd_pred = total_pred * ratio
    c_cpl_pred = total_pred * (1.0 - ratio)
    return c_gnd_pred, c_cpl_pred


def distribute_cpl_to_pairs(c_cpl_total: float,
                             pair_features: List[dict],
                             bs_factor: float = BS_FACTOR_DEFAULT,
                             min_pair_fF: float = 0.0) -> List[Tuple[str, float]]:
    """Distribute c_cpl_total across pairs using geometric weighting.

    pair_features: list of dicts (one per aggressor) with at minimum
        'aggressor_net', 'mean_dist', 'lat_overlap_total', 'bs_overlap_total',
        'agg_eps_mean'

    Returns list of (aggressor_net, c_pair_pred).
    """
    if not pair_features or c_cpl_total <= 0:
        return []

    weights = []
    for p in pair_features:
        d = max(float(p.get("mean_dist", 1.0)), DIST_REG_UM)
        lat = float(p.get("lat_overlap_total", 0.0))
        bs = float(p.get("bs_overlap_total", 0.0))
        eps = float(p.get("agg_eps_mean", p.get("target_eps_mean", 3.0)))
        w = (lat + bs_factor * bs) * eps / (d * d)
        # If no overlap at all, fall back to inv-d²
        if w <= 0:
            w = eps / (d * d)
        weights.append(w)

    weights = np.array(weights, dtype=np.float64)
    total_w = weights.sum()
    if total_w <= 0:
        # Equal split
        share = c_cpl_total / len(pair_features)
        return [(p["aggressor_net"], share) for p in pair_features]

    weights = weights / total_w
    raw = [(p["aggressor_net"], float(c_cpl_total * w)) for p, w in zip(pair_features, weights)]
    if min_pair_fF <= 0:
        return raw

    # Drop pairs below threshold; redistribute their mass proportionally to kept pairs.
    kept = [(n, v) for n, v in raw if v >= min_pair_fF]
    dropped_mass = sum(v for n, v in raw if v < min_pair_fF)
    if not kept:
        return raw  # all below threshold, fall back
    if dropped_mass > 0:
        kept_total = sum(v for _, v in kept)
        scale = (kept_total + dropped_mass) / kept_total
        kept = [(n, v * scale) for n, v in kept]
    return kept


def load_pair_features_design(parquet_path: Path) -> Dict[str, List[dict]]:
    """Load per-design pair_features parquet, group by target_net.

    Returns: {target_net: [pair_dict, ...]} where pair_dict has the keys needed
    by distribute_cpl_to_pairs.
    """
    df = pd.read_parquet(parquet_path)
    out: Dict[str, List[dict]] = {}
    for tgt, sub in df.groupby("target_net"):
        rows = sub.to_dict(orient="records")
        out[tgt] = rows
    return out


def assemble_net_records(features_df: pd.DataFrame,
                          total_pred: np.ndarray,
                          pair_groups: Dict[str, List[dict]],
                          c_gnd_pred: np.ndarray,
                          c_cpl_pred: np.ndarray,
                          total_r: Optional[np.ndarray] = None,
                          port_xy: Optional[np.ndarray] = None,
                          design_name: Optional[str] = None) -> List[dict]:
    """Build SPEF records ready for LumpedSPEFWriter.write."""
    records = []
    for i, net_name in enumerate(features_df["net_name"].tolist()):
        c_cpl = float(c_cpl_pred[i])
        if c_cpl <= 0:
            pairs_pred = []
        elif net_name in pair_groups:
            pairs_pred = distribute_cpl_to_pairs(c_cpl, pair_groups[net_name])
        else:
            pairs_pred = []
        rec = {
            "name": net_name,
            "total_cap": float(total_pred[i]),
            "c_gnd": float(c_gnd_pred[i]),
            "pairs": pairs_pred,
            "total_r": float(total_r[i]) if total_r is not None else 0.0,
            "port_xy": tuple(port_xy[i]) if port_xy is not None else (0.0, 0.0),
        }
        records.append(rec)
    return records
