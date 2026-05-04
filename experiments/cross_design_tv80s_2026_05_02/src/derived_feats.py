"""Derived features computed on-the-fly without rebuilding parquets."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add cheap interaction / ratio features from existing columns."""
    df = df.copy()
    eps = 1e-6
    # Ratios
    df["d_compact_total_per_metal"] = df["compact_total_fF"] / (df["tgt_total_metal_area_um2"] + eps)
    df["d_aggressor_pressure"] = df["agg_total_count"] / (df["tgt_n_cuboids"] + eps)
    df["d_pwr_to_agg_area"] = df["pwr_total_metal_area_um2"] / (df["agg_total_metal_area_um2"] + eps)
    df["d_cpl_density"] = df["cpl_n_pairs"] / (df["tgt_n_cuboids"] + eps)
    # Layer span
    layer_cols = ["tgt_count_M1","tgt_count_M2","tgt_count_M3","tgt_count_M4",
                  "tgt_count_M5","tgt_count_M6","tgt_count_M7","tgt_count_M8","tgt_count_M9p"]
    if all(c in df.columns for c in layer_cols):
        layer_arr = df[layer_cols].to_numpy()
        df["d_n_layers_present"] = (layer_arr > 0).sum(axis=1).astype(np.float32)
        df["d_layer_max_idx"] = np.argmax(layer_arr[:, ::-1] > 0, axis=1)  # rightmost layer with cuboids
        df["d_layer_min_idx"] = np.argmax(layer_arr > 0, axis=1)
        df["d_layer_span"] = df["d_layer_max_idx"] - df["d_layer_min_idx"]
    # Cap-relevant size metrics
    df["d_log_metal_area"] = np.log1p(df["tgt_total_metal_area_um2"])
    df["d_log_wire_length"] = np.log1p(df["tgt_wire_length_um"])
    df["d_log_compact_total"] = np.log1p(df["compact_total_fF"])
    # Density per area
    df["d_aggressor_per_area"] = df["agg_total_count"] / (df["tgt_total_metal_area_um2"] + eps)
    df["d_compact_cpl_frac"] = df["compact_cpl_fF"] / (df["compact_total_fF"] + eps)
    # Coupling proximity intensity
    df["d_cpl_intensity_below_1um"] = df["cpl_n_below_1um"] / (df["cpl_n_pairs"] + eps)
    df["d_lateral_to_broadside_ratio"] = df["cpl_total_lateral_overlap_um2"] / (df["cpl_total_broadside_overlap_um2"] + eps)
    df["d_inv_d_per_pair"] = df["cpl_sum_inv_d"] / (df["cpl_n_pairs"] + eps)
    # Layer-aggressor matching
    if "agg_count_M1" in df.columns and "tgt_count_M1" in df.columns:
        df["d_agg_target_layer_correlation"] = np.zeros(len(df), dtype=np.float32)
        for L in ["M1","M2","M3","M4","M5","M6","M7","M8","M9p"]:
            tcol = f"tgt_count_{L}"; acol = f"agg_count_{L}"
            if tcol in df.columns and acol in df.columns:
                df["d_agg_target_layer_correlation"] += np.minimum(df[tcol], df[acol] / 100.0)
    return df
