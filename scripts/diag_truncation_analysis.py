#!/usr/bin/env python3
"""
Quantify cuboid truncation caused by NF_PAD_TO_CUBOIDS=768.

Usage:
    source tool.env && python3 scripts/diag_truncation_analysis.py

This script scans the predefined validation manifest, reconstructs the same
cuboid assembly order used by src/data/datasets.py
([target, core aggressors, voxelized context]), measures how often the final
assembled tile would exceed 768 cuboids, writes a per-tile CSV summary, and
prints aggregate correlations against net-level error metrics.
"""

import gzip
import math
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, "/home/jslee/projects/PINNPEX")


REPO_ROOT = Path("/home/jslee/projects/PINNPEX")
VAL_MANIFEST = REPO_ROOT / "output_intel22/active_learning/cache/predefined_valid_subset.csv"
HURDLE_CSV = REPO_ROOT / "output_intel22/active_learning/v3_netlevel/diagnose_hurdle.csv"
DATA_ROOT = Path("/data/PEX_SSL/data/processed/intel22")
OUTPUT_DIR = REPO_ROOT / "output_intel22/diag"
OUTPUT_CSV = OUTPUT_DIR / "truncation_analysis.csv"

NF_PAD_TO_CUBOIDS = 768
WINDOW_SIZE = (4.0, 4.0, 20.0)
TILING_OVERLAP = 0.5
CORE_RADIUS_UM = 2.0
VOXEL_GRID = np.array([1.0, 1.0, 0.1], dtype=np.float32)
MAX_WARNING_REPEATS = 10
_warning_counts = {}
_protocol5_warned = False


def print_header(title):
    print(f"\n=== {title} ===")


def iter_with_progress(rows, desc):
    try:
        from tqdm import tqdm

        return tqdm(rows, total=len(rows), desc=desc)
    except Exception:
        def generator():
            total = len(rows)
            for idx, item in enumerate(rows, 1):
                if idx == 1 or idx % 250 == 0 or idx == total:
                    print(f"[{desc}] {idx}/{total}")
                yield item

        return generator()


def warn(msg):
    count = _warning_counts.get(msg, 0) + 1
    _warning_counts[msg] = count
    if count <= MAX_WARNING_REPEATS:
        warnings.warn(msg, RuntimeWarning)
    elif count == MAX_WARNING_REPEATS + 1:
        warnings.warn(f"{msg} [further repeats suppressed]", RuntimeWarning)


def safe_percentile(values, q):
    if len(values) == 0:
        return math.nan
    return float(np.percentile(values, q))


def compute_core_ratios(cuboids, abs_geo, origin):
    if len(cuboids) == 0:
        return np.zeros(0, dtype=np.float32)

    w = cuboids[:, 3].astype(np.float32)
    h = cuboids[:, 4].astype(np.float32)
    d = cuboids[:, 5].astype(np.float32)
    lx = abs_geo[:, 0].astype(np.float32) - float(origin[0])
    ly = abs_geo[:, 1].astype(np.float32) - float(origin[1])

    core_hw_x = (WINDOW_SIZE[0] - TILING_OVERLAP) / 2.0
    core_hw_y = (WINDOW_SIZE[1] - TILING_OVERLAP) / 2.0

    l_max = np.max(cuboids[:, 3:6], axis=1).astype(np.float32)
    c_len = np.zeros(len(cuboids), dtype=np.float32)

    mask_x = (w >= h) & (w >= d)
    mask_y = (h > w) & (h >= d)
    mask_z = ~(mask_x | mask_y)

    if np.any(mask_x):
        upper = np.minimum(lx[mask_x] + w[mask_x] / 2.0, core_hw_x)
        lower = np.maximum(lx[mask_x] - w[mask_x] / 2.0, -core_hw_x)
        c_len[mask_x] = np.maximum(upper - lower, 0.0)

    if np.any(mask_y):
        upper = np.minimum(ly[mask_y] + h[mask_y] / 2.0, core_hw_y)
        lower = np.maximum(ly[mask_y] - h[mask_y] / 2.0, -core_hw_y)
        c_len[mask_y] = np.maximum(upper - lower, 0.0)

    if np.any(mask_z):
        via_mask = (np.abs(lx) <= core_hw_x) & (np.abs(ly) <= core_hw_y)
        valid_via = mask_z & via_mask
        c_len[valid_via] = d[valid_via]

    safe_l_max = np.clip(l_max, 1e-6, None)
    return np.where(l_max > 0, c_len / safe_l_max, 0.0).astype(np.float32)


def estimate_post_voxel_count(cuboids, names, abs_geo, origin, target_net):
    names_arr = np.asarray(names)
    if len(cuboids) != len(names_arr) or len(cuboids) != len(abs_geo):
        raise ValueError("cuboids / cuboid_net_names / abs_geometries length mismatch")

    tgt_mask = names_arr == target_net
    n_target = int(np.count_nonzero(tgt_mask))

    aggr_mask = ~tgt_mask
    aggr_names = names_arr[aggr_mask]
    n_aggr_unique_raw = int(len({str(x) for x in aggr_names.tolist() if str(x) not in {"", "PAD"}}))

    if not np.any(aggr_mask):
        return n_target, n_aggr_unique_raw, n_target

    core_ratios = compute_core_ratios(cuboids, abs_geo, origin)
    del core_ratios  # Included for parity with dataset logic; not needed afterward.

    tgt_tensor = cuboids[tgt_mask]
    aggr_tensor = cuboids[aggr_mask]

    if len(tgt_tensor) > 0:
        target_center = tgt_tensor[:, :3].mean(axis=0)
    else:
        target_center = np.zeros(3, dtype=np.float32)

    dist_xy = np.linalg.norm(aggr_tensor[:, :2] - target_center[:2], axis=1)
    is_core = dist_xy <= CORE_RADIUS_UM
    n_core = int(np.count_nonzero(is_core))

    ctx_tensor = aggr_tensor[~is_core]
    n_ctx_vox = 0
    if len(ctx_tensor) > 0:
        grid_hash = np.round(ctx_tensor[:, :3] / VOXEL_GRID).astype(np.int64)
        n_ctx_vox = int(len(np.unique(grid_hash, axis=0)))

    n_estimated_post_vox = int(n_target + n_core + n_ctx_vox)
    return n_target, n_aggr_unique_raw, n_estimated_post_vox


def analyze_tile(row):
    global _protocol5_warned
    design_name = row.get("design_name")
    sample_filename = row.get("sample_filename")
    tile_idx = row.get("tile_idx")
    net_name = row.get("net_name")

    result = {
        "net_name": net_name,
        "design_name": design_name,
        "tile_idx": tile_idx,
        "n_raw": math.nan,
        "n_target": math.nan,
        "n_aggr_unique_raw": math.nan,
        "n_estimated_post_vox": math.nan,
        "was_truncated": False,
        "dropped_cuboids": math.nan,
    }

    if pd.isna(design_name) or pd.isna(sample_filename):
        warn(f"Skipping manifest row with missing design_name/sample_filename: tile_idx={tile_idx}, net={net_name}")
        return result

    pkl_path = DATA_ROOT / str(design_name) / str(sample_filename)
    if not pkl_path.exists():
        warn(f"Missing tile file: {pkl_path}")
        return result

    try:
        with gzip.open(pkl_path, "rb") as fh:
            data = pickle.load(fh)
    except Exception as exc:
        if "unsupported pickle protocol: 5" in str(exc):
            if not _protocol5_warned:
                warn("Encountered pickle protocol 5, but the current Python runtime cannot decode it; subsequent protocol-5 tile warnings are suppressed")
                _protocol5_warned = True
            return result
        warn(f"Failed to load {pkl_path}: {exc}")
        return result

    try:
        cuboids = np.asarray(data["cuboids"])
        names = list(data.get("cuboid_net_names", []))
        abs_geo = np.asarray(data.get("abs_geometries", np.zeros((len(cuboids), 6), dtype=np.float32)))
        origin = data.get("origin", [0.0, 0.0, 0.0])
    except Exception as exc:
        warn(f"Malformed payload in {pkl_path}: {exc}")
        return result

    if cuboids.ndim != 2 or cuboids.shape[1] < 6:
        warn(f"Unexpected cuboid shape in {pkl_path}: {cuboids.shape}")
        return result

    if len(cuboids) == 0:
        result.update(
            {
                "n_raw": 0,
                "n_target": 0,
                "n_aggr_unique_raw": 0,
                "n_estimated_post_vox": 0,
                "was_truncated": False,
                "dropped_cuboids": 0,
            }
        )
        return result

    try:
        n_target, n_aggr_unique_raw, n_post = estimate_post_voxel_count(
            cuboids=cuboids,
            names=names,
            abs_geo=abs_geo,
            origin=origin,
            target_net=str(net_name),
        )
    except Exception as exc:
        warn(f"Failed to analyze {pkl_path}: {exc}")
        return result

    dropped = max(int(n_post) - NF_PAD_TO_CUBOIDS, 0)
    result.update(
        {
            "n_raw": int(len(cuboids)),
            "n_target": int(n_target),
            "n_aggr_unique_raw": int(n_aggr_unique_raw),
            "n_estimated_post_vox": int(n_post),
            "was_truncated": bool(dropped > 0),
            "dropped_cuboids": int(dropped),
        }
    )
    return result


def print_summary_table(summary_df):
    display_cols = ["group", "n_nets", "mean_rel_err", "median_rel_err", "mean_dropped", "median_dropped"]
    if summary_df.empty:
        print("No summary rows available.")
        return
    print(summary_df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print_header("Inputs")
    print(f"Validation manifest: {VAL_MANIFEST}")
    print(f"Diagnose hurdle CSV: {HURDLE_CSV}")
    print(f"Tile data root      : {DATA_ROOT}")
    print(f"NF_PAD_TO_CUBOIDS   : {NF_PAD_TO_CUBOIDS}")
    print(f"Python version      : {sys.version.split()[0]}")

    if sys.version_info < (3, 8):
        warn("Python < 3.8 cannot load pickle protocol 5; tile loads may all be skipped")

    if not VAL_MANIFEST.exists():
        warn(f"Validation manifest not found: {VAL_MANIFEST}")
        pd.DataFrame(
            columns=[
                "net_name",
                "design_name",
                "tile_idx",
                "n_raw",
                "n_target",
                "n_aggr_unique_raw",
                "n_estimated_post_vox",
                "was_truncated",
                "dropped_cuboids",
            ]
        ).to_csv(OUTPUT_CSV, index=False)
        print(f"Wrote empty output to {OUTPUT_CSV}")
        return

    val_df = pd.read_csv(VAL_MANIFEST)
    required = {"sample_filename", "design_name", "tile_idx", "net_name"}
    missing_cols = required - set(val_df.columns)
    if missing_cols:
        raise ValueError(f"Validation manifest missing required columns: {sorted(missing_cols)}")

    records = []
    rows = [row for _, row in val_df.iterrows()]
    for row in iter_with_progress(rows, "tiles"):
        records.append(analyze_tile(row))

    out_df = pd.DataFrame(records)
    out_df.to_csv(OUTPUT_CSV, index=False)

    print_header("Tile-Level Output")
    print(f"Wrote {len(out_df)} rows to {OUTPUT_CSV}")

    valid_drop = out_df["dropped_cuboids"].dropna()
    truncated_mask = out_df["was_truncated"].fillna(False).astype(bool)
    trunc_rate = 100.0 * float(truncated_mask.mean()) if len(out_df) else math.nan

    print_header("Overall Truncation")
    print(f"truncation_rate = {trunc_rate:.2f}%")
    if len(valid_drop) > 0:
        print(f"truncated_tiles = {int(truncated_mask.sum())}/{len(out_df)}")
    else:
        print("No valid tiles were analyzed.")

    print_header("Dropped Cuboids Distribution")
    drop_values = valid_drop[valid_drop > 0].to_numpy()
    if len(drop_values) == 0:
        print("No truncated tiles; all dropped_cuboids percentiles are 0.")
    else:
        pct_df = pd.DataFrame(
            {
                "percentile": [50, 75, 90, 95, 99],
                "dropped_cuboids": [safe_percentile(drop_values, q) for q in [50, 75, 90, 95, 99]],
            }
        )
        print(pct_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print_header("Net-Level Correlation")
    net_tile_df = (
        out_df.groupby("net_name", dropna=False)
        .agg(
            dropped_cuboids_per_net=("dropped_cuboids", lambda s: float(np.nansum(s))),
            any_truncated=("was_truncated", lambda s: bool(pd.Series(s).fillna(False).any())),
            n_truncated_tiles=("was_truncated", lambda s: int(pd.Series(s).fillna(False).sum())),
            n_tiles=("tile_idx", "count"),
        )
        .reset_index()
    )

    merged_df = None
    if HURDLE_CSV.exists():
        hurdle_df = pd.read_csv(HURDLE_CSV)
        if "net_name" not in hurdle_df.columns or "rel_err" not in hurdle_df.columns:
            warn(f"Skipping hurdle merge; required columns missing in {HURDLE_CSV}")
        else:
            merged_df = hurdle_df.merge(net_tile_df, on="net_name", how="left")
            merged_df["dropped_cuboids_per_net"] = merged_df["dropped_cuboids_per_net"].fillna(0.0)
            merged_df["any_truncated"] = merged_df["any_truncated"].fillna(False)
            merged_df["n_truncated_tiles"] = merged_df["n_truncated_tiles"].fillna(0).astype(int)

            corr_df = merged_df[["dropped_cuboids_per_net", "rel_err"]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(corr_df) >= 2 and corr_df["dropped_cuboids_per_net"].nunique() > 1 and corr_df["rel_err"].nunique() > 1:
                r_value, p_value = pearsonr(corr_df["dropped_cuboids_per_net"], corr_df["rel_err"])
                print(f"Pearson r(dropped_cuboids_per_net, rel_err) = {r_value:.4f} (p={p_value:.4g}, n={len(corr_df)})")
            else:
                print("Pearson correlation unavailable: insufficient variation after merge.")

            print_header("MAPE Comparison")
            summary_rows = []
            for label, mask in [
                ("truncated_nets", merged_df["any_truncated"].astype(bool)),
                ("non_truncated_nets", ~merged_df["any_truncated"].astype(bool)),
            ]:
                subset = merged_df[mask]
                summary_rows.append(
                    {
                        "group": label,
                        "n_nets": int(len(subset)),
                        "mean_rel_err": float(subset["rel_err"].mean()) if len(subset) else math.nan,
                        "median_rel_err": float(subset["rel_err"].median()) if len(subset) else math.nan,
                        "mean_dropped": float(subset["dropped_cuboids_per_net"].mean()) if len(subset) else math.nan,
                        "median_dropped": float(subset["dropped_cuboids_per_net"].median()) if len(subset) else math.nan,
                    }
                )
            summary_df = pd.DataFrame(summary_rows)
            print_summary_table(summary_df)
    else:
        warn(f"Diagnose hurdle CSV not found: {HURDLE_CSV}")
        print("Skipping Pearson / MAPE comparison.")

    print_header("Summary Table")
    summary = {
        "n_tiles_total": int(len(out_df)),
        "n_tiles_valid": int(out_df["n_estimated_post_vox"].notna().sum()),
        "n_tiles_truncated": int(truncated_mask.sum()),
        "truncation_rate_pct": trunc_rate,
        "mean_dropped_all_tiles": float(valid_drop.mean()) if len(valid_drop) else math.nan,
        "mean_dropped_truncated_tiles": float(drop_values.mean()) if len(drop_values) else 0.0,
        "max_dropped_cuboids": float(drop_values.max()) if len(drop_values) else 0.0,
    }
    print(pd.DataFrame([summary]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if merged_df is not None:
        overlap_nets = int(merged_df["dropped_cuboids_per_net"].notna().sum())
        print(f"\nMerged hurdle rows: {len(merged_df)}")
        print(f"Nets with truncation diagnostics after merge: {overlap_nets}")


if __name__ == "__main__":
    main()
