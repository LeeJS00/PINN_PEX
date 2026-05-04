#!/usr/bin/env python3
"""01_extract_train_features.py — extract per-pair features for TRAIN designs.

Strategy: only extract features for (target, aggressor) pairs that appear
in the golden parquet for that design. Subsample the golden pairs per
design to keep total feature-extraction tractable on a single CPU.

Skips intel22_ldpc_decoder_802_3an_f3 (56K topology pkls, far too heavy
for a 90-min smoke timebox).
"""
from __future__ import annotations
import gzip
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

REPO = Path("/home/jslee/projects/PINNPEX")
sys.path.insert(0, str(REPO))

from configs.config_v3 import LAYERS_INFO_PATH
from src.preprocessing.layer_parser import LayerInfoParser

sys.path.insert(0, str(REPO / "pex_v3" / "joint_pareto" / "allocators" / "cpl"))
import per_pair_residual as ppr


# Designs ordered by topology pkl count (smallest first to maximise time-budget yield)
TRAIN_DESIGNS = [
    "intel22_gcd_f3",                     #   290 pkls
    "intel22_spi_top_f3",                 # 1.7K
    "intel22_mc_top_f3",                  # 3.9K
    "intel22_usbf_top_f3",                # 7.7K
    "intel22_ibex_core_f3",               # 11.9K
    "intel22_aes_cipher_top_f3",          # 12.0K
    "intel22_wb_conmax_top_f3",           # 17.7K
    # "intel22_vga_enh_top_f3",           # 34.5K — SKIP for smoke (too slow seg load)
    # "intel22_ldpc_decoder_802_3an_f3",  # 56.2K — SKIP for smoke
]

GOLDEN_DIR = Path("/data/PINNPEX/data/processed_v3/intel22/per_pair_golden")
TOPO_ROOT = Path("/data/PINNPEX/data/processed_v3/intel22")
OUT_DIR = Path("/home/jslee/projects/PINNPEX/pex_v3/joint_pareto/experiments/exp_013_per_pair/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Subsampling
MAX_GOLDEN_PAIRS_PER_DESIGN = 100_000


def _load_one_pkl(path):
    try:
        with gzip.open(path, "rb") as f:
            d = pickle.load(f)
    except Exception:
        return None
    gs = d.get("global_segments", [])
    own_net = None
    for s in gs:
        if "net_name" in s and s.get("type") == "WIRE":
            own_net = s["net_name"]
            break
    if own_net is None:
        return None
    own_segs = [s for s in gs if s.get("type") == "WIRE" and s.get("net_name") == own_net]
    arr = ppr._segs_to_arr(own_segs)
    if arr is None:
        return None
    return (own_net, arr)


def load_segments_for_design(design: str, n_workers: int = 8) -> dict[str, np.ndarray]:
    topo_dir = TOPO_ROOT / design / "topology"
    paths = sorted(topo_dir.glob(f"{design}___topo_*.pkl.gz"))
    out = {}
    if n_workers <= 1 or len(paths) < 200:
        for p in paths:
            r = _load_one_pkl(p)
            if r is not None:
                out[r[0]] = r[1]
        return out
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        chunksize = max(1, min(64, len(paths) // (n_workers * 8)))
        for r in pool.imap_unordered(_load_one_pkl, paths, chunksize=chunksize):
            if r is not None:
                out[r[0]] = r[1]
    return out


def main():
    layer_info = LayerInfoParser(LAYERS_INFO_PATH).parse()
    metal_props = ppr.metal_layer_props(layer_info)
    print(f">>> metal layers: {sorted(metal_props.keys())}")

    rng = np.random.default_rng(0)

    all_rows = []
    for design in TRAIN_DESIGNS:
        gpath = GOLDEN_DIR / f"{design}.parquet"
        if not gpath.exists():
            print(f"  ! missing golden for {design}, skip")
            continue
        t0 = time.time()
        gdf = pd.read_parquet(gpath)
        a = gdf["target_net"].values.astype(object)
        b = gdf["aggressor_net"].values.astype(object)
        lo = np.where(a <= b, a, b)
        hi = np.where(a <= b, b, a)
        gdf["key_lo"] = lo
        gdf["key_hi"] = hi
        gpairs_df = gdf.groupby(["key_lo", "key_hi"], sort=False)["c_pair_fF"].sum().reset_index()
        n_total = len(gpairs_df)
        if n_total > MAX_GOLDEN_PAIRS_PER_DESIGN:
            sub_idx = rng.choice(n_total, size=MAX_GOLDEN_PAIRS_PER_DESIGN, replace=False)
            gpairs_df = gpairs_df.iloc[sub_idx]
        gpairs = list(zip(gpairs_df["key_lo"].values, gpairs_df["key_hi"].values, gpairs_df["c_pair_fF"].values))
        del gdf, gpairs_df, a, b, lo, hi
        print(f">>> {design}: {n_total:,} unique golden pairs (sampled {len(gpairs):,}) in {time.time()-t0:.1f}s")

        t1 = time.time()
        nets = load_segments_for_design(design, n_workers=8)
        print(f"    loaded {len(nets)} nets segments in {time.time()-t1:.1f}s")

        n_match = n_skip = 0
        rows = []
        t2 = time.time()
        last_report = t2
        for i, (na, nb, c_g) in enumerate(gpairs):
            if na not in nets or nb not in nets:
                n_skip += 1
                continue
            feat = ppr.extract_pair_features_fast(nets[na], nets[nb], metal_props, cutoff_um=5.0)
            if feat is None:
                n_skip += 1
                continue
            feat["c_golden_pair_fF"] = float(c_g)
            feat["design_name"] = design
            feat["target_net"] = na
            feat["aggressor_net"] = nb
            rows.append(feat)
            n_match += 1
            if time.time() - last_report > 30:
                print(f"      progress {i+1}/{len(gpairs)}  matched={n_match}  ({time.time()-t2:.1f}s)")
                last_report = time.time()
        print(f"    matched {n_match} pairs (skipped {n_skip}), feat-extract {time.time()-t2:.1f}s")
        all_rows.extend(rows)
        # Incremental snapshot per design — survives kill
        if all_rows:
            snap = pd.DataFrame(all_rows)
            snap.to_parquet(OUT_DIR / "train_pairs.parquet", index=False)
            print(f"    saved incremental snapshot ({len(snap):,} rows)")
        # Free memory: nets dict is ~1GB on big designs
        del nets

    df = pd.DataFrame(all_rows)
    print(f"\n>>> total rows: {len(df):,}")
    out = OUT_DIR / "train_pairs.parquet"
    df.to_parquet(out, index=False)
    print(f"saved {out}")

    if len(df):
        ratio = df["c_golden_pair_fF"] / np.maximum(df["c_analytic_pair_fF"], 1e-9)
        print(f"  golden / analytic ratio:  median={ratio.median():.3f}  mean={ratio.mean():.3f}")
        print(f"    p10={np.percentile(ratio, 10):.3f}  p90={np.percentile(ratio, 90):.3f}")
        ape_an = 100 * np.abs(df["c_analytic_pair_fF"] - df["c_golden_pair_fF"]) / np.maximum(df["c_golden_pair_fF"], 1e-9)
        print(f"  analytic-only MAPE:  mean={ape_an.mean():.2f}%  median={ape_an.median():.2f}%")


if __name__ == "__main__":
    main()
