#!/usr/bin/env python3
"""03_smoke_tv80s.py — apply trained residual model on tv80s.

Two evaluation tracks:
  A. Per-pair MAPE on golden ∩ predicted (positive recall)
  B. Per-pair MAPE on top-K candidates from KD-tree (the OPERATIONAL setting)

Track A uses pairs where (target, aggressor) is BOTH in golden parquet
and in our extracted candidate set.

We extract candidate pairs by, for each target net, walking its segments
and finding aggressors in the design within max_dist_um=5.0 via KD-tree
(matches v10 baseline allocator).
"""
from __future__ import annotations
import gzip
import pickle
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

REPO = Path("/home/jslee/projects/PINNPEX")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pex_v3" / "joint_pareto" / "allocators" / "cpl"))

from configs.config_v3 import LAYERS_INFO_PATH
from src.preprocessing.layer_parser import LayerInfoParser
import per_pair_residual as ppr

DESIGN = "intel22_tv80s_f3"
GOLDEN = Path(f"/data/PINNPEX/data/processed_v3/intel22/per_pair_golden/{DESIGN}.parquet")
TOPO_DIR = Path(f"/data/PINNPEX/data/processed_v3/intel22/{DESIGN}/topology")
OUT_DIR = Path("/home/jslee/projects/PINNPEX/pex_v3/joint_pareto/experiments/exp_013_per_pair/results")


def load_design_segments(topo_dir: Path) -> dict[str, np.ndarray]:
    paths = sorted(topo_dir.glob("*.pkl.gz"))
    out: dict[str, list] = defaultdict(list)
    own_net_for_path: dict[str, str] = {}
    for p in paths:
        try:
            with gzip.open(p, "rb") as f:
                d = pickle.load(f)
        except Exception:
            continue
        gs = d.get("global_segments", [])
        own_net = None
        for s in gs:
            if "net_name" in s and s.get("type") == "WIRE":
                own_net = s["net_name"]
                break
        if own_net is None:
            continue
        own_segs = [s for s in gs if s.get("type") == "WIRE" and s.get("net_name") == own_net]
        out[own_net].extend(own_segs)
    arr_dict = {}
    for net, segs in out.items():
        arr = ppr._segs_to_arr(segs)
        if arr is not None:
            arr_dict[net] = arr
    return arr_dict


def main():
    layer_info = LayerInfoParser(LAYERS_INFO_PATH).parse()
    metal_props = ppr.metal_layer_props(layer_info)

    print(f">>> loading model")
    import lightgbm as lgb
    booster = lgb.Booster(model_file=str(OUT_DIR / "residual_model.lgb"))

    print(f">>> loading {DESIGN} segments")
    t0 = time.time()
    nets = load_design_segments(TOPO_DIR)
    print(f"    {len(nets)} nets in {time.time()-t0:.1f}s")

    print(f">>> loading golden parquet")
    gdf = pd.read_parquet(GOLDEN)
    gdf["key"] = gdf.apply(lambda r: tuple(sorted([r["target_net"], r["aggressor_net"]])), axis=1)
    g_pairs = gdf.groupby("key")["c_pair_fF"].sum().to_dict()
    print(f"    {len(g_pairs):,} golden pairs")

    print(f">>> building global KD-tree on all segments")
    # Flatten: each row in big_arr corresponds to a (net_name, seg_idx).
    big_records = []
    big_coords = []
    for net, arr in nets.items():
        for i in range(len(arr)):
            big_records.append((net, i))
            big_coords.append((arr[i, 1], arr[i, 2]))
    big_coords = np.asarray(big_coords)
    tree = cKDTree(big_coords)
    print(f"    {len(big_records)} segments")

    print(f">>> finding candidate pairs per target via KD-tree (max_dist=5.0μm)")
    t1 = time.time()
    candidate_pairs: set[tuple[str, str]] = set()
    for tgt_net, arr in nets.items():
        # one query per segment of target
        for i in range(len(arr)):
            x, y = arr[i, 1], arr[i, 2]
            idxs = tree.query_ball_point((x, y), r=5.0)
            for j in idxs:
                other_net, _ = big_records[j]
                if other_net == tgt_net:
                    continue
                key = tuple(sorted([tgt_net, other_net]))
                candidate_pairs.add(key)
    print(f"    {len(candidate_pairs):,} unique candidate pairs in {time.time()-t1:.1f}s")

    # Now extract features for each candidate pair (a SUBSET, since we don't need every aggressor — top-K could be smaller)
    print(f">>> extracting features for candidate pairs")
    t2 = time.time()
    rows = []
    for (na, nb) in candidate_pairs:
        if na not in nets or nb not in nets:
            continue
        feat = ppr.extract_pair_features_fast(nets[na], nets[nb], metal_props, cutoff_um=5.0)
        if feat is None:
            continue
        feat["target_net"] = na
        feat["aggressor_net"] = nb
        feat["c_golden_pair_fF"] = float(g_pairs.get((na, nb), 0.0))
        rows.append(feat)
    df = pd.DataFrame(rows)
    print(f"    extracted {len(df):,} feature rows in {time.time()-t2:.1f}s")

    # Run model
    print(f">>> running booster.predict")
    pred = ppr.predict_per_pair(df, booster)
    df["c_pred_pair_fF"] = pred

    # Save
    df.to_parquet(OUT_DIR / "tv80s_pair_predictions.parquet", index=False)

    # Eval: only on pairs that EXIST in golden
    has_golden = df["c_golden_pair_fF"] > 1e-6
    df_g = df[has_golden].copy()
    print()
    print(f"=== tv80s per-pair MAPE on common pairs (golden ∩ predicted) ===")
    print(f"    n_common   = {len(df_g):,}")
    print(f"    n_golden   = {len(g_pairs):,}")
    print(f"    coverage   = {100*len(df_g)/len(g_pairs):.1f}%")
    g = df_g["c_golden_pair_fF"].values
    p_pred = df_g["c_pred_pair_fF"].values
    p_an = df_g["c_analytic_pair_fF"].values
    ape_pred = 100 * np.abs(p_pred - g) / np.maximum(g, 1e-9)
    ape_an = 100 * np.abs(p_an - g) / np.maximum(g, 1e-9)
    print(f"    ANALYTIC      mean={ape_an.mean():7.2f}%  median={np.median(ape_an):7.2f}%  p90={np.percentile(ape_an,90):7.2f}%")
    print(f"    AN+RESIDUAL   mean={ape_pred.mean():7.2f}%  median={np.median(ape_pred):7.2f}%  p90={np.percentile(ape_pred,90):7.2f}%")

    # Stratified
    print()
    print("    stratified by golden c_pair (fF):")
    edges = [0, 0.001, 0.005, 0.01, 0.05, 0.1, np.inf]
    labels = ["<0.001", "0.001-0.005", "0.005-0.01", "0.01-0.05", "0.05-0.1", ">=0.1"]
    idx = np.clip(np.digitize(g, edges) - 1, 0, len(labels) - 1)
    for i, lb in enumerate(labels):
        m = idx == i
        if m.sum() > 0:
            print(f"      {lb:>14s}: n={m.sum():>6d}  AN.mean={ape_an[m].mean():7.2f}%  PRED.mean={ape_pred[m].mean():7.2f}%  PRED.median={np.median(ape_pred[m]):7.2f}%")

    # Save metrics
    metrics = pd.Series({
        "n_pred": len(df),
        "n_golden": len(g_pairs),
        "n_common": int(len(df_g)),
        "coverage_pct": 100*len(df_g)/len(g_pairs),
        "analytic_mape_mean": float(ape_an.mean()),
        "analytic_mape_median": float(np.median(ape_an)),
        "pred_mape_mean": float(ape_pred.mean()),
        "pred_mape_median": float(np.median(ape_pred)),
        "pred_mape_p90": float(np.percentile(ape_pred, 90)),
    })
    metrics.to_csv(OUT_DIR / "tv80s_smoke_metrics.csv", header=False)
    print()
    print(f"saved {OUT_DIR/'tv80s_smoke_metrics.csv'}")


if __name__ == "__main__":
    main()
