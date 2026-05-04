#!/usr/bin/env python3
"""diag_v10_per_pair.py — measure per-pair coupling MAPE on v10 SPEF (tv80s).

Inputs:
  - golden parquet: /data/PINNPEX/data/processed_v3/intel22/per_pair_golden/intel22_tv80s_f3.parquet
  - predicted SPEF: pex_v3/output/spef_e2e_fast_v10/intel22_tv80s_f3_HERO_v10.spef

Output: per-pair MAPE distribution + stratified by golden c_pair magnitude
(matches sister codebase per_pair_compare.py logic but with current v10 SPEF).
"""
from __future__ import annotations
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

GOLDEN_PARQUET = Path("/data/PINNPEX/data/processed_v3/intel22/per_pair_golden/intel22_tv80s_f3.parquet")
PRED_SPEF = Path("/home/jslee/projects/PINNPEX/pex_v3/output/spef_e2e_fast_v10/intel22_tv80s_f3_HERO_v10.spef")
OUT_DIR = Path("/home/jslee/projects/PINNPEX/pex_v3/joint_pareto/experiments/exp_013_per_pair")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_pred_per_pair(spef_path: Path) -> dict:
    """Streaming parse — return dict {(target, agg) sorted: c_pair_fF}."""
    pairs = defaultdict(float)
    current = None
    in_cap = False
    pair_sums_for_net = defaultdict(float)
    with open(spef_path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("//"):
                continue
            if s.startswith("*D_NET"):
                # Flush previous
                if current is not None:
                    for agg, v in pair_sums_for_net.items():
                        if agg == current:
                            continue
                        if v < 1e-6:
                            continue
                        key = tuple(sorted([current, agg]))
                        pairs[key] += v
                pair_sums_for_net = defaultdict(float)
                current = s.split()[1]
                in_cap = False
                continue
            if s == "*CAP":
                in_cap = True
                continue
            if s == "*RES" or s.startswith("*END"):
                if current is not None:
                    for agg, v in pair_sums_for_net.items():
                        if agg == current:
                            continue
                        if v < 1e-6:
                            continue
                        key = tuple(sorted([current, agg]))
                        pairs[key] += v
                pair_sums_for_net = defaultdict(float)
                in_cap = False
                if s.startswith("*END"):
                    current = None
                continue
            if not in_cap or current is None:
                continue
            parts = s.split()
            if len(parts) >= 4 and ":" in parts[2]:
                # Coupling: id node_a node_b c_val   where parts[1] is target node, parts[2] is aggressor node
                try:
                    c_val = float(parts[3])
                except ValueError:
                    continue
                agg_node = parts[2]
                if ":" in agg_node:
                    head, _, _ = agg_node.rpartition(":")
                    agg_net = head
                else:
                    agg_net = agg_node
                pair_sums_for_net[agg_net] += c_val
    return dict(pairs)


def main():
    print(f">>> Loading golden parquet: {GOLDEN_PARQUET}")
    gdf = pd.read_parquet(GOLDEN_PARQUET)
    # Symmetric key
    gdf["key"] = gdf.apply(lambda r: tuple(sorted([r["target_net"], r["aggressor_net"]])), axis=1)
    g_pairs = gdf.groupby("key")["c_pair_fF"].sum().to_dict()
    print(f"    golden pairs (symmetric-merged): {len(g_pairs):,}")

    print(f">>> Parsing predicted SPEF: {PRED_SPEF}")
    p_pairs = parse_pred_per_pair(PRED_SPEF)
    print(f"    predicted pairs: {len(p_pairs):,}")

    common = set(g_pairs.keys()) & set(p_pairs.keys())
    print(f"    common pairs: {len(common):,} ({100*len(common)/len(g_pairs):.1f}% golden coverage)")

    # Compute per-pair MAPE
    g_arr = np.array([g_pairs[k] for k in common])
    p_arr = np.array([p_pairs[k] for k in common])
    nz = g_arr > 1e-6
    ape = 100 * np.abs(p_arr - g_arr) / np.maximum(g_arr, 1e-6)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, nz.sum(), nz.sum())
        boots.append(ape[nz][idx].mean())
    lo, hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)

    print()
    print(f"=== v10 per-pair coupling MAPE ({nz.sum()} common pairs, golden > 1e-6) ===")
    print(f"  mean    = {ape[nz].mean():.3f}%  CI=[{lo:.3f}, {hi:.3f}]")
    print(f"  median  = {np.median(ape[nz]):.3f}%")
    print(f"  p90     = {np.percentile(ape[nz], 90):.2f}%")
    print(f"  p99     = {np.percentile(ape[nz], 99):.2f}%")
    rel = (p_arr - g_arr) / np.maximum(g_arr, 1e-6) * 100
    print(f"  signed bias mean = {rel[nz].mean():+.2f}%")

    print()
    print("=== Stratified by golden c_pair (fF) ===")
    edges = [0, 0.001, 0.005, 0.01, 0.05, 0.1, np.inf]
    labels = ["<0.001", "0.001-0.005", "0.005-0.01", "0.01-0.05", "0.05-0.1", ">=0.1"]
    idx = np.clip(np.digitize(g_arr, edges) - 1, 0, len(labels) - 1)
    for i, lb in enumerate(labels):
        m = (idx == i) & nz
        if m.sum() > 0:
            print(f"  {lb:>14s}: n={m.sum():>6d}  mape_mean={ape[m].mean():7.2f}%  median={np.median(ape[m]):6.2f}%  p90={np.percentile(ape[m], 90):7.2f}%")

    # Save
    out_csv = OUT_DIR / "v10_per_pair_metrics.csv"
    pd.Series({
        "n_pred": len(p_pairs),
        "n_golden": len(g_pairs),
        "n_common": len(common),
        "coverage_pct": 100*len(common)/len(g_pairs),
        "mape_mean": float(ape[nz].mean()),
        "mape_median": float(np.median(ape[nz])),
        "mape_p90": float(np.percentile(ape[nz], 90)),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "signed_bias_mean": float(rel[nz].mean()),
    }).to_csv(out_csv, header=False)
    print(f"\nsaved {out_csv}")


if __name__ == "__main__":
    main()
