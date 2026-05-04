"""Compare per-(target, aggressor) coupling cap between predicted and golden SPEF.

Output:
  - per-pair MAPE (mean / median / p90), with bootstrap CI
  - per-pair coverage (how many golden pairs are in our prediction)
  - top mispredicted pairs
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent.parent
sys.path.insert(0, str(_WS))

from configs import cfg

# Load PINNPEX root's compare_spef as parser
_spec = importlib.util.spec_from_file_location(
    "compare_spef_pinpex",
    str(_WS.parent.parent / "src" / "evaluation" / "compare_spef.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
parse_spef_with_coordinates = _mod.parse_spef_with_coordinates


GOLDEN = Path("/home/jslee/projects/PEX_SSL/golden_data/spef_data/intel22/intel22_tv80s_f3_starrc.spef")
PRED = cfg.OUTPUT_DIR / "spef_e2e" / "tv80s_predicted_v1.spef"


def aggregate_pairs(net_dict):
    """Returns dict {(target, aggressor): c_pair} aggregating over all node-level entries."""
    pairs = {}
    for tgt, info in net_dict.items():
        for node_id, agg_caps in info.get("cpl_caps", {}).items():
            for agg_name, val in agg_caps.items():
                key = tuple(sorted([tgt, agg_name]))  # symmetric
                pairs[key] = pairs.get(key, 0.0) + float(val)
    # If both directions appear, sum and divide by 2 for true symmetric coupling cap?
    # Actually golden lists each pair once; if duplicated due to symmetric storage,
    # divide by 2. We sort the key first — duplicates are identical.
    return pairs


def main():
    print(f"Parsing predicted: {PRED}")
    p = parse_spef_with_coordinates(PRED)
    print(f"Parsing golden:   {GOLDEN}")
    g = parse_spef_with_coordinates(GOLDEN)

    p_pairs = aggregate_pairs(p)
    g_pairs = aggregate_pairs(g)
    print(f"\nPredicted pairs: {len(p_pairs)}")
    print(f"Golden pairs:    {len(g_pairs)}")

    # Coverage: golden pairs we have in prediction
    common = set(g_pairs.keys()) & set(p_pairs.keys())
    print(f"Common pairs:    {len(common)} ({100*len(common)/len(g_pairs):.1f}% of golden)")

    # MAPE on common pairs
    g_arr = np.array([g_pairs[k] for k in common])
    p_arr = np.array([p_pairs[k] for k in common])
    nz = g_arr > 1e-6
    ape = 100 * np.abs(p_arr - g_arr) / np.maximum(g_arr, 1e-6)

    rng = np.random.default_rng(0)
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, nz.sum(), nz.sum())
        boots.append(ape[nz][idx].mean())
    lo = np.percentile(boots, 2.5); hi = np.percentile(boots, 97.5)

    print(f"\n=== Per-pair coupling MAPE on {nz.sum()} common pairs ===")
    print(f"  mean={ape[nz].mean():.3f}%  CI=[{lo:.3f}, {hi:.3f}]")
    print(f"  median={np.median(ape[nz]):.3f}%")
    print(f"  p90={np.percentile(ape[nz], 90):.2f}%")
    print(f"  p99={np.percentile(ape[nz], 99):.2f}%")

    # Bias
    rel = (p_arr - g_arr) / np.maximum(g_arr, 1e-6) * 100
    print(f"  mean signed bias: {rel[nz].mean():+.2f}%")

    # Stratified by golden c_pair magnitude
    print("\n=== Per-pair MAPE stratified by golden c_pair (fF) ===")
    edges = [0, 0.001, 0.005, 0.01, 0.05, 0.1, np.inf]
    labels = ["<0.001", "0.001-0.005", "0.005-0.01", "0.01-0.05", "0.05-0.1", ">=0.1"]
    idx = np.clip(np.digitize(g_arr, edges) - 1, 0, len(labels) - 1)
    for i, lb in enumerate(labels):
        m = (idx == i) & nz
        if m.sum() > 0:
            print(f"  {lb:>14s}: n={m.sum():>6d}  mape_mean={ape[m].mean():.2f}%  median={np.median(ape[m]):.2f}%")

    # Coverage gap analysis
    only_in_golden = set(g_pairs.keys()) - set(p_pairs.keys())
    only_in_pred = set(p_pairs.keys()) - set(g_pairs.keys())
    print(f"\n=== Coverage gaps ===")
    print(f"  pairs only in golden: {len(only_in_golden)} ({100*len(only_in_golden)/len(g_pairs):.1f}%)")
    print(f"  pairs only in pred:   {len(only_in_pred)} ({100*len(only_in_pred)/len(p_pairs):.1f}%)")
    if only_in_golden:
        miss_vals = np.array([g_pairs[k] for k in list(only_in_golden)[:10000]])
        print(f"    mass missed (sum c_pair): {miss_vals.sum():.3f} fF "
              f"(% of total golden cpl mass: {100*miss_vals.sum()/sum(g_pairs.values()):.2f}%)")

    # Save metrics CSV
    metrics = {
        "n_pairs_predicted": len(p_pairs),
        "n_pairs_golden": len(g_pairs),
        "n_pairs_common": len(common),
        "coverage_pct": 100 * len(common) / len(g_pairs),
        "mape_mean": ape[nz].mean(),
        "mape_median": np.median(ape[nz]),
        "mape_p90": np.percentile(ape[nz], 90),
        "ci_lo": lo,
        "ci_hi": hi,
        "bias_mean_pct": rel[nz].mean(),
    }
    pd.Series(metrics).to_csv(cfg.REPORTS_DIR / "spef_e2e_pair_metrics.csv", header=False)
    print(f"\nsaved {cfg.REPORTS_DIR / 'spef_e2e_pair_metrics.csv'}")


if __name__ == "__main__":
    main()
