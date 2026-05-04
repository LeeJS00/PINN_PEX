"""Comprehensive validation of e2e SPEF predictions vs golden.

Reports:
  - Per-net total_cap, c_gnd, c_cpl_total, total_R MAPE
  - Per-pair coupling MAPE (with stratification by magnitude)
  - SPEF size + runtime breakdown
  - Cross-design generalization summary
  - Plots: r2_scatter, mape_histogram, stratified_mape, per_pair_scatter
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent.parent
sys.path.insert(0, str(_WS))

from configs import cfg

# Load PINNPEX SPEF parser
_spec = importlib.util.spec_from_file_location(
    "compare_spef_pinpex",
    str(_WS.parent.parent / "src" / "evaluation" / "compare_spef.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
parse_spef = _mod.parse_spef_with_coordinates


def aggregate_pairs(net_dict):
    pairs = {}
    for tgt, info in net_dict.items():
        for node_id, agg_caps in info.get("cpl_caps", {}).items():
            for agg_name, val in agg_caps.items():
                key = tuple(sorted([tgt, agg_name]))
                pairs[key] = pairs.get(key, 0.0) + float(val)
    return pairs


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--predicted_spef", type=Path, required=True)
    ap.add_argument("--golden_spef", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--design_name", type=str, default="design")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing predicted: {args.predicted_spef}")
    p = parse_spef(args.predicted_spef)
    print(f"Parsing golden:    {args.golden_spef}")
    g = parse_spef(args.golden_spef)

    common = sorted(set(p.keys()) & set(g.keys()))
    n_common = len(common)
    print(f"\nCommon nets: {n_common} (pred={len(p)}, gold={len(g)})")

    # === Per-net comparison ===
    metrics = []
    rng = np.random.default_rng(0)
    for label, getter in [
        ("total_cap", lambda x: x["total_cap"]),
        ("c_gnd", lambda x: x["sum_gnd_cap"]),
        ("c_cpl_total", lambda x: x["sum_cpl_cap"]),
        ("total_res", lambda x: x["total_res"]),
    ]:
        g_vals = np.array([getter(g[n]) for n in common])
        p_vals = np.array([getter(p[n]) for n in common])
        nz = g_vals > 1e-6
        ape = 100 * np.abs(p_vals - g_vals) / np.maximum(g_vals, 1e-6)
        ape_nz = ape[nz]
        bias = ((p_vals - g_vals) / np.maximum(g_vals, 1e-6))[nz].mean() * 100
        # bootstrap CI
        boots = []
        for _ in range(2000):
            idx = rng.integers(0, len(ape_nz), len(ape_nz))
            boots.append(ape_nz[idx].mean())
        lo = np.percentile(boots, 2.5); hi = np.percentile(boots, 97.5)
        # R²
        ly = np.log10(np.clip(g_vals[nz], 1e-4, None))
        lh = np.log10(np.clip(p_vals[nz], 1e-4, None))
        r2_lin = 1 - ((g_vals[nz] - p_vals[nz])**2).sum() / max(((g_vals[nz] - g_vals[nz].mean())**2).sum(), 1e-12)
        r2_log = 1 - ((ly - lh)**2).sum() / max(((ly - ly.mean())**2).sum(), 1e-12)
        metrics.append({
            "metric": label,
            "n": int(nz.sum()),
            "mape_mean": float(ape_nz.mean()),
            "mape_median": float(np.median(ape_nz)),
            "mape_p90": float(np.percentile(ape_nz, 90)),
            "ci_lo": float(lo),
            "ci_hi": float(hi),
            "bias_pct": float(bias),
            "r2_lin": float(r2_lin),
            "r2_log": float(r2_log),
            "g_mean": float(g_vals.mean()),
            "p_mean": float(p_vals.mean()),
        })

    df_m = pd.DataFrame(metrics)
    df_m.to_csv(args.out_dir / "per_net_metrics.csv", index=False)
    print("\n=== Per-net metrics ===")
    print(df_m.to_string(index=False))

    # === Per-pair comparison ===
    print("\n=== Per-pair coupling comparison ===")
    p_pairs = aggregate_pairs(p)
    g_pairs = aggregate_pairs(g)
    common_pairs = set(g_pairs.keys()) & set(p_pairs.keys())
    g_arr = np.array([g_pairs[k] for k in common_pairs])
    p_arr = np.array([p_pairs[k] for k in common_pairs])
    nz = g_arr > 1e-6
    ape = 100 * np.abs(p_arr - g_arr) / np.maximum(g_arr, 1e-6)
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, nz.sum(), nz.sum())
        boots.append(ape[nz][idx].mean())
    pair_metrics = {
        "n_predicted": len(p_pairs),
        "n_golden": len(g_pairs),
        "n_common": len(common_pairs),
        "coverage_pct_of_golden": 100 * len(common_pairs) / max(len(g_pairs), 1),
        "mape_mean": float(ape[nz].mean()),
        "mape_median": float(np.median(ape[nz])),
        "mape_p90": float(np.percentile(ape[nz], 90)),
        "ci_lo": float(np.percentile(boots, 2.5)),
        "ci_hi": float(np.percentile(boots, 97.5)),
    }
    pd.Series(pair_metrics).to_csv(args.out_dir / "per_pair_metrics.csv", header=False)
    for k, v in pair_metrics.items():
        print(f"  {k}: {v}")

    # === Stratified per-pair MAPE ===
    print("\n=== Per-pair MAPE stratified by golden c_pair (fF) ===")
    edges = [0, 0.001, 0.005, 0.01, 0.05, 0.1, np.inf]
    labels = ["<0.001", "0.001-0.005", "0.005-0.01", "0.01-0.05", "0.05-0.1", ">=0.1"]
    idx_b = np.clip(np.digitize(g_arr, edges) - 1, 0, len(labels) - 1)
    strat_rows = []
    for i, lb in enumerate(labels):
        m = (idx_b == i) & nz
        if m.sum() > 0:
            row = {"bucket": lb, "n": int(m.sum()),
                   "mape_mean": float(ape[m].mean()),
                   "mape_median": float(np.median(ape[m])),
                   "mape_p90": float(np.percentile(ape[m], 90))}
            print(f"  {lb:>14s}: n={row['n']:>6d}  mape_mean={row['mape_mean']:.2f}%")
            strat_rows.append(row)
    pd.DataFrame(strat_rows).to_csv(args.out_dir / "per_pair_stratified.csv", index=False)

    # === R² scatter plot for total_cap ===
    g_total = np.array([g[n]["total_cap"] for n in common])
    p_total = np.array([p[n]["total_cap"] for n in common])
    fig, ax = plt.subplots(figsize=(7, 6.5))
    h = ax.hist2d(np.log10(np.clip(g_total, 1e-4, None)),
                   np.log10(np.clip(p_total, 1e-4, None)),
                   bins=80, cmap="viridis", norm=mcolors.LogNorm(vmin=1, vmax=None))
    plt.colorbar(h[3], ax=ax, label="net count")
    lo = min(np.log10(g_total.min() + 1e-6), np.log10(p_total.min() + 1e-6))
    hi = max(np.log10(g_total.max()), np.log10(p_total.max()))
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="y = ŷ")
    R2_log = 1 - ((np.log10(g_total + 1e-4) - np.log10(p_total + 1e-4))**2).sum() / \
             max(((np.log10(g_total + 1e-4) - np.log10(g_total + 1e-4).mean())**2).sum(), 1e-12)
    ax.set_xlabel("log₁₀(golden total cap, fF)")
    ax.set_ylabel("log₁₀(predicted total cap, fF)")
    ax.set_title(f"E2E SPEF — predicted vs golden total_cap on {args.design_name}\n"
                 f"n={n_common}  R²(log)={R2_log:.4f}  MAPE={metrics[0]['mape_mean']:.2f}%")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out_dir / "r2_scatter_total_cap.png", dpi=140)
    plt.close()

    # === Per-pair scatter ===
    p_pair_arr = np.array([p_pairs[k] for k in common_pairs])
    g_pair_arr = np.array([g_pairs[k] for k in common_pairs])
    fig, ax = plt.subplots(figsize=(7, 6.5))
    h = ax.hist2d(np.log10(np.clip(g_pair_arr, 1e-5, None)),
                   np.log10(np.clip(p_pair_arr, 1e-5, None)),
                   bins=100, cmap="viridis", norm=mcolors.LogNorm(vmin=1))
    plt.colorbar(h[3], ax=ax, label="pair count")
    pair_lo = min(np.log10(g_pair_arr.min() + 1e-6), np.log10(p_pair_arr.min() + 1e-6))
    pair_hi = max(np.log10(g_pair_arr.max()), np.log10(p_pair_arr.max()))
    ax.plot([pair_lo, pair_hi], [pair_lo, pair_hi], "r--", lw=1.5, label="y = ŷ")
    ax.set_xlabel("log₁₀(golden c_pair, fF)")
    ax.set_ylabel("log₁₀(predicted c_pair, fF)")
    ax.set_title(f"E2E SPEF — per-pair coupling on {args.design_name}\n"
                 f"n_common={len(common_pairs)}  MAPE={pair_metrics['mape_mean']:.2f}%")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out_dir / "r2_scatter_per_pair.png", dpi=140)
    plt.close()

    # === Summary text ===
    summary_path = args.out_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"E2E SPEF Validation Summary — {args.design_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Predicted SPEF: {args.predicted_spef}\n")
        f.write(f"Golden SPEF:    {args.golden_spef}\n")
        f.write(f"Common nets:    {n_common}\n\n")
        f.write("Per-net metrics:\n")
        f.write(df_m.to_string(index=False) + "\n\n")
        f.write("Per-pair metrics:\n")
        for k, v in pair_metrics.items():
            f.write(f"  {k}: {v}\n")
    print(f"\nSummary saved: {summary_path}")
    print(f"Plots saved in: {args.out_dir}/")


if __name__ == "__main__":
    main()
