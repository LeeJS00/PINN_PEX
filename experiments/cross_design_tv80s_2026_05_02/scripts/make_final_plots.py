"""Generate publication-quality plots for the final report.

Plots:
  1. r2_scatter.png — log-log predicted vs true with R^2, identity line, density coloring
  2. mape_histogram.png — APE distribution with median/mean/p90/p99 markers
  3. stratified_mape.png — bar chart of mean MAPE by cap bucket
  4. ensemble_evolution.png — line chart of MAPE across passes 1-7
  5. r2_per_bucket.png — small-multiples scatter, one subplot per cap bucket

All plots use the canonical best ensemble (super_ensemble_test.csv = 7.9852% MAPE).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent
sys.path.insert(0, str(_WS))

from configs import cfg


def r2(y, yhat):
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def r2_log(y, yhat):
    ly = np.log10(np.clip(y, 1e-4, None))
    lh = np.log10(np.clip(yhat, 1e-4, None))
    return r2(ly, lh)


def make_plots():
    rd = cfg.REPORTS_DIR
    plots_dir = rd / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(rd / "super_ensemble_test.csv")
    y = df["y_true"].to_numpy()
    yhat = df["y_pred"].to_numpy()
    ape = 100 * np.abs(yhat - y) / np.maximum(y, 1e-3)

    R2 = r2(y, yhat)
    R2_log = r2_log(y, yhat)
    mape_mean = ape.mean()
    mape_median = np.median(ape)
    mape_p90 = np.percentile(ape, 90)
    mape_p99 = np.percentile(ape, 99)

    print(f"R^2 (linear): {R2:.4f}")
    print(f"R^2 (log10):  {R2_log:.4f}")
    print(f"MAPE mean: {mape_mean:.3f}%, median: {mape_median:.3f}%, p90: {mape_p90:.2f}%, p99: {mape_p99:.2f}%")

    # === Plot 1: log-log R^2 scatter ===
    fig, ax = plt.subplots(figsize=(7, 6.5))
    h = ax.hist2d(np.log10(np.clip(y, 1e-4, None)),
                  np.log10(np.clip(yhat, 1e-4, None)),
                  bins=80, cmap="viridis", norm=colors.LogNorm(vmin=1, vmax=None))
    cb = plt.colorbar(h[3], ax=ax, label="net count (log scale)")
    lo = min(np.log10(y.min()), np.log10(yhat.min()))
    hi = max(np.log10(y.max()), np.log10(yhat.max()))
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="y = ŷ", alpha=0.8)
    ax.set_xlabel("log₁₀(true total cap, fF)")
    ax.set_ylabel("log₁₀(predicted total cap, fF)")
    ax.set_title(f"Cross-design tv80s: predicted vs true total cap\n"
                 f"n={len(y):,}  R²(log)={R2_log:.4f}  R²(lin)={R2:.4f}  "
                 f"MAPE={mape_mean:.2f}%")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "r2_scatter.png", dpi=140)
    plt.close()
    print(f"saved {plots_dir / 'r2_scatter.png'}")

    # === Plot 2: MAPE histogram ===
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 50, 101)
    ax.hist(np.clip(ape, 0, 50), bins=bins, color="steelblue", edgecolor="white", linewidth=0.3)
    for v, name, color in [(mape_median, "median", "green"),
                            (mape_mean, "mean", "red"),
                            (mape_p90, "p90", "orange"),
                            (mape_p99, "p99", "purple")]:
        ax.axvline(v, color=color, lw=1.5, ls="--", label=f"{name}={v:.2f}%")
    ax.set_xlabel("absolute percentage error (%)")
    ax.set_ylabel("net count")
    ax.set_title(f"APE distribution on tv80s test (n={len(y):,})\n"
                 "(values >50% clipped to last bin)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "mape_histogram.png", dpi=140)
    plt.close()
    print(f"saved {plots_dir / 'mape_histogram.png'}")

    # === Plot 3: stratified MAPE bar chart ===
    bucket_edges = [0, 0.1, 0.2, 0.5, 1.0, 5.0, np.inf]
    bucket_labels = ["<0.1", "0.1-0.2", "0.2-0.5", "0.5-1", "1-5", "≥5"]
    bucket_idx = np.digitize(y, bucket_edges) - 1
    bucket_idx = np.clip(bucket_idx, 0, len(bucket_labels) - 1)
    means = []; medians = []; counts = []
    for i in range(len(bucket_labels)):
        m = bucket_idx == i
        if m.sum() == 0:
            means.append(0); medians.append(0); counts.append(0)
        else:
            means.append(ape[m].mean())
            medians.append(np.median(ape[m]))
            counts.append(m.sum())
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(bucket_labels))
    w = 0.4
    bars1 = ax.bar(x - w/2, means, w, label="mean MAPE", color="steelblue", edgecolor="white")
    bars2 = ax.bar(x + w/2, medians, w, label="median MAPE", color="coral", edgecolor="white")
    for i, (m, n) in enumerate(zip(means, counts)):
        ax.text(i, max(means[i], medians[i]) + 0.3, f"n={n:,}", ha="center", fontsize=9, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels)
    ax.set_xlabel("true total cap (fF) bucket")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Stratified MAPE by cap magnitude — ENS_super_ensemble")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(plots_dir / "stratified_mape.png", dpi=140)
    plt.close()
    print(f"saved {plots_dir / 'stratified_mape.png'}")

    # === Plot 4: ensemble evolution across passes ===
    evolution = [
        ("LGBM ×1\n(v1, leak)", 7.7),
        ("LGBM ×1\n(leak fix)", 9.6),
        ("LGBM ×5+ENS\n(v2)", 9.65),
        ("v2+ResMLP/GBDT\n30 mdl", 9.14),
        ("v3+ResMLP\n×5 (36)", 8.84),
        ("+GBDT/CAT\n+nova val (62)", 8.66),
        ("+DeepSet ×5\n(70)", 8.40),
        ("+DeepSet ×10\n(75)", 8.40),
        ("Pass 3:\nval_tuned", 8.047),
        ("Pass 5:\n1D b=12", 7.995),
        ("Pass 6:\n1D mean", 7.9931),
        ("Pass 7:\n1D+2D super", 7.9852),
    ]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(evolution))
    vals = [e[1] for e in evolution]
    labels = [e[0] for e in evolution]
    ax.plot(x, vals, "o-", lw=1.7, markersize=7, color="darkblue")
    ax.axhline(4.0, color="green", ls=":", lw=1.5, label="goal: 4% MAPE")
    ax.axhline(7.9852, color="red", ls="--", lw=1.5, label="achieved: 7.9852%")
    ax.fill_between(x, vals, vals[-1], where=[v > vals[-1] for v in vals],
                    alpha=0.1, color="blue", interpolate=True)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.2f}%", (i, v), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, fontsize=8.5)
    ax.set_ylabel("Test mean MAPE (%)")
    ax.set_title("MAPE evolution: feature-engineering + ensembling vs the 4% goal")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(3.0, 10.5)
    plt.tight_layout()
    plt.savefig(plots_dir / "ensemble_evolution.png", dpi=140)
    plt.close()
    print(f"saved {plots_dir / 'ensemble_evolution.png'}")

    # === Plot 5: per-bucket scatter small-multiples ===
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for i, (lo_, hi_, label) in enumerate([
        (0, 0.1, "<0.1 fF"), (0.1, 0.2, "0.1-0.2 fF"), (0.2, 0.5, "0.2-0.5 fF"),
        (0.5, 1.0, "0.5-1 fF"), (1.0, 5.0, "1-5 fF"), (5.0, np.inf, "≥5 fF")]):
        ax = axes[i // 3, i % 3]
        m = (y >= lo_) & (y < hi_)
        if m.sum() == 0:
            ax.set_visible(False); continue
        yi = y[m]; yh = yhat[m]
        if len(yi) > 1:
            r2_b = r2(yi, yh)
            mape_b = (100 * np.abs(yh - yi) / np.maximum(yi, 1e-3)).mean()
        else:
            r2_b = float("nan"); mape_b = float("nan")
        ax.scatter(yi, yh, s=8, alpha=0.4, color="steelblue", edgecolors="none")
        if len(yi) > 0:
            mn = min(yi.min(), yh.min())
            mx = max(yi.max(), yh.max())
            ax.plot([mn, mx], [mn, mx], "r--", lw=1, alpha=0.6)
        ax.set_xlabel("true (fF)")
        ax.set_ylabel("predicted (fF)")
        ax.set_title(f"{label}  n={m.sum()}  R²={r2_b:.3f}  MAPE={mape_b:.2f}%", fontsize=10)
        ax.grid(True, alpha=0.3)
        if hi_ > 1:
            ax.set_xscale("log"); ax.set_yscale("log")
    plt.suptitle("Per-bucket predicted vs true scatter — ENS_super_ensemble", fontsize=12)
    plt.tight_layout()
    plt.savefig(plots_dir / "per_bucket_scatter.png", dpi=140)
    plt.close()
    print(f"saved {plots_dir / 'per_bucket_scatter.png'}")

    # === Plot 6: residual analysis ===
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # 6a: residual % vs true cap (log-x)
    ax = axes[0]
    resid_pct = 100 * (yhat - y) / np.maximum(y, 1e-3)
    ax.scatter(y, resid_pct, s=6, alpha=0.3, color="steelblue", edgecolors="none")
    ax.axhline(0, color="red", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("true total cap (fF)")
    ax.set_ylabel("relative residual (predicted − true) / true × 100%")
    ax.set_title("Signed residual vs true cap")
    ax.set_ylim(-100, 100)
    ax.grid(True, alpha=0.3)
    # 6b: bias by bucket
    ax = axes[1]
    bucket_means = []
    for i in range(len(bucket_labels)):
        m = bucket_idx == i
        if m.sum() == 0:
            bucket_means.append(0)
        else:
            bucket_means.append((100 * (yhat[m] - y[m]) / np.maximum(y[m], 1e-3)).mean())
    colors_b = ["red" if v < 0 else "blue" for v in bucket_means]
    ax.bar(bucket_labels, bucket_means, color=colors_b, edgecolor="white", alpha=0.8)
    ax.axhline(0, color="black", lw=1)
    for i, v in enumerate(bucket_means):
        ax.text(i, v + (1 if v >= 0 else -1), f"{v:+.1f}%", ha="center", fontsize=9)
    ax.set_xlabel("true total cap bucket (fF)")
    ax.set_ylabel("mean signed bias (%)")
    ax.set_title("Per-bucket bias (negative = under-prediction)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(plots_dir / "residual_analysis.png", dpi=140)
    plt.close()
    print(f"saved {plots_dir / 'residual_analysis.png'}")

    # Save metrics summary
    summary = {
        "n_test": len(y),
        "mape_mean": mape_mean,
        "mape_median": mape_median,
        "mape_p90": mape_p90,
        "mape_p99": mape_p99,
        "r2_log": R2_log,
        "r2_linear": R2,
    }
    pd.Series(summary).to_csv(rd / "final_metrics.csv", header=False)
    print(f"saved {rd / 'final_metrics.csv'}")
    print(f"\n=== Final metrics ===")
    for k, v in summary.items():
        print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    make_plots()
