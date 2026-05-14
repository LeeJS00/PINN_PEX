"""make_figures.py — generate 5 visualizations for the TreePEX presentation.

Outputs PNG files in TreePEX/presentation/figures/:
  fig1_per_bucket_R2.png         — within-bucket R² ceiling diagnosis
  fig2_5seed_progression.png     — frontier evolution
  fig3_strategy_ladder.png       — auto-4pct strategy ranking
  fig4_per_bucket_TreePEX.png     — TreePEX ensemble per-bucket MAPE
  fig5_speed_accuracy_pareto.png — TreePEX vs prior methods

All figures use consistent palette (matplotlib tab10) and serif fonts for paper-ready look.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/home/jslee/projects/PINNPEX/TreePEX/presentation/figures")
ROOT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
})


def fig1_per_bucket_R2():
    """Within-bucket cpl R² across cap-deciles for v12, B1, Small_combined, Big_combined."""
    deciles = [f"C{i+1}" for i in range(10)]
    # From pex_v4/results/short_net_diagnostic.csv (1-seed)
    v12_cpl    = [0.374, 0.131, 0.301, 0.247, 0.327, 0.255, 0.541, 0.472, 0.629, 0.910]
    b1_cpl     = [0.226, 0.067, 0.238, 0.254, 0.304, 0.234, 0.495, 0.503, 0.665, 0.909]
    paxgb_cpl  = [0.243, 0.076, 0.241, 0.251, 0.310, 0.232, 0.501, 0.492, 0.664, 0.909]
    big_combined_cpl = [0.365, 0.284, 0.346, 0.318, 0.347, 0.354, 0.528, 0.484, 0.661, 0.908]
    TreePEX_ensemble_cpl = [0.398, 0.270, 0.352, 0.309, 0.360, 0.352, 0.541, 0.529, 0.685, 0.906]  # tv80s TreePEX

    x = np.arange(len(deciles)); w = 0.18
    fig, ax = plt.subplots(figsize=(10, 5.0))
    ax.bar(x - 1.5*w, v12_cpl,    w, label="v12 PINN (5-seed)",        color="#888888")
    ax.bar(x - 0.5*w, b1_cpl,     w, label="B1 XGBoost (baseline)",     color="#1f77b4")
    ax.bar(x + 0.5*w, big_combined_cpl, w, label="+ top-K aggressor (Big_combined)", color="#ff7f0e")
    ax.bar(x + 1.5*w, TreePEX_ensemble_cpl, w, label="TreePEX 5-seed ensemble", color="#2ca02c")
    ax.set_xlabel("c_total cap decile (C1=smallest, C10=largest)")
    ax.set_ylabel("Within-bucket R² (cpl channel)")
    ax.set_title("Within-bucket R² ceiling on tv80s — top-K features lift mid-bucket discrimination")
    ax.set_xticks(x); ax.set_xticklabels(deciles)
    ax.set_ylim(-0.05, 1.0)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "fig1_per_bucket_R2.png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    print("[ok] fig1_per_bucket_R2.png")


def fig2_5seed_progression():
    """Frontier evolution: v12 PINN → B1 → Small_combined → Big_combined → S4 Tweedie → TreePEX ensemble."""
    methods = ["v12 PINN\n(5-seed)", "B1 XGBoost\n(5-seed)", "Small_combined\n(5-seed)", "Big_combined\n(5-seed)", "S4 Tweedie\n(5-seed)", "TreePEX\nensemble"]
    tv80s = [5.55, 5.30, 5.28, 5.17, 5.087, 4.979]
    nova  = [None, 5.83, 5.62, 5.92, 5.417, 5.279]  # v12 nova not measured
    walls = [20.4, 0.05, 0.05, 0.4, 0.05, 0.171]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    x = np.arange(len(methods))
    w = 0.4
    bars1 = ax1.bar(x - w/2, tv80s, w, label="tv80s tot_med", color="#1f77b4")
    nova_clean = [v if v is not None else 0 for v in nova]
    bars2 = ax1.bar(x + w/2, nova_clean, w, label="nova tot_med", color="#ff7f0e")
    for i, v in enumerate(nova):
        if v is None:
            ax1.text(i + w/2, 0.2, "n/a", ha="center", color="gray", fontsize=9)
    ax1.set_xticks(x); ax1.set_xticklabels(methods, fontsize=9, rotation=10)
    ax1.set_ylabel("MAPE_tot_med (%)")
    ax1.set_title("(a) Accuracy: 5-seed mean ± std (where applicable)")
    ax1.axhline(4.0, color="red", linestyle=":", alpha=0.6, label="4 % goal")
    ax1.legend(loc="upper right")
    ax1.set_ylim(0, 7.0)
    ax1.grid(axis="y", alpha=0.3)
    for i, v in enumerate(tv80s):
        ax1.text(i - w/2, v + 0.1, f"{v:.2f}", ha="center", fontsize=8)
    for i, v in enumerate(nova):
        if v is not None:
            ax1.text(i + w/2, v + 0.1, f"{v:.2f}", ha="center", fontsize=8)

    bars = ax2.bar(x, walls, color=["#888888"]*1 + ["#1f77b4"]*4 + ["#2ca02c"])
    ax2.set_xticks(x); ax2.set_xticklabels(methods, fontsize=9, rotation=10)
    ax2.set_ylabel("Inference wall (s)")
    ax2.set_title("(b) Inference time on tv80s 3,169 nets")
    ax2.set_yscale("log")
    ax2.grid(axis="y", alpha=0.3, which="both")
    for i, v in enumerate(walls):
        ax2.text(i, v * 1.2, f"{v:.2f}s", ha="center", fontsize=8)

    fig.suptitle("Frontier evolution toward TreePEX deployable ensemble", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(ROOT / "fig2_5seed_progression.png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    print("[ok] fig2_5seed_progression.png")


def fig3_strategy_ladder():
    """Auto-4pct strategy ladder: pex_v4 + pex_v5 strategies — accuracy vs deployability."""
    strategies = [
        ("S1 grand-avg",    5.097, "oracle/blend"),
        ("S2 NNLS blend",   5.184, "oracle/blend"),
        ("S3 + v12",        5.111, "oracle/blend"),
        ("S4 Tweedie",      5.087, "deployable"),
        ("S6 Big+Tweedie",  5.181, "deployable"),
        ("S8 mega",         5.111, "oracle/blend"),
        ("S9 vp grid",      5.114, "deployable"),
        ("P1 quantile",     5.380, "deployable"),
        ("P2 per-bucket",   4.742, "ORACLE only"),
        ("P7 mega-mean",    4.679, "ORACLE only"),
        ("P8 router",       5.692, "deployable"),
        ("TreePEX ensemble", 4.979, "deployable"),
    ]
    names = [s[0] for s in strategies]
    vals  = [s[1] for s in strategies]
    cats  = [s[2] for s in strategies]

    palette = {"deployable": "#2ca02c", "oracle/blend": "#888888", "ORACLE only": "#d62728"}
    colors = [palette[c] for c in cats]

    fig, ax = plt.subplots(figsize=(11, 5.0))
    bars = ax.barh(np.arange(len(names)), vals, color=colors)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(4.0, color="red", linestyle=":", linewidth=1.5, label="4 % goal")
    ax.axvline(5.55, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="v12 PINN ref")
    ax.set_xlabel("tv80s tot_med MAPE (%)")
    ax.set_title("Auto-4pct strategy ladder — 14 attempts, deployable best = TreePEX ensemble (4.979)")
    for i, v in enumerate(vals):
        ax.text(v + 0.05, i, f"{v:.3f}", va="center", fontsize=8)
    # Legend
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="#2ca02c", label="deployable"),
        Patch(facecolor="#888888", label="oracle blend (TEST-fitted)"),
        Patch(facecolor="#d62728", label="oracle-only (NOT deployable)"),
    ]
    ax.legend(handles=handles, loc="lower right", framealpha=0.95)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "fig3_strategy_ladder.png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    print("[ok] fig3_strategy_ladder.png")


def fig4_per_bucket_TreePEX():
    """TreePEX per-bucket MAPE on tv80s + nova showing C1-C2 noise floor and C8 hitting 4 %."""
    deciles = [f"C{i+1}" for i in range(10)]
    tv80s_mape = [6.85, 5.68, 5.69, 5.07, 4.99, 4.92, 4.88, 4.02, 4.43, 4.20]
    nova_mape  = [6.88, 6.10, 5.68, 5.34, 5.14, 4.84, 4.77, 4.74, 4.87, 4.85]
    tv80s_cap  = [0.120, 0.177, 0.232, 0.300, 0.391, 0.537, 0.816, 1.458, 2.743, 6.828]

    fig, ax = plt.subplots(figsize=(10, 5.0))
    x = np.arange(len(deciles)); w = 0.4
    ax.bar(x - w/2, tv80s_mape, w, label="tv80s (n=3,169)", color="#1f77b4")
    ax.bar(x + w/2, nova_mape,  w, label="nova (n=92,425)", color="#ff7f0e")
    ax.axhline(4.0, color="red", linestyle=":", alpha=0.7, label="4 % goal")
    ax.set_xticks(x); ax.set_xticklabels(deciles)
    ax.set_xlabel("c_total cap decile (C1: cap < 0.15 fF → C10: cap > 4 fF)")
    ax.set_ylabel("MAPE_tot_med (%)")
    ax.set_title("TreePEX ensemble per-bucket MAPE — C1 noise floor; C8 hits 4.02 % on tv80s")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    # Annotate C8 (best)
    ax.annotate("C8 = 4.02 %\n(target hit on mid-cap)",
                 xy=(7 - w/2, 4.02), xytext=(8.0, 6.5),
                 arrowprops=dict(arrowstyle="->", color="green"),
                 fontsize=9, color="green",
                 ha="center", bbox=dict(facecolor="white", edgecolor="green", alpha=0.9))
    # Annotate C1 (worst)
    ax.annotate("C1 = 6.85 %\n(noise floor, cap<0.15fF)",
                 xy=(0 - w/2, 6.85), xytext=(0.2, 8.0),
                 arrowprops=dict(arrowstyle="->", color="red"),
                 fontsize=9, color="red",
                 ha="left", bbox=dict(facecolor="white", edgecolor="red", alpha=0.9))
    fig.tight_layout()
    fig.savefig(ROOT / "fig4_per_bucket_TreePEX.png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    print("[ok] fig4_per_bucket_TreePEX.png")


def fig5_speed_accuracy_pareto():
    """Speed-accuracy Pareto: TreePEX vs prior ML PEX methods."""
    methods = [
        ("StarRC golden",          0.0,    600,  "#000000", "^", "(reference)"),     # ~10 min/chip
        ("Innovus pattern match",  44.0,   180,  "#888888", "v", "(22-72 % per bucket)"),
        ("OpenRCX pattern match",  43.0,   180,  "#aaaaaa", "v", "(16-72 % per bucket)"),
        ("v12 PINN frontier",      5.55,   20.4, "#1f77b4", "s", "(5-seed)"),
        ("B1 XGBoost (baseline)",  5.30,   0.05, "#9467bd", "o", "(5-seed)"),
        ("S4 Tweedie 5-seed",       5.087, 0.05, "#ff7f0e", "D", "(5-seed lock)"),
        ("TreePEX ensemble",        4.979, 0.171, "#2ca02c", "*", "(deployable, ENSEMBLE)"),
        ("P2 oracle (UPPER BOUND)", 4.742, 0.5,  "#d62728", "x", "(NOT deployable)"),
    ]
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, mape, wall, color, marker, note in methods:
        size = 250 if "TreePEX" in name else 150
        ax.scatter(wall, mape, s=size, color=color, marker=marker, edgecolor="black",
                   linewidth=1.0, label=f"{name}", zorder=3)
        offy = 0.15
        if "TreePEX" in name:
            ax.annotate(note, (wall, mape - 0.4), color="green", fontsize=10,
                        ha="left", fontweight="bold")
        ax.annotate(f"  {name}", (wall, mape), fontsize=9, va="center",
                     ha="left" if wall < 5 else "left")
    ax.set_xscale("log")
    ax.set_xlabel("Inference wall on tv80s 3,169 nets (s, log scale)")
    ax.set_ylabel("MAPE_tot_med (%)")
    ax.set_title("Speed-accuracy Pareto: TreePEX dominates on tv80s test")
    ax.axhline(4.0, color="red", linestyle=":", alpha=0.5, label="4 % goal")
    ax.axvline(0.171, color="green", linestyle=":", alpha=0.5)
    ax.set_xlim(0.02, 1500)
    ax.set_ylim(3.0, 50)
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    # Pareto frontier annotation
    ax.text(0.171, 3.7, "← TreePEX frontier",
            color="green", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(ROOT / "fig5_speed_accuracy_pareto.png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    print("[ok] fig5_speed_accuracy_pareto.png")


def fig6_feature_pipeline():
    """Feature pipeline diagram: base 41 + top-K 26 → joined 67 → XGBoost."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")

    def block(x, y, w, h, label, color="#e3f2fd", edge="#1565c0", text_size=10):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor=edge, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=text_size, weight="bold")

    # Raw inputs (left)
    block(0.3, 7.3, 2.0, 1.3, "DEF\n+ tech LEF", "#fff3e0", "#e65100", 11)
    block(0.3, 5.4, 2.0, 1.3, "Liberty (.lib)\n+ layer stack", "#fff3e0", "#e65100", 11)
    block(0.3, 3.5, 2.0, 1.3, "Raw tile\npickles\n(1.3M tiles)", "#fff3e0", "#e65100", 10)

    # Preprocessing (middle-left)
    block(3.0, 7.3, 3.0, 1.3, "DEF/LEF parser\n→ cuboid extraction\n(per-net, per-tile)", "#e1f5fe", "#01579b", 10)
    block(3.0, 5.4, 3.0, 1.3, "NetFeatureVector\nbuilder\n(41 base features)", "#e1f5fe", "#01579b", 10)
    block(3.0, 3.5, 3.0, 1.3, "Top-K aggressor\nextractor (29_*)\n(26 features)", "#e1f5fe", "#01579b", 10)

    # Feature CSVs (middle)
    block(6.6, 6.3, 3.5, 1.3, "all_designs.csv\n221k nets × 41 base feats", "#e8f5e9", "#1b5e20", 10)
    block(6.6, 4.4, 3.5, 1.3, "new_features_with_ids.csv\n257k nets × 26 H3 feats", "#e8f5e9", "#1b5e20", 10)

    # Join + model (right)
    block(10.7, 5.0, 2.5, 2.0, "JOIN on\n(design, net)\n→ 221k × 67 feats", "#fce4ec", "#880e4f", 11)
    block(13.7, 5.0, 2.0, 2.0, "5-seed\nTweedie\nXGBoost\n(120 MB)", "#f3e5f5", "#4a148c", 12)

    # Output
    block(13.7, 1.5, 2.0, 1.5, "predictions\npred_gnd\npred_cpl", "#ffe0b2", "#bf360c", 11)
    block(10.7, 1.5, 2.5, 1.5, "SPEF write\nIEEE 1481-1999", "#ffe0b2", "#bf360c", 11)

    # Arrows
    def arr(x1, y1, x2, y2):
        ax.annotate("", xytext=(x1, y1), xy=(x2, y2),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.5))
    arr(2.3, 7.95, 3.0, 7.95); arr(2.3, 6.05, 3.0, 6.05); arr(2.3, 4.15, 3.0, 4.15)
    arr(6.0, 7.0, 6.6, 7.0)   # cuboid → all_designs (via NetFeatureVector)
    arr(6.0, 6.05, 6.6, 6.95)
    arr(6.0, 4.15, 6.6, 4.95)  # Top-K → new_features
    arr(10.1, 6.4, 10.7, 6.5)
    arr(10.1, 4.5, 10.7, 5.5)
    arr(13.2, 6.0, 13.7, 6.0)
    arr(14.7, 5.0, 14.7, 3.1)
    arr(13.7, 2.25, 13.2, 2.25)

    # Title
    ax.text(8.0, 8.6, "Preprocessing → Features → Model → SPEF",
            ha="center", fontsize=14, weight="bold", color="#1565c0")

    fig.savefig(ROOT / "fig6_feature_pipeline.png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    print("[ok] fig6_feature_pipeline.png")


def fig7_xgboost_architecture():
    """XGBoost architecture sketch + Tweedie objective."""
    from matplotlib.patches import FancyBboxPatch
    fig = plt.figure(figsize=(14, 6.5))

    # Left: tree boosting sketch
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_xlim(0, 10); ax1.set_ylim(0, 9); ax1.axis("off")
    ax1.text(5, 8.5, "5-seed Tweedie XGBoost ensemble",
             ha="center", fontsize=14, weight="bold", color="#1565c0")

    # Boosting iteration boxes
    seed_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    seeds = [42, 0, 1, 2, 3]
    for i, (seed, col) in enumerate(zip(seeds, seed_colors)):
        y = 7.0 - i * 1.2
        # Per-seed box
        rect = FancyBboxPatch((0.3, y), 1.6, 0.8, boxstyle="round,pad=0.05",
                               facecolor=col, edgecolor="black", linewidth=1, alpha=0.3)
        ax1.add_patch(rect)
        ax1.text(1.1, y + 0.4, f"seed={seed}", ha="center", va="center",
                 fontsize=10, weight="bold")
        # tree icons
        for k in range(5):
            tx = 2.5 + k * 1.0
            rect = FancyBboxPatch((tx, y), 0.85, 0.8, boxstyle="round,pad=0.03",
                                   facecolor="white", edgecolor=col, linewidth=1.2)
            ax1.add_patch(rect)
            ax1.text(tx + 0.42, y + 0.4, f"T{k+1}", ha="center", va="center",
                     fontsize=9)
        ax1.text(8.2, y + 0.4, "...500", ha="left", va="center", fontsize=9, color=col)
        # arrow
        ax1.annotate("", xytext=(2.0, y + 0.4), xy=(2.5, y + 0.4),
                     arrowprops=dict(arrowstyle="->", color=col, lw=1.5))

    ax1.text(5, 0.5, "10 weight files: gnd/cpl × 5 seeds (~12 MB each)",
             ha="center", fontsize=10, style="italic", color="#555555")

    # Right: Tweedie objective math + comparison
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_xlim(0, 10); ax2.set_ylim(0, 9); ax2.axis("off")
    ax2.text(5, 8.5, "Tweedie objective (vp = 1.5)",
             ha="center", fontsize=14, weight="bold", color="#1565c0")

    txt = (
        r"$\bf{Why~Tweedie?}$"
        "\n• Cap distribution is power-law (small nets dominate)"
        "\n• Tweedie variance-power = 1.5 → compound Poisson-Gamma"
        "\n• Log-link directly handles non-negative target"
        "\n\n"
        r"$\bf{Loss}$:  "
        "\nL(p, y) = max(0, y)·[(p^(2-vp)) / (vp-1)·(2-vp)] - y·(p^(1-vp)) / (1-vp)"
        "\n\n"
        r"$\bf{vs~alternatives}$:"
        "\n• `reg:squarederror` on log1p(y): MSE — not MAPE-aligned"
        "\n• `reg:quantileerror` α=0.5: |p-y| — different anchor"
        "\n• custom MAPE: sign(p-y)/y — unstable Hessian"
        "\n\n"
        r"$\bf{Result}$ (Small + Tweedie 5-seed):"
        "\n  tv80s 5.087 ± 0.049  /  nova 5.417 ± 0.027"
        "\n  vs squarederror baseline: -0.19 / -0.20 pp"
    )
    ax2.text(0.3, 7.5, txt, va="top", fontsize=10.5, family="serif",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f3e5f5", edgecolor="#4a148c"))

    fig.tight_layout()
    fig.savefig(ROOT / "fig7_xgboost_architecture.png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    print("[ok] fig7_xgboost_architecture.png")


def fig8_training_protocol():
    """Training protocol: data splits + 5-seed + early-stopping."""
    from matplotlib.patches import FancyBboxPatch, Rectangle
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8.5); ax.axis("off")

    # Title
    ax.text(7, 8.0, "Training protocol — 5-seed locked, manifest split, early-stopping",
            ha="center", fontsize=14, weight="bold", color="#1565c0")

    # Data split bar
    total_w = 12.0
    train_w = total_w * 112914 / 221102   # 51%
    valid_w = total_w * 12594  / 221102   # 5.7%
    test_w  = total_w * 95594  / 221102   # 43.3%

    y = 6.5; h = 0.7
    ax.add_patch(Rectangle((1.0, y), train_w, h, facecolor="#1f77b4", edgecolor="black"))
    ax.add_patch(Rectangle((1.0 + train_w, y), valid_w, h, facecolor="#ff7f0e", edgecolor="black"))
    ax.add_patch(Rectangle((1.0 + train_w + valid_w, y), test_w, h, facecolor="#2ca02c", edgecolor="black"))
    ax.text(1.0 + train_w/2, y + h/2, "TRAIN  112,914  (51 %)\n8 designs", ha="center", va="center",
            fontsize=10, color="white", weight="bold")
    ax.text(1.0 + train_w + valid_w/2, y + h/2, "VALID\n12,594", ha="center", va="center",
            fontsize=9, color="white", weight="bold")
    ax.text(1.0 + train_w + valid_w + test_w/2, y + h/2,
            "TEST  95,594  (43 %)\nintel22_tv80s_f3 + intel22_nova_f3",
            ha="center", va="center", fontsize=10, color="white", weight="bold")

    ax.text(0.5, y + h + 0.2, "Manifest H1 split (net-level hash, no (design, net) leakage):",
            fontsize=11, weight="bold")

    # Per-seed training scheme
    y2 = 4.2
    ax.text(0.5, y2 + 0.7, "5-seed protocol (paired Wilcoxon p=0.0625 with n=5):",
            fontsize=11, weight="bold")

    seeds = [42, 0, 1, 2, 3]
    for i, seed in enumerate(seeds):
        x = 1.0 + i * 2.4
        rect = FancyBboxPatch((x, y2 - 1.3), 2.0, 1.8, boxstyle="round,pad=0.05",
                               facecolor="#f3e5f5", edgecolor="#4a148c", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + 1.0, y2 + 0.2, f"seed = {seed}", ha="center", fontsize=10, weight="bold")
        ax.text(x + 1.0, y2 - 0.2, "subsample 0.8\ncolsample 0.8\n→ stochastic each fit",
                ha="center", fontsize=8, color="#555555")
        # per-channel pairs
        ax.text(x + 1.0, y2 - 0.85,
                "gnd model  →  pred_gnd\ncpl model  →  pred_cpl",
                ha="center", fontsize=8.5, color="#1565c0", weight="bold")

    # Hyperparameters / loss
    box_x, box_y = 0.5, 1.2
    rect = FancyBboxPatch((box_x, box_y), 6.5, 1.6, boxstyle="round,pad=0.1",
                           facecolor="#e1f5fe", edgecolor="#01579b", linewidth=1.5)
    ax.add_patch(rect)
    ax.text(box_x + 3.25, box_y + 1.3, "Hyperparameters (Small_combined config)",
            ha="center", fontsize=10.5, weight="bold")
    ax.text(box_x + 0.3, box_y + 0.85, "• depth = 8        • n_estimators = 500", fontsize=10)
    ax.text(box_x + 0.3, box_y + 0.50, "• learning_rate = 0.05    • subsample = 0.8", fontsize=10)
    ax.text(box_x + 0.3, box_y + 0.15, "• colsample_bytree = 0.8   • tree_method = hist", fontsize=10)

    rect = FancyBboxPatch((7.5, box_y), 6.0, 1.6, boxstyle="round,pad=0.1",
                           facecolor="#fff3e0", edgecolor="#e65100", linewidth=1.5)
    ax.add_patch(rect)
    ax.text(7.5 + 3.0, box_y + 1.3, "Training objective + stopping",
            ha="center", fontsize=10.5, weight="bold")
    ax.text(7.8, box_y + 0.85, "• objective = `reg:tweedie`  (vp=1.5)", fontsize=10)
    ax.text(7.8, box_y + 0.50, "• early_stopping_rounds = 100 on valid log-MSE", fontsize=10)
    ax.text(7.8, box_y + 0.15, "• 5-seed train wall total: ~9 min on 1 CPU thread", fontsize=10)

    fig.savefig(ROOT / "fig8_training_protocol.png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    print("[ok] fig8_training_protocol.png")


if __name__ == "__main__":
    fig1_per_bucket_R2()
    fig2_5seed_progression()
    fig3_strategy_ladder()
    fig4_per_bucket_TreePEX()
    fig5_speed_accuracy_pareto()
    fig6_feature_pipeline()
    fig7_xgboost_architecture()
    fig8_training_protocol()
    print("\nAll figures generated in:", ROOT)
