"""Diagnose total_R prediction errors on tv80s (v7 baseline).

Goals:
  1. Length-stratified MAPE (Q1..Q4 by total wirelength).
  2. Per-layer-mix outliers (which nets are mis-predicted, what does their layer mix look like?).
  3. Compare predicted R vs analytic (sheet_R x wirelength) — quantify how
     much of the error is recoverable just by an analytic per-layer model
     (no via R, no model nonlinearity), and how much is via / topology.
  4. Bias breakdown by net length tertile — is the -4.77% bias uniform or
     concentrated in long nets?
  5. Layer-transition proxy: nets whose wirelength spans many layers (=many vias)
     should have larger error if via R is the dominant residual.

Outputs (in reports/spef_e2e_R_diag/):
  - r_diag_summary.csv   : per-bucket and per-stratum metrics
  - r_diag_outliers.csv  : top-100 worst nets with feature snapshot
  - r_layer_mix.csv      : layer mix vs error correlation
  - r_diag.txt           : human-readable summary
  - 4 PNG plots
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent.parent
sys.path.insert(0, str(_WS))

from configs import cfg

# --- SPEF parser (PINNPEX) ---
_spec = importlib.util.spec_from_file_location(
    "compare_spef_pinpex",
    str(_WS.parent.parent / "src" / "evaluation" / "compare_spef.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
parse_spef = _mod.parse_spef_with_coordinates


# Per-layer sheet R (ohm/sq) — same defaults compute_resistance.py uses.
SHEET_R = {
    "M1": 1.5,
    "M2": 0.42, "M3": 0.42, "M4": 0.42, "M5": 0.42,
    "M6": 0.32, "M7": 0.32, "M8": 0.32,
    "M9p": 0.18,
}
# Approximate per-layer wire width (μm). Used to convert wirelength to "n_squares".
# These match intel22 standard cell wiring widths roughly.
LAYER_WIDTH = {
    "M1": 0.040, "M2": 0.044, "M3": 0.044, "M4": 0.044, "M5": 0.044,
    "M6": 0.080, "M7": 0.080, "M8": 0.160, "M9p": 0.320,
}


def analytic_R(row) -> float:
    """Sum_{layer} sheet_R[layer] * wirelength[layer] / width[layer]."""
    R = 0.0
    for L in SHEET_R:
        wl = float(row.get(f"tgt_wirelen_{L}", 0.0))
        if wl > 0:
            R += SHEET_R[L] * wl / LAYER_WIDTH[L]
    return R


def n_layers_used(row) -> int:
    return int(sum(1 for L in SHEET_R if float(row.get(f"tgt_wirelen_{L}", 0.0)) > 1e-6))


def main():
    out_dir = _WS / "reports" / "spef_e2e_R_diag"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = _WS / "output" / "spef_e2e" / "tv80s_FINAL.spef"
    gold_path = Path("/home/jslee/projects/PEX_SSL/golden_data/spef_data/intel22/intel22_tv80s_f3_starrc.spef")

    print(f"[1/4] Parsing SPEFs ...", flush=True)
    p = parse_spef(pred_path)
    g = parse_spef(gold_path)
    common = sorted(set(p.keys()) & set(g.keys()))

    rows = []
    for n in common:
        rows.append({
            "net_name": n,
            "R_pred": float(p[n]["total_res"]),
            "R_gold": float(g[n]["total_res"]),
            "total_cap_gold": float(g[n]["total_cap"]),
        })
    df_r = pd.DataFrame(rows)

    print(f"[2/4] Loading tv80s features ...", flush=True)
    feat_path = _WS / "cache" / "features_v3" / "intel22_tv80s_f3.parquet"
    df_f = pd.read_parquet(feat_path)
    df = df_r.merge(df_f, on="net_name", how="inner")
    print(f"  joined nets: {len(df)} (pred={len(df_r)}, feat={len(df_f)})")

    df = df[df["R_gold"] > 0.1].reset_index(drop=True).copy()
    df["ape"] = 100.0 * (df["R_pred"] - df["R_gold"]).abs() / df["R_gold"]
    df["err_signed_pct"] = 100.0 * (df["R_pred"] - df["R_gold"]) / df["R_gold"]

    df["R_analytic_uncalib"] = df.apply(analytic_R, axis=1)
    # Calibrate analytic to match golden median (so we compare shapes, not levels).
    cal = df["R_gold"].median() / df["R_analytic_uncalib"].median()
    df["R_analytic"] = df["R_analytic_uncalib"] * cal
    df["ape_analytic"] = 100.0 * (df["R_analytic"] - df["R_gold"]).abs() / df["R_gold"]

    df["n_layers"] = df.apply(n_layers_used, axis=1)
    df["log_wl"] = np.log10(df["tgt_wire_length_um"].clip(lower=1e-3))

    overall_mape = df["ape"].mean()
    overall_bias = df["err_signed_pct"].mean()
    analytic_mape = df["ape_analytic"].mean()
    print(f"\n  v7 ensemble : MAPE={overall_mape:.3f}%  bias={overall_bias:+.3f}%  n={len(df)}")
    print(f"  pure analytic (sheet_R * wirelen, calib): MAPE={analytic_mape:.3f}%")
    print(f"    => analytic ceiling tells us how much of the error is in")
    print(f"       layer mix / wirelength accuracy alone (no via, no topology).")

    # ------------------------------------------------------------------
    # 1. Length-stratified
    # ------------------------------------------------------------------
    print(f"\n[3/4] Length-stratified MAPE (quartiles by tgt_wire_length_um)", flush=True)
    df["wl_q"] = pd.qcut(df["tgt_wire_length_um"], q=4,
                          labels=["Q1_short", "Q2", "Q3", "Q4_long"])
    strat_rows = []
    for q, sub in df.groupby("wl_q", observed=True):
        strat_rows.append({
            "stratum": str(q),
            "n": len(sub),
            "wl_min_um": float(sub["tgt_wire_length_um"].min()),
            "wl_max_um": float(sub["tgt_wire_length_um"].max()),
            "wl_median_um": float(sub["tgt_wire_length_um"].median()),
            "R_gold_median": float(sub["R_gold"].median()),
            "ape_mean": float(sub["ape"].mean()),
            "ape_median": float(sub["ape"].median()),
            "bias_pct": float(sub["err_signed_pct"].mean()),
            "ape_analytic_mean": float(sub["ape_analytic"].mean()),
        })
    df_strat = pd.DataFrame(strat_rows)
    print(df_strat.to_string(index=False))

    # ------------------------------------------------------------------
    # 2. n_layers stratification (proxy for via count)
    # ------------------------------------------------------------------
    print(f"\n  Layer-count stratification (n_layers = number of layers used by net)", flush=True)
    layer_rows = []
    for nl, sub in df.groupby("n_layers"):
        layer_rows.append({
            "n_layers": int(nl),
            "n": len(sub),
            "ape_mean": float(sub["ape"].mean()),
            "ape_median": float(sub["ape"].median()),
            "bias_pct": float(sub["err_signed_pct"].mean()),
            "wl_median_um": float(sub["tgt_wire_length_um"].median()),
            "R_gold_median": float(sub["R_gold"].median()),
        })
    df_layers = pd.DataFrame(layer_rows).sort_values("n_layers").reset_index(drop=True)
    print(df_layers.to_string(index=False))

    # ------------------------------------------------------------------
    # 3. Layer-mix vs error correlation
    # ------------------------------------------------------------------
    print(f"\n  Per-layer wirelength fraction vs ape (top correlations)", flush=True)
    cors = []
    for L in SHEET_R:
        col = f"tgt_wirelen_{L}"
        if col in df.columns:
            frac = df[col] / df["tgt_wire_length_um"].clip(lower=1e-6)
            r = np.corrcoef(frac, df["err_signed_pct"])[0, 1]
            cors.append({"layer": L, "corr_frac_vs_signed_err": float(r),
                          "frac_mean": float(frac.mean())})
    df_cor = pd.DataFrame(cors).sort_values("corr_frac_vs_signed_err",
                                              key=lambda s: s.abs(), ascending=False)
    print(df_cor.to_string(index=False))

    # ------------------------------------------------------------------
    # 4. Worst-100 outliers
    # ------------------------------------------------------------------
    print(f"\n[4/4] Top-20 outlier nets:", flush=True)
    show_cols = ["net_name", "R_gold", "R_pred", "ape", "err_signed_pct",
                  "tgt_wire_length_um", "n_layers"]
    worst = df.nlargest(100, "ape")
    print(worst[show_cols].head(20).to_string(index=False))

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    df_strat.to_csv(out_dir / "r_diag_length_stratified.csv", index=False)
    df_layers.to_csv(out_dir / "r_diag_layer_count_stratified.csv", index=False)
    df_cor.to_csv(out_dir / "r_diag_layer_mix_corr.csv", index=False)
    worst.to_csv(out_dir / "r_diag_outliers_top100.csv", index=False)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    # (a) scatter pred vs gold, colored by ape
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(df["R_gold"], df["R_pred"], c=df["ape"].clip(upper=80),
                    s=4, alpha=0.5, cmap="viridis")
    plt.colorbar(sc, ax=ax, label="ape (%) clipped at 80")
    lo, hi = df["R_gold"].min(), df["R_gold"].max()
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("R_golden (Ω)"); ax.set_ylabel("R_predicted (Ω)")
    ax.set_title(f"v7 R: pred vs gold  (n={len(df)}, MAPE={overall_mape:.2f}%, bias={overall_bias:+.2f}%)")
    ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_dir / "r_scatter.png", dpi=140); plt.close()

    # (b) ape vs wirelength (signed error)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["tgt_wire_length_um"], df["err_signed_pct"], s=4, alpha=0.4)
    ax.axhline(0, color="r", lw=1)
    ax.set_xscale("log"); ax.set_xlabel("wirelength (μm)")
    ax.set_ylabel("signed error % (R_pred - R_gold) / R_gold * 100")
    ax.set_ylim(-100, 200)
    ax.set_title("Bias vs wirelength")
    ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_dir / "r_bias_vs_wl.png", dpi=140); plt.close()

    # (c) ape vs n_layers (boxplot)
    fig, ax = plt.subplots(figsize=(8, 5))
    nl_groups = [df.loc[df["n_layers"] == nl, "ape"].values
                  for nl in sorted(df["n_layers"].unique())]
    ax.boxplot(nl_groups, labels=[str(n) for n in sorted(df["n_layers"].unique())],
                showfliers=False)
    ax.set_xlabel("number of distinct layers used"); ax.set_ylabel("APE (%)")
    ax.set_title("APE distribution by layer-count (proxy for via count)")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(out_dir / "r_ape_by_nlayers.png", dpi=140); plt.close()

    # (d) v7 vs analytic — does analytic explain most variance?
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df["R_analytic"], df["R_gold"], s=4, alpha=0.4, label="analytic vs gold")
    ax.scatter(df["R_pred"], df["R_gold"], s=4, alpha=0.4, label="v7 pred vs gold", c="orange")
    lo, hi = df["R_gold"].min(), df["R_gold"].max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("predicted R"); ax.set_ylabel("R_golden")
    ax.set_title(f"Analytic ceiling vs v7\nanalytic MAPE={analytic_mape:.2f}%  v7 MAPE={overall_mape:.2f}%")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_dir / "r_analytic_vs_v7.png", dpi=140); plt.close()

    # ------------------------------------------------------------------
    # Text summary
    # ------------------------------------------------------------------
    summary_path = out_dir / "r_diag.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("v7 total_R diagnostic (intel22_tv80s_f3) — 2026-05-02\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"n nets   : {len(df)}\n")
        f.write(f"v7  MAPE : {overall_mape:.3f}%  (bias {overall_bias:+.3f}%)\n")
        f.write(f"analytic : {analytic_mape:.3f}%  (sheet_R * wirelen / width, calib)\n")
        f.write(f"  -> analytic captures level + wirelength but ignores via R / topology;\n")
        f.write(f"     the gap (analytic - v7) is what 145-dim hand features + ensemble\n")
        f.write(f"     bought us; the gap (v7 - perfect) is what via/topology features\n")
        f.write(f"     could potentially recover.\n\n")
        f.write("Length-stratified (quartiles by tgt_wire_length_um):\n")
        f.write(df_strat.to_string(index=False))
        f.write("\n\nLayer-count stratification (proxy for via count):\n")
        f.write(df_layers.to_string(index=False))
        f.write("\n\nLayer-mix correlation with signed error:\n")
        f.write(df_cor.to_string(index=False))
        f.write("\n\nTop-20 outliers:\n")
        f.write(worst[show_cols].head(20).to_string(index=False))
        f.write("\n")
    print(f"\nWrote: {summary_path}")
    print(f"Plots in: {out_dir}/")


if __name__ == "__main__":
    main()
