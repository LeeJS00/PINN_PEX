#!/usr/bin/env python3
"""
26_cgnd_error_analysis.py — C_gnd error origin analysis.

Why does Mesh PINN gnd MAPE plateau at 19-22% on cross-design test?

Analyses:
  1. Per-net (gnd error) vs (features) Spearman/Pearson correlation
  2. Quartile breakdown: which features correlate with high gnd error
  3. Per-design / per-layer-mix stratification
  4. Outlier nets (worst 50): geometric / circuit pattern signature
  5. Cancellation hypothesis: is gnd error correlated with cpl error?
     (if yes, model is trading gnd ↓ for cpl ↑ — calibration could fix)

Output: structured report `pex_v3/paper/CGND_ERROR_ANALYSIS.md`
"""
from __future__ import annotations
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "pex_v3"))

import numpy as np
import pandas as pd

# Mesh-curriculum 5-seed ensemble predictions on test split
MESH_TEST = (_PROJECT_ROOT / "pex_v3" / "output" / "phase1_mesh_5seed_ensemble"
             / "ensemble_predictions_test.csv")
# v3 features for richer per-net info
FEATURES_CSV = "/data/PINNPEX/data/processed_v3/intel22/features/all_designs.csv"
OUT_MD = _PROJECT_ROOT / "pex_v3" / "paper" / "CGND_ERROR_ANALYSIS.md"


def main() -> None:
    print(f">>> Loading mesh predictions: {MESH_TEST}")
    pred = pd.read_csv(MESH_TEST)
    print(f"  shape: {pred.shape}")

    print(f">>> Loading v3 features: {FEATURES_CSV}")
    feats = pd.read_csv(FEATURES_CSV)
    feats_test = feats[feats.design_name.isin(["intel22_nova_f3", "intel22_tv80s_f3"])]
    print(f"  test feats: {feats_test.shape}")

    df = pred.merge(feats_test, on=["design_name", "net_name"], how="inner",
                    suffixes=("", "_feat"))
    print(f">>> joined: {df.shape}")

    # Per-net error metrics
    eps = 1e-3
    df["gnd_rel_err"] = (df.pred_gnd_fF - df.golden_gnd_fF).abs() / df.golden_gnd_fF.clip(lower=eps)
    df["cpl_rel_err"] = (df.pred_cpl_fF - df.golden_cpl_fF).abs() / df.golden_cpl_fF.clip(lower=eps)
    df["total_rel_err"] = (df.pred_total_fF - df.golden_total_fF).abs() / df.golden_total_fF.clip(lower=eps)
    df["gnd_signed"] = (df.pred_gnd_fF - df.golden_gnd_fF) / df.golden_gnd_fF.clip(lower=eps)

    print(f"\n>>> baseline gnd MAPE: median {df.gnd_rel_err.median()*100:.3f}%  mean {df.gnd_rel_err.mean()*100:.3f}%")

    md = []
    md.append("# C_gnd Error Origin Analysis\n\n")
    md.append("_Mesh-curriculum 5-seed ensemble on cross-design test (95,594 nets)_\n\n")
    md.append(f"## Baseline\n\n")
    md.append(f"- gnd MAPE: median **{df.gnd_rel_err.median()*100:.3f}%**, mean {df.gnd_rel_err.mean()*100:.3f}%\n")
    md.append(f"- cpl MAPE: median **{df.cpl_rel_err.median()*100:.3f}%**, mean {df.cpl_rel_err.mean()*100:.3f}%\n")
    md.append(f"- total MAPE: median {df.total_rel_err.median()*100:.3f}%\n\n")

    # ---- 1. gnd_rel_err vs cpl_rel_err correlation ----
    print()
    print("=" * 60)
    print("[1] gnd error ↔ cpl error correlation (cancellation hypothesis)")
    print("=" * 60)
    from scipy.stats import spearmanr, pearsonr
    sp = spearmanr(df.gnd_rel_err, df.cpl_rel_err)
    pe = pearsonr(df.gnd_rel_err, df.cpl_rel_err)
    print(f"  Spearman ρ: {sp.correlation:.4f}  p={sp.pvalue:.2e}")
    print(f"  Pearson  r: {pe.statistic:.4f}  p={pe.pvalue:.2e}")
    md.append(f"## 1. gnd error vs cpl error correlation\n\n")
    md.append(f"- Spearman ρ = {sp.correlation:.4f} (p={sp.pvalue:.2e})\n")
    md.append(f"- Pearson r  = {pe.statistic:.4f} (p={pe.pvalue:.2e})\n\n")
    md.append(f"**Interpretation**: ")
    if abs(sp.correlation) > 0.3:
        md.append("Strong positive correlation suggests gnd/cpl errors share root cause (geometry, model capacity), NOT trading off.\n\n")
    elif abs(sp.correlation) > 0.1:
        md.append("Mild correlation; some shared cause but channels somewhat independent.\n\n")
    else:
        md.append("Near-zero correlation; gnd error is independent of cpl error.\n\n")

    # ---- 2. Per-feature correlation with gnd error ----
    print()
    print("=" * 60)
    print("[2] Feature ↔ gnd error correlation (top contributors)")
    print("=" * 60)
    # Choose numeric features
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude target/error columns
    exclude = {"gnd_rel_err", "cpl_rel_err", "total_rel_err", "gnd_signed",
               "pred_gnd_fF", "pred_cpl_fF", "pred_total_fF",
               "golden_gnd_fF", "golden_cpl_fF", "golden_total_fF",
               "c_gnd_fF", "c_cpl_total_fF"}
    feat_cols = [c for c in num_cols if c not in exclude]
    corrs = []
    for c in feat_cols:
        try:
            v = df[c].fillna(0)
            if v.std() < 1e-9:
                continue
            sp_c = spearmanr(v, df.gnd_rel_err).correlation
            pe_c = pearsonr(v.fillna(0), df.gnd_rel_err).statistic
            corrs.append((c, abs(sp_c), sp_c, pe_c))
        except Exception:
            continue
    corrs.sort(key=lambda x: -x[1])  # absolute Spearman desc
    print(f"  Top 15 features by |Spearman ρ| with gnd_rel_err:")
    print(f"  {'feature':<35} {'|ρ|':>8} {'ρ':>9} {'r':>9}")
    md.append(f"## 2. Top features by Spearman |ρ| vs gnd error\n\n")
    md.append(f"| feature | \\|Spearman ρ\\| | Spearman ρ | Pearson r |\n|---|---:|---:|---:|\n")
    for c, abs_sp, sp_c, pe_c in corrs[:15]:
        print(f"  {c:<35} {abs_sp:>8.4f} {sp_c:>9.4f} {pe_c:>9.4f}")
        md.append(f"| `{c}` | {abs_sp:.4f} | {sp_c:.4f} | {pe_c:.4f} |\n")
    md.append("\n")

    # ---- 3. Quartile breakdown ----
    print()
    print("=" * 60)
    print("[3] gnd error quartile vs key features (median per quartile)")
    print("=" * 60)
    df["gnd_err_quartile"] = pd.qcut(df.gnd_rel_err, 4, labels=["Q1(low)", "Q2", "Q3", "Q4(high)"])
    # Pick top 6 corr features
    key_features = [c for c, _, _, _ in corrs[:6]]
    print(f"  features: {key_features}")
    qb = df.groupby("gnd_err_quartile", observed=True)[key_features + ["gnd_rel_err"]].median()
    print(qb.to_string(float_format="%.3f"))
    md.append(f"## 3. Quartile breakdown (median feature value per gnd-error quartile)\n\n")
    md.append(f"| feature | Q1 (low err) | Q2 | Q3 | Q4 (high err) |\n|---|---:|---:|---:|---:|\n")
    for f in key_features + ["gnd_rel_err"]:
        if f in qb.columns:
            vals = [qb.loc[q, f] for q in ["Q1(low)", "Q2", "Q3", "Q4(high)"]]
            md.append(f"| `{f}` | {vals[0]:.3f} | {vals[1]:.3f} | {vals[2]:.3f} | {vals[3]:.3f} |\n")
    md.append("\n")

    # ---- 4. Per-design ----
    print()
    print("=" * 60)
    print("[4] Per-design gnd MAPE")
    print("=" * 60)
    pd_md = []
    pd_md.append(f"## 4. Per-design breakdown\n\n")
    pd_md.append(f"| design | n_nets | gnd median | gnd mean | cpl median | total median |\n|---|---:|---:|---:|---:|---:|\n")
    for design, sub in df.groupby("design_name"):
        line = f"  {design:<30} n={len(sub):>7,}  gnd median={sub.gnd_rel_err.median()*100:>6.3f}%  cpl median={sub.cpl_rel_err.median()*100:>6.3f}%"
        print(line)
        pd_md.append(f"| {design} | {len(sub):,} | {sub.gnd_rel_err.median()*100:.3f}% | {sub.gnd_rel_err.mean()*100:.3f}% | {sub.cpl_rel_err.median()*100:.3f}% | {sub.total_rel_err.median()*100:.3f}% |\n")
    md.extend(pd_md)
    md.append("\n")

    # ---- 5. Layer mix stratification ----
    print()
    print("=" * 60)
    print("[5] Layer mix stratification")
    print("=" * 60)
    layer_cols = [c for c in df.columns if c.startswith("layer_hist_")]
    if layer_cols:
        # Dominant layer per net
        df["dominant_layer"] = df[layer_cols].idxmax(axis=1).str.replace("layer_hist_", "")
        print(f"  Dominant layer distribution (test):")
        print(f"  {df.dominant_layer.value_counts().to_dict()}")
        md.append(f"## 5. Layer-mix stratification (gnd MAPE by dominant layer)\n\n")
        md.append(f"| dominant layer | n_nets | gnd median | gnd mean | cpl median |\n|---|---:|---:|---:|---:|\n")
        for layer, sub in df.groupby("dominant_layer"):
            print(f"    {layer:<10} n={len(sub):>7,}  gnd median={sub.gnd_rel_err.median()*100:>6.3f}%  cpl={sub.cpl_rel_err.median()*100:>6.3f}%")
            md.append(f"| `{layer}` | {len(sub):,} | {sub.gnd_rel_err.median()*100:.3f}% | {sub.gnd_rel_err.mean()*100:.3f}% | {sub.cpl_rel_err.median()*100:.3f}% |\n")
    md.append("\n")

    # ---- 6. Top 50 outlier nets ----
    print()
    print("=" * 60)
    print("[6] Top 50 outlier nets (highest gnd error)")
    print("=" * 60)
    top50 = df.nlargest(50, "gnd_rel_err")
    print(f"  Top 50 stats:")
    print(f"    gnd error: median {top50.gnd_rel_err.median()*100:.3f}%  max {top50.gnd_rel_err.max()*100:.3f}%")
    print(f"    typical features (median):")
    md.append(f"## 6. Outlier characterization (top 50 highest gnd error)\n\n")
    md.append(f"| feature | top-50 median | overall median | ratio |\n|---|---:|---:|---:|\n")
    for f in key_features:
        t50_med = top50[f].median()
        all_med = df[f].median()
        ratio = t50_med / all_med if all_med != 0 else float("nan")
        print(f"      {f}: {t50_med:.3f} (overall {all_med:.3f}, ratio {ratio:.2f}x)")
        md.append(f"| `{f}` | {t50_med:.3f} | {all_med:.3f} | {ratio:.2f}× |\n")
    md.append("\n")

    # ---- 7. Sign analysis: is gnd_pred consistently under or over? ----
    print()
    print("=" * 60)
    print("[7] Signed gnd error (pred over/under)")
    print("=" * 60)
    print(f"  median signed: {df.gnd_signed.median()*100:.3f}%")
    print(f"  mean signed:   {df.gnd_signed.mean()*100:.3f}%")
    print(f"  % nets where pred < golden (under-prediction): {(df.gnd_signed < 0).mean()*100:.1f}%")
    print(f"  % nets where pred > golden (over-prediction):  {(df.gnd_signed > 0).mean()*100:.1f}%")
    md.append(f"## 7. Signed error (under vs over)\n\n")
    md.append(f"- median signed error: **{df.gnd_signed.median()*100:.3f}%**\n")
    md.append(f"- mean signed error:   {df.gnd_signed.mean()*100:.3f}%\n")
    md.append(f"- % nets where pred < golden: **{(df.gnd_signed < 0).mean()*100:.1f}%**\n")
    md.append(f"- % nets where pred > golden: **{(df.gnd_signed > 0).mean()*100:.1f}%**\n\n")

    # Save markdown
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("".join(md))
    print(f"\n✅ report → {OUT_MD}")


if __name__ == "__main__":
    main()
