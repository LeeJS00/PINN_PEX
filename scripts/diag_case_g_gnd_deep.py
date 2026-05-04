#!/usr/bin/env python3
"""
scripts/diag_case_g_gnd_deep.py

Case G — GND root-cause deep analysis.

Phase A's Case 3 surfaced a systematic GND under-prediction (mean signed
error -0.3 to -0.6 fF across all models). This script drills into the cause:

  1. Underprediction bias — distribution of pred_gnd / y_gnd ratio. Where
     are we losing capacitance? Per-design and per-net-magnitude breakdown.
  2. Per-layer cuboid GND distribution. Phase A showed M6+ cuboids predict
     ~0 fF — verify whether that reflects data (no big cuboids exist) or
     model under-prediction.
  3. KCL balance: pred_total ≈ pred_gnd + Σ pred_cpl ? Per-net residual.
  4. Power vs signal cuboid contribution. The CPL→GND lumping in evaluator
     may not match StarRC's segment-level allocation.
  5. Net total decomposition: how much of net's GND comes from each layer.

Usage:
  python3 scripts/diag_case_g_gnd_deep.py \\
      --dumps v10b dspinn_v1_new dspinn_v2
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/home/jslee/projects/PINNPEX")

import numpy as np


def load_dumps(names: list[str], al_root: Path) -> dict[str, dict]:
    dumps: dict[str, dict] = {}
    for name in names:
        path = al_root / name / 'eval_dump.npz'
        if not path.exists():
            print(f"⚠ skip {name}: {path}")
            continue
        d = dict(np.load(str(path), allow_pickle=True))
        dumps[name] = d
    if not dumps:
        sys.exit("❌ No dumps loaded.")
    for name, d in dumps.items():
        names_d = d['target_names'].astype(str)
        order = np.argsort(names_d)
        for k in ['pred_total', 'pred_gnd', 'pred_cpl',
                  'y_total', 'y_gnd', 'y_cpl', 'valid_aggr',
                  'designs', 'net_size', 'net_z_mean', 'target_names']:
            d[k] = d[k][order]
    return dumps


LAYER_BUCKETS = [
    ('PRE_M1', 0.00, 0.40),
    ('M1',     0.40, 0.62),
    ('M2',     0.62, 0.75),
    ('M3',     0.75, 0.90),
    ('M4',     0.90, 1.05),
    ('M5',     1.05, 1.20),
    ('M6',     1.20, 1.50),
    ('M7',     1.50, 4.50),
    ('M8',     4.50, 6.00),
    ('TOP',    6.00, 20.00),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dumps', nargs='+', required=True)
    ap.add_argument('--output_root', default='/home/jslee/projects/PINNPEX/output_intel22')
    args = ap.parse_args()

    output_root = Path(args.output_root)
    al_root = output_root / 'active_learning'
    out_dir = al_root / 'diag_phase_a'
    out_dir.mkdir(parents=True, exist_ok=True)

    dumps = load_dumps(args.dumps, al_root)
    md = ["# Case G — GND Root-Cause Deep Analysis", ""]

    ref = list(dumps.values())[0]
    md.append(f"_Validation: {ref['y_total'].shape[0]} nets, "
              f"{ref['cuboid_z'].size:,} cuboids_")
    md.append("")

    # =========================================================================
    # 1. Pred/Golden ratio distribution per design + per net-magnitude
    # =========================================================================
    md.append("## 1. GND under-prediction ratio (pred_gnd / y_gnd)")
    md.append("")
    md.append("Ratio < 1 means the model under-predicts GND. Mean ratio across "
              "nets shows the systematic bias direction.")
    md.append("")
    md.append("### Overall ratio distribution")
    md.append("")
    md.append("| Model         | Mean | Median |  P10 |  P50 |  P90 | %nets ratio<0.5 | %nets ratio>1.5 |")
    md.append("|---------------|-----:|-------:|-----:|-----:|-----:|----------------:|----------------:|")
    for name, d in dumps.items():
        pos = d['y_gnd'] > 0.005
        if pos.sum() == 0:
            continue
        ratio = d['pred_gnd'][pos] / (d['y_gnd'][pos] + 1e-6)
        under = (ratio < 0.5).mean() * 100
        over = (ratio > 1.5).mean() * 100
        md.append(f"| {name:<13} | {ratio.mean():>4.2f} | {np.median(ratio):>5.2f} | "
                  f"{np.percentile(ratio, 10):>4.2f} | "
                  f"{np.percentile(ratio, 50):>4.2f} | "
                  f"{np.percentile(ratio, 90):>4.2f} | "
                  f"{under:>14.1f}% | "
                  f"{over:>14.1f}% |")
    md.append("")

    md.append("### Per-design mean ratio")
    md.append("")
    designs = ref['designs'].astype(str)
    unique_designs = sorted(np.unique(designs))
    md.append("| Design | N nets | "
              + " | ".join(f"{n}" for n in dumps.keys()) + " |")
    md.append("|--------|-------:|" + "|".join(["----:"] * len(dumps)) + "|")
    for design in unique_designs:
        m = (designs == design)
        n = int(m.sum())
        if n == 0:
            continue
        row = [f" {design} ", f" {n} "]
        for name, d in dumps.items():
            pos = (d['y_gnd'][m] > 0.005)
            if pos.sum() == 0:
                row.append("    n/a ")
                continue
            r = d['pred_gnd'][m][pos] / (d['y_gnd'][m][pos] + 1e-6)
            row.append(f" {r.mean():>4.2f} ")
        md.append("|" + "|".join(row) + "|")
    md.append("")

    md.append("### Ratio vs y_gnd magnitude (by quartile of y_gnd)")
    md.append("")
    qy = np.quantile(ref['y_gnd'][ref['y_gnd'] > 0.005], [0.25, 0.5, 0.75])
    md.append(f"_Quartile cuts (fF): Q1={qy[0]:.3f}, Q2={qy[1]:.3f}, Q3={qy[2]:.3f}_")
    md.append("")
    md.append("| Magnitude bin | N nets | "
              + " | ".join(f"{n} mean ratio" for n in dumps.keys()) + " |")
    md.append("|---------------|-------:|"
              + "|".join(["-----:"] * len(dumps)) + "|")
    for label, lo, hi in [(f"≤Q1 ({qy[0]:.3f}fF)", 0.005, qy[0]),
                          (f"Q1-Q2 (≤{qy[1]:.3f}fF)", qy[0], qy[1]),
                          (f"Q2-Q3 (≤{qy[2]:.3f}fF)", qy[1], qy[2]),
                          (f"Q3+", qy[2], 1e9)]:
        m = (ref['y_gnd'] > lo) & (ref['y_gnd'] <= hi)
        n = int(m.sum())
        row = [f" {label} ", f" {n} "]
        for name, d in dumps.items():
            if n == 0:
                row.append("    n/a ")
                continue
            r = d['pred_gnd'][m] / (d['y_gnd'][m] + 1e-6)
            row.append(f" {r.mean():>4.2f} ")
        md.append("|" + "|".join(row) + "|")
    md.append("")

    # =========================================================================
    # 2. Per-layer cuboid GND distribution + cuboid count
    # =========================================================================
    md.append("## 2. Per-layer cuboid GND prediction analysis")
    md.append("")
    md.append("Phase A Case 3 showed M6+ cuboids predict ≈0 fF. This breaks down "
              "whether (a) few cuboids exist on those layers, or (b) the model "
              "predicts zero. Also reports learned per-layer cap density (if available).")
    md.append("")
    md.append("| Layer | z range | N cuboids | "
              + " | ".join(f"{n} mean | {n} %=0" for n in dumps.keys()) + " |")
    md.append("|-------|---------|-----------|"
              + "|".join(["-----:|-----:"] * len(dumps)) + "|")
    for label, lo, hi in LAYER_BUCKETS:
        z = ref['cuboid_z']
        in_b = (z >= lo) & (z < hi)
        n = int(in_b.sum())
        if n == 0:
            row = [f" {label} ", f" {lo:.2f}-{hi:.2f} ", f" 0 "]
            for _ in dumps:
                row.extend(["    — ", "    — "])
            md.append("|" + "|".join(row) + "|")
            continue
        row = [f" {label} ", f" {lo:.2f}-{hi:.2f} ", f" {n:,} "]
        for name, d in dumps.items():
            g = d['cuboid_gnd'][in_b]
            mean = float(g.mean())
            pzero = float((g < 1e-9).mean() * 100)
            row.extend([f" {mean:>6.4f} fF ", f" {pzero:>4.1f}% "])
        md.append("|" + "|".join(row) + "|")
    md.append("")

    # =========================================================================
    # 3. KCL balance: pred_total vs (pred_gnd + Σ pred_cpl)
    # =========================================================================
    md.append("## 3. KCL balance: pred_total vs pred_gnd + Σ pred_cpl")
    md.append("")
    md.append("Each model's pred_total should equal pred_gnd plus the sum of all "
              "valid pred_cpl entries. Discrepancy points to aggregation bugs in "
              "the evaluator pipeline.")
    md.append("")
    md.append("| Model         | Mean total | Mean (gnd+Σcpl) | Mean signed diff | RMSE | Max |abs diff| |")
    md.append("|---------------|-----------:|----------------:|-----------------:|-----:|----------------:|")
    for name, d in dumps.items():
        pos = d['y_total'] > 0.005
        cpl_sum = (d['pred_cpl'] * d['valid_aggr'].astype(np.float32)).sum(axis=1)
        rebuilt = d['pred_gnd'] + cpl_sum
        diff = d['pred_total'] - rebuilt
        md.append(f"| {name:<13} | "
                  f"{d['pred_total'][pos].mean():>9.3f} fF | "
                  f"{rebuilt[pos].mean():>13.3f} fF | "
                  f"{diff[pos].mean():>14.4f} fF | "
                  f"{float(np.sqrt(np.mean(diff[pos]**2))):>4.3f} | "
                  f"{float(np.abs(diff[pos]).max()):>11.3f} fF |")
    md.append("")

    # =========================================================================
    # 4. Net-level pred_gnd vs y_gnd distribution + correlation
    # =========================================================================
    md.append("## 4. Net-level pred_gnd vs y_gnd correlation")
    md.append("")
    md.append("| Model         | Pearson r | Spearman ρ (rank) | log-Pearson r | Slope (linfit) |")
    md.append("|---------------|----------:|------------------:|--------------:|---------------:|")
    from scipy.stats import spearmanr
    for name, d in dumps.items():
        pos = d['y_gnd'] > 0.005
        if pos.sum() < 2:
            continue
        p = d['pred_gnd'][pos]
        t = d['y_gnd'][pos]
        r = float(np.corrcoef(p, t)[0, 1]) if p.std() > 0 else float('nan')
        try:
            ρ, _ = spearmanr(p, t)
            ρ_str = f"{float(ρ):>13.3f}"
        except Exception:
            ρ_str = "          n/a"
        log_r = float(np.corrcoef(np.log1p(p.clip(min=0)), np.log1p(t.clip(min=0)))[0, 1])
        # Slope: linear regression pred = a * golden + b
        if t.std() > 0:
            slope = float(np.polyfit(t, p, 1)[0])
        else:
            slope = float('nan')
        md.append(f"| {name:<13} | {r:>9.3f} | {ρ_str} | {log_r:>13.3f} | {slope:>13.3f} |")
    md.append("")
    md.append("_Slope = 1.0 means perfectly calibrated. <1.0 means systematic "
              "magnitude under-prediction. >1.0 means over-prediction._")
    md.append("")

    # =========================================================================
    # 5. Per-layer GND attribution within each net total
    # =========================================================================
    md.append("## 5. Within-net GND distribution by layer (cuboid sums)")
    md.append("")
    md.append("How does each model's GND prediction split across metal layers? "
              "Reveals whether upper metals (M6-M8) are entirely missing.")
    md.append("")
    cuboid_to_net = ref['cuboid_to_net']
    n_nets = ref['y_total'].shape[0]
    md.append("| Layer | "
              + " | ".join(f"{n} mean fraction" for n in dumps.keys())
              + " |")
    md.append("|-------|"
              + "|".join(["----------------:"] * len(dumps))
              + "|")
    for label, lo, hi in LAYER_BUCKETS:
        z = ref['cuboid_z']
        in_layer = (z >= lo) & (z < hi)
        if in_layer.sum() == 0:
            row = [f" {label} "] + ["       —" for _ in dumps]
            md.append("|" + "|".join(row) + "|")
            continue
        row = [f" {label} "]
        for name, d in dumps.items():
            # Sum cuboid GND per net for this layer
            layer_gnd_per_net = np.zeros(n_nets, dtype=np.float32)
            np.add.at(layer_gnd_per_net, cuboid_to_net[in_layer], d['cuboid_gnd'][in_layer])
            # Total cuboid GND per net
            total_cuboid_gnd_per_net = np.zeros(n_nets, dtype=np.float32)
            np.add.at(total_cuboid_gnd_per_net, cuboid_to_net, d['cuboid_gnd'])
            # Fraction of total
            frac = layer_gnd_per_net / total_cuboid_gnd_per_net.clip(min=1e-9)
            mean_frac = float(frac[total_cuboid_gnd_per_net > 0].mean()
                              if (total_cuboid_gnd_per_net > 0).any() else 0)
            row.append(f" {mean_frac * 100:>13.2f}% ")
        md.append("|" + "|".join(row) + "|")
    md.append("")

    # =========================================================================
    # 6. Summary interpretations
    # =========================================================================
    md.append("## 6. Summary findings")
    md.append("")
    findings = []
    # 1. Underprediction direction
    for name, d in dumps.items():
        pos = d['y_gnd'] > 0.005
        if pos.sum() == 0:
            continue
        ratio = d['pred_gnd'][pos] / (d['y_gnd'][pos] + 1e-6)
        median_r = float(np.median(ratio))
        if median_r < 0.7:
            findings.append(f"- ⚠ **{name}** median pred/golden ratio = {median_r:.2f} → systematic under-prediction. "
                            "Causes: (a) gnd_modifier saturating low, (b) layer_scale_phys_gnd init under-shoots, "
                            "or (c) some cuboids ignored entirely.")
    # 2. Layer dominance
    md.append("")
    md.append("**Per-design ratio observations:**")
    for design in unique_designs:
        m = (designs == design)
        ratios = {}
        for name, d in dumps.items():
            pos = (d['y_gnd'][m] > 0.005)
            if pos.sum() == 0:
                continue
            ratios[name] = float(np.median(
                d['pred_gnd'][m][pos] / (d['y_gnd'][m][pos] + 1e-6)))
        if ratios:
            md.append(f"  - {design}: " +
                      ", ".join(f"{k}={v:.2f}" for k, v in ratios.items()))
    md.append("")
    if findings:
        md.append("**Critical findings:**")
        md.extend(findings)

    out_md = out_dir / 'report_case_g_gnd_deep.md'
    out_md.write_text('\n'.join(md))
    print('\n'.join(md))
    print(f"\nReport: {out_md}")


if __name__ == '__main__':
    main()
