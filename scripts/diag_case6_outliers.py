#!/usr/bin/env python3
"""
scripts/diag_case6_outliers.py

Case 6 — Per-net outlier identification.

Identifies the worst-performing nets and re-evaluates aggregate metrics
after their removal. Tests whether the global metric is dominated by a
few catastrophic outliers.

Usage:
  python3 scripts/diag_case6_outliers.py \\
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
    md = ["# Case 6 — Per-net Outlier Analysis", ""]

    ref = list(dumps.values())[0]
    y_total = ref['y_total']
    n_nets = y_total.shape[0]
    pos_total = y_total > 0.005
    md.append(f"_Validation: {n_nets} nets, {int(pos_total.sum())} with golden total > 0.005 fF_")
    md.append("")

    # =========================================================================
    # Per-net total MAPE distribution + worst nets per model
    # =========================================================================
    md.append("## 1. Per-net total MAPE distribution")
    md.append("")
    md.append("| Model         | Mean | P50 (median) | P75 | P90 | P95 | P99 | Max |")
    md.append("|---------------|-----:|-------------:|----:|----:|----:|----:|----:|")
    for name, d in dumps.items():
        mape = np.abs(d['pred_total'][pos_total] - y_total[pos_total]) / (y_total[pos_total] + 1e-6) * 100
        md.append(f"| {name:<13} | {mape.mean():>4.1f}% | "
                  f"{np.percentile(mape, 50):>10.1f}% | "
                  f"{np.percentile(mape, 75):>3.1f}% | "
                  f"{np.percentile(mape, 90):>3.1f}% | "
                  f"{np.percentile(mape, 95):>3.1f}% | "
                  f"{np.percentile(mape, 99):>3.1f}% | "
                  f"{mape.max():>3.1f}% |")
    md.append("")

    # =========================================================================
    # Re-evaluate after trimming worst K nets
    # =========================================================================
    md.append("## 2. Aggregate MAPE after trimming top-K worst nets per model")
    md.append("")
    md.append("(Trimming removes the K nets with highest individual MAPE before "
              "averaging. Tests whether outliers dominate the headline number.)")
    md.append("")
    md.append("| Trim K | "
              + " | ".join(f"{n} MAPE" for n in dumps.keys()) + " |")
    md.append("|-------:|"
              + "|".join(["----:"] * len(dumps)) + "|")
    for K in [0, 5, 10, 20, 50, 100]:
        row = [f" {K} "]
        for name, d in dumps.items():
            mape = np.abs(d['pred_total'][pos_total] - y_total[pos_total]) / (y_total[pos_total] + 1e-6) * 100
            if K > 0:
                idx_sorted = np.argsort(-mape)  # descending — highest first
                kept = np.ones_like(mape, dtype=bool)
                kept[idx_sorted[:K]] = False
                trimmed = mape[kept]
            else:
                trimmed = mape
            row.append(f" {trimmed.mean():>5.2f}% ")
        md.append("|" + "|".join(row) + "|")
    md.append("")

    # =========================================================================
    # Top 10 worst nets per model
    # =========================================================================
    md.append("## 3. Top 10 worst nets (by MAPE) per model")
    md.append("")
    for name, d in dumps.items():
        md.append(f"### {name}")
        md.append("")
        mape = np.abs(d['pred_total'][pos_total] - y_total[pos_total]) / (y_total[pos_total] + 1e-6) * 100
        sorted_idx = np.argsort(-mape)[:10]
        # Extract metadata for these worst nets
        all_idx = np.where(pos_total)[0]
        worst_global = all_idx[sorted_idx]
        md.append("| Rank | Design | Net | y_total (fF) | pred_total | MAPE | net_size |")
        md.append("|-----:|--------|-----|-------------:|-----------:|-----:|---------:|")
        for rank, gi in enumerate(worst_global, 1):
            md.append(f"| {rank:>4} | "
                      f"{d['designs'][gi]} | "
                      f"{d['target_names'][gi]} | "
                      f"{d['y_total'][gi]:>11.4f} | "
                      f"{d['pred_total'][gi]:>9.4f} | "
                      f"{mape[sorted_idx[rank - 1]]:>4.1f}% | "
                      f"{d['net_size'][gi]:>7} |")
        md.append("")

    # =========================================================================
    # Same nets across models — find common bad nets
    # =========================================================================
    md.append("## 4. Common worst nets across models (top-50 intersection)")
    md.append("")
    common_set = None
    for name, d in dumps.items():
        mape = np.abs(d['pred_total'][pos_total] - y_total[pos_total]) / (y_total[pos_total] + 1e-6) * 100
        worst_idx = np.argsort(-mape)[:50]
        all_idx = np.where(pos_total)[0]
        worst_global = set(all_idx[worst_idx].tolist())
        if common_set is None:
            common_set = worst_global
        else:
            common_set = common_set & worst_global
    md.append(f"_Nets that are top-50 worst in **ALL** models: {len(common_set)}_")
    md.append("")
    if common_set:
        md.append("| Design | Net | y_total | "
                  + " | ".join(f"{n} pred" for n in dumps.keys()) + " |")
        md.append("|--------|-----|--------:|"
                  + "|".join(["-----:"] * len(dumps)) + "|")
        for gi in sorted(common_set):
            row = [
                f" {ref['designs'][gi]} ",
                f" {ref['target_names'][gi]} ",
                f" {ref['y_total'][gi]:.3f} ",
            ]
            for name, d in dumps.items():
                row.append(f" {d['pred_total'][gi]:.3f} ")
            md.append("|" + "|".join(row) + "|")
    md.append("")

    out_md = out_dir / 'report_case6_outliers.md'
    out_md.write_text('\n'.join(md))
    print('\n'.join(md))
    print(f"\nReport: {out_md}")


if __name__ == '__main__':
    main()
