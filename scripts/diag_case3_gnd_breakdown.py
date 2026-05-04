#!/usr/bin/env python3
"""
scripts/diag_case3_gnd_breakdown.py

Case 3 — GND error breakdown (per-design / per-layer / per-net-size).

For each model dump:
  1. Per-design: net MAPE, GND SMAPE, CPL SMAPE — which designs regress?
  2. Per-layer (z bucket): GND magnitude error per cuboid bucket
  3. Per-net-size: MAPE/GND quartiles by cuboid count
  4. Net-level scatter: pred_gnd vs y_gnd, residual distribution

Usage:
  python3 scripts/diag_case3_gnd_breakdown.py \\
      --dumps v10b dspinn_v1_new dspinn_v2
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/home/jslee/projects/PINNPEX")

import numpy as np


def load_dumps(names: list[str], al_root: Path) -> dict[str, dict]:
    dumps = {}
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


def smape_pos(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    pos = target > 0.005
    if pos.sum() == 0:
        return float('nan')
    p, t = pred[pos], target[pos]
    return float((2.0 * np.abs(p - t) / (np.abs(p) + np.abs(t) + eps)).mean() * 100.0)


def mape_pos(pred: np.ndarray, target: np.ndarray) -> float:
    pos = target >= 0.005
    if pos.sum() == 0:
        return float('nan')
    return float(np.mean(np.abs(pred[pos] - target[pos]) /
                         (target[pos] + 1e-6)) * 100.0)


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

    md = ["# Case 3 — GND Error Breakdown", ""]

    # --- 1. Per-design ---
    md.append("## 1. Per-design net MAPE / GND SMAPE / CPL SMAPE")
    md.append("")
    ref = list(dumps.values())[0]
    designs = ref['designs'].astype(str)
    unique_designs = sorted(np.unique(designs))
    for design in unique_designs:
        md.append(f"### {design}")
        md.append("")
        mask = (designs == design)
        n = int(mask.sum())
        if n == 0:
            continue
        md.append(f"_{n} nets_")
        md.append("")
        md.append("| Model         | Net MAPE | GND SMAPE | CPL SMAPE |")
        md.append("|---------------|---------:|----------:|----------:|")
        for name, d in dumps.items():
            mape = mape_pos(d['pred_total'][mask], d['y_total'][mask])
            gnd_s = smape_pos(d['pred_gnd'][mask], d['y_gnd'][mask])
            v_d = d['valid_aggr'][mask].astype(bool)
            p_cpl = d['pred_cpl'][mask][v_d]
            t_cpl = d['y_cpl'][mask][v_d]
            cpl_s = (
                float((2.0 * np.abs(p_cpl - t_cpl) /
                       (np.abs(p_cpl) + np.abs(t_cpl) + 1e-6)).mean() * 100.0)
                if t_cpl.size > 0 else float('nan')
            )
            md.append(f"| {name:<13} | {mape:>7.2f}% | {gnd_s:>8.2f}% | {cpl_s:>8.2f}% |")
        md.append("")

    # --- 2. Per-layer GND error ---
    md.append("## 2. Per-layer GND magnitude per cuboid (z bucket)")
    md.append("")
    md.append("Buckets approximate intel22 metal layer z-positions. Each model's "
              "predicted c_gnd_seg per cuboid is summarized.")
    md.append("")
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
    md.append("| Layer | z-range (μm) | "
              + " | ".join(f"{n} mean GND" for n in dumps.keys()) + " |")
    md.append("|-------|--------------|"
              + "|".join(["-------------:"] * len(dumps)) + "|")
    for label, lo, hi in LAYER_BUCKETS:
        row = [f" {label} ", f" {lo:>4.2f}–{hi:<4.2f} "]
        for name, d in dumps.items():
            z = d['cuboid_z']
            g = d['cuboid_gnd']
            in_bucket = (z >= lo) & (z < hi)
            if in_bucket.sum() == 0:
                row.append("           — ")
            else:
                vals = g[in_bucket]
                row.append(f" {vals.mean():>6.4f} fF (n={int(in_bucket.sum()):,}) ")
        md.append("|" + "|".join(row) + "|")
    md.append("")

    # --- 3. Per-net-size MAPE quartiles ---
    md.append("## 3. Per-net-size MAPE (cuboid count quartiles)")
    md.append("")
    sizes = ref['net_size']
    qs = np.quantile(sizes, [0.25, 0.5, 0.75])
    bins = [
        (f"≤Q1 ({qs[0]:.0f})",   0,           qs[0]),
        (f"Q1-Q2 (≤{qs[1]:.0f})", qs[0],      qs[1]),
        (f"Q2-Q3 (≤{qs[2]:.0f})", qs[1],      qs[2]),
        (f"Q3+",                   qs[2],     1e9),
    ]
    md.append("| Size bin | N nets | "
              + " | ".join(f"{n} MAPE" for n in dumps.keys()) + " |")
    md.append("|----------|-------:|"
              + "|".join(["----:"] * len(dumps)) + "|")
    for label, lo, hi in bins:
        in_bin = (sizes > lo) & (sizes <= hi)
        n = int(in_bin.sum())
        if n == 0:
            continue
        row = [f" {label} ", f" {n} "]
        for name, d in dumps.items():
            mape = mape_pos(d['pred_total'][in_bin], d['y_total'][in_bin])
            row.append(f" {mape:>5.2f}% ")
        md.append("|" + "|".join(row) + "|")
    md.append("")

    # --- 4. Net-level GND residuals ---
    md.append("## 4. Net-level GND residual statistics")
    md.append("")
    md.append("| Model         | Mean signed err | Mean abs err | RMSE | Max abs err |")
    md.append("|---------------|----------------:|-------------:|-----:|------------:|")
    for name, d in dumps.items():
        pos = d['y_gnd'] > 0.005
        if pos.sum() == 0:
            continue
        err = d['pred_gnd'][pos] - d['y_gnd'][pos]
        md.append(f"| {name:<13} | "
                  f"{float(err.mean()):>13.4f} fF | "
                  f"{float(np.abs(err).mean()):>10.4f} fF | "
                  f"{float(np.sqrt(np.mean(err**2))):>5.3f} | "
                  f"{float(np.abs(err).max()):>9.3f} fF |")
    md.append("")

    # --- 5. Net-level totals (MAPE/SMAPE per net) distribution ---
    md.append("## 5. Per-net total MAPE distribution (P10/P50/P90/P99)")
    md.append("")
    md.append("| Model         |  P10 |  P50 |  P90 |  P99 |")
    md.append("|---------------|-----:|-----:|-----:|-----:|")
    for name, d in dumps.items():
        pos = d['y_total'] > 0.005
        if pos.sum() == 0:
            continue
        mape = np.abs(d['pred_total'][pos] - d['y_total'][pos]) / (d['y_total'][pos] + 1e-6) * 100
        md.append(f"| {name:<13} | "
                  f"{np.percentile(mape, 10):>4.1f}% | "
                  f"{np.percentile(mape, 50):>4.1f}% | "
                  f"{np.percentile(mape, 90):>4.1f}% | "
                  f"{np.percentile(mape, 99):>4.1f}% |")
    md.append("")

    out_md = out_dir / 'report_case3_gnd_breakdown.md'
    out_md.write_text('\n'.join(md))
    print('\n'.join(md))
    print(f"\nReport: {out_md}")


if __name__ == '__main__':
    main()
