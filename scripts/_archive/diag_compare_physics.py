#!/usr/bin/env python3
"""
scripts/diag_compare_physics.py

Compare physics-only baseline vs learned-modifier predictions across models.
Determines whether learned MLP corrections are HELPING or HURTING.

Usage:
  python3 scripts/diag_compare_physics.py
"""
import sys
from pathlib import Path

sys.path.insert(0, "/home/jslee/projects/PINNPEX")

import numpy as np


def smape_pos(pred, target, eps=1e-6):
    pos = target > 0.005
    if pos.sum() == 0:
        return float('nan')
    p, t = pred[pos], target[pos]
    return float((2.0 * np.abs(p - t) / (np.abs(p) + np.abs(t) + eps)).mean() * 100.0)


def mape_pos(pred, target):
    pos = target >= 0.005
    if pos.sum() == 0:
        return float('nan')
    return float(np.mean(np.abs(pred[pos] - target[pos]) /
                         (target[pos] + 1e-6)) * 100.0)


def cpl_smape(pred_cpl, y_cpl, valid_aggr, eps=1e-6):
    valid = valid_aggr.astype(bool)
    if valid.sum() == 0:
        return float('nan')
    p, t = pred_cpl[valid], y_cpl[valid]
    return float((2.0 * np.abs(p - t) / (np.abs(p) + np.abs(t) + eps)).mean() * 100.0)


def cpl_ratio_median(pred_cpl, y_cpl, valid_aggr):
    valid = valid_aggr.astype(bool)
    pred_sum = (pred_cpl * valid).sum(axis=1)
    gold_sum = (y_cpl * valid).sum(axis=1)
    pos = gold_sum > 0.001
    if pos.sum() == 0:
        return float('nan')
    return float(np.median(pred_sum[pos] / (gold_sum[pos] + 1e-6)) * 100)


def gnd_ratio_median(pred_gnd, y_gnd):
    pos = y_gnd > 0.005
    if pos.sum() == 0:
        return float('nan')
    return float(np.median(pred_gnd[pos] / (y_gnd[pos] + 1e-6)) * 100)


def evaluate(name, d):
    valid = d['valid_aggr']
    return {
        'label': name,
        'net_mape':       mape_pos(d['pred_total'], d['y_total']),
        'tot_smape':      smape_pos(d['pred_total'], d['y_total']),
        'gnd_smape':      smape_pos(d['pred_gnd'],   d['y_gnd']),
        'cpl_smape':      cpl_smape(d['pred_cpl'], d['y_cpl'], valid),
        'gnd_ratio_med':  gnd_ratio_median(d['pred_gnd'], d['y_gnd']),
        'cpl_ratio_med':  cpl_ratio_median(d['pred_cpl'], d['y_cpl'], valid),
    }


def main():
    al_root = Path('/home/jslee/projects/PINNPEX/output_intel22/active_learning')

    # Load by target_name and align
    cases = [
        ('v10b (physics-only)', al_root / 'v10b' / 'eval_dump_physics.npz'),
        ('v10b (trained)',      al_root / 'v10b' / 'eval_dump.npz'),
        ('dspinn_v1_new (trained)', al_root / 'dspinn_v1_new' / 'eval_dump.npz'),
        ('dspinn_v2 (trained, 5k step)', al_root / 'dspinn_v2' / 'eval_dump.npz'),
    ]
    rows = []
    for label, path in cases:
        if not path.exists():
            print(f"⚠ skip {label}: {path}")
            continue
        d = dict(np.load(str(path), allow_pickle=True))
        # Reorder by target_names
        names = d['target_names'].astype(str)
        order = np.argsort(names)
        for k in ['pred_total', 'pred_gnd', 'pred_cpl',
                  'y_total', 'y_gnd', 'y_cpl', 'valid_aggr',
                  'designs', 'net_size', 'net_z_mean', 'target_names']:
            d[k] = d[k][order]
        rows.append(evaluate(label, d))

    print()
    print("# Physics-Only vs Trained Comparison")
    print()
    print(f"{'Model':<35} | {'Net MAPE':>9} | {'Tot SMAPE':>9} | {'GND SMAPE':>9} | {'CPL SMAPE':>9} | {'GND ratio':>9} | {'CPL ratio':>9}")
    print(f"{'-'*35} | {'-'*9} | {'-'*9} | {'-'*9} | {'-'*9} | {'-'*9} | {'-'*9}")
    for r in rows:
        print(f"{r['label']:<35} | "
              f"{r['net_mape']:>8.2f}% | "
              f"{r['tot_smape']:>8.2f}% | "
              f"{r['gnd_smape']:>8.2f}% | "
              f"{r['cpl_smape']:>8.2f}% | "
              f"{r['gnd_ratio_med']:>8.1f}% | "
              f"{r['cpl_ratio_med']:>8.1f}%")
    print()

    # Save markdown
    out_md = al_root / 'diag_phase_a' / 'report_physics_compare.md'
    md = ["# Physics-Only Baseline vs Trained Models", ""]
    md.append("Tests whether learned MLP corrections (gnd_modifier, cpl_modifier, "
              "cpl_residual) are reducing or increasing error vs the rule-based "
              "physics formula alone.")
    md.append("")
    md.append("Physics-only mode (`--physics_only`) zeros the last linear of "
              "gnd_mlp and cpl_mlp, so gnd_modifier = exp(0) = 1.0 and "
              "cpl_modifier = exp(0) = 1.0. layer_scale_phys_gnd is also reset "
              "to physics-calibrated init values.")
    md.append("")
    md.append("| Model | Net MAPE | Tot SMAPE | GND SMAPE | CPL SMAPE | GND ratio (med) | CPL ratio (med) |")
    md.append("|-------|---------:|----------:|----------:|----------:|----------------:|----------------:|")
    for r in rows:
        md.append(f"| {r['label']:<33} | "
                  f"{r['net_mape']:>7.2f}% | "
                  f"{r['tot_smape']:>8.2f}% | "
                  f"{r['gnd_smape']:>8.2f}% | "
                  f"{r['cpl_smape']:>8.2f}% | "
                  f"{r['gnd_ratio_med']:>14.1f}% | "
                  f"{r['cpl_ratio_med']:>14.1f}% |")
    md.append("")
    out_md.write_text('\n'.join(md))
    print(f"Report: {out_md}")


if __name__ == '__main__':
    main()
