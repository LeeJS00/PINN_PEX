#!/usr/bin/env python3
"""
scripts/diag_case5_topology.py

Case 5 — Topology mismatch fairness check.

The PINN tool is "SPEF-free": it emits per-net total caps and per-aggressor
CPL, but the segment topology may differ from StarRC's. This diagnostic
checks whether the per-net comparison is genuinely fair:

  1. Aggressor coverage — for each net, how many predicted nonzero edges
     exist outside the golden valid_aggr set (= leaked predictions to
     untracked aggressors)?
  2. Aggressor identity match — among golden valid aggressors, what
     fraction received any predicted contribution at all?
  3. Per-net CPL sum: golden vs predicted total CPL across valid + invalid
     aggressor columns. Where does predicted "extra" CPL go?
  4. Edge count mismatch: model emits more or fewer edges than golden has?

Usage:
  python3 scripts/diag_case5_topology.py \\
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
    md = ["# Case 5 — Topology Mismatch Fairness Check", ""]
    md.append("Each model's pred_cpl is a (net, MAX_AGGR) matrix. Golden y_cpl "
              "is the same shape. valid_aggr marks the columns where the golden "
              "SPEF reports a nonzero CPL value. Below we check whether "
              "predicted CPL leaks outside the valid set, and whether the model "
              "covers the golden aggressors.")
    md.append("")

    for name, d in dumps.items():
        md.append(f"## {name}")
        md.append("")
        valid = d['valid_aggr'].astype(bool)
        pred = d['pred_cpl']
        gold = d['y_cpl']

        n_nets, max_aggr = pred.shape
        n_valid_total = int(valid.sum())
        n_invalid_total = int((~valid).sum())

        # 1. Predicted nonzero edges outside valid set ("leaked")
        nonzero_pred = pred > 1e-6
        leaked = nonzero_pred & ~valid
        leak_count = int(leaked.sum())
        leak_per_net = leaked.sum(axis=1)
        leaked_mass = float(pred[leaked].sum())
        valid_mass = float(pred[valid].sum())
        gold_mass = float(gold[valid].sum())

        md.append("### Predicted CPL distribution")
        md.append("")
        md.append("| Quantity | Value |")
        md.append("|----------|------:|")
        md.append(f"| Nets | {n_nets} |")
        md.append(f"| Max aggressors per net (MAX_AGGR) | {max_aggr} |")
        md.append(f"| Valid (golden has CPL) entries  | {n_valid_total:,} |")
        md.append(f"| Invalid (golden=0) entries      | {n_invalid_total:,} |")
        md.append(f"| Predicted nonzero in **valid**  | {int((nonzero_pred & valid).sum()):,} |")
        md.append(f"| Predicted nonzero in **invalid** (leakage) | {leak_count:,} |")
        md.append(f"| Σ pred CPL on valid columns     | {valid_mass:.3f} fF |")
        md.append(f"| Σ pred CPL on invalid columns (leaked mass) | {leaked_mass:.3f} fF |")
        md.append(f"| Σ golden CPL                    | {gold_mass:.3f} fF |")
        md.append(f"| **Leak ratio** (mass leaked / total pred) | "
                  f"{leaked_mass / (leaked_mass + valid_mass + 1e-9) * 100:.2f}% |")
        md.append("")

        # 2. Coverage: among valid aggressors, what fraction have non-zero prediction?
        covered = valid & nonzero_pred
        cov_per_net = covered.sum(axis=1) / valid.sum(axis=1).clip(min=1)
        md.append("### Aggressor coverage (predicted nonzero on valid columns)")
        md.append("")
        md.append("| Stat | Value |")
        md.append("|------|------:|")
        md.append(f"| Mean coverage   | {cov_per_net.mean() * 100:.1f}% |")
        md.append(f"| Median coverage | {float(np.median(cov_per_net)) * 100:.1f}% |")
        md.append(f"| P10 coverage    | {float(np.percentile(cov_per_net, 10)) * 100:.1f}% |")
        md.append(f"| P90 coverage    | {float(np.percentile(cov_per_net, 90)) * 100:.1f}% |")
        md.append("")

        # 3. Per-net leak distribution
        md.append("### Per-net leakage (extra predicted edges)")
        md.append("")
        md.append("| Stat | Value |")
        md.append("|------|------:|")
        md.append(f"| Nets with leak count = 0  | {int((leak_per_net == 0).sum()):,} ({(leak_per_net == 0).mean() * 100:.1f}%) |")
        md.append(f"| Nets with leak count ≥ 1  | {int((leak_per_net >= 1).sum()):,} ({(leak_per_net >= 1).mean() * 100:.1f}%) |")
        md.append(f"| Nets with leak count ≥ 10 | {int((leak_per_net >= 10).sum()):,} |")
        md.append(f"| Max leak count per net    | {int(leak_per_net.max())} |")
        md.append(f"| Mean leak per net (when >0) | "
                  f"{leak_per_net[leak_per_net > 0].mean() if (leak_per_net > 0).any() else 0:.1f} |")
        md.append("")

        # 4. Per-net pred CPL sum vs golden
        pred_cpl_sum = (pred * valid).sum(axis=1)
        gold_cpl_sum = (gold * valid).sum(axis=1)
        leaked_cpl_sum = (pred * (~valid)).sum(axis=1)
        net_pos = gold_cpl_sum > 0.005
        if net_pos.sum() > 0:
            ratio = pred_cpl_sum[net_pos] / (gold_cpl_sum[net_pos] + 1e-6)
            leak_ratio = leaked_cpl_sum[net_pos] / (gold_cpl_sum[net_pos] + 1e-6)
            md.append("### Per-net CPL totals (on nets with nonzero golden CPL)")
            md.append("")
            md.append(f"| Stat | Value |")
            md.append(f"|------|------:|")
            md.append(f"| Pred Σ CPL / Golden Σ CPL — mean   | {ratio.mean():.3f} |")
            md.append(f"| Pred Σ CPL / Golden Σ CPL — median | {float(np.median(ratio)):.3f} |")
            md.append(f"| Leaked CPL / Golden CPL — mean | {leak_ratio.mean():.3f} |")
            md.append(f"| Leaked CPL / Golden CPL — median | {float(np.median(leak_ratio)):.3f} |")
            md.append("")

    # Cross-model summary
    md.append("## Cross-model leak summary")
    md.append("")
    md.append("| Model | Total leak edges | Total leak mass | Leak fraction | Mean coverage |")
    md.append("|-------|-----------------:|----------------:|--------------:|--------------:|")
    for name, d in dumps.items():
        valid = d['valid_aggr'].astype(bool)
        pred = d['pred_cpl']
        nonzero = pred > 1e-6
        leaked = nonzero & ~valid
        valid_mass = float(pred[valid].sum())
        leaked_mass = float(pred[leaked].sum())
        cov = (valid & nonzero).sum(axis=1) / valid.sum(axis=1).clip(min=1)
        md.append(f"| {name:<13} | "
                  f"{int(leaked.sum()):>14,} | "
                  f"{leaked_mass:>11.3f} fF | "
                  f"{leaked_mass / (leaked_mass + valid_mass + 1e-9) * 100:>11.2f}% | "
                  f"{cov.mean() * 100:>11.1f}% |")
    md.append("")

    md.append("## Interpretation")
    md.append("")
    md.append("- **Coverage** = fraction of golden aggressors that received nonzero "
              "prediction. Low coverage → topology mismatch (model not finding the "
              "edge at all).")
    md.append("- **Leak fraction** = predicted CPL mass that landed in invalid "
              "columns (where golden=0). Low = SPEF-free comparison stays fair.")
    md.append("- For SPEF-free fairness: as long as leak fraction is low (<10%) "
              "and coverage is high (>70%), per-net comparison via valid_aggr_mask "
              "is genuine.")

    out_md = out_dir / 'report_case5_topology.md'
    out_md.write_text('\n'.join(md))
    print('\n'.join(md))
    print(f"\nReport: {out_md}")


if __name__ == '__main__':
    main()
