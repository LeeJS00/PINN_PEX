#!/usr/bin/env python3
"""
scripts/diag_case2_cpl_distribution.py

Case 2 — CPL SMAPE 분포 분해.

For each model dump:
  - Per-aggressor SMAPE distribution (mean/median/percentiles)
  - Filter sweep: SMAPE retained as we threshold golden CPL > τ
  - Top-K Jaccard agreement between predicted and golden top-K aggressors per net
  - Pearson r and rank correlation between pred_cpl and y_cpl on valid edges

Usage:
  python3 scripts/diag_case2_cpl_distribution.py \\
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
            print(f"⚠  skip {name}: {path}")
            continue
        d = dict(np.load(str(path), allow_pickle=True))
        dumps[name] = d
    if not dumps:
        sys.exit("❌ No dumps loaded.")
    # Reorder by target_name so labels align across dumps.
    for name, d in dumps.items():
        names_d = d['target_names'].astype(str)
        order = np.argsort(names_d)
        for k in ['pred_total', 'pred_gnd', 'pred_cpl',
                  'y_total', 'y_gnd', 'y_cpl', 'valid_aggr',
                  'designs', 'net_size', 'net_z_mean', 'target_names']:
            d[k] = d[k][order]
    return dumps


def per_edge_smape(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return 2.0 * np.abs(pred - target) / (np.abs(pred) + np.abs(target) + eps) * 100.0


def topk_agreement(pred_cpl: np.ndarray, y_cpl: np.ndarray,
                   valid_aggr: np.ndarray, k: int = 3) -> tuple[float, float]:
    """For each net, compute top-K (by golden CPL magnitude) Jaccard with the
    predicted top-K. Returns (mean Jaccard, fraction nets with at least one
    valid aggressor)."""
    n_nets = pred_cpl.shape[0]
    jaccards = []
    for i in range(n_nets):
        valid_idx = np.where(valid_aggr[i])[0]
        if valid_idx.size < k:
            continue
        gt_topk = valid_idx[np.argsort(-y_cpl[i, valid_idx])[:k]]
        pr_topk = valid_idx[np.argsort(-pred_cpl[i, valid_idx])[:k]]
        inter = len(set(gt_topk) & set(pr_topk))
        union = len(set(gt_topk) | set(pr_topk))
        jaccards.append(inter / union if union > 0 else 0.0)
    if not jaccards:
        return float('nan'), 0.0
    return float(np.mean(jaccards)), len(jaccards) / n_nets


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

    md = ["# Case 2 — CPL SMAPE Distribution Decomposition", ""]
    md.append("_Note: each dump uses its own per-batch aggressor budget so the "
              "aggressor column ordering may differ across dumps. Per-edge "
              "metrics use each dump's own valid_aggr mask, which is fair for "
              "distribution analysis._")
    md.append("")

    # Per-aggressor SMAPE distribution — use each dump's OWN valid mask.
    md.append("## 1. Per-aggressor SMAPE distribution (each dump's valid edges)")
    md.append("")
    md.append("| Model         | Mean | Median |  P10 |  P50 |  P90 |  P99 | N edges |")
    md.append("|---------------|-----:|-------:|-----:|-----:|-----:|-----:|--------:|")
    for name, d in dumps.items():
        v_d = d['valid_aggr'].astype(bool)
        sm = per_edge_smape(d['pred_cpl'][v_d], d['y_cpl'][v_d])
        md.append(f"| {name:<13} | {sm.mean():>4.1f}% | {np.median(sm):>5.1f}% | "
                  f"{np.percentile(sm,10):>4.1f}% | {np.percentile(sm,50):>4.1f}% | "
                  f"{np.percentile(sm,90):>4.1f}% | {np.percentile(sm,99):>4.1f}% | "
                  f"{sm.size:>6,} |")
    md.append("")

    # Filter sweep — per dump
    md.append("## 2. Filter sweep: SMAPE on edges with golden CPL > τ")
    md.append("")
    thresholds = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    for name, d in dumps.items():
        v_d = d['valid_aggr'].astype(bool)
        full_cpl_y = d['y_cpl'][v_d]
        full_cpl_p = d['pred_cpl'][v_d]
        n_total = full_cpl_y.size
        md.append(f"### {name} (valid edges = {n_total:,})")
        md.append("")
        md.append("| τ (fF) | N edges retained | Retain % | Mean SMAPE | Median SMAPE |")
        md.append("|-------:|-----------------:|---------:|-----------:|-------------:|")
        for τ in thresholds:
            keep = full_cpl_y > τ
            if keep.sum() == 0:
                continue
            sm = per_edge_smape(full_cpl_p[keep], full_cpl_y[keep])
            md.append(f"| {τ:>6.3f} | {int(keep.sum()):>15,} | "
                      f"{keep.sum() / n_total * 100:>7.1f}% | "
                      f"{sm.mean():>9.2f}% | "
                      f"{np.median(sm):>11.2f}% |")
        md.append("")

    # Top-K Jaccard
    md.append("## 3. Top-K agreement (predicted vs golden top-K aggressors per net)")
    md.append("")
    md.append("| Model         | Top-1 Jaccard | Top-3 Jaccard | Top-5 Jaccard | Top-10 Jaccard |")
    md.append("|---------------|--------------:|--------------:|--------------:|---------------:|")
    for name, d in dumps.items():
        valid_d = d['valid_aggr'].astype(bool)
        row = [f"{name:<13}"]
        for k in [1, 3, 5, 10]:
            j, _ = topk_agreement(d['pred_cpl'], d['y_cpl'], valid_d, k=k)
            row.append(f"{j:>13.3f}")
        md.append(f"| {' | '.join(row)} |")
    md.append("")

    # Pearson + rank correlation on valid edges
    md.append("## 4. Correlation between predicted and golden CPL (valid edges)")
    md.append("")
    md.append("| Model         | Pearson r (raw) | Pearson r (log1p) | Spearman ρ |")
    md.append("|---------------|----------------:|------------------:|-----------:|")
    from scipy.stats import spearmanr  # may not be installed
    for name, d in dumps.items():
        v_d = d['valid_aggr'].astype(bool)
        p = d['pred_cpl'][v_d]
        t = d['y_cpl'][v_d]
        if p.size < 2:
            md.append(f"| {name:<13} | n/a | n/a | n/a |")
            continue
        # Pearson raw
        r_raw = float(np.corrcoef(p, t)[0, 1]) if p.std() > 0 else float('nan')
        # Pearson on log1p (handles long tail)
        r_log = float(np.corrcoef(np.log1p(p.clip(min=0)), np.log1p(t.clip(min=0)))[0, 1])
        # Spearman
        try:
            ρ, _ = spearmanr(p, t)
            ρ_str = f"{float(ρ):>10.3f}"
        except Exception:
            ρ_str = "      n/a"
        md.append(f"| {name:<13} | {r_raw:>15.3f} | {r_log:>17.3f} | {ρ_str} |")
    md.append("")

    # Quick interpretation
    md.append("## Interpretation")
    md.append("")
    # Find τ at which mean SMAPE drops below 100% for v2 (best model)
    if 'dspinn_v2' in dumps:
        d = dumps['dspinn_v2']
        v_d = d['valid_aggr'].astype(bool)
        full_y = d['y_cpl'][v_d]
        full_p = d['pred_cpl'][v_d]
        for τ in thresholds:
            keep = full_y > τ
            if keep.sum() == 0:
                continue
            sm = per_edge_smape(full_p[keep], full_y[keep])
            if sm.mean() < 100:
                md.append(f"- **dspinn_v2 mean SMAPE drops below 100% at τ={τ} fF** "
                          f"({sm.mean():.1f}%, {keep.sum() / full_y.size * 100:.0f}% of edges retained).")
                break
        # Top-K significance
        j3, _ = topk_agreement(d['pred_cpl'], d['y_cpl'], d['valid_aggr'].astype(bool), k=3)
        if j3 > 0.4:
            md.append(f"- dspinn_v2 Top-3 Jaccard = {j3:.2f} → distribution learning works "
                      f"(Top-K aggressors mostly correct, magnitude is the residual issue).")

    out_md = out_dir / 'report_case2_cpl_distribution.md'
    out_md.write_text('\n'.join(md))
    print('\n'.join(md))
    print(f"\nReport: {out_md}")


if __name__ == '__main__':
    main()
