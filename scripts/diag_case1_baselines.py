#!/usr/bin/env python3
"""
scripts/diag_case1_baselines.py

Case 1 — Trivial baseline SMAPE floor:
  Compares trained model SMAPE vs simple non-learned baselines to determine
  whether the persistent CPL SMAPE ceiling (320–376%) is a metric artifact
  or a genuine model failure.

Baselines:
  - Constant: predict the mean of training set Y_total / Y_gnd / Y_cpl
  - Zero CPL: predict Y_total/Y_gnd from data, but pred_cpl = 0 everywhere
  - Random small: small uniform random predictions
  - Per-net oracle (sum-only): pred_total = Y_total exactly, distribute CPL
    uniformly across valid aggressors (knows total but not edge structure)

For each baseline + the trained model dump, computes:
  - Net-level MAPE (via standard formula)
  - Total / GND / CPL SMAPE

Outputs:
  output_intel22/active_learning/diag_phase_a/report_case1_baselines.md

Usage:
  python3 scripts/diag_case1_baselines.py \\
      --dumps v10b dspinn_v1_new dspinn_v2
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/home/jslee/projects/PINNPEX")

import numpy as np


def smape(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    """Symmetric MAPE (%): 2*|p-t| / (|p|+|t|) * 100, masked to non-zero target."""
    pred = np.asarray(pred).ravel()
    target = np.asarray(target).ravel()
    if pred.shape != target.shape:
        return float('nan')
    pos = target > 0.005
    if pos.sum() == 0:
        return float('nan')
    p, t = pred[pos], target[pos]
    return float((2.0 * np.abs(p - t) / (np.abs(p) + np.abs(t) + eps)).mean() * 100.0)


def net_mape(pred_total: np.ndarray, y_total: np.ndarray) -> float:
    pos = y_total >= 0.005
    if pos.sum() == 0:
        return float('nan')
    return float(np.mean(np.abs(pred_total[pos] - y_total[pos]) /
                         (y_total[pos] + 1e-6)) * 100.0)


def cpl_smape_masked(pred_cpl: np.ndarray, y_cpl: np.ndarray,
                     valid_aggr: np.ndarray, eps: float = 1e-6) -> float:
    """Per-aggressor SMAPE, masked to valid (golden cpl > 0) entries only."""
    pred_flat   = pred_cpl[valid_aggr]
    target_flat = y_cpl[valid_aggr]
    if target_flat.size == 0:
        return float('nan')
    return float((2.0 * np.abs(pred_flat - target_flat) /
                  (np.abs(pred_flat) + np.abs(target_flat) + eps)).mean() * 100.0)


def evaluate_predictions(label: str,
                         pred_total: np.ndarray,
                         pred_gnd:   np.ndarray,
                         pred_cpl:   np.ndarray,
                         y_total:    np.ndarray,
                         y_gnd:      np.ndarray,
                         y_cpl:      np.ndarray,
                         valid_aggr: np.ndarray) -> dict:
    return {
        'label':      label,
        'net_mape':   net_mape(pred_total, y_total),
        'tot_smape':  smape(pred_total, y_total),
        'gnd_smape':  smape(pred_gnd, y_gnd),
        'cpl_smape':  cpl_smape_masked(pred_cpl, y_cpl, valid_aggr),
    }


def make_baselines(y_total, y_gnd, y_cpl, valid_aggr,
                   rng: np.random.Generator) -> dict[str, dict]:
    """Compute trivial baselines from golden labels alone."""
    out = {}

    # 1. Constant — predict mean over the validation set itself
    const_total = np.full_like(y_total, y_total.mean())
    const_gnd   = np.full_like(y_gnd, y_gnd.mean())
    cpl_mean = (y_cpl * valid_aggr.astype(np.float32))[valid_aggr.astype(bool)].mean()
    const_cpl = np.full_like(y_cpl, cpl_mean) * valid_aggr.astype(np.float32)
    out['Constant_mean'] = evaluate_predictions(
        'Constant_mean', const_total, const_gnd, const_cpl,
        y_total, y_gnd, y_cpl, valid_aggr.astype(bool))

    # 2. Zero CPL — keep total/gnd at golden, predict CPL=0
    zero_cpl = np.zeros_like(y_cpl)
    out['ZeroCPL_oracle_gnd'] = evaluate_predictions(
        'ZeroCPL_oracle_gnd', y_total, y_gnd, zero_cpl,
        y_total, y_gnd, y_cpl, valid_aggr.astype(bool))

    # 3. Random small — random ~mean magnitude
    rand_total = rng.uniform(0.1, y_total.max() * 0.5, size=y_total.shape).astype(np.float32)
    rand_gnd   = rng.uniform(0.1, y_gnd.max() * 0.5, size=y_gnd.shape).astype(np.float32)
    rand_cpl   = rng.uniform(0.0, cpl_mean * 2.0, size=y_cpl.shape).astype(np.float32) * valid_aggr.astype(np.float32)
    out['Random_uniform'] = evaluate_predictions(
        'Random_uniform', rand_total, rand_gnd, rand_cpl,
        y_total, y_gnd, y_cpl, valid_aggr.astype(bool))

    # 4. Per-net oracle — knows total exactly, distributes uniformly across valid aggressors
    n_valid_aggr = valid_aggr.sum(axis=1, keepdims=True).clip(min=1)
    oracle_cpl_sum = y_cpl.sum(axis=1, keepdims=True)  # known per-net CPL total
    oracle_cpl = (oracle_cpl_sum / n_valid_aggr) * valid_aggr.astype(np.float32)
    out['Oracle_sum'] = evaluate_predictions(
        'Oracle_sum', y_total, y_gnd, oracle_cpl,
        y_total, y_gnd, y_cpl, valid_aggr.astype(bool))

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dumps', nargs='+', required=True,
                    help='Model names whose eval_dump.npz files exist '
                         'under output_intel22/active_learning/<name>/')
    ap.add_argument('--output_root', default='/home/jslee/projects/PINNPEX/output_intel22')
    args = ap.parse_args()

    output_root = Path(args.output_root)
    al_root = output_root / 'active_learning'
    out_dir = al_root / 'diag_phase_a'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all dumps; baselines computed from the FIRST dump's labels (all dumps
    # use the same val set so labels should match — sanity-check this).
    dumps: dict[str, dict] = {}
    for name in args.dumps:
        path = al_root / name / 'eval_dump.npz'
        if not path.exists():
            print(f"⚠  skip {name}: {path} missing")
            continue
        d = dict(np.load(str(path), allow_pickle=True))
        dumps[name] = d

    if not dumps:
        print("❌ No dumps loaded.")
        sys.exit(1)

    first_name = list(dumps.keys())[0]
    ref = dumps[first_name]
    # NetGroupedSampler shuffles nets per run — reorder every dump by target_name
    # so all dumps reference the same nets in the same order.
    ref_names = ref['target_names'].astype(str)
    sort_idx_ref = np.argsort(ref_names)
    for name, d in dumps.items():
        names = d['target_names'].astype(str)
        sort_idx = np.argsort(names)
        for k in ['pred_total', 'pred_gnd', 'pred_cpl',
                  'y_total', 'y_gnd', 'y_cpl', 'valid_aggr',
                  'designs', 'net_size', 'net_z_mean', 'target_names']:
            d[k] = d[k][sort_idx]
    ref = dumps[first_name]
    y_total = ref['y_total']
    y_gnd   = ref['y_gnd']
    y_cpl   = ref['y_cpl']
    valid_aggr_bool = ref['valid_aggr'].astype(bool)
    n_nets = y_total.shape[0]
    print(f"Loaded {len(dumps)} dump(s); {n_nets} nets, MAX_AGGR={y_cpl.shape[1]} "
          f"(reordered by target_name)")

    # Baselines (label-only, no model)
    rng = np.random.default_rng(42)
    baseline_rows = make_baselines(y_total, y_gnd, y_cpl,
                                   ref['valid_aggr'].astype(np.int8), rng)

    # Trained model rows
    model_rows: dict[str, dict] = {}
    for name, d in dumps.items():
        # Sanity: label arrays should be identical across dumps
        assert np.allclose(d['y_total'], y_total, atol=1e-6), f"label mismatch in {name}"
        model_rows[name] = evaluate_predictions(
            name,
            d['pred_total'], d['pred_gnd'], d['pred_cpl'],
            d['y_total'], d['y_gnd'], d['y_cpl'],
            d['valid_aggr'].astype(bool))

    # Render report
    md = []
    md.append("# Case 1 — Trivial Baseline SMAPE Floor")
    md.append("")
    md.append(f"_Validation set: {n_nets} nets, MAX_AGGR={y_cpl.shape[1]}_")
    md.append(f"_Golden CPL stats: nonzero edges={int(valid_aggr_bool.sum()):,}, "
              f"mean={float(y_cpl[valid_aggr_bool].mean()):.4f} fF, "
              f"median={float(np.median(y_cpl[valid_aggr_bool])):.4f} fF, "
              f"max={float(y_cpl[valid_aggr_bool].max()):.4f} fF_")
    md.append("")
    md.append("## Trivial baselines (no learned model)")
    md.append("")
    md.append("| Baseline                | Net MAPE | Tot SMAPE | GND SMAPE | CPL SMAPE |")
    md.append("|-------------------------|---------:|----------:|----------:|----------:|")
    for k, r in baseline_rows.items():
        md.append(f"| {r['label']:<23} | "
                  f"{r['net_mape']:>7.2f}% | "
                  f"{r['tot_smape']:>8.2f}% | "
                  f"{r['gnd_smape']:>8.2f}% | "
                  f"{r['cpl_smape']:>8.2f}% |")
    md.append("")
    md.append("## Trained models")
    md.append("")
    md.append("| Model                   | Net MAPE | Tot SMAPE | GND SMAPE | CPL SMAPE |")
    md.append("|-------------------------|---------:|----------:|----------:|----------:|")
    for name, r in model_rows.items():
        md.append(f"| {r['label']:<23} | "
                  f"{r['net_mape']:>7.2f}% | "
                  f"{r['tot_smape']:>8.2f}% | "
                  f"{r['gnd_smape']:>8.2f}% | "
                  f"{r['cpl_smape']:>8.2f}% |")
    md.append("")
    # Quick interpretation
    cpl_floor = max(baseline_rows['Oracle_sum']['cpl_smape'],
                    baseline_rows['ZeroCPL_oracle_gnd']['cpl_smape'])
    md.append("## Interpretation")
    md.append("")
    if cpl_floor > 200:
        md.append(f"- ⚠  **Oracle baselines also produce CPL SMAPE > 200%** — "
                  "indicates that even with perfect knowledge of net-level CPL totals, "
                  "the per-aggressor SMAPE metric reports very large numbers. "
                  "This is a strong signal that **CPL SMAPE 320% is a metric artifact**, "
                  "not an architectural failure of DS-PINN.")
    else:
        md.append(f"- ZeroCPL oracle (knows GND, predicts 0 CPL): {baseline_rows['ZeroCPL_oracle_gnd']['cpl_smape']:.1f}% — "
                  "this is the SMAPE 'floor' for getting CPL completely wrong.")
        md.append(f"- Sum-oracle: {baseline_rows['Oracle_sum']['cpl_smape']:.1f}%.")

    if model_rows:
        any_model = list(model_rows.values())[0]
        if any_model['cpl_smape'] > baseline_rows['Constant_mean']['cpl_smape']:
            md.append(f"- ⚠  **Trained model CPL SMAPE ({any_model['cpl_smape']:.1f}%) is "
                      f"WORSE than constant-mean baseline ({baseline_rows['Constant_mean']['cpl_smape']:.1f}%)** — "
                      "the model is not learning useful CPL distribution at all by this metric.")
        else:
            md.append(f"- Model CPL SMAPE beats Constant baseline — model has *some* CPL signal.")

    out_md = out_dir / 'report_case1_baselines.md'
    out_md.write_text('\n'.join(md))
    print('\n'.join(md))
    print(f"\nReport: {out_md}")


if __name__ == '__main__':
    main()
