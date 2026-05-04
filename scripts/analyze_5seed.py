"""
Analyze the 5-seed measurement protocol output.

For each (variant, seed):
  - Parse al_5seed_<variant>_seed<N>.log to extract per-step MAPE, CPL ratio,
    composite score, and BEST flag history.
  - Identify the final BEST (lowest net_mape encountered).

Aggregate per variant (typically 5 seeds):
  - Median, IQR (p25, p75), min, max for (best_mape, best_cpl_ratio,
    best_composite, last_step_mape, last_step_cpl_ratio).

Statistical comparison:
  - Mann-Whitney U for distributional differences:
      v3_baseline vs v4_full_calib
      v3_baseline vs v5_gnd_only
      v4_full_calib vs v5_gnd_only

Output: prints a table to stdout, saves CSV under
output_intel22/active_learning/m5_summary/.

Usage:
    python3 scripts/analyze_5seed.py
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "output_intel22"
CKPT_DIR = ROOT / "output_intel22" / "active_learning"
OUT_DIR = ROOT / "output_intel22" / "active_learning" / "m5_summary"


VARIANTS = ['v3_baseline', 'v4_full_calib', 'v5_gnd_only']
SEEDS = [0, 1, 2, 3, 4]


# Log line patterns
RE_STEP = re.compile(r'^\>\>\> \[FineTuner\] Step (\d+):')
RE_MAPE = re.compile(r'Net-level MAPE\s+:\s+([\d.]+)%')
RE_CPL_RATIO = re.compile(r'CPL ratio \(med\)\s+:\s+([\d.]+)%')
RE_COMPOSITE = re.compile(r'Composite Score:\s+([\d.]+)\s*(🌟 BEST!)?')
RE_REAL_TOT_SMAPE = re.compile(r'True SMAPE \[%\]\s+->\s+Tot:\s+([\d.]+)')
RE_REAL_GND_SMAPE = re.compile(r'GND:\s+([\d.]+)\s+\|\s+CPL')
RE_REAL_CPL_SMAPE = re.compile(r'CPL \(per-edge\):\s+([\d.]+)')


def parse_log(log_path: Path) -> pd.DataFrame:
    """Parse step-level metrics from a 5seed log file.

    Returns DataFrame with columns: step, net_mape, cpl_ratio_med, composite,
    is_best, real_tot_smape, real_gnd_smape, real_cpl_smape.
    """
    rows: list[dict] = []
    if not log_path.exists():
        return pd.DataFrame()

    cur: dict = {}
    with open(log_path) as f:
        for line in f:
            m = RE_STEP.search(line)
            if m:
                if cur:
                    rows.append(cur)
                cur = {'step': int(m.group(1))}
                continue
            if not cur: continue
            m = RE_MAPE.search(line)
            if m: cur['net_mape'] = float(m.group(1)); continue
            m = RE_CPL_RATIO.search(line)
            if m: cur['cpl_ratio_med'] = float(m.group(1)); continue
            m = RE_COMPOSITE.search(line)
            if m:
                cur['composite'] = float(m.group(1))
                cur['is_best'] = bool(m.group(2))
                continue
            m = RE_REAL_TOT_SMAPE.search(line)
            if m: cur['real_tot_smape'] = float(m.group(1)); continue
            m = RE_REAL_GND_SMAPE.search(line)
            if m: cur['real_gnd_smape'] = float(m.group(1)); continue
            m = RE_REAL_CPL_SMAPE.search(line)
            if m: cur['real_cpl_smape'] = float(m.group(1)); continue
    if cur: rows.append(cur)

    return pd.DataFrame(rows)


def aggregate(df_per_run: pd.DataFrame) -> dict:
    """For one (variant, seed) DataFrame, return summary metrics."""
    if df_per_run.empty:
        return {'best_mape': float('nan'), 'best_cpl_ratio': float('nan'),
                'best_composite': float('nan'), 'last_step': 0,
                'last_mape': float('nan'), 'n_steps': 0}
    # "Best" = row with min net_mape across the run (matches finetuner's
    # save criterion).
    valid = df_per_run.dropna(subset=['net_mape'])
    if valid.empty:
        best_idx = -1
    else:
        best_idx = valid['net_mape'].idxmin()
    best = df_per_run.loc[best_idx] if best_idx != -1 else None
    last  = df_per_run.iloc[-1]
    return {
        'best_mape':       float(best['net_mape'])      if best is not None and pd.notna(best.get('net_mape')) else float('nan'),
        'best_cpl_ratio':  float(best['cpl_ratio_med']) if best is not None and pd.notna(best.get('cpl_ratio_med')) else float('nan'),
        'best_composite':  float(best['composite'])     if best is not None and pd.notna(best.get('composite')) else float('nan'),
        'best_step':       int(best['step'])            if best is not None else 0,
        'last_step':       int(last['step']),
        'last_mape':       float(last.get('net_mape', float('nan'))) if pd.notna(last.get('net_mape', np.nan)) else float('nan'),
        'last_cpl_ratio':  float(last.get('cpl_ratio_med', float('nan'))) if pd.notna(last.get('cpl_ratio_med', np.nan)) else float('nan'),
        'n_steps':         len(df_per_run),
    }


def per_variant_stats(rows: list[dict]) -> dict:
    """Compute median, p25, p75, min, max across N seeds for a variant."""
    df = pd.DataFrame(rows)
    out = {'n_seeds': len(df)}
    for col in ['best_mape', 'best_cpl_ratio', 'best_composite']:
        if col not in df.columns or df[col].isna().all():
            continue
        v = df[col].dropna()
        out[f'{col}_median'] = float(v.median())
        out[f'{col}_p25']    = float(v.quantile(0.25))
        out[f'{col}_p75']    = float(v.quantile(0.75))
        out[f'{col}_min']    = float(v.min())
        out[f'{col}_max']    = float(v.max())
        out[f'{col}_mean']   = float(v.mean())
        out[f'{col}_std']    = float(v.std())
    return out


def mann_whitney(a: list[float], b: list[float]) -> tuple[float, float]:
    """Two-sided Mann-Whitney U test. Returns (U, p)."""
    try:
        from scipy.stats import mannwhitneyu
    except ImportError:
        return float('nan'), float('nan')
    a = [x for x in a if not (isinstance(x, float) and (x != x))]  # drop NaN
    b = [x for x in b if not (isinstance(x, float) and (x != x))]
    if len(a) < 2 or len(b) < 2:
        return float('nan'), float('nan')
    u, p = mannwhitneyu(a, b, alternative='two-sided')
    return float(u), float(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', nargs='+', default=VARIANTS)
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--strict', action='store_true',
                    help='Fail if any (variant, seed) log is missing or has 0 steps.')
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f">>> Parsing logs across {args.variants} × seeds {args.seeds}")
    per_run_records: list[dict] = []
    raw_step_dfs: dict = {}
    for variant in args.variants:
        for seed in args.seeds:
            log = LOG_DIR / f"al_5seed_{variant}_seed{seed}.log"
            df = parse_log(log)
            raw_step_dfs[(variant, seed)] = df
            agg = aggregate(df)
            agg['variant'] = variant; agg['seed'] = seed
            agg['log_path'] = str(log)
            per_run_records.append(agg)
            if df.empty:
                tag = '(missing/empty)'
                if args.strict:
                    print(f"  [FAIL] {log.name} {tag}", file=sys.stderr)
                    return 1
                else:
                    print(f"  [WARN] {log.name} {tag}")
            else:
                print(f"  {variant} seed={seed}: {len(df)} step reports, "
                      f"best_mape={agg['best_mape']:.2f}% at step {agg['best_step']}")

    df_runs = pd.DataFrame(per_run_records)
    df_runs.to_csv(OUT_DIR / "per_run.csv", index=False)
    print(f"\n  per-run CSV: {OUT_DIR / 'per_run.csv'}")

    # Per-variant stats
    print(f"\n>>> Per-variant aggregates ({len(args.seeds)} seeds expected):")
    variant_records: list[dict] = []
    for variant in args.variants:
        sub = df_runs[df_runs['variant'] == variant]
        rec = {'variant': variant}
        rec.update(per_variant_stats(sub.to_dict('records')))
        variant_records.append(rec)
    df_variant = pd.DataFrame(variant_records)
    cols_to_show = ['variant', 'n_seeds',
                    'best_mape_median', 'best_mape_p25', 'best_mape_p75',
                    'best_mape_min', 'best_mape_max',
                    'best_cpl_ratio_median', 'best_cpl_ratio_p25', 'best_cpl_ratio_p75',
                    'best_composite_median']
    cols_to_show = [c for c in cols_to_show if c in df_variant.columns]
    print(df_variant[cols_to_show].round(2).to_string(index=False))
    df_variant.to_csv(OUT_DIR / "per_variant.csv", index=False)
    print(f"\n  per-variant CSV: {OUT_DIR / 'per_variant.csv'}")

    # Pairwise Mann-Whitney on best_mape
    print(f"\n>>> Mann-Whitney U (two-sided) on best_mape across seeds:")
    pairs: list[dict] = []
    var_map = {v: df_runs[df_runs['variant'] == v]['best_mape'].dropna().tolist()
                for v in args.variants}
    for i, va in enumerate(args.variants):
        for vb in args.variants[i+1:]:
            u, p = mann_whitney(var_map[va], var_map[vb])
            sig = ''
            if not (p != p):  # not NaN
                if p < 0.01: sig = '** (p<0.01)'
                elif p < 0.05: sig = '* (p<0.05)'
                elif p < 0.1: sig = '. (p<0.10)'
                else: sig = 'ns'
            print(f"  {va} vs {vb}: U={u:.1f} p={p:.4f}  {sig}")
            pairs.append({'variant_a': va, 'variant_b': vb, 'u': u, 'p_value': p})
    pd.DataFrame(pairs).to_csv(OUT_DIR / "mann_whitney.csv", index=False)

    print(f"\n>>> Done. Outputs under {OUT_DIR}/")
    return 0


if __name__ == '__main__':
    sys.exit(main())
