"""
SPEF gnd/cpl decomposition analysis for tv80s 3-tool comparison.

Standard SPEF *CAP block:
    <id> <node>            <value>   → 3 tokens after id = ground cap (single node)
    <id> <node1> <node2>   <value>   → 4 tokens after id = coupling cap (pair)

Each tool's per-net (gnd, cpl, total) is the sum of its *CAP entries by token count.
Compare against StarRC golden to reveal:
  - per-net total cap MAPE (apples-to-apples)
  - per-net gnd-only MAPE  (tool's gnd-labeled vs StarRC's gnd-labeled)
  - per-net cpl-only MAPE  (tool's cpl-labeled vs StarRC's cpl-labeled)
  - gnd/cpl ratio: how each tool partitions total cap into the two channels

Outputs:
  - Console table: per-tool aggregate metrics
  - JSON: {tool: {per_net: {gnd, cpl, total}, agg_metrics}}
  - PINN comparison row using same eval_logger seed-0 (combined IS+CN + LGBM)
"""
from __future__ import annotations
import json
import re
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd


def parse_spef_cap_breakdown(spef_path: Path):
    """Stream-parse SPEF; return per-net (gnd, cpl, total) in fF.

    Detects token count after the CAP entry id:
      3 tokens (id, node, val)        → gnd
      4 tokens (id, n1, n2, val)      → cpl
    """
    nets = {}   # net_name → {'gnd': fF, 'cpl': fF, 'total': fF}
    name_map = {}

    c_unit_factor = 1.0
    in_name_map = False
    in_dnet = False
    in_cap = False
    current_net = None
    cur_gnd = 0.0
    cur_cpl = 0.0
    n_gnd_entries = 0
    n_cpl_entries = 0

    with open(spef_path, 'r') as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue

            if stripped.startswith('*C_UNIT'):
                if 'PF' in stripped: c_unit_factor = 1000.0
                elif 'FF' in stripped: c_unit_factor = 1.0
                elif 'NF' in stripped: c_unit_factor = 1e6
                continue

            if stripped.startswith('*NAME_MAP'):
                in_name_map = True
                continue

            if in_name_map:
                m = re.match(r'\*(\d+)\s+(\S+)', stripped)
                if m:
                    name_map[m.group(1)] = m.group(2)
                    continue
                else:
                    in_name_map = False  # fall through

            if stripped.startswith('*D_NET'):
                if current_net is not None:
                    nets[current_net] = {
                        'gnd': cur_gnd * c_unit_factor,
                        'cpl': cur_cpl * c_unit_factor,
                        'total': (cur_gnd + cur_cpl) * c_unit_factor,
                        'n_gnd_entries': n_gnd_entries,
                        'n_cpl_entries': n_cpl_entries,
                    }
                parts = stripped.split()
                raw = parts[1].lstrip('*')
                current_net = name_map.get(raw, raw)
                cur_gnd = 0.0
                cur_cpl = 0.0
                n_gnd_entries = 0
                n_cpl_entries = 0
                in_cap = False
                in_dnet = True
                continue

            if stripped.startswith('*END'):
                if current_net is not None:
                    nets[current_net] = {
                        'gnd': cur_gnd * c_unit_factor,
                        'cpl': cur_cpl * c_unit_factor,
                        'total': (cur_gnd + cur_cpl) * c_unit_factor,
                        'n_gnd_entries': n_gnd_entries,
                        'n_cpl_entries': n_cpl_entries,
                    }
                current_net = None
                in_cap = False
                in_dnet = False
                continue

            if in_dnet:
                if stripped.startswith('*CAP'):
                    in_cap = True
                    continue
                if stripped.startswith('*RES') or stripped.startswith('*CONN'):
                    in_cap = False
                    continue
                if in_cap:
                    parts = stripped.split()
                    # Skip if first token isn't an integer ID
                    try:
                        _ = int(parts[0])
                    except ValueError:
                        continue
                    # Token layout: id, ...nodes..., value  (value always last)
                    if len(parts) == 3:
                        try:
                            cur_gnd += float(parts[2])
                            n_gnd_entries += 1
                        except ValueError:
                            pass
                    elif len(parts) >= 4:
                        try:
                            cur_cpl += float(parts[-1])
                            n_cpl_entries += 1
                        except ValueError:
                            pass

    return nets, c_unit_factor


def main():
    base = Path('/home/jslee/projects/PEX_SSL/golden_data/spef_data/intel22_')
    files = {
        'starrc':  base / 'intel22_tv80s_nonamemap_starrc.spef',
        'innovus': base / 'intel22_tv80s_nonamemap_innovus.spef',
        'openrcx': base / 'intel22_tv80s_nonamemap_openrcx.spef',
    }

    parsed = {}
    for tool, path in files.items():
        t0 = time.time()
        nets, cunit = parse_spef_cap_breakdown(path)
        elapsed = time.time() - t0
        print(f'{tool:>8}: {len(nets):>6} nets parsed in {elapsed:.2f}s, C_UNIT factor → fF: {cunit}')
        parsed[tool] = nets

    common = sorted(set(parsed['starrc'].keys()) & set(parsed['innovus'].keys()) & set(parsed['openrcx'].keys()))
    print(f'\n{len(common)} common nets across all 3 tools (tv80s)\n')

    # Build arrays
    arrays = {}
    for tool in ['starrc', 'innovus', 'openrcx']:
        arrays[tool] = {
            'gnd': np.array([parsed[tool][n]['gnd'] for n in common]),
            'cpl': np.array([parsed[tool][n]['cpl'] for n in common]),
            'total': np.array([parsed[tool][n]['total'] for n in common]),
            'n_gnd_entries': np.array([parsed[tool][n]['n_gnd_entries'] for n in common]),
            'n_cpl_entries': np.array([parsed[tool][n]['n_cpl_entries'] for n in common]),
        }

    sr = arrays['starrc']

    # Numerical noise filter
    EPS = 1e-3
    mask_total = sr['total'] > EPS
    mask_gnd = sr['gnd'] > EPS
    mask_cpl = sr['cpl'] > EPS

    print(f'Filter: starrc total > {EPS} fF: {mask_total.sum()}/{len(common)} nets')
    print(f'        starrc gnd   > {EPS} fF: {mask_gnd.sum()}')
    print(f'        starrc cpl   > {EPS} fF: {mask_cpl.sum()}')
    print()

    print('=' * 90)
    print('Per-net MAPE vs StarRC golden  (median, mean, p95)')
    print('=' * 90)
    print(f'{"Tool":<10} {"channel":<8} {"n":>6} {"median":>9} {"mean":>9} {"p95":>9}')
    print('-' * 90)
    rows = []
    for tool in ['innovus', 'openrcx']:
        for ch, msk in [('total', mask_total), ('gnd', mask_gnd), ('cpl', mask_cpl)]:
            sr_v = sr[ch][msk]
            tl_v = arrays[tool][ch][msk]
            err = np.abs(tl_v - sr_v) / sr_v
            row = {
                'tool': tool, 'channel': ch, 'n': int(msk.sum()),
                'median': float(np.median(err)),
                'mean': float(np.mean(err)),
                'p95': float(np.percentile(err, 95)),
            }
            rows.append(row)
            print(f'{tool:<10} {ch:<8} {msk.sum():>6} {row["median"]*100:>8.3f}% {row["mean"]*100:>8.3f}% {row["p95"]*100:>8.3f}%')
    print()

    print('=' * 90)
    print('Per-tool gnd:cpl decomposition ratio (% of total cap that is gnd)')
    print('=' * 90)
    for tool in ['starrc', 'innovus', 'openrcx']:
        a = arrays[tool]
        gnd_frac = a['gnd'][mask_total] / a['total'][mask_total]
        n_g_med = float(np.median(a['n_gnd_entries']))
        n_c_med = float(np.median(a['n_cpl_entries']))
        print(f'  {tool:<10} median gnd_frac = {np.median(gnd_frac)*100:5.1f}%  '
              f'mean = {np.mean(gnd_frac)*100:5.1f}%  '
              f'(median *CAP entries: {n_g_med:.0f} gnd / {n_c_med:.0f} cpl)')
    print()

    print('=' * 90)
    print('Per-net total cap CORRELATION (R²) vs StarRC')
    print('=' * 90)
    for tool in ['innovus', 'openrcx']:
        for ch in ['total', 'gnd', 'cpl']:
            msk = mask_total if ch == 'total' else (mask_gnd if ch == 'gnd' else mask_cpl)
            sr_v = sr[ch][msk]
            tl_v = arrays[tool][ch][msk]
            if len(sr_v) < 2:
                continue
            ss_res = float(np.sum((tl_v - sr_v) ** 2))
            ss_tot = float(np.sum((sr_v - sr_v.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
            print(f'  {tool:<10} {ch:<8} R² = {r2:.6f}')
    print()

    # PINN comparison on same tv80s nets — use seed 0 of best stack + LGBM
    print('=' * 90)
    print('PINN best-stack (Combined IS+CN + LGBM 8-feat) vs StarRC on tv80s')
    print('=' * 90)
    pinn_dir = Path('/home/jslee/projects/PINNPEX/pex_v3/output/ablation/HybridPexV3MeshInputSubsetClampNorm')
    final_dir = Path('/home/jslee/projects/PINNPEX/pex_v3/experiments/auto_optimize_2026_05_03/outputs/final_hero')

    # Use eval_logger but apply LGBM cal in line if not already
    pinn_results = {}
    for s in range(5):
        tst = pd.read_parquet(pinn_dir / f'seed{s}/eval_logger_test.parquet')
        # Filter to tv80s
        msk_tv = tst['design'].values == 'intel22_tv80s_f3'
        tv = tst[msk_tv].copy()
        # Load corrected predictions from final_hero npz
        npz = final_dir / f'seed{s}/corrected_predictions.npz'
        if npz.exists():
            corr = np.load(npz)
            # corrected_predictions.npz is full test set (95594), aligned with tst order
            corr_tv_g = corr['gnd_pred'][msk_tv]
            corr_tv_c = corr['cpl_pred'][msk_tv]
        else:
            corr_tv_g = tv['gnd_pred'].values
            corr_tv_c = tv['cpl_pred'].values

        gold_g = tv['gnd_gold'].values
        gold_c = tv['cpl_gold'].values
        gold_tot = gold_g + gold_c
        pred_tot = corr_tv_g + corr_tv_c

        msk_pos = gold_tot > 1e-3
        err_tot = np.abs(pred_tot - gold_tot)[msk_pos] / gold_tot[msk_pos]
        msk_g = gold_g > 1e-3
        msk_c = gold_c > 1e-3
        err_g = np.abs(corr_tv_g - gold_g)[msk_g] / gold_g[msk_g]
        err_c = np.abs(corr_tv_c - gold_c)[msk_c] / gold_c[msk_c]
        pinn_results[s] = {
            'n_total': int(msk_pos.sum()),
            'total_median': float(np.median(err_tot)),
            'gnd_median': float(np.median(err_g)),
            'cpl_median': float(np.median(err_c)),
        }

    # Aggregate across 5 seeds
    for ch in ['total', 'gnd', 'cpl']:
        med_5 = [pinn_results[s][f'{ch}_median'] for s in range(5)]
        print(f'  PINN best-stack {ch:<8} 5-seed median: {np.median(med_5)*100:.3f}% '
              f'(seeds: {[f"{x*100:.2f}%" for x in med_5]})')

    print()
    print('=' * 90)
    print('FINAL TV80S LEADERBOARD (per-net cap MAPE vs StarRC golden, median)')
    print('=' * 90)
    print(f'{"Tool":<28} {"total":>8} {"gnd":>8} {"cpl":>8}  notes')
    print('-' * 90)
    pinn_rows = {ch: float(np.median([pinn_results[s][f'{ch}_median'] for s in range(5)])) for ch in ['total', 'gnd', 'cpl']}
    print(f'{"PINN best-stack (this work)":<28} {pinn_rows["total"]*100:>7.3f}% {pinn_rows["gnd"]*100:>7.3f}% {pinn_rows["cpl"]*100:>7.3f}%  44.7K NN + LGBM 8-feat cal')
    for tool in ['innovus', 'openrcx']:
        rows_t = {r['channel']: r['median'] for r in rows if r['tool'] == tool}
        print(f'{tool:<28} {rows_t["total"]*100:>7.3f}% {rows_t["gnd"]*100:>7.3f}% {rows_t["cpl"]*100:>7.3f}%  pattern matching')

    out_path = Path('/home/jslee/projects/PINNPEX/pex_v3/experiments/auto_optimize_2026_05_03/reports/spef_3tool_analysis_tv80s.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        'design': 'tv80s',
        'n_common_nets': len(common),
        'mape_rows': rows,
        'pinn_5seed': pinn_rows,
        'pinn_per_seed': pinn_results,
        'gnd_fraction_median': {
            tool: float(np.median(arrays[tool]['gnd'][mask_total] / arrays[tool]['total'][mask_total]))
            for tool in ['starrc', 'innovus', 'openrcx']
        },
        'n_cap_entries_median': {
            tool: {
                'gnd': float(np.median(arrays[tool]['n_gnd_entries'])),
                'cpl': float(np.median(arrays[tool]['n_cpl_entries'])),
            }
            for tool in ['starrc', 'innovus', 'openrcx']
        },
    }, open(out_path, 'w'), indent=2)
    print(f'\nReport saved to: {out_path}')


if __name__ == '__main__':
    main()
