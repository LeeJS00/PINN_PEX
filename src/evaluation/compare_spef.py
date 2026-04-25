# src/evaluation/compare_spef.py
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
import csv

def parse_spef_with_coordinates(spef_path):
    nets = {}
    current_net = None
    in_conn, in_cap, in_res = False, False, False
    
    with open(spef_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('//'): continue
            
            if line.startswith('*D_NET'):
                tokens = line.split()
                current_net = tokens[1]
                nets[current_net] = {
                    'total_cap': float(tokens[2]) if tokens[2].replace('.','',1).isdigit() else 0.0,
                    'total_res': 0.0,  
                    'sum_gnd_cap': 0.0, 
                    'sum_cpl_cap': 0.0, 
                    'nodes': {},
                    'gnd_caps': defaultdict(float),
                    'cpl_caps': defaultdict(lambda: defaultdict(float))
                }
                in_conn, in_cap, in_res = False, False, False
                continue
                
            if not current_net: continue
            
            if line.startswith('*CONN'):
                in_conn, in_cap, in_res = True, False, False
                continue
            elif line.startswith('*CAP'):
                in_conn, in_cap, in_res = False, True, False
                continue
            elif line.startswith('*RES'):
                in_conn, in_cap, in_res = False, False, True
                continue
            elif line.startswith('*END'):
                in_conn, in_cap, in_res = False, False, False
                continue
                
            if in_conn and line.startswith('*'):
                tokens = line.split()
                if '*C' in tokens:
                    idx = tokens.index('*C')
                    if idx + 2 < len(tokens):
                        try:
                            nets[current_net]['nodes'][tokens[1]] = (float(tokens[idx+1]), float(tokens[idx+2]))
                        except ValueError:
                            pass
                            
            if in_cap and not line.startswith('*'):
                tokens = line.split()
                if len(tokens) == 3: # Ground Cap
                    cap_val = float(tokens[2])
                    nets[current_net]['gnd_caps'][tokens[1]] += cap_val
                    nets[current_net]['sum_gnd_cap'] += cap_val
                elif len(tokens) == 4: # Coupling Cap
                    cap_val = float(tokens[3])
                    nets[current_net]['cpl_caps'][tokens[1]][tokens[2].split(':')[0]] += cap_val
                    nets[current_net]['sum_cpl_cap'] += cap_val

            if in_res and not line.startswith('*'):
                tokens = line.split()
                if len(tokens) >= 4:
                    nets[current_net]['total_res'] += float(tokens[3])

    return nets

def compute_metrics(gold_arr, pred_arr):
    gold_arr, pred_arr = np.array(gold_arr), np.array(pred_arr)
    valid_mask = np.isfinite(gold_arr) & np.isfinite(pred_arr)
    g_valid, p_valid = gold_arr[valid_mask], pred_arr[valid_mask]
    
    if len(g_valid) == 0: return float('nan'), float('nan'), float('nan')
        
    safe_g = np.where(g_valid == 0, 1e-6, g_valid)
    mape = np.mean(np.abs(p_valid - g_valid) / safe_g) * 100
    rmse = np.sqrt(np.mean((p_valid - g_valid)**2))
    
    if len(g_valid) > 1 and np.var(g_valid) > 1e-12 and np.var(p_valid) > 1e-12:
        r2 = np.corrcoef(g_valid, p_valid)[0, 1]**2
    else: r2 = 0.0
        
    return mape, r2, rmse

def compare_spefs(golden_spef_path, pred_spef_path, output_dir=None):
    print(f"\n{'='*80}")
    print(f"🧬 DEEP DIVE PEX AUTOPSY & EVALUATION (Expert Mode)")
    print(f"{'='*80}")
    
    gold_data = parse_spef_with_coordinates(golden_spef_path)
    pred_data = parse_spef_with_coordinates(pred_spef_path)
    common_nets = set(gold_data.keys()).intersection(set(pred_data.keys()))
    
    print(f"\n[1] Structural Integrity")
    print(f"  - Golden Nets : {len(gold_data)} | Pred Nets : {len(pred_data)} | Common : {len(common_nets)}")
    if len(common_nets) == 0: return

    # --- 1. GLOBAL BALANCE SHEET (칩 전체 전하 보존 검증) ---
    g_tot_sum = sum(n['total_cap'] for n in gold_data.values())
    p_tot_sum = sum(n['total_cap'] for n in pred_data.values())
    g_gnd_sum = sum(n['sum_gnd_cap'] for n in gold_data.values())
    p_gnd_sum = sum(n['sum_gnd_cap'] for n in pred_data.values())
    g_cpl_sum = sum(n['sum_cpl_cap'] for n in gold_data.values())
    p_cpl_sum = sum(n['sum_cpl_cap'] for n in pred_data.values())

    print(f"\n[2] FULL-CHIP KCL BALANCE SHEET (Did we lose charge?)")
    print(f"  {'Type':<12} | {'Golden (fF)':<15} | {'Predicted (fF)':<15} | {'Diff (fF)':<12} | {'Ratio'}")
    print(f"  {'-'*75}")
    print(f"  {'Total Cap':<12} | {g_tot_sum:<15.4f} | {p_tot_sum:<15.4f} | {p_tot_sum-g_tot_sum:<12.4f} | {p_tot_sum/max(g_tot_sum, 1e-6):.2f}x")
    print(f"  {'Ground Cap':<12} | {g_gnd_sum:<15.4f} | {p_gnd_sum:<15.4f} | {p_gnd_sum-g_gnd_sum:<12.4f} | {p_gnd_sum/max(g_gnd_sum, 1e-6):.2f}x")
    print(f"  {'Coupling Cap':<12} | {g_cpl_sum:<15.4f} | {p_cpl_sum:<15.4f} | {p_cpl_sum-g_cpl_sum:<12.4f} | {p_cpl_sum/max(g_cpl_sum, 1e-6):.2f}x")

    if abs((p_tot_sum) - (p_gnd_sum + p_cpl_sum)) > 1.0:
        print(f"  🚨 [WARNING] Prediction SPEF breaks KCL! (Total != Gnd + Cpl)")

    # --- 2. MICRO-ANALYSIS DATA PREP ---
    m_gold_tot_cap, m_pred_tot_cap = [], []
    m_gold_gnd_cap, m_pred_gnd_cap = [], []
    m_gold_cpl_cap, m_pred_cpl_cap = [], []
    m_gold_tot_res, m_pred_tot_res = [], []
    
    error_records = []

    for net in common_nets:
        g = gold_data[net]
        p = pred_data[net]
        
        m_gold_tot_cap.append(g['total_cap']); m_pred_tot_cap.append(p['total_cap'])
        m_gold_gnd_cap.append(g['sum_gnd_cap']); m_pred_gnd_cap.append(p['sum_gnd_cap'])
        m_gold_cpl_cap.append(g['sum_cpl_cap']); m_pred_cpl_cap.append(p['sum_cpl_cap'])
        m_gold_tot_res.append(g['total_res']); m_pred_tot_res.append(p['total_res'])
        
        err_tot = abs(g['total_cap'] - p['total_cap'])
        error_records.append({
            'net': net,
            'g_tot': g['total_cap'], 'p_tot': p['total_cap'], 'err_tot': err_tot,
            'g_gnd': g['sum_gnd_cap'], 'p_gnd': p['sum_gnd_cap'],
            'g_cpl': g['sum_cpl_cap'], 'p_cpl': p['sum_cpl_cap'],
            'g_res': g['total_res'], 'p_res': p['total_res']
        })

    # --- 3. METRICS ---
    tot_mape, tot_r2, tot_rmse = compute_metrics(m_gold_tot_cap, m_pred_tot_cap)
    gnd_mape, gnd_r2, gnd_rmse = compute_metrics(m_gold_gnd_cap, m_pred_gnd_cap)
    cpl_mape, cpl_r2, cpl_rmse = compute_metrics(m_gold_cpl_cap, m_pred_cpl_cap)
    res_mape, res_r2, res_rmse = compute_metrics(m_gold_tot_res, m_pred_tot_res)

    print(f"\n[3] Macroscopic Features Analysis (Net-Level)")
    print(f"  [Total Capacitance]  MAPE: {tot_mape:6.2f}% | R^2: {tot_r2:.4f} | RMSE: {tot_rmse:.4f} fF")
    print(f"  [Ground Capacitance] MAPE: {gnd_mape:6.2f}% | R^2: {gnd_r2:.4f} | RMSE: {gnd_rmse:.4f} fF")
    print(f"  [Coupling Cap (Sum)] MAPE: {cpl_mape:6.2f}% | R^2: {cpl_r2:.4f} | RMSE: {cpl_rmse:.4f} fF")
    print(f"  [Total Resistance]   MAPE: {res_mape:6.2f}% | R^2: {res_r2:.4f} | RMSE: {res_rmse:.4f} Ohms")

    # --- 4. DEEP DIVE: TOP 10 WORST NETS ---
    print(f"\n[4] DEEP DIVE: Top 10 Nets by Absolute Total Cap Error")
    df_err = pd.DataFrame(error_records)
    # df_err_sorted = df_err.sort_values(by='err_tot', ascending=False).head(10)
    df_err_sorted = df_err.sort_values(by='g_tot', ascending=True).head(10)
    
    
    print(f"  {'Net Name':<25} | {'Gold(Tot/Gnd/Cpl) (fF)':<28} | {'Pred(Tot/Gnd/Cpl) (fF)':<28} | {'Err (fF)'}")
    print(f"  {'-'*100}")
    for _, r in df_err_sorted.iterrows():
        g_str = f"{r['g_tot']:.3f} / {r['g_gnd']:.3f} / {r['g_cpl']:.3f}"
        p_str = f"{r['p_tot']:.3f} / {r['p_gnd']:.3f} / {r['p_cpl']:.3f}"
        print(f"  {r['net'][:25]:<25} | {g_str:<28} | {p_str:<28} | {r['err_tot']:.3f}")

    # --- 4b. LENGTH-STRATIFIED MAPE ---
    print(f"\n[4b] Length-Stratified MAPE Analysis")
    df_err['mape'] = df_err['err_tot'] / (df_err['g_tot'].clip(lower=1e-6)) * 100

    # By total cap bucket (strong proxy for wire length in routed designs)
    cap_bins   = [0, 1, 5, 20, float('inf')]
    cap_labels = ['<1fF', '1-5fF', '5-20fF', '>20fF']
    df_err['cap_bin'] = pd.cut(df_err['g_tot'], bins=cap_bins, labels=cap_labels)
    cap_stats = df_err.groupby('cap_bin', observed=True)['mape'].agg(['mean', 'median', 'count'])
    print(f"  By total_cap (net length proxy):")
    for lbl, row in cap_stats.iterrows():
        bar = '█' * max(1, int(row['mean'] / 5))
        print(f"    {lbl:8s} {bar:20s} MAPE={row['mean']:6.2f}%  median={row['median']:6.2f}%  n={int(row['count'])}")

    # By total resistance quartile (direct wire-length proxy)
    res_nonzero = df_err[df_err['g_res'] > 0]
    if len(res_nonzero) > 4:
        res_nonzero = res_nonzero.copy()
        res_nonzero['res_bin'] = pd.qcut(
            res_nonzero['g_res'], q=4,
            labels=['Q1(short)', 'Q2', 'Q3', 'Q4(long)'], duplicates='drop')
        res_stats = res_nonzero.groupby('res_bin', observed=True)['mape'].agg(['mean', 'median', 'count'])
        print(f"  By total_res quartile (Ω, wire-length proxy):")
        res_bounds = res_nonzero.groupby('res_bin', observed=True)['g_res'].agg(['min', 'max'])
        for lbl, row in res_stats.iterrows():
            lo, hi = res_bounds.loc[lbl, 'min'], res_bounds.loc[lbl, 'max']
            print(f"    {lbl:12s} [{lo:6.1f}–{hi:6.1f}Ω]  MAPE={row['mean']:6.2f}%  median={row['median']:6.2f}%  n={int(row['count'])}")

    # --- 5. TOPOLOGY (AGGRESSOR) MISMATCH ANALYSIS ---
    worst_net = df_err_sorted.iloc[0]['net']
    print(f"\n[5] AGGRESSOR MISMATCH ANALYSIS: The Worst Net [{worst_net}]")
    
    g_cpls = gold_data[worst_net]['cpl_caps']
    p_cpls = pred_data[worst_net]['cpl_caps']
    
    g_aggrs = defaultdict(float)
    for node, aggr_dict in g_cpls.items():
        for a_net, a_val in aggr_dict.items(): g_aggrs[a_net] += a_val
            
    p_aggrs = defaultdict(float)
    for node, aggr_dict in p_cpls.items():
        for a_net, a_val in aggr_dict.items(): p_aggrs[a_net] += a_val

    all_aggrs = set(g_aggrs.keys()).union(set(p_aggrs.keys()))
    aggr_records = []
    for a in all_aggrs:
        aggr_records.append({
            'aggr_name': a, 
            'gold_val': g_aggrs.get(a, 0.0), 
            'pred_val': p_aggrs.get(a, 0.0)
        })
        
    df_aggr = pd.DataFrame(aggr_records)
    df_aggr['diff'] = abs(df_aggr['gold_val'] - df_aggr['pred_val'])
    df_aggr = df_aggr.sort_values(by='diff', ascending=False).head(10)
    
    print(f"  Top 10 Aggressors for {worst_net}:")
    print(f"  {'Aggressor Net Name':<30} | {'Golden (fF)':<12} | {'Pred (fF)':<12} | {'Diff (fF)'}")
    print(f"  {'-'*75}")
    for _, r in df_aggr.iterrows():
        print(f"  {r['aggr_name'][:30]:<30} | {r['gold_val']:<12.5f} | {r['pred_val']:<12.5f} | {r['diff']:.5f}")

    print("\n" + "="*80)
    
    # Optional: Save detailed report to CSV
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"spef_comparison_report.csv"
        df_err.to_csv(report_path, index=False)
        print(f"Detailed report saved to: {report_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--golden', type=str, required=True)
    parser.add_argument('--pred', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    args = parser.parse_args()
    compare_spefs(args.golden, args.pred, args.out_dir)