"""
Verify Innovus/OpenRCX MAPE on tv80s vs CSV-reported numbers.
Parse each SPEF, extract per-net total cap (sum of *CAP entries), compute MAPE vs StarRC golden.
"""
import re
import sys
import time
from pathlib import Path
from collections import defaultdict
import numpy as np


def parse_spef_caps(spef_path):
    """Stream-parse SPEF; return {net_name: {'gnd': float, 'cpl': float, 'total': float}} in fF.
    
    SPEF *CAP block:
      <id> <node> <value>             → ground cap entry (total += value)
      <id> <node1> <node2> <value>    → coupling cap entry
    Detection: count tokens after id (2 tokens = gnd, 3 tokens = cpl).
    
    Convert C_UNIT to fF (StarRC = 1.0 FF, Innovus = 1.0 PF (1000 fF), OpenRCX likely PF).
    """
    nets = {}
    name_map = {}  # *id → name
    
    with open(spef_path, 'r') as f:
        # Find C_UNIT
        c_unit_factor = 1.0  # default fF
        in_name_map = False
        in_dnet = False
        in_cap = False
        current_net = None
        cur_gnd = 0.0
        cur_cpl = 0.0
        
        for line in f:
            line = line.rstrip()
            if line.startswith('*C_UNIT'):
                if 'PF' in line: c_unit_factor = 1000.0  # PF→fF
                elif 'FF' in line: c_unit_factor = 1.0
                elif 'NF' in line: c_unit_factor = 1e6
                continue
            if line.startswith('*NAME_MAP'):
                in_name_map = True
                continue
            if in_name_map:
                m = re.match(r'\*(\d+)\s+(\S+)', line)
                if m:
                    name_map[m.group(1)] = m.group(2)
                elif line.startswith('*D_NET') or line.startswith('*PORTS'):
                    in_name_map = False
            if line.startswith('*D_NET'):
                if current_net is not None:
                    nets[current_net] = {'gnd': cur_gnd * c_unit_factor,
                                        'cpl': cur_cpl * c_unit_factor,
                                        'total': (cur_gnd + cur_cpl) * c_unit_factor}
                parts = line.split()
                # parts[1] = net id (with leading *) or name
                raw = parts[1].lstrip('*')
                current_net = name_map.get(raw, raw)
                cur_gnd = 0.0
                cur_cpl = 0.0
                in_cap = False
                in_dnet = True
                continue
            if line.startswith('*END'):
                if current_net is not None:
                    nets[current_net] = {'gnd': cur_gnd * c_unit_factor,
                                        'cpl': cur_cpl * c_unit_factor,
                                        'total': (cur_gnd + cur_cpl) * c_unit_factor}
                current_net = None
                in_cap = False
                in_dnet = False
                continue
            if in_dnet:
                if line.startswith('*CAP'):
                    in_cap = True
                    continue
                if line.startswith('*RES') or line.startswith('*CONN'):
                    in_cap = False
                    continue
                if in_cap:
                    s = line.strip()
                    if not s: continue
                    parts = s.split()
                    if len(parts) == 3:
                        try:
                            cur_gnd += float(parts[2])
                        except: pass
                    elif len(parts) >= 4:
                        try:
                            cur_cpl += float(parts[-1])
                        except: pass
        if current_net is not None:
            nets[current_net] = {'gnd': cur_gnd * c_unit_factor,
                                'cpl': cur_cpl * c_unit_factor,
                                'total': (cur_gnd + cur_cpl) * c_unit_factor}
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
        nets, cunit = parse_spef_caps(path)
        print(f'{tool:>8}: {len(nets):>6} nets parsed in {time.time()-t0:.1f}s, C_UNIT factor → fF: {cunit}')
        parsed[tool] = nets
    
    # Find common nets
    common = set(parsed['starrc'].keys()) & set(parsed['innovus'].keys()) & set(parsed['openrcx'].keys())
    print(f'\n{len(common)} common nets across all 3 tools')
    
    # Compute per-net MAPE: |tool_total - starrc_total| / starrc_total
    # Skip nets with starrc_total < threshold (numerical noise)
    common = sorted(common)
    sr = np.array([parsed['starrc'][n]['total'] for n in common])
    inv = np.array([parsed['innovus'][n]['total'] for n in common])
    ocx = np.array([parsed['openrcx'][n]['total'] for n in common])
    
    # Filter out tiny-cap nets (numerical noise) — use 1e-3 fF threshold
    mask = sr > 1e-3
    sr_f = sr[mask]
    inv_f = inv[mask]
    ocx_f = ocx[mask]
    print(f'After filter (starrc total > 1e-3 fF): {mask.sum()} nets')
    
    inv_mape = np.abs(inv_f - sr_f) / sr_f
    ocx_mape = np.abs(ocx_f - sr_f) / sr_f
    
    print(f'\n=== tv80s per-net total-cap MAPE vs StarRC golden ===')
    print(f'  Innovus median = {np.median(inv_mape)*100:.3f}%, mean = {np.mean(inv_mape)*100:.3f}%, p95 = {np.percentile(inv_mape, 95)*100:.3f}%')
    print(f'  OpenRCX median = {np.median(ocx_mape)*100:.3f}%, mean = {np.mean(ocx_mape)*100:.3f}%, p95 = {np.percentile(ocx_mape, 95)*100:.3f}%')
    print(f'\n  CSV reported:')
    print(f'    Innovus  4.869% (vs measured median {np.median(inv_mape)*100:.3f}%)')
    print(f'    OpenRCX  7.605% (vs measured median {np.median(ocx_mape)*100:.3f}%)')
    
    # Per-channel
    sr_g = np.array([parsed['starrc'][n]['gnd'] for n in common])[mask]
    inv_g = np.array([parsed['innovus'][n]['gnd'] for n in common])[mask]
    ocx_g = np.array([parsed['openrcx'][n]['gnd'] for n in common])[mask]
    sr_c = np.array([parsed['starrc'][n]['cpl'] for n in common])[mask]
    inv_c = np.array([parsed['innovus'][n]['cpl'] for n in common])[mask]
    ocx_c = np.array([parsed['openrcx'][n]['cpl'] for n in common])[mask]
    
    g_mask = sr_g > 1e-3
    c_mask = sr_c > 1e-3
    print(f'\n  gnd-only MAPE (n={g_mask.sum()}):')
    print(f'    Innovus  median = {np.median(np.abs(inv_g[g_mask] - sr_g[g_mask]) / sr_g[g_mask])*100:.3f}%')
    print(f'    OpenRCX  median = {np.median(np.abs(ocx_g[g_mask] - sr_g[g_mask]) / sr_g[g_mask])*100:.3f}%')
    print(f'  cpl-only MAPE (n={c_mask.sum()}):')
    print(f'    Innovus  median = {np.median(np.abs(inv_c[c_mask] - sr_c[c_mask]) / sr_c[c_mask])*100:.3f}%')
    print(f'    OpenRCX  median = {np.median(np.abs(ocx_c[c_mask] - sr_c[c_mask]) / sr_c[c_mask])*100:.3f}%')


if __name__ == '__main__':
    main()
