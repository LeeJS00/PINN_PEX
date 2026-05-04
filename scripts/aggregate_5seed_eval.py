"""
Aggregate heteroscedastic + OOD evaluation across all 5-seed checkpoints.

For each (variant, seed) m5_*/best_model.pth:
  - Run net-centric forward over validation designs (heteroscedastic check)
  - Run net-centric forward over TEST_DEFS (OOD generalization)
  - Extract per-quartile and per-design metrics

Per variant, aggregate median + IQR + min/max across the N seeds.
Compute Mann-Whitney U for v3 vs v4, v3 vs v5, v4 vs v5 on each metric.

Outputs:
  output_intel22/active_learning/m5_summary/eval_per_seed.csv
  output_intel22/active_learning/m5_summary/eval_per_variant.csv
  output_intel22/active_learning/m5_summary/eval_quartile.csv

Usage:
    python3 scripts/aggregate_5seed_eval.py --gpu 0 --max_nets_per_design 200
"""
from __future__ import annotations
import argparse
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import configs.config as cfg
OUT_DIR = ROOT / "output_intel22" / "active_learning" / "m5_summary"
CKPT_BASE = ROOT / "output_intel22" / "active_learning"


VARIANTS = ['v3_baseline', 'v4_full_calib', 'v5_gnd_only']
POWER_NETS = {'vss', 'vdd', 'vcc', 'gnd', 'vssx', 'vccx'}


def discover_ckpts() -> dict:
    """Return {variant: {seed: path}}."""
    out = {v: {} for v in VARIANTS}
    for variant in VARIANTS:
        for seed in range(5):
            p = CKPT_BASE / f"m5_{variant}_seed{seed}" / "best_model.pth"
            if p.exists():
                out[variant][seed] = p
    return out


def run_eval_one_ckpt(ckpt_path: Path, label: str, designs: list[str],
                      gpu: int, max_nets: int, batch_size: int) -> pd.DataFrame:
    """Run net-centric forward on designs, return per-net (label, design,
    net, gnd_pred_fF, gnd_gold_fF, cpl_pred_fF, cpl_gold_fF) DataFrame."""
    import torch
    from torch.utils.data import DataLoader
    from src.models.neural_field import DeepPEX_Model
    from src.data.datasets import robust_collate
    from src.data.calibration_extractor import _PhysicsOnlyDataset
    from src.evaluation.compare_spef import parse_spef_with_coordinates

    cfg._use_dspinn = True
    cfg._use_gino = False
    device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'

    model = DeepPEX_Model(cfg).to(device)
    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    cur = model.state_dict()
    keep = {k: v for k, v in state.items() if k in cur and v.shape == cur[k].shape}
    model.load_state_dict(keep, strict=False)
    model.eval()

    manifest = pd.read_csv(Path(cfg.PROCESSED_DIR) / "dataset_manifest.csv")
    rng = np.random.default_rng(seed=42)
    rows: list[dict] = []

    for design in designs:
        d_rows = manifest[manifest['design_name'] == design].reset_index(drop=True)
        if len(d_rows) == 0: continue
        unique_nets = d_rows['net_name'].drop_duplicates().to_numpy()
        n = min(max_nets, len(unique_nets))
        chosen = rng.choice(unique_nets, n, replace=False)
        sub = d_rows[d_rows['net_name'].isin(chosen)].reset_index(drop=True)

        spef_path = None
        for sp in list(cfg.TRAIN_SPEFS) + list(cfg.TEST_SPEFS):
            if design in sp.stem:
                spef_path = sp; break
        if spef_path is None: continue
        spef = parse_spef_with_coordinates(spef_path)

        ds = _PhysicsOnlyDataset(sub, pad_size=cfg.NF_PAD_TO_CUBOIDS)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            collate_fn=robust_collate, num_workers=2, pin_memory=True)

        per_net_pred_gnd: dict = defaultdict(float)
        per_net_pred_cpl: dict = defaultdict(float)

        with torch.no_grad():
            for batch in loader:
                if batch is None: continue
                cuboids, mask, labels_dict, meta_dict = batch
                cuboids = cuboids.to(device, non_blocking=True)
                mask    = mask.to(device, non_blocking=True)
                A_tgt        = labels_dict['A_tgt'].to(device, non_blocking=True)
                core_ratios  = labels_dict['core_ratios'].to(device, non_blocking=True)

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    preds = model(cuboids, mask, compute_coupling=True)
                B = cuboids.shape[0]
                target_net_names  = meta_dict['target_net_name']
                cuboid_name_lists = meta_dict['cuboid_net_names']

                c_gnd_seg = preds['c_gnd_seg'].float()
                tile_gnd  = (c_gnd_seg * A_tgt * core_ratios).sum(dim=1).cpu().numpy()
                for b in range(B):
                    nn = target_net_names[b].replace('\\', '')
                    per_net_pred_gnd[nn] += float(tile_gnd[b])

                sparse = preds['sparse_cpl']
                if sparse['b_idx'].numel() == 0:
                    continue
                b_idx   = sparse['b_idx'].long()
                src_idx = sparse['src_idx'].long()
                dst_idx = sparse['dst_idx'].long()
                c_cpl   = sparse['c_cpl'].float()
                cr_src = core_ratios[b_idx, src_idx]
                cr_dst = core_ratios[b_idx, dst_idx]
                tgt_at_src = A_tgt[b_idx, src_idx] > 0
                cr_eff = torch.where(tgt_at_src, cr_src, cr_dst)
                edge_cpl = c_cpl * cr_eff
                b_idx_cpu = b_idx.cpu().numpy()
                dst_idx_cpu = dst_idx.cpu().numpy()
                edge_cpl_cpu = edge_cpl.cpu().numpy()
                for e in range(b_idx_cpu.shape[0]):
                    b = int(b_idx_cpu[e])
                    di = int(dst_idx_cpu[e])
                    names = cuboid_name_lists[b]
                    if di >= len(names): continue
                    dst_net = str(names[di]).replace('\\', '')
                    nn = target_net_names[b].replace('\\', '')
                    contrib = float(edge_cpl_cpu[e])
                    if dst_net.lower() in POWER_NETS:
                        per_net_pred_gnd[nn] += contrib
                    else:
                        per_net_pred_cpl[nn] += contrib

        for net, sp in spef.items():
            net_clean = net.replace('\\', '')
            gnd_gold = float(sum(sp['gnd_caps'].values()))
            cpl_gold = float(sum(sum(d.values()) for d in sp['cpl_caps'].values()))
            if gnd_gold <= 0.005: continue
            pred_gnd = per_net_pred_gnd.get(net_clean, 0.0)
            pred_cpl = per_net_pred_cpl.get(net_clean, 0.0)
            if pred_gnd <= 0: continue
            rows.append({
                'label': label, 'design': design, 'net': net_clean,
                'gnd_gold_fF': gnd_gold, 'gnd_pred_fF': pred_gnd,
                'cpl_gold_fF': cpl_gold, 'cpl_pred_fF': pred_cpl,
                'total_gold_fF': gnd_gold + cpl_gold,
                'total_pred_fF': pred_gnd + pred_cpl,
            })
    df = pd.DataFrame(rows)
    return df


def per_seed_summary(df: pd.DataFrame, label: str) -> dict:
    """Per-ckpt aggregate metrics."""
    if df.empty: return {'label': label}
    per_design: list[dict] = []
    all_ratios = []
    all_gnd_gold, all_gnd_pred = [], []
    all_cpl_gold, all_cpl_pred = [], []
    for design, g in df.groupby('design'):
        cpl_mask = g['cpl_gold_fF'] > 0.005
        per_design.append({
            'design':   design,
            'n_nets':   len(g),
            'gnd_chip_ratio':   float(g['gnd_pred_fF'].sum() / g['gnd_gold_fF'].sum()),
            'cpl_chip_ratio':   float(g['cpl_pred_fF'].sum() / max(g['cpl_gold_fF'].sum(), 1e-9)),
            'total_chip_ratio': float(g['total_pred_fF'].sum() / g['total_gold_fF'].sum()),
            'gnd_mape':         float(((g['gnd_pred_fF'] - g['gnd_gold_fF']).abs() / g['gnd_gold_fF']).mean()),
            'cpl_mape':         float(((g.loc[cpl_mask, 'cpl_pred_fF'] - g.loc[cpl_mask, 'cpl_gold_fF']).abs()
                                       / g.loc[cpl_mask, 'cpl_gold_fF']).mean()) if cpl_mask.any() else float('nan'),
            'total_mape':       float(((g['total_pred_fF'] - g['total_gold_fF']).abs() / g['total_gold_fF']).mean()),
        })
    pd_summary = pd.DataFrame(per_design)
    # Slope, Pearson r over all designs
    gnd_arr = df['gnd_gold_fF'].to_numpy()
    pred_arr = df['gnd_pred_fF'].to_numpy()
    slope, intercept = np.polyfit(gnd_arr, pred_arr, 1)
    pearson_r = float(df[['gnd_gold_fF', 'gnd_pred_fF']].corr().iloc[0, 1])
    return {
        'label': label,
        'gnd_chip_ratio_mean':   float(pd_summary['gnd_chip_ratio'].mean()),
        'cpl_chip_ratio_mean':   float(pd_summary['cpl_chip_ratio'].mean()),
        'total_chip_ratio_mean': float(pd_summary['total_chip_ratio'].mean()),
        'gnd_mape_mean':         float(pd_summary['gnd_mape'].mean()),
        'cpl_mape_mean':         float(pd_summary['cpl_mape'].mean()),
        'total_mape_mean':       float(pd_summary['total_mape'].mean()),
        'slope':                 float(slope),
        'pearson_r':             pearson_r,
        'per_design':            pd_summary.to_dict('records'),
    }


def aggregate_variant(per_seed_records: list[dict]) -> dict:
    """Compute median/IQR/min/max across seeds for a variant."""
    if not per_seed_records: return {}
    df = pd.DataFrame(per_seed_records)
    cols = ['gnd_chip_ratio_mean', 'cpl_chip_ratio_mean', 'total_chip_ratio_mean',
            'gnd_mape_mean', 'cpl_mape_mean', 'total_mape_mean',
            'slope', 'pearson_r']
    out = {'n_seeds': len(df)}
    for c in cols:
        if c not in df.columns or df[c].isna().all(): continue
        v = df[c].dropna()
        out[f'{c}_median'] = float(v.median())
        out[f'{c}_p25']    = float(v.quantile(0.25))
        out[f'{c}_p75']    = float(v.quantile(0.75))
        out[f'{c}_min']    = float(v.min())
        out[f'{c}_max']    = float(v.max())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--max_nets_per_design', type=int, default=200)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--ind_designs', nargs='+', default=None,
                    help='In-distribution designs (default: cfg.AL_PREDEFINED_DESIGNS).')
    ap.add_argument('--ood_designs', nargs='+', default=None,
                    help='OOD designs (default: cfg.TEST_DEFS).')
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ind_designs = args.ind_designs or list(cfg.AL_PREDEFINED_DESIGNS)
    ood_designs = args.ood_designs or [p.stem for p in cfg.TEST_DEFS]
    print(f">>> in-dist designs: {ind_designs}")
    print(f">>> OOD designs:     {ood_designs}")

    ckpts = discover_ckpts()
    n_total = sum(len(s) for s in ckpts.values())
    print(f">>> Discovered {n_total} ckpts across {len(VARIANTS)} variants")
    for v, ss in ckpts.items():
        print(f"  {v}: seeds {sorted(ss)}")

    # --- Per-seed evaluation -------------------------------------------------
    per_seed_ind: list[dict] = []
    per_seed_ood: list[dict] = []
    raw_ind_dfs: list[pd.DataFrame] = []
    raw_ood_dfs: list[pd.DataFrame] = []

    for variant, seed_paths in ckpts.items():
        for seed, path in sorted(seed_paths.items()):
            label = f"{variant}_seed{seed}"
            print(f"\n>>> Eval {label} ind...")
            t0 = time.time()
            df_ind = run_eval_one_ckpt(path, label, ind_designs,
                                        args.gpu, args.max_nets_per_design,
                                        args.batch_size)
            print(f"  ind: {len(df_ind)} nets, {time.time()-t0:.1f}s")
            raw_ind_dfs.append(df_ind)
            ind_summary = per_seed_summary(df_ind, label)
            ind_summary['variant'] = variant; ind_summary['seed'] = seed
            ind_summary['split'] = 'ind'
            per_seed_ind.append(ind_summary)

            t0 = time.time()
            df_ood = run_eval_one_ckpt(path, label, ood_designs,
                                        args.gpu, args.max_nets_per_design,
                                        args.batch_size)
            print(f"  ood: {len(df_ood)} nets, {time.time()-t0:.1f}s")
            raw_ood_dfs.append(df_ood)
            ood_summary = per_seed_summary(df_ood, label)
            ood_summary['variant'] = variant; ood_summary['seed'] = seed
            ood_summary['split'] = 'ood'
            per_seed_ood.append(ood_summary)

    # Merge raw and write
    if raw_ind_dfs:
        pd.concat(raw_ind_dfs, ignore_index=True).to_csv(OUT_DIR / "eval_raw_ind.csv", index=False)
    if raw_ood_dfs:
        pd.concat(raw_ood_dfs, ignore_index=True).to_csv(OUT_DIR / "eval_raw_ood.csv", index=False)

    df_per_seed = pd.DataFrame(
        [{k: v for k, v in r.items() if k != 'per_design'} for r in per_seed_ind + per_seed_ood]
    )
    df_per_seed.to_csv(OUT_DIR / "eval_per_seed.csv", index=False)

    # --- Per-variant aggregates ----------------------------------------------
    rows: list[dict] = []
    for split, recs in [('ind', per_seed_ind), ('ood', per_seed_ood)]:
        for variant in VARIANTS:
            sub = [r for r in recs if r.get('variant') == variant]
            if not sub: continue
            agg = aggregate_variant(sub)
            agg['variant'] = variant; agg['split'] = split
            rows.append(agg)
    df_variant = pd.DataFrame(rows)
    df_variant.to_csv(OUT_DIR / "eval_per_variant.csv", index=False)

    print(f"\n>>> Per-variant summary (median values across seeds):")
    cols = ['variant', 'split', 'n_seeds',
            'total_mape_mean_median', 'gnd_chip_ratio_mean_median',
            'cpl_chip_ratio_mean_median', 'slope_median', 'pearson_r_median']
    cols = [c for c in cols if c in df_variant.columns]
    print(df_variant[cols].round(3).to_string(index=False))

    # --- Statistical tests on key metrics ------------------------------------
    try:
        from scipy.stats import mannwhitneyu
    except ImportError:
        print("scipy unavailable; skipping Mann-Whitney")
        return 0

    print(f"\n>>> Mann-Whitney U on total_mape_mean:")
    for split, recs in [('ind', per_seed_ind), ('ood', per_seed_ood)]:
        var_data = {v: [r['total_mape_mean'] for r in recs if r.get('variant') == v
                        and 'total_mape_mean' in r] for v in VARIANTS}
        for i, va in enumerate(VARIANTS):
            for vb in VARIANTS[i+1:]:
                a, b = var_data[va], var_data[vb]
                if len(a) < 2 or len(b) < 2:
                    print(f"  [{split}] {va} vs {vb}: insufficient data")
                    continue
                u, p = mannwhitneyu(a, b, alternative='two-sided')
                sig = '** ' if p < 0.01 else ('* ' if p < 0.05 else ('. ' if p < 0.1 else 'ns'))
                print(f"  [{split}] {va} vs {vb}: U={u:.1f} p={p:.4f}  {sig}")

    print(f"\n>>> Done. Outputs under {OUT_DIR}/")
    return 0


if __name__ == '__main__':
    sys.exit(main())
