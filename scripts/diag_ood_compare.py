"""
Diag: OOD comparison of multiple checkpoints on TEST_DEFS (nova_f3, tv80s_f3).

Loads each ckpt, runs net-centric forward over test designs, aggregates per-net
GND + CPL using finetuner-style aggregation, reports:
    - per-design net MAPE
    - per-design chip_gnd ratio (Σpred / Σgold)
    - per-design chip_cpl ratio
    - per-quartile-of-y_gnd ratio (heteroscedastic check on OOD)
    - linear fit slope (target 1.0)

This is the **primary contribution measure** for v3 vs v4 vs v5 — both v3 and
v4 trained on TRAIN_DEFS only. nova/tv80s are completely held-out.

Usage:
    python3 scripts/diag_ood_compare.py \\
        --ckpt v3=output_intel22/active_learning/dspinn_v3/best_model.pth \\
        --ckpt v4=output_intel22/active_learning/v4_distillinit/best_model.pth \\
        --ckpt v5=output_intel22/active_learning/v5_calib_gnd_only/best_model.pth \\
        --gpu 0 --max_nets_per_design 300

The CALIBRATION_INIT_PATH that matters: each ckpt was trained with one
specific JSON. We DON'T need to re-set it for inference — the calibration
values are baked into the saved layer_scale_phys_gnd / cpl_layer_pair_log_scale
parameters in the ckpt. We just need cfg.CALIBRATION_INIT_PATH to point to
*something* (or nothing) at model construction; the loaded ckpt overwrites
those tensors anyway.
"""
from __future__ import annotations
import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import configs.config as cfg


POWER_NETS = {'vss', 'vdd', 'vcc', 'gnd', 'vssx', 'vccx'}


def evaluate_ckpt(ckpt_path: str, label: str, designs: list[str], gpu: int,
                  max_nets_per_design: int, batch_size: int) -> pd.DataFrame:
    """Run forward over each design's net-centric subset, return per-net rows."""
    import torch
    from torch.utils.data import DataLoader
    from src.models.neural_field import DeepPEX_Model
    from src.data.datasets import robust_collate
    from src.data.calibration_extractor import _PhysicsOnlyDataset
    from src.evaluation.compare_spef import parse_spef_with_coordinates

    cfg._use_dspinn = True
    cfg._use_gino = False
    device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'

    print(f"\n>>> Evaluating {label} on {designs}")
    print(f"  ckpt: {ckpt_path}")

    model = DeepPEX_Model(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    cur = model.state_dict()
    keep = {k: v for k, v in state.items() if k in cur and v.shape == cur[k].shape}
    model.load_state_dict(keep, strict=False)
    print(f"  loaded {len(keep)} tensors")
    model.eval()

    manifest = pd.read_csv(Path(cfg.PROCESSED_DIR) / "dataset_manifest.csv")
    rng = np.random.default_rng(seed=42)
    rows: list[dict] = []

    for design in designs:
        d_rows = manifest[manifest['design_name'] == design].reset_index(drop=True)
        if len(d_rows) == 0:
            print(f"  [SKIP] {design}: no tiles in manifest")
            continue

        unique_nets = d_rows['net_name'].drop_duplicates().to_numpy()
        n = min(max_nets_per_design, len(unique_nets))
        chosen = rng.choice(unique_nets, n, replace=False)
        sub = d_rows[d_rows['net_name'].isin(chosen)].reset_index(drop=True)
        print(f"  {design}: {n} nets, {len(sub)} tiles")

        # Find SPEF
        spef_path = None
        for sp in list(cfg.TRAIN_SPEFS) + list(cfg.TEST_SPEFS):
            if design in sp.stem:
                spef_path = sp; break
        if spef_path is None:
            print(f"    [SKIP] no SPEF")
            continue
        spef = parse_spef_with_coordinates(spef_path)

        ds = _PhysicsOnlyDataset(sub, pad_size=cfg.NF_PAD_TO_CUBOIDS)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            collate_fn=robust_collate, num_workers=4, pin_memory=True)

        per_net_pred_gnd: dict = defaultdict(float)
        per_net_pred_cpl: dict = defaultdict(float)

        t0 = time.time()
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

        print(f"    forward: {time.time() - t0:.1f}s")

        # Build per-net rows
        for net, sp in spef.items():
            net_clean = net.replace('\\', '')
            gnd_gold = float(sum(sp['gnd_caps'].values()))
            cpl_gold = float(sum(sum(d.values()) for d in sp['cpl_caps'].values()))
            if gnd_gold <= 0.005: continue
            pred_gnd = per_net_pred_gnd.get(net_clean, 0.0)
            pred_cpl = per_net_pred_cpl.get(net_clean, 0.0)
            if pred_gnd <= 0: continue   # not in walked set
            rows.append({
                'label': label, 'design': design, 'net': net_clean,
                'gnd_gold_fF': gnd_gold, 'gnd_pred_fF': pred_gnd,
                'cpl_gold_fF': cpl_gold, 'cpl_pred_fF': pred_cpl,
                'total_gold_fF': gnd_gold + cpl_gold,
                'total_pred_fF': pred_gnd + pred_cpl,
            })

    df = pd.DataFrame(rows)
    return df


def summarize_per_design(df: pd.DataFrame) -> pd.DataFrame:
    """Per (label, design): chip ratios, MAPE, n_nets."""
    rows: list[dict] = []
    for (label, design), g in df.groupby(['label', 'design']):
        # MAPE: |pred - gold| / gold per net, mean
        gnd_mape = float(((g['gnd_pred_fF'] - g['gnd_gold_fF']).abs() / g['gnd_gold_fF']).mean())
        cpl_mask = g['cpl_gold_fF'] > 0.005
        cpl_mape = float(((g.loc[cpl_mask, 'cpl_pred_fF'] - g.loc[cpl_mask, 'cpl_gold_fF']).abs()
                          / g.loc[cpl_mask, 'cpl_gold_fF']).mean()) if cpl_mask.any() else float('nan')
        tot_mape = float(((g['total_pred_fF'] - g['total_gold_fF']).abs() / g['total_gold_fF']).mean())
        rows.append({
            'label':  label,
            'design': design,
            'n_nets': len(g),
            'gnd_chip_ratio':   float(g['gnd_pred_fF'].sum() / g['gnd_gold_fF'].sum()),
            'cpl_chip_ratio':   float(g['cpl_pred_fF'].sum() / max(g['cpl_gold_fF'].sum(), 1e-9)),
            'total_chip_ratio': float(g['total_pred_fF'].sum() / g['total_gold_fF'].sum()),
            'gnd_mape':         gnd_mape,
            'cpl_mape':         cpl_mape,
            'total_mape':       tot_mape,
        })
    return pd.DataFrame(rows)


def quartile_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per (label, quartile): median ratio, slope."""
    rows: list[dict] = []
    for label, g in df.groupby('label'):
        # Quartiles based on this label's golden distribution
        q = g['gnd_gold_fF'].quantile([0.0, 0.25, 0.5, 0.75, 1.0]).to_list()
        bins = pd.cut(g['gnd_gold_fF'],
                      bins=[-np.inf, q[1], q[2], q[3], np.inf],
                      labels=['Q1', 'Q2', 'Q3', 'Q4'])
        for qname, gq in g.groupby(bins, observed=True):
            ratio = gq['gnd_pred_fF'] / gq['gnd_gold_fF'].clip(lower=1e-9)
            rows.append({
                'label': label, 'quartile': str(qname), 'n': len(gq),
                'gnd_gold_min_fF': float(gq['gnd_gold_fF'].min()),
                'gnd_gold_max_fF': float(gq['gnd_gold_fF'].max()),
                'ratio_median': float(ratio.median()),
                'ratio_p25': float(ratio.quantile(0.25)),
                'ratio_p75': float(ratio.quantile(0.75)),
                'chip_ratio': float(gq['gnd_pred_fF'].sum() / gq['gnd_gold_fF'].sum()),
            })
        # Slope and Pearson r
        slope, intercept = np.polyfit(g['gnd_gold_fF'].to_numpy(),
                                       g['gnd_pred_fF'].to_numpy(), 1)
        r = g[['gnd_gold_fF', 'gnd_pred_fF']].corr().iloc[0, 1]
        rows.append({
            'label': label, 'quartile': 'ALL', 'n': len(g),
            'gnd_gold_min_fF': float(g['gnd_gold_fF'].min()),
            'gnd_gold_max_fF': float(g['gnd_gold_fF'].max()),
            'ratio_median': float((g['gnd_pred_fF'] / g['gnd_gold_fF'].clip(lower=1e-9)).median()),
            'ratio_p25': float('nan'), 'ratio_p75': float('nan'),
            'chip_ratio': float(g['gnd_pred_fF'].sum() / g['gnd_gold_fF'].sum()),
            'slope': float(slope), 'pearson_r': float(r),
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', action='append', required=True,
                    help='label=path/to/best_model.pth (can repeat).')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--max_nets_per_design', type=int, default=300)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--designs', nargs='+', default=None,
                    help='Override default TEST_DEFS designs.')
    ap.add_argument('--out_dir', type=str,
                    default='output_intel22/active_learning/ood_compare')
    args = ap.parse_args()

    designs = args.designs or [p.stem for p in cfg.TEST_DEFS]
    print(f">>> OOD comparison on designs: {designs}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[pd.DataFrame] = []
    for spec in args.ckpt:
        if '=' not in spec:
            print(f"  [SKIP] {spec} — must be label=path")
            continue
        label, path = spec.split('=', 1)
        df = evaluate_ckpt(path, label, designs, args.gpu,
                            args.max_nets_per_design, args.batch_size)
        all_rows.append(df)

    if not all_rows:
        print("FAIL: no evaluations completed")
        return 1

    df_all = pd.concat(all_rows, ignore_index=True)
    raw_csv = out_dir / "ood_raw.csv"
    df_all.to_csv(raw_csv, index=False)
    print(f"\n  raw csv: {raw_csv}")

    print(f"\n>>> Per-design summary:")
    per_design = summarize_per_design(df_all)
    print(per_design.round(3).to_string(index=False))
    per_design.to_csv(out_dir / "ood_per_design.csv", index=False)

    print(f"\n>>> Per-quartile heteroscedastic (across all OOD designs):")
    qsum = quartile_summary(df_all)
    print(qsum.round(3).to_string(index=False))
    qsum.to_csv(out_dir / "ood_quartile.csv", index=False)

    return 0


if __name__ == '__main__':
    sys.exit(main())
