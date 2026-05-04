"""
Diag: per-quartile-of-y_gnd heteroscedastic plot.

Mirrors `docs/dspinn_development_log.md §3.4` table:
    Quartile of y_gnd   v10b ratio   v1_new   v2
    ≤Q1 (≤0.41 fF)      1.39 (over)  1.07     1.58 (over)
    Q1-Q2               1.47         1.29     1.53
    Q2-Q3               1.07         0.96     1.11
    Q3+ (large nets)    0.72 (under) 0.59     0.73 (under)

The motivating problem: model knows *where* GND is (Pearson r ≈ 0.85) but
not *how much* — slope = 0.6. If data-driven calibration helped, expect
quartile ratios to flatten toward 1.0.

Usage:
    python3 scripts/diag_quartile_heteroscedastic.py \\
        --ckpt output_intel22/active_learning/v4_distillinit/best_model.pth \\
        --gpu 4 --label v4_distillinit
"""
from __future__ import annotations
import argparse
import json
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True,
                    help='Path to model best_model.pth.')
    ap.add_argument('--gpu', type=int, default=4)
    ap.add_argument('--label', type=str, required=True,
                    help='Run label for the report (e.g. v3, v4_distillinit).')
    ap.add_argument('--use_dspinn', action='store_true', default=True)
    ap.add_argument('--max_nets_per_design', type=int, default=300,
                    help='Net-centric walk: full coverage per net.')
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--designs', nargs='+', default=None,
                    help='Designs to evaluate (default: cfg.AL_PREDEFINED_DESIGNS).')
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from src.models.neural_field import DeepPEX_Model
    from src.data.datasets import robust_collate
    from src.data.calibration_extractor import _PhysicsOnlyDataset
    from src.evaluation.compare_spef import parse_spef_with_coordinates

    cfg._use_dspinn = bool(args.use_dspinn)
    cfg._use_gino = False
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    print(f">>> Heteroscedastic per-quartile analysis: {args.label}")
    print(f"  ckpt: {args.ckpt}")
    print(f"  device: {device}")

    model = DeepPEX_Model(cfg).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    cur = model.state_dict()
    keep = {k: v for k, v in state.items() if k in cur and v.shape == cur[k].shape}
    model.load_state_dict(keep, strict=False)
    print(f"  loaded {len(keep)} tensors (filtered from {len(state)})")
    model.eval()

    designs = args.designs or list(cfg.AL_PREDEFINED_DESIGNS)
    print(f"  designs: {designs}")

    manifest = pd.read_csv(Path(cfg.PROCESSED_DIR) / "dataset_manifest.csv")

    # Resolve SPEF for each design.
    def find_spef(design: str) -> Path | None:
        for sp in list(cfg.TRAIN_SPEFS) + list(cfg.TEST_SPEFS):
            if design in sp.stem:
                return sp
        return None

    rng = np.random.default_rng(seed=42)
    rows: list[dict] = []   # (label, design, net, gnd_pred, gnd_gold)

    for design in designs:
        d_rows = manifest[manifest['design_name'] == design].reset_index(drop=True)
        if len(d_rows) == 0:
            print(f"  [SKIP] {design}: no tiles")
            continue
        unique_nets = d_rows['net_name'].drop_duplicates().to_numpy()
        n = min(args.max_nets_per_design, len(unique_nets))
        chosen = rng.choice(unique_nets, n, replace=False)
        sub = d_rows[d_rows['net_name'].isin(chosen)].reset_index(drop=True)
        print(f"\n>>> {design}: {n} nets, {len(sub)} tiles")

        spef_path = find_spef(design)
        if spef_path is None:
            print(f"  [SKIP] no SPEF for {design}")
            continue
        spef = parse_spef_with_coordinates(spef_path)

        ds = _PhysicsOnlyDataset(sub, pad_size=cfg.NF_PAD_TO_CUBOIDS)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=robust_collate, num_workers=4, pin_memory=True)

        per_net_pred_gnd: dict = defaultdict(float)
        t0 = time.time()
        with torch.no_grad():
            for batch in loader:
                if batch is None: continue
                cuboids, mask, labels_dict, meta_dict = batch
                cuboids = cuboids.to(device, non_blocking=True)
                mask    = mask.to(device, non_blocking=True)
                A_tgt        = labels_dict['A_tgt'].to(device, non_blocking=True)
                core_ratios  = labels_dict['core_ratios'].to(device, non_blocking=True)

                preds = model(cuboids, mask, compute_coupling=True)
                B = cuboids.shape[0]
                target_net_names  = meta_dict['target_net_name']
                cuboid_name_lists = meta_dict['cuboid_net_names']

                # GND aggregation: c_gnd_seg × A_tgt × core_ratios per tile.
                c_gnd_seg = preds['c_gnd_seg'].float()
                tile_gnd  = (c_gnd_seg * A_tgt * core_ratios).sum(dim=1).cpu().numpy()
                for b in range(B):
                    nn = target_net_names[b].replace('\\', '')
                    per_net_pred_gnd[nn] += float(tile_gnd[b])

                # CPL aggregation with power-net lumping (mirrors finetuner.py:506-513)
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
                    if dst_net.lower() in POWER_NETS:
                        per_net_pred_gnd[target_net_names[b].replace('\\', '')] += float(edge_cpl_cpu[e])

        print(f"  forward: {time.time() - t0:.1f}s")

        # Compare to golden
        for net, sp in spef.items():
            net_clean = net.replace('\\', '')
            gold = float(sum(sp['gnd_caps'].values()))
            if gold <= 0.005: continue
            pred = per_net_pred_gnd.get(net_clean, None)
            if pred is None or pred <= 0: continue
            rows.append({
                'label':  args.label,
                'design': design,
                'net':    net_clean,
                'gnd_gold_fF': gold,
                'gnd_pred_fF': pred,
                'ratio': pred / gold,
            })

    if not rows:
        print("FAIL: no nets with valid pred and gold")
        return 1

    df = pd.DataFrame(rows)
    print(f"\n>>> Aggregated {len(df)} nets across {df['design'].nunique()} designs")

    # Quartile binning
    q = df['gnd_gold_fF'].quantile([0.0, 0.25, 0.5, 0.75, 1.0]).to_list()
    print(f"  y_gnd quartile bounds (fF): {[f'{v:.3f}' for v in q]}")

    bins = pd.cut(df['gnd_gold_fF'],
                  bins=[-np.inf, q[1], q[2], q[3], np.inf],
                  labels=['Q1 (smallest)', 'Q1-Q2', 'Q2-Q3', 'Q3+ (largest)'])
    summary = df.groupby(bins, observed=True).agg(
        n_nets=('ratio', 'count'),
        ratio_mean=('ratio', 'mean'),
        ratio_median=('ratio', 'median'),
        ratio_p25=('ratio', lambda x: x.quantile(0.25)),
        ratio_p75=('ratio', lambda x: x.quantile(0.75)),
        chip_ratio=('gnd_pred_fF', 'sum'),
    )
    summary['chip_ratio'] = summary['chip_ratio'] / df.groupby(bins, observed=True)['gnd_gold_fF'].sum()

    print(f"\n>>> Heteroscedastic per-quartile (model={args.label}):")
    print(summary.round(3).to_string())

    # Pearson r and slope
    pearson_r = df[['gnd_gold_fF', 'gnd_pred_fF']].corr().iloc[0, 1]
    # Slope via linear fit
    slope, intercept = np.polyfit(df['gnd_gold_fF'].to_numpy(),
                                   df['gnd_pred_fF'].to_numpy(), 1)
    print(f"\n  Pearson r:  {pearson_r:.3f}")
    print(f"  Linear fit slope:  {slope:.3f}  (target: 1.0)")
    print(f"  Linear fit intercept:  {intercept:.4f} fF")

    # Save CSV next to ckpt
    out_dir = Path(args.ckpt).parent
    out_csv = out_dir / f"hetero_quartile_{args.label}.csv"
    summary_full = summary.copy()
    summary_full['pearson_r'] = pearson_r
    summary_full['slope'] = slope
    summary_full['intercept_fF'] = intercept
    summary_full.to_csv(out_csv)
    raw_csv = out_dir / f"hetero_raw_{args.label}.csv"
    df.to_csv(raw_csv, index=False)
    print(f"\n  summary csv: {out_csv}")
    print(f"  raw csv:     {raw_csv}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
