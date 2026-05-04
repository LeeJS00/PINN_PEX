"""
Diag: Sanity-check a calibration_init.json by reading it back through the
model (reproducing the exact init path used at training time) and comparing
predicted vs golden per-net cap on holdout designs.

For each holdout design:
    pred_gnd_net = Σ_target_cuboid: c_gnd_seg @ physics_only with calibrated
                   layer_scale_phys_gnd, then aggregated like finetuner.py:486
    pred_cpl_per_aggr = Σ_edge: c_cpl @ physics_only with calibrated
                        cpl_layer_pair_log_scale, then aggregated like
                        finetuner.py:493 + power-net lumping.

Reports: per-design GND ratio (Σpred / Σgold) and per-aggressor CPL median ratio.
Aborts with non-zero exit code if any holdout design's GND ratio falls outside
[0.5, 2.0] or per-aggressor CPL median ratio is outside [0.3, 3.0].

Usage:
    python3 scripts/diag_calibration_check.py \\
        --calibration /data/PINNPEX/data/processed/intel22/calibration_init.json \\
        --gpu 4
"""
from __future__ import annotations
import argparse
import json
import math
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import configs.config as cfg
from src.evaluation.compare_spef import parse_spef_with_coordinates


POWER_NETS = {'vss', 'vdd', 'vcc', 'gnd', 'vssx', 'vccx'}


def load_design_spef(design: str) -> dict:
    """Find SPEF for a design among TRAIN_SPEFS + TEST_SPEFS, parse it."""
    candidates = list(cfg.TRAIN_SPEFS) + list(cfg.TEST_SPEFS)
    for sp in candidates:
        if design in sp.stem:
            return parse_spef_with_coordinates(sp)
    raise FileNotFoundError(f"No SPEF found for design {design}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--calibration', type=str, required=True,
                    help='Path to calibration_init.json (will be loaded via cfg.CALIBRATION_INIT_PATH).')
    ap.add_argument('--gpu', type=int, default=4)
    ap.add_argument('--max_tiles_per_design', type=int, default=1500,
                    help='Cap tiles per holdout design for speed.')
    ap.add_argument('--max_nets_per_design', type=int, default=200,
                    help='Net-centric: walk all tiles of N random nets per holdout design '
                         '(preferred over --max_tiles_per_design for accurate per-net prediction).')
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--abort_on_fail', action='store_true', default=True,
                    help='Exit 1 if ratio thresholds violated.')
    ap.add_argument('--gnd_ratio_low', type=float, default=0.5)
    ap.add_argument('--gnd_ratio_high', type=float, default=2.0)
    ap.add_argument('--cpl_ratio_low', type=float, default=0.3)
    ap.add_argument('--cpl_ratio_high', type=float, default=3.0)
    args = ap.parse_args()

    print(f">>> Loading calibration JSON: {args.calibration}")
    with open(args.calibration) as f:
        calib = json.load(f)
    holdout_designs = list(calib['source']['designs_holdout'])
    print(f"  holdout designs: {holdout_designs}")
    K = len(calib['metal_z_anchors_um'])
    print(f"  K layers: {K}")
    print(f"  calibrated cpl_diag (softplus value): {calib['cpl_pair_diag_value']:.3f}")
    print(f"  calibrated cpl_cross (softplus value): {calib['cpl_pair_cross_value']:.3f}")

    # Set cfg path so the model loads it
    cfg.CALIBRATION_INIT_PATH = args.calibration

    import torch
    from torch.utils.data import DataLoader
    from src.models.neural_field import DeepPEX_Model
    from src.data.datasets import robust_collate
    from src.data.calibration_extractor import _PhysicsOnlyDataset

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    cfg._use_dspinn = False
    cfg._use_gino = False
    model = DeepPEX_Model(cfg).to(device)
    fr = model.flux_router

    # Verify init was loaded from JSON (not hardcoded fallback)
    actual_density = torch.nn.functional.softplus(fr.layer_scale_phys_gnd).detach().cpu().numpy()
    expected_density = np.asarray(calib['gnd_density_fF_per_um2'])
    diff = np.abs(actual_density - expected_density)
    print(f"  density init max-deviation from JSON: {diff.max():.4e}  "
          f"(should be ≈0 if JSON was loaded)")

    actual_cpl_diag = torch.nn.functional.softplus(fr.cpl_layer_pair_log_scale.diag()).detach().cpu().numpy()
    print(f"  cpl pair diag mean (after init): {actual_cpl_diag.mean():.3f} "
          f"(JSON says {calib['cpl_pair_diag_value']:.3f})")

    # Apply physics-only modifiers (same as diag_eval_dump.py:90-96)
    with torch.no_grad():
        fr.gnd_mlp[-1].weight.zero_()
        fr.gnd_mlp[-1].bias.zero_()
        fr.cpl_mlp[-1].weight.zero_()
        fr.cpl_mlp[-1].bias.copy_(torch.tensor([0.0, -10.0], device=device))
    model.eval()

    # Walk holdout tiles per design
    manifest = pd.read_csv(Path(cfg.PROCESSED_DIR) / "dataset_manifest.csv")
    z_anchors_t = fr.metal_z_anchors.detach().to(device)

    overall_status = True
    summary_rows = []

    rng = np.random.default_rng(seed=42)
    for design in holdout_designs:
        print(f"\n>>> Holdout design: {design}")
        d_rows = manifest[manifest['design_name'] == design].reset_index(drop=True)
        if len(d_rows) == 0:
            print(f"  [SKIP] no tiles for {design} in manifest")
            continue
        if args.max_nets_per_design is not None:
            unique_nets = d_rows['net_name'].drop_duplicates().to_numpy()
            n_pick = min(args.max_nets_per_design, len(unique_nets))
            chosen = rng.choice(unique_nets, n_pick, replace=False)
            sub = d_rows[d_rows['net_name'].isin(chosen)].reset_index(drop=True)
            print(f"  net-centric walk: {n_pick} nets, {len(sub)} tiles (full coverage per net)")
        else:
            sub = d_rows.head(args.max_tiles_per_design).reset_index(drop=True)
            print(f"  tile-head walk: {len(sub)} tiles (partial net coverage — not recommended)")

        ds = _PhysicsOnlyDataset(sub, pad_size=cfg.NF_PAD_TO_CUBOIDS)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=robust_collate, num_workers=4, pin_memory=True)

        # Per-net aggregation buffers (mirror finetuner.py:486-513)
        per_net_pred_gnd: dict = defaultdict(float)
        per_net_pred_cpl_per_aggr: dict = defaultdict(lambda: defaultdict(float))

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

                # GND aggregation: c_gnd_seg * A_tgt * core_ratios summed per tile
                c_gnd_seg = preds['c_gnd_seg'].float()
                tile_gnd  = (c_gnd_seg * A_tgt * core_ratios).sum(dim=1).cpu().numpy()
                for b in range(B):
                    nn = target_net_names[b].replace('\\', '')
                    per_net_pred_gnd[nn] += float(tile_gnd[b])

                # CPL aggregation with power-net lumping
                sparse = preds['sparse_cpl']
                if sparse['b_idx'].numel() == 0:
                    continue
                b_idx   = sparse['b_idx'].long()
                src_idx = sparse['src_idx'].long()
                dst_idx = sparse['dst_idx'].long()
                c_cpl   = sparse['c_cpl'].float()
                # core_ratio_eff
                cr_src = core_ratios[b_idx, src_idx]
                cr_dst = core_ratios[b_idx, dst_idx]
                tgt_at_src = A_tgt[b_idx, src_idx] > 0
                cr_eff = torch.where(tgt_at_src, cr_src, cr_dst)
                edge_cpl = c_cpl * cr_eff

                # Move to CPU
                b_idx_cpu = b_idx.cpu().numpy()
                src_idx_cpu = src_idx.cpu().numpy()
                dst_idx_cpu = dst_idx.cpu().numpy()
                edge_cpl_cpu = edge_cpl.cpu().numpy()

                for e in range(b_idx_cpu.shape[0]):
                    b = int(b_idx_cpu[e])
                    si = int(src_idx_cpu[e]); di = int(dst_idx_cpu[e])
                    names = cuboid_name_lists[b]
                    if di >= len(names): continue
                    dst_net = str(names[di]).replace('\\', '')
                    src_net = str(names[si]).replace('\\', '')
                    target_net = target_net_names[b].replace('\\', '')
                    contrib = float(edge_cpl_cpu[e])
                    if dst_net.lower() in POWER_NETS:
                        per_net_pred_gnd[target_net] += contrib
                    else:
                        if src_net != target_net: continue
                        per_net_pred_cpl_per_aggr[target_net][dst_net] += contrib

        print(f"  forward+aggregation: {time.time() - t0:.1f}s")

        # Compare to golden
        spef = load_design_spef(design)
        golden_gnd = {n: float(sum(d['gnd_caps'].values())) for n, d in spef.items()}
        golden_cpl = defaultdict(lambda: defaultdict(float))
        for n, d in spef.items():
            for node, aggrs in d['cpl_caps'].items():
                for aggr, cap in aggrs.items():
                    golden_cpl[n][aggr.replace('\\', '')] += float(cap)

        # Per-net ratio: pred_gnd / gold_gnd
        gnd_ratios = []
        gnd_pred_total = 0.0; gnd_gold_total = 0.0
        for net, gold in golden_gnd.items():
            if gold <= 0.005: continue
            pred = per_net_pred_gnd.get(net.replace('\\', ''), None)
            if pred is None: continue   # not in our walked tiles
            gnd_pred_total += pred
            gnd_gold_total += gold
            gnd_ratios.append(pred / gold)

        # Per-aggressor CPL ratio (median)
        cpl_ratios = []
        cpl_pred_total = 0.0; cpl_gold_total = 0.0
        for net, aggr_dict in golden_cpl.items():
            for aggr, gold in aggr_dict.items():
                if gold <= 0.005: continue
                pred = per_net_pred_cpl_per_aggr.get(net.replace('\\', ''), {}).get(aggr, None)
                if pred is None or pred <= 0: continue
                cpl_pred_total += pred
                cpl_gold_total += gold
                cpl_ratios.append(pred / gold)

        gnd_med = float(np.median(gnd_ratios)) if gnd_ratios else float('nan')
        cpl_med = float(np.median(cpl_ratios)) if cpl_ratios else float('nan')
        gnd_chip_ratio = gnd_pred_total / max(gnd_gold_total, 1e-9)
        cpl_chip_ratio = cpl_pred_total / max(cpl_gold_total, 1e-9)

        print(f"  GND per-net ratio (median): {gnd_med:.3f}  (n={len(gnd_ratios)})")
        print(f"  GND chip ratio (Σpred/Σgold): {gnd_chip_ratio:.3f}")
        print(f"  CPL per-aggr ratio (median): {cpl_med:.3f}  (n={len(cpl_ratios)})")
        print(f"  CPL chip ratio (Σpred/Σgold): {cpl_chip_ratio:.3f}")

        gnd_ok = (args.gnd_ratio_low <= gnd_med <= args.gnd_ratio_high) if np.isfinite(gnd_med) else False
        cpl_ok = (args.cpl_ratio_low <= cpl_med <= args.cpl_ratio_high) if np.isfinite(cpl_med) else False
        if not (gnd_ok and cpl_ok):
            overall_status = False
            print(f"  [WARN] thresholds violated: GND_ok={gnd_ok}, CPL_ok={cpl_ok}")
        else:
            print(f"  ✓ ratios within [{args.gnd_ratio_low}, {args.gnd_ratio_high}] / "
                  f"[{args.cpl_ratio_low}, {args.cpl_ratio_high}]")

        summary_rows.append({
            'design': design, 'gnd_ratio_med': gnd_med, 'gnd_chip_ratio': gnd_chip_ratio,
            'cpl_ratio_med': cpl_med, 'cpl_chip_ratio': cpl_chip_ratio,
            'n_gnd_nets': len(gnd_ratios), 'n_cpl_pairs': len(cpl_ratios),
            'gnd_ok': gnd_ok, 'cpl_ok': cpl_ok,
        })

    print("\n>>> Summary:")
    print(pd.DataFrame(summary_rows).to_string(index=False))

    if not overall_status and args.abort_on_fail:
        print("\nFAIL: at least one holdout design out of acceptable ratio range.")
        return 1
    print("\nPASS: all holdout designs within thresholds.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
