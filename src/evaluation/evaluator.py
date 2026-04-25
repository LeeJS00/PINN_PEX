# ==========================================================
# FILE: src/evaluation/evaluator.py
# ==========================================================
import torch
import pandas as pd
import numpy as np
import argparse
import time
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import gzip
import pickle
import sys
from scipy.spatial import cKDTree

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.neural_field import DeepPEX_Model
# [CRITICAL FIX] SSLDataset은 meta_dict가 없으므로 Base Dataset 사용
from src.data.datasets import NeuralFieldEvalDataset, robust_collate
from src.utils.spef_writer import AutonomousGraphBuilder, SPEFWriter
from src.preprocessing.layer_parser import LayerInfoParser
from src.preprocessing.lef_parser import LefParser
from src.utils.naming import sanitize_name
import configs.config as cfg
from src.utils.profiler import RuntimeProfiler
from src.evaluation.compare_spef import parse_spef_with_coordinates, compute_metrics


POWER_NETS = {'vss', 'vdd', 'vcc', 'gnd', 'vssx', 'vccx'}


def _run_inference_and_accumulate(model, loader, DEVICE):
    """
    Runs model inference over all tiles and accumulates per-cuboid predictions
    into cuboid_accumulators[net][geo_key] = {gnd, cpl, weight_sum, abs_geo}.
    Returns (cuboid_accumulators, stage2_stats).
    """
    cuboid_accumulators = defaultdict(
        lambda: defaultdict(lambda: {'gnd': 0.0, 'cpl': defaultdict(float), 'abs_geo': None, 'weight_sum': 0.0})
    )
    stats = {
        'n_tiles': 0, 'n_target_cuboids': 0,
        'n_edges_total': 0, 'n_tiles_zero_edges': 0,
        'gpu_time': 0.0, 'total_time': 0.0,
    }
    t_wall0 = time.time()

    with torch.no_grad():
        for batch in tqdm(loader, desc="  [Stage 2] Inference"):
            if batch is None:
                continue
            cuboids, mask, meta_dict = batch
            cuboids, mask = cuboids.to(DEVICE), mask.to(DEVICE)
            if cuboids.shape[-1] > 9:
                cuboids = cuboids[..., :9]

            t_gpu = time.time()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                preds = model(cuboids, mask, compute_coupling=True)
            stats['gpu_time'] += time.time() - t_gpu

            B = cuboids.shape[0]
            c_gnd_seg = torch.nan_to_num(preds['c_gnd_seg'], nan=0.0).cpu().numpy()

            sparse_cpl = preds['sparse_cpl']
            has_edges = len(sparse_cpl['b_idx']) > 0
            if has_edges:
                b_idx  = sparse_cpl['b_idx'].cpu().numpy()
                src_idx = sparse_cpl['src_idx'].cpu().numpy()
                dst_idx = sparse_cpl['dst_idx'].cpu().numpy()
                c_cpl   = torch.nan_to_num(sparse_cpl['c_cpl'], nan=0.0).cpu().numpy()
                stats['n_edges_total'] += len(b_idx)
                n_batches_with_edges = len(np.unique(b_idx))
                stats['n_tiles_zero_edges'] += B - n_batches_with_edges
            else:
                stats['n_tiles_zero_edges'] += B

            stats['n_tiles'] += B

            for b in range(B):
                target_net   = meta_dict['target_net_name'][b]
                abs_geos     = meta_dict['abs_geometries'][b].numpy()
                names        = np.array(meta_dict['cuboid_net_names'][b])
                core_ratios  = meta_dict['core_ratios'][b]

                semantic_types = cuboids[b, :, 6].cpu().numpy()
                is_wire   = semantic_types > 0.8
                is_target = (names == target_net) & (core_ratios > 0) & is_wire
                target_indices = np.where(is_target)[0]
                if len(target_indices) == 0:
                    continue
                stats['n_target_cuboids'] += len(target_indices)

                # --- ground cap accumulation ---
                for idx in target_indices:
                    geo_key = tuple(np.round(abs_geos[idx][:6], 6).tolist())
                    node = cuboid_accumulators[target_net][geo_key]
                    if node['abs_geo'] is None:
                        node['abs_geo'] = abs_geos[idx]
                    node['weight_sum'] += float(core_ratios[idx])
                    node['gnd']        += float(c_gnd_seg[b, idx] * core_ratios[idx])

                # --- coupling cap accumulation ---
                if not has_edges:
                    continue
                edge_mask = b_idx == b
                if not edge_mask.any():
                    continue

                e_src, e_dst = src_idx[edge_mask], dst_idx[edge_mask]
                e_cpl_raw    = c_cpl[edge_mask]

                valid = is_target[e_src] & (e_cpl_raw * core_ratios[e_src] > 1e-6)
                e_src = e_src[valid]; e_dst = e_dst[valid]
                e_val = e_cpl_raw[valid] * core_ratios[e_src]

                aggr_names = np.array([str(n).lower() for n in names[e_dst]])
                not_filtered = ~np.isin(
                    aggr_names,
                    list(POWER_NETS) + ["unknown", "pad", "unknown_pin", target_net.lower()]
                )
                e_src = e_src[not_filtered]; e_dst = e_dst[not_filtered]; e_val = e_val[not_filtered]

                # node_cpl_dict: cuboid_idx → {aggr_net → sum_val}
                node_cpl_dict = defaultdict(lambda: defaultdict(float))
                net_totals    = defaultdict(float)
                for s, d, v in zip(e_src, e_dst, e_val):
                    net_totals[names[d]] += float(v)
                for s, d, v in zip(e_src, e_dst, e_val):
                    if net_totals[names[d]] >= 1e-6:
                        node_cpl_dict[s][names[d]] += float(v)

                for idx in target_indices:
                    if idx not in node_cpl_dict:
                        continue
                    geo_key = tuple(np.round(abs_geos[idx][:6], 6).tolist())
                    for a_net, a_val in node_cpl_dict[idx].items():
                        cuboid_accumulators[target_net][geo_key]['cpl'][a_net] += float(a_val)

    stats['total_time'] = time.time() - t_wall0
    return cuboid_accumulators, stats


def _build_net_predictions(cuboid_accumulators):
    """
    Applies Voronoi weight normalization and returns:
      net_pred[net_name] = {gnd, cpl_total, cpl_by_net}
    """
    net_pred = {}
    for target_net, cuboid_dict in cuboid_accumulators.items():
        total_gnd   = 0.0
        cpl_by_net  = defaultdict(float)
        for node in cuboid_dict.values():
            w = max(node['weight_sum'], 1e-12)
            total_gnd += node['gnd'] / w
            for a_net, a_val in node['cpl'].items():
                cpl_by_net[a_net] += a_val / w
        net_pred[target_net] = {
            'gnd':       total_gnd,
            'cpl_total': sum(cpl_by_net.values()),
            'cpl_by_net': dict(cpl_by_net),
        }
    return net_pred


def run_direct_eval(model, test_df, DATA_DIR, MODEL_DIR, DEVICE, design_filter=None):
    """
    Direct MAPE evaluation (no SPEF file written).
    Prints 6-stage analysis and saves per-net CSV report.
    design_filter: if set, only evaluate that design name.
    """
    GOLDEN_SPEF_DIR = Path(cfg.SPEF_DIR)

    for design_name, group_df in test_df.groupby('design_name'):
        if design_filter and design_filter not in design_name:
            continue
        print(f"\n{'='*72}")
        print(f"  Design : {design_name}  ({len(group_df)} tiles)")
        print(f"{'='*72}")

        # ── Stage 1: Load golden SPEF ─────────────────────────────────────
        t0 = time.time()
        golden_path = GOLDEN_SPEF_DIR / f"{design_name}_starrc.spef"
        if not golden_path.exists():
            golden_path = GOLDEN_SPEF_DIR / f"{design_name}.spef"
        if not golden_path.exists():
            print(f"  ⚠  No golden SPEF found at {GOLDEN_SPEF_DIR} for {design_name}. Skipping.")
            continue

        gold_data = parse_spef_with_coordinates(str(golden_path))
        t_gold = time.time() - t0

        tile_per_net = group_df.groupby('net_name').size()
        print(f"\n[Stage 1] Golden SPEF loaded: {len(gold_data)} nets  ({t_gold:.2f}s)")
        print(f"  Manifest tiles : {len(group_df):,}  |  unique nets : {tile_per_net.shape[0]:,}")
        print(f"  Tiles/net      : min={tile_per_net.min()}  median={tile_per_net.median():.0f}  max={tile_per_net.max()}")

        # ── Stage 2: Inference ────────────────────────────────────────────
        dataset = NeuralFieldEvalDataset(DATA_DIR, group_df, pad_size=cfg.NF_PAD_TO_CUBOIDS)
        loader  = torch.utils.data.DataLoader(
            dataset, batch_size=16, collate_fn=robust_collate,
            num_workers=8, pin_memory=True
        )

        cuboid_accumulators, s2 = _run_inference_and_accumulate(model, loader, DEVICE)

        avg_edges = s2['n_edges_total'] / max(s2['n_tiles'], 1)
        print(f"\n[Stage 2] Inference complete  (wall {s2['total_time']:.1f}s, GPU {s2['gpu_time']:.1f}s)")
        print(f"  Tiles processed   : {s2['n_tiles']:,}")
        print(f"  Target cuboids    : {s2['n_target_cuboids']:,}")
        print(f"  Total edges       : {s2['n_edges_total']:,}  (avg {avg_edges:.1f}/tile)")
        print(f"  Tiles w/o edges   : {s2['n_tiles_zero_edges']:,}  ({100*s2['n_tiles_zero_edges']/max(s2['n_tiles'],1):.1f}%)")
        print(f"  Throughput        : {s2['n_tiles']/max(s2['gpu_time'],1e-6):.0f} tiles/s (GPU)")

        # ── Stage 3: Net aggregation ──────────────────────────────────────
        t0 = time.time()
        net_pred = _build_net_predictions(cuboid_accumulators)
        t_agg = time.time() - t0

        nets_with_gnd = sum(1 for v in net_pred.values() if v['gnd']       > 1e-9)
        nets_with_cpl = sum(1 for v in net_pred.values() if v['cpl_total'] > 1e-9)

        gold_gnd_sum = sum(v['sum_gnd_cap'] for v in gold_data.values())
        gold_cpl_sum = sum(v['sum_cpl_cap'] for v in gold_data.values())
        pred_gnd_sum = sum(v['gnd']         for v in net_pred.values())
        pred_cpl_sum = sum(v['cpl_total']   for v in net_pred.values())

        gold_only  = set(gold_data) - set(net_pred)
        pred_only  = set(net_pred)  - set(gold_data)
        common_nets = set(gold_data) & set(net_pred)

        print(f"\n[Stage 3] Net aggregation  ({t_agg:.2f}s)")
        print(f"  Nets predicted    : {len(net_pred):,}  (gnd≥0: {nets_with_gnd:,}  cpl≥0: {nets_with_cpl:,})")
        print(f"  Coverage          : common={len(common_nets):,}  gold_only={len(gold_only):,}  pred_only={len(pred_only):,}")
        print(f"\n  Chip-level balance:")
        print(f"  {'Type':<13} | {'Golden (fF)':>13} | {'Predicted (fF)':>14} | {'Ratio':>7}")
        print(f"  {'-'*57}")
        print(f"  {'Ground Cap':<13} | {gold_gnd_sum:>13.3f} | {pred_gnd_sum:>14.3f} | {pred_gnd_sum/max(gold_gnd_sum,1e-9):>7.4f}x")
        print(f"  {'Coupling Cap':<13} | {gold_cpl_sum:>13.3f} | {pred_cpl_sum:>14.3f} | {pred_cpl_sum/max(gold_cpl_sum,1e-9):>7.4f}x")

        # ── Stage 4: MAPE + stratification ───────────────────────────────
        records = []
        g_tot, p_tot, g_gnd, p_gnd, g_cpl, p_cpl = [], [], [], [], [], []

        for net in common_nets:
            g    = gold_data[net]
            pred = net_pred[net]
            p_total = pred['gnd'] + pred['cpl_total']
            g_tot.append(g['total_cap']);    p_tot.append(p_total)
            g_gnd.append(g['sum_gnd_cap']); p_gnd.append(pred['gnd'])
            g_cpl.append(g['sum_cpl_cap']); p_cpl.append(pred['cpl_total'])
            records.append({
                'net': net, 'design': design_name,
                'g_tot': g['total_cap'],    'p_tot': p_total,
                'g_gnd': g['sum_gnd_cap'],  'p_gnd': pred['gnd'],
                'g_cpl': g['sum_cpl_cap'],  'p_cpl': pred['cpl_total'],
                'err_tot': abs(g['total_cap'] - p_total),
                'mape': abs(g['total_cap'] - p_total) / max(g['total_cap'], 1e-6) * 100,
            })

        tot_mape, tot_r2, tot_rmse = compute_metrics(g_tot, p_tot)
        gnd_mape, gnd_r2, gnd_rmse = compute_metrics(g_gnd, p_gnd)
        cpl_mape, cpl_r2, cpl_rmse = compute_metrics(g_cpl, p_cpl)

        print(f"\n[Stage 4] Net-level metrics  ({len(common_nets):,} common nets)")
        print(f"  {'Metric':<16} | {'MAPE%':>8} | {'R²':>7} | {'RMSE (fF)':>10}")
        print(f"  {'-'*50}")
        print(f"  {'Total Cap':<16} | {tot_mape:>8.3f} | {tot_r2:>7.4f} | {tot_rmse:>10.5f}")
        print(f"  {'Ground Cap':<16} | {gnd_mape:>8.3f} | {gnd_r2:>7.4f} | {gnd_rmse:>10.5f}")
        print(f"  {'Coupling Cap':<16} | {cpl_mape:>8.3f} | {cpl_r2:>7.4f} | {cpl_rmse:>10.5f}")

        df = pd.DataFrame(records)

        # By golden total cap bucket (net-length proxy)
        cap_bins   = [0, 1, 5, 20, float('inf')]
        cap_labels = ['<1 fF', '1-5 fF', '5-20 fF', '>20 fF']
        df['cap_bin'] = pd.cut(df['g_tot'], bins=cap_bins, labels=cap_labels)
        cap_stats = df.groupby('cap_bin', observed=True)['mape'].agg(['mean', 'median', 'count'])
        print(f"\n  By golden cap bucket (net-length proxy):")
        for lbl, row in cap_stats.iterrows():
            bar = '█' * max(1, min(24, int(row['mean'] / 5)))
            print(f"    {lbl:8s}  {bar:<24s}  mean={row['mean']:7.2f}%  median={row['median']:7.2f}%  n={int(row['count']):,}")

        # By coupling fraction (gnd-dominant vs cpl-dominant)
        df['cpl_frac'] = df['g_cpl'] / df['g_tot'].clip(lower=1e-6)
        df['cpl_bin']  = pd.cut(df['cpl_frac'],
                                bins=[0, 0.2, 0.5, 0.8, 1.01],
                                labels=['GND-dom', 'Mixed-GND', 'Mixed-CPL', 'CPL-dom'])
        cpl_stats = df.groupby('cpl_bin', observed=True)['mape'].agg(['mean', 'median', 'count'])
        print(f"\n  By coupling fraction (CPL / Total in golden):")
        for lbl, row in cpl_stats.iterrows():
            bar = '█' * max(1, min(24, int(row['mean'] / 5)))
            print(f"    {lbl:12s}  {bar:<24s}  mean={row['mean']:7.2f}%  median={row['median']:7.2f}%  n={int(row['count']):,}")

        # ── Stage 5: Top-20 worst nets ────────────────────────────────────
        df_worst = df.sort_values('mape', ascending=False).head(20)
        print(f"\n[Stage 5] Top-20 Worst Nets by MAPE")
        hdr = f"  {'Net':<30}  {'MAPE%':>7}  {'G_tot':>7}  {'P_tot':>7}  {'G_gnd':>7}  {'P_gnd':>7}  {'G_cpl':>7}  {'P_cpl':>7}"
        print(hdr)
        print(f"  {'-'*(len(hdr)-2)}")
        for _, r in df_worst.iterrows():
            print(f"  {r['net'][:30]:<30}  {r['mape']:>7.2f}  "
                  f"{r['g_tot']:>7.4f}  {r['p_tot']:>7.4f}  "
                  f"{r['g_gnd']:>7.4f}  {r['p_gnd']:>7.4f}  "
                  f"{r['g_cpl']:>7.4f}  {r['p_cpl']:>7.4f}")

        # ── Stage 6: Aggressor mismatch for the single worst net ──────────
        worst_net = df_worst.iloc[0]['net']
        print(f"\n[Stage 6] Aggressor Mismatch — Worst Net: '{worst_net}'")

        g_aggrs = defaultdict(float)
        for node_cpl in gold_data[worst_net]['cpl_caps'].values():
            for a_net, val in node_cpl.items():
                g_aggrs[a_net] += val
        p_aggrs = net_pred[worst_net]['cpl_by_net']

        all_aggrs = set(g_aggrs) | set(p_aggrs)
        df_aggr = pd.DataFrame([{
            'aggr': a,
            'gold': g_aggrs.get(a, 0.0),
            'pred': p_aggrs.get(a, 0.0),
        } for a in all_aggrs])
        df_aggr['diff'] = (df_aggr['gold'] - df_aggr['pred']).abs()
        df_aggr = df_aggr.sort_values('diff', ascending=False).head(10)

        print(f"  {'Aggressor':<40}  {'Gold(fF)':>9}  {'Pred(fF)':>9}  {'|Diff|(fF)':>10}")
        print(f"  {'-'*74}")
        for _, r in df_aggr.iterrows():
            print(f"  {r['aggr'][:40]:<40}  {r['gold']:>9.5f}  {r['pred']:>9.5f}  {r['diff']:>10.5f}")

        # ── Save CSV ──────────────────────────────────────────────────────
        out_csv = MODEL_DIR / f"{design_name}_eval_report.csv"
        df.to_csv(out_csv, index=False)
        print(f"\n  ✅  Report saved → {out_csv}")

    print(f"\n{'='*72}")


def evaluate_model(args, spef_write=False):
    GPU_ID = args.gpu if args.gpu is not None else cfg.GPU_ID
    DEVICE = f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu"
    DATA_DIR = Path(cfg.PROCESSED_DIR)
    MODEL_DIR = Path(cfg.OUTPUT_DIR) / "active_learning" / f"{args.model_name}"
    eval_profiler = RuntimeProfiler(MODEL_DIR / "eval_macro_runtime.csv")

    model = DeepPEX_Model(cfg).to(DEVICE)
    best_ckpt = MODEL_DIR / "best_model.pth"
    if best_ckpt.exists():
        print(f">>> Loading checkpoint: {best_ckpt}")
        state_dict = torch.load(best_ckpt, map_location=DEVICE, weights_only=True)
    else:
        print("❌ No checkpoint found for evaluation.")
        return

    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict, strict=False)
    model.eval()

    manifest_path = DATA_DIR / "dataset_manifest.csv"
    if not manifest_path.exists():
        return
    manifest_df = pd.read_csv(manifest_path)
    test_df = manifest_df[manifest_df['split'] == 'test'].reset_index(drop=True) \
        if 'split' in manifest_df.columns else manifest_df

    if spef_write:
        print("\n" + "="*60)
        print(" 🚀 [Microscopic Faraday Dumping & SPEF Gen Mode] 🚀")
        print("="*60)

        layer_map = LayerInfoParser(cfg.LAYERS_INFO_PATH).parse()
        tech_lef = LefParser(cfg.TECH_LEF_PATH).parse()
        eval_profiler = RuntimeProfiler(MODEL_DIR / "eval_spef_runtime.csv")
        for design_name, group_df in test_df.groupby('design_name'):
            if design_name != "intel22_tv80s_f3": continue

            topo_cache = {}
            global_top_ports = set()
            for topo_file in (DATA_DIR / design_name / "topology").rglob(f"*topo_*.pkl.gz"):
                net_stem = topo_file.name.replace(".pkl.gz", "")
                if "topo_" in net_stem:
                    topo_cache[net_stem.split("topo_")[-1]] = topo_file

            if topo_cache:
                with gzip.open(list(topo_cache.values())[0], 'rb') as f:
                    global_top_ports = set(pickle.load(f).get('top_ports', []))

            print(f"\n>>> Processing Design: {design_name} ({len(group_df)} tiles)")

            dataset = NeuralFieldEvalDataset(DATA_DIR, group_df, pad_size=4096)
            loader = torch.utils.data.DataLoader(dataset, batch_size=64, collate_fn=robust_collate, num_workers=8, pin_memory=True)

            distributed_nodes = defaultdict(lambda: defaultdict(lambda: {'gnd': 0.0, 'cpl': defaultdict(float), 'abs_geo': None}))
            cuboid_accumulators = defaultdict(lambda: defaultdict(lambda: {'gnd': 0.0, 'cpl': defaultdict(float), 'abs_geo': None, 'weight_sum': 0.0}))

            with torch.no_grad():
                for batch in tqdm(loader, desc="Inferencing Distributed Caps"):
                    if batch is None: continue
                    eval_profiler.start("GPU_Inference")
                    cuboids, mask, meta_dict = batch
                    cuboids, mask = cuboids.to(DEVICE), mask.to(DEVICE)
                    if cuboids.shape[-1] > 9: cuboids = cuboids[..., :9]

                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        preds = model(cuboids, mask, compute_coupling=True)
                    eval_profiler.stop("GPU_Inference")

                    eval_profiler.start("Tensor_to_Node_Mapping")
                    B = cuboids.shape[0]
                    c_gnd_seg = torch.nan_to_num(preds['c_gnd_seg'], nan=0.0).cpu().numpy()

                    sparse_cpl = preds['sparse_cpl']
                    b_idx, src_idx, dst_idx = sparse_cpl['b_idx'].cpu().numpy(), sparse_cpl['src_idx'].cpu().numpy(), sparse_cpl['dst_idx'].cpu().numpy()
                    c_cpl = torch.nan_to_num(sparse_cpl['c_cpl'], nan=0.0).cpu().numpy()

                    for b in range(B):
                        target_net = meta_dict['target_net_name'][b]
                        abs_geos = meta_dict['abs_geometries'][b].numpy()
                        names = np.array(meta_dict['cuboid_net_names'][b])
                        core_ratios = meta_dict['core_ratios'][b]

                        semantic_types = cuboids[b, :, 6].cpu().numpy()
                        is_wire = semantic_types > 0.8

                        is_target = (names == target_net) & (core_ratios > 0) & is_wire
                        target_indices = np.where(is_target)[0]
                        if len(target_indices) == 0: continue

                        edge_mask = (b_idx == b)
                        e_src, e_dst, e_cpl = src_idx[edge_mask], dst_idx[edge_mask], c_cpl[edge_mask]
                        valid_edge_mask = is_target[e_src] & (e_cpl * core_ratios[e_src] > 1e-6)
                        e_src, e_dst, e_cpl_val = e_src[valid_edge_mask], e_dst[valid_edge_mask], e_cpl[valid_edge_mask] * core_ratios[e_src[valid_edge_mask]]
                        aggr_names = np.array([str(n).lower() for n in names[e_dst]])

                        is_dummy = np.isin(aggr_names, ["unknown", "pad", "unknown_pin", target_net.lower()])
                        is_power = np.isin(aggr_names, list(POWER_NETS))
                        is_valid = ~(is_dummy | is_power)

                        node_cpl_dict = defaultdict(lambda: defaultdict(float))
                        net_to_net_sums = defaultdict(float)
                        valid_src, valid_dst, valid_val = e_src[is_valid], e_dst[is_valid], e_cpl_val[is_valid]

                        for d, v in zip(valid_dst, valid_val):
                            net_to_net_sums[names[d]] += float(v)

                        for s, d, v in zip(valid_src, valid_dst, valid_val):
                            aggr_name = names[d]
                            if net_to_net_sums[aggr_name] >= 1e-6:
                                node_cpl_dict[s][aggr_name] += float(v)

                        for idx in target_indices:
                            geo_key = tuple(np.round(abs_geos[idx][:6], 6).tolist())
                            node = cuboid_accumulators[target_net][geo_key]
                            if node['abs_geo'] is None:
                                node['abs_geo'] = abs_geos[idx]
                            node['weight_sum'] += float(core_ratios[idx])
                            node['gnd'] += float(c_gnd_seg[b, idx] * core_ratios[idx])

                            cpl_data = node_cpl_dict[idx]
                            for a_net, a_val in cpl_data.items():
                                node['cpl'][a_net] += float(a_val)
                        eval_profiler.stop("Tensor_to_Node_Mapping")

            for target_net, cuboid_dict in cuboid_accumulators.items():
                for cuboid_data in cuboid_dict.values():
                    weight_sum = max(cuboid_data['weight_sum'], 1e-12)
                    cx, cy, cz = cuboid_data['abs_geo'][:3]
                    DBU = 2000
                    spatial_hash = (int(round(cx*DBU)), int(round(cy*DBU)), int(round(cz*DBU)))
                    node = distributed_nodes[target_net][spatial_hash]
                    if node['abs_geo'] is None:
                        node['abs_geo'] = cuboid_data['abs_geo']
                    node['gnd'] += cuboid_data['gnd'] / weight_sum
                    for a_net, a_val in cuboid_data['cpl'].items():
                        node['cpl'][a_net] += a_val / weight_sum

            out_spef_path = MODEL_DIR / f"{design_name}_autonomous.spef"
            print(f">>> Streaming Autonomous Topology to {out_spef_path.name}...")

            with open(out_spef_path, 'w') as spef_file:
                spef_writer = SPEFWriter(file_handle=spef_file, design_name=design_name, top_ports=global_top_ports)
                eval_profiler.start("CPU_KDTree_SPEF_Assembly")
                spef_writer.write_header()

                for safe_net_name, topo_path in tqdm(topo_cache.items(), desc="Writing SPEF Nets"):
                    try:
                        with gzip.open(topo_path, 'rb') as f:
                            topo_data = pickle.load(f)
                        original_segments = topo_data['global_segments']
                        if not original_segments: continue
                    except Exception:
                        continue

                    net_name = safe_net_name
                    for seg in original_segments:
                        if 'net_name' in seg:
                            net_name = seg['net_name']
                            break

                    node_dict = distributed_nodes.get(net_name, {})

                    MAX_LEN = 1.0
                    frac_segments = []
                    for seg in original_segments:
                        if seg.get('type') == 'WIRE':
                            p1, p2 = np.array(seg['start']), np.array(seg['end'])
                            dist = np.linalg.norm(p1 - p2)
                            if dist > MAX_LEN:
                                num_splits = int(np.ceil(dist / MAX_LEN))
                                for i in range(num_splits):
                                    new_seg = seg.copy()
                                    new_seg['start'] = (p1 + (p2 - p1) * (i / num_splits)).tolist()
                                    new_seg['end'] = (p1 + (p2 - p1) * ((i + 1) / num_splits)).tolist()
                                    frac_segments.append(new_seg)
                            else:
                                frac_segments.append(seg)
                        else:
                            frac_segments.append(seg)

                    topo_nodes_2d = set()
                    for seg in frac_segments:
                        stype = seg.get('type')
                        if stype == 'WIRE':
                            topo_nodes_2d.add(tuple(seg['start']))
                            topo_nodes_2d.add(tuple(seg['end']))
                        elif stype == 'VIA':
                            topo_nodes_2d.add(tuple(seg['pos']))
                        elif stype in ['PIN', 'INST_PORT'] and 'pos' in seg:
                            r = seg['pos']
                            topo_nodes_2d.add(((r[0]+r[2])/2.0, (r[1]+r[3])/2.0))
                        elif stype == 'RECT' and 'rect' in seg:
                            r = seg['rect']
                            topo_nodes_2d.add(((r[0]+r[2])/2.0, (r[1]+r[3])/2.0))

                    topo_nodes_list = list(topo_nodes_2d)
                    if not topo_nodes_list: continue

                    DBU = 2000
                    tree = cKDTree(np.array(topo_nodes_list))
                    smart_node_dict = defaultdict(lambda: {'gnd': 0.0, 'cpl': defaultdict(float), 'abs_geo': None})

                    for ml_hash, ml_data in node_dict.items():
                        ml_coord_3d = ml_data['abs_geo'][:3] if ml_data['abs_geo'] is not None else np.array(ml_hash) / DBU
                        ml_coord_2d = ml_coord_3d[:2]
                        ml_z = ml_coord_3d[2]

                        k_neighbors = min(2, len(topo_nodes_list))
                        dists, idxs = tree.query(ml_coord_2d, k=k_neighbors)

                        if k_neighbors == 1 or np.isscalar(dists):
                            w1, w2, idx1, idx2 = 1.0, 0.0, int(idxs), None
                        elif dists[0] < 1e-6:
                            w1, w2, idx1, idx2 = 1.0, 0.0, int(idxs[0]), None
                        else:
                            inv_d1, inv_d2 = 1.0 / dists[0], 1.0 / dists[1]
                            w1, w2 = inv_d1 / (inv_d1 + inv_d2), inv_d2 / (inv_d1 + inv_d2)
                            idx1, idx2 = int(idxs[0]), int(idxs[1])

                        n1_coord_2d = topo_nodes_list[idx1]
                        n1_hash = (int(round(n1_coord_2d[0]*DBU)), int(round(n1_coord_2d[1]*DBU)), int(round(ml_z*DBU)))
                        smart_node_dict[n1_hash]['gnd'] += ml_data['gnd'] * w1
                        if smart_node_dict[n1_hash]['abs_geo'] is None:
                            smart_node_dict[n1_hash]['abs_geo'] = np.array([n1_coord_2d[0], n1_coord_2d[1], ml_z])
                        for a_net, a_val in ml_data['cpl'].items():
                            smart_node_dict[n1_hash]['cpl'][a_net] += a_val * w1

                        if w2 > 0:
                            n2_coord_2d = topo_nodes_list[idx2]
                            n2_hash = (int(round(n2_coord_2d[0]*DBU)), int(round(n2_coord_2d[1]*DBU)), int(round(ml_z*DBU)))
                            smart_node_dict[n2_hash]['gnd'] += ml_data['gnd'] * w2
                            if smart_node_dict[n2_hash]['abs_geo'] is None:
                                smart_node_dict[n2_hash]['abs_geo'] = np.array([n2_coord_2d[0], n2_coord_2d[1], ml_z])
                            for a_net, a_val in ml_data['cpl'].items():
                                smart_node_dict[n2_hash]['cpl'][a_net] += a_val * w2

                    graph_builder = AutonomousGraphBuilder(
                        net_name=net_name, global_segments=frac_segments,
                        ml_distributed_nodes=smart_node_dict, top_ports=global_top_ports,
                        layer_info=layer_map, tech_lef=tech_lef
                    )
                    spef_writer.stream_autonomous_net(graph_builder)
                eval_profiler.stop("CPU_KDTree_SPEF_Assembly")

            eval_profiler.save_and_reset("Eval_SPEF_Gen", f"Design_{design_name}")
            print(f"✅ Autonomous SPEF Streamed successfully to: {out_spef_path}")

    else:
        # Direct MAPE evaluation (no SPEF file written)
        run_direct_eval(model, test_df, DATA_DIR, MODEL_DIR, DEVICE,
                        design_filter=getattr(args, 'design', None))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--spef_write', action='store_true')
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--design', type=str, default=None,
                        help='Substring filter for design name (e.g. tv80s)')
    args = parser.parse_args()
    import time
    start = time.time()
    evaluate_model(args, spef_write=args.spef_write)
    print(f"Total time: {time.time() - start:.2f} seconds")
