"""
Diagnose the MAPE hurdle: per-net predictions aggregated across all batches.
Each net's tiles may span multiple batches (batch_size=4), so we sum the
partial scatter_add contributions across batches to get the full net prediction.
"""
import os, sys, ast
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import configs.config as cfg
from src.models.neural_field import DeepPEX_Model
from src.data.datasets import NeuralFieldFinetuneDataset, robust_collate
from src.data.replay_buffer import NetGroupedSampler  # correct sampler: batch_nets=1
from torch.utils.data import DataLoader

GPU = int(os.environ.get('GPU', '0'))
DEVICE = f'cuda:{GPU}'
MODEL_DIR = Path(f"{cfg.OUTPUT_DIR}/active_learning/v3_netlevel")
CKPT = MODEL_DIR / "best_model.pth"
VALID_CSV = Path(f"{cfg.OUTPUT_DIR}/active_learning/cache/predefined_valid_subset.csv")

print(f">>> Using GPU {GPU}, ckpt: {CKPT}")

model = DeepPEX_Model(cfg).to(DEVICE)
state = torch.load(CKPT, map_location=DEVICE, weights_only=True)
# Strip torch.compile's _orig_mod. prefix if present
state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
current = model.state_dict()
filtered = {k: v for k, v in state.items() if k in current and v.shape == current[k].shape}
model.load_state_dict(filtered, strict=False)
model.eval()
print(f"  loaded {len(filtered)}/{len(state)} tensors from checkpoint")

valid_df = pd.read_csv(VALID_CSV)
valid_df['coupled_caps'] = valid_df['coupled_caps'].apply(ast.literal_eval)
print(f"  valid tiles: {len(valid_df)}, nets: {valid_df['net_name'].nunique()}")

dataset = NeuralFieldFinetuneDataset(str(cfg.PROCESSED_DIR), valid_df, pad_size=1024)
sampler = NetGroupedSampler(valid_df, batch_nets=1, max_tiles_per_batch=128)
loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=robust_collate,
                    num_workers=2, pin_memory=True)

# Aggregate per-net predictions across ALL batches
net_pred_sum = defaultdict(float)
net_gt     = {}
net_tile_count_seen = defaultdict(int)

with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
    for batch_idx, batch in enumerate(loader):
        if batch is None: continue
        cuboids, mask, labels_dict, meta_dict = batch
        cuboids = cuboids.to(DEVICE)[..., :9]
        mask = mask.to(DEVICE)
        A_tgt = labels_dict['A_tgt'].to(DEVICE)
        Y_total = labels_dict['Y_total'].to(DEVICE)
        core_ratios = labels_dict['core_ratios'].to(DEVICE)
        batch_net_ids = labels_dict['batch_net_ids'].to(DEVICE)
        num_nets = labels_dict['num_unique_nets']
        frw_matrix = labels_dict.get('frw_ratio_matrix', None)
        if frw_matrix is not None: frw_matrix = frw_matrix.to(DEVICE)

        preds = model(cuboids, mask, compute_coupling=True, frw_ratio_matrix=frw_matrix)
        c_total_phys = preds['c_total_phys'].float()

        pred_total = torch.zeros(num_nets, dtype=torch.float32, device=DEVICE).scatter_add_(
            0, batch_net_ids, torch.sum(c_total_phys * A_tgt * core_ratios, dim=1))

        net_ids_cpu = batch_net_ids.cpu().numpy()
        target_nets_batch = [meta_dict['target_net_name'][b].replace('\\', '')
                             for b in range(len(meta_dict['target_net_name']))]
        # Map batch-local net_id → global net_name
        nid_to_name = {}
        for b in range(len(target_nets_batch)):
            nid_to_name[int(net_ids_cpu[b])] = target_nets_batch[b]

        pred_total_cpu = pred_total.cpu().numpy()
        Y_total_cpu = Y_total.cpu().numpy()
        for nid, nname in nid_to_name.items():
            net_pred_sum[nname] += float(pred_total_cpu[nid])
            net_gt[nname] = float(Y_total_cpu[nid])  # Y_total is per-net, same each batch
            net_tile_count_seen[nname] += sum(1 for x in net_ids_cpu if x == nid)

        if batch_idx % 50 == 0:
            print(f"  batch {batch_idx}/{len(sampler)} done")

# Build per-net diagnostic dataframe
records = []
for nname in net_gt:
    gt = net_gt[nname]
    pr = net_pred_sum[nname]
    records.append({'net_name': nname, 'gt_total': gt, 'pred_total': pr,
                    'n_tiles_seen': net_tile_count_seen[nname]})
pred_df = pd.DataFrame(records)

net_to_design = valid_df.drop_duplicates('net_name').set_index('net_name')['design_name']
pred_df['design_name'] = pred_df['net_name'].map(net_to_design)
pred_df['abs_err'] = (pred_df['pred_total'] - pred_df['gt_total']).abs()
pred_df['rel_err'] = pred_df['abs_err'] / pred_df['gt_total'].clip(lower=0.005) * 100
pred_df['signed_err_pct'] = (pred_df['pred_total'] - pred_df['gt_total']) / pred_df['gt_total'].clip(lower=0.005) * 100

net_tile_counts = valid_df.groupby('net_name').size().rename('n_tiles_total')
pred_df = pred_df.merge(net_tile_counts, left_on='net_name', right_index=True)

def count_aggr(name):
    sub = valid_df[valid_df['net_name'] == name]
    all_aggr = set()
    for cc in sub['coupled_caps']:
        all_aggr.update(cc.keys())
    return len(all_aggr)
pred_df['n_aggr'] = pred_df['net_name'].apply(count_aggr)

print("\n" + "="*80)
print(f"OVERALL: {len(pred_df)} UNIQUE nets, MAPE = {pred_df['rel_err'].mean():.2f}%, median = {pred_df['rel_err'].median():.2f}%")
print(f"Mean signed error: {pred_df['signed_err_pct'].mean():.2f}% (+=over-pred, -=under-pred)")

print("\n=== PER-DESIGN MAPE ===")
print(pred_df.groupby('design_name').agg(
    n_nets=('net_name', 'count'),
    mape=('rel_err', 'mean'),
    median_rel=('rel_err', 'median'),
    signed=('signed_err_pct', 'mean'),
    gt_cap_sum=('gt_total', 'sum'),
    abs_err_sum=('abs_err', 'sum'),
).round(2).to_string())

print("\n=== MAPE BY NET CAP MAGNITUDE ===")
bins = [0, 5, 20, 100, 500, float('inf')]
labels = ['<5fF', '5-20', '20-100', '100-500', '>500fF']
pred_df['cap_bin'] = pd.cut(pred_df['gt_total'], bins=bins, labels=labels)
print(pred_df.groupby('cap_bin', observed=True).agg(
    count=('net_name', 'count'),
    mape=('rel_err', 'mean'),
    signed=('signed_err_pct', 'mean'),
    total_true_cap=('gt_total', 'sum'),
    total_err=('abs_err', 'sum'),
).round(2).to_string())

print("\n=== MAPE BY NET TILE COUNT ===")
bins = [0, 5, 15, 30, 60, float('inf')]
labels = ['1-5', '6-15', '16-30', '31-60', '>60 tiles']
pred_df['tile_bin'] = pd.cut(pred_df['n_tiles_total'], bins=bins, labels=labels)
print(pred_df.groupby('tile_bin', observed=True).agg(
    count=('net_name', 'count'),
    mape=('rel_err', 'mean'),
    signed=('signed_err_pct', 'mean'),
).round(2).to_string())

print("\n=== MAPE BY AGGRESSOR DENSITY ===")
bins = [0, 50, 200, 500, 1000, float('inf')]
labels = ['<50', '50-200', '200-500', '500-1000', '>1000']
pred_df['aggr_bin'] = pd.cut(pred_df['n_aggr'], bins=bins, labels=labels)
print(pred_df.groupby('aggr_bin', observed=True).agg(
    count=('net_name', 'count'),
    mape=('rel_err', 'mean'),
    signed=('signed_err_pct', 'mean'),
).round(2).to_string())

print("\n=== TOP 10 WORST NETS (by absolute error) ===")
top_err = pred_df.sort_values('abs_err', ascending=False).head(10)
print(top_err[['net_name', 'design_name', 'gt_total', 'pred_total', 'abs_err', 'rel_err',
               'n_tiles_total', 'n_aggr']].to_string(index=False))

print("\n=== TOP 10 WORST NETS (relative error, gt>5fF) ===")
top_rel = pred_df[pred_df['gt_total'] > 5].sort_values('rel_err', ascending=False).head(10)
print(top_rel[['net_name', 'design_name', 'gt_total', 'pred_total', 'abs_err', 'rel_err',
               'n_tiles_total', 'n_aggr']].to_string(index=False))

# Pred/GT ratio distribution
print("\n=== PREDICTION CALIBRATION (pred/gt ratio) ===")
pred_df['ratio'] = pred_df['pred_total'] / pred_df['gt_total'].clip(lower=0.01)
print(f"Ratio stats: median={pred_df['ratio'].median():.2f}, mean={pred_df['ratio'].mean():.2f}")
print(f"Ratio <0.5 (severe under-pred): {(pred_df['ratio']<0.5).sum()}/{len(pred_df)}")
print(f"Ratio >2.0 (severe over-pred):  {(pred_df['ratio']>2.0).sum()}/{len(pred_df)}")
print(f"Ratio in [0.8, 1.2] (good):     {((pred_df['ratio']>=0.8)&(pred_df['ratio']<=1.2)).sum()}/{len(pred_df)}")

out_path = MODEL_DIR / 'diagnose_hurdle.csv'
pred_df.to_csv(out_path, index=False)
print(f"\n💾 Saved per-net predictions to {out_path}")
