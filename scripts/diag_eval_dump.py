#!/usr/bin/env python3
"""
scripts/diag_eval_dump.py

Run a single validation forward pass on best_model.pth and dump per-net +
per-cuboid arrays to NPZ for downstream Phase A diagnostic analyses
(case 1 baselines, case 2 CPL distribution, case 3 GND breakdown).

Auto-detects whether the checkpoint contains DS-PINN MacroDensityFNO
weights and configures cfg._use_dspinn accordingly.

Usage:
  python3 scripts/diag_eval_dump.py --model_name v10b --gpu 2
  python3 scripts/diag_eval_dump.py --model_name dspinn_v1_new --gpu 2
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/home/jslee/projects/PINNPEX")

import numpy as np
import pandas as pd
import torch

import configs.config as cfg
from src.models.neural_field import DeepPEX_Model
from src.preprocessing.layer_parser import LayerInfoParser
from src.physics.materials import BEOLMaterialStack
from src.active_learning.oracle import FullChipPEXOracle
from src.data.replay_buffer import DesignLevelReplayBuffer


def autodetect_dspinn(ckpt_path: Path) -> bool:
    state = torch.load(str(ckpt_path), map_location='cpu', weights_only=True)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    return any('macro_density_fno' in k for k in state.keys())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name', required=True)
    ap.add_argument('--gpu', type=int, default=2)
    ap.add_argument('--ckpt_filename', default='best_model.pth')
    ap.add_argument('--physics_only', action='store_true',
                    help='Zero out learned MLP corrections so the model emits '
                         'pure rule-based physics predictions: gnd_modifier=1.0, '
                         'cpl_modifier=1.0, cpl_residual≈0. layer_scale_phys_gnd '
                         'is reset to physics-calibrated init values.')
    ap.add_argument('--out_suffix', default='',
                    help='Suffix for the output NPZ filename (e.g. "_physics" → eval_dump_physics.npz).')
    args = ap.parse_args()

    DEVICE = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)

    al_root = Path(cfg.OUTPUT_DIR) / "active_learning"
    al_dir = al_root / args.model_name
    ckpt_path = al_dir / args.ckpt_filename
    if not ckpt_path.exists():
        print(f"❌ Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    use_dspinn = autodetect_dspinn(ckpt_path)
    print(f">>> {args.model_name}: auto-detected use_dspinn = {use_dspinn}")
    cfg._use_dspinn = use_dspinn
    cfg._use_gino = False

    # Build model
    model = DeepPEX_Model(cfg).to(DEVICE)

    # Shape-filtered load (mirrors run_active_learning).
    # Strip torch.compile's "_orig_mod." prefix when present so v10b ckpts
    # (saved during compiled training) load against the eager model used here.
    state = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=True)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    current_state = model.state_dict()
    filtered = {k: v for k, v in state.items()
                if k in current_state and v.shape == current_state[k].shape}
    dropped = [k for k in state if k not in filtered]
    model.load_state_dict(filtered, strict=False)
    print(f">>> Loaded ckpt: {len(filtered)} kept, {len(dropped)} shape-filtered.")

    if args.physics_only:
        print(">>> [PHYSICS-ONLY MODE] Disabling learned MLP corrections.")
        fr = model.flux_router if hasattr(model, 'flux_router') else model._orig_mod.flux_router
        with torch.no_grad():
            # gnd_mlp: zero last layer → gnd_modifier = exp(0) = 1.0
            fr.gnd_mlp[-1].weight.zero_()
            fr.gnd_mlp[-1].bias.zero_()
            # cpl_mlp: zero last layer weight, set bias = [0, -10] → modifier=1, residual≈0
            fr.cpl_mlp[-1].weight.zero_()
            fr.cpl_mlp[-1].bias.copy_(torch.tensor([0.0, -10.0], device=fr.cpl_mlp[-1].bias.device))
            # Reset per-layer GND density to physics-calibrated init
            init_gnd = fr._make_gnd_cap_density_init().to(fr.layer_scale_phys_gnd.device)
            fr.layer_scale_phys_gnd.copy_(init_gnd)
            # Reset per-layer-pair CPL log scale to zero (softplus(0)≈0.693)
            if hasattr(fr, 'cpl_layer_pair_log_scale'):
                fr.cpl_layer_pair_log_scale.zero_()
            if hasattr(fr, 'layer_scale_phys_cpl'):
                fr.layer_scale_phys_cpl.fill_(-1.45)  # original physics init
            # Reset fringe scale
            init_fringe = fr._make_gnd_fringe_scale_init().to(fr.gnd_fringe_scale.device)
            fr.gnd_fringe_scale.copy_(init_fringe)
            # Disable explicit VSS rail edges (reset vss_gnd_scale to soft init)
            if hasattr(fr, 'vss_gnd_scale'):
                fr.vss_gnd_scale.fill_(-3.0)
        print(">>> Physics base reset: gnd_mlp/cpl_mlp output zeroed, layer scales → init.")

    model.eval()

    # Build val_loader from predefined cache (same path as run_active_learning's
    # FAST_ENGINEERING_MODE).
    cache_dir = al_root / "cache"
    val_cache_path = cache_dir / "predefined_valid_subset.csv"
    if not val_cache_path.exists():
        print(f"❌ Predefined valid cache not found: {val_cache_path}")
        sys.exit(1)
    val_df = pd.read_csv(val_cache_path)

    oracle = FullChipPEXOracle(al_root)
    val_buffer = DesignLevelReplayBuffer(max_designs=10)
    def_map = {p.stem: p for p in cfg.TRAIN_DEFS + cfg.TEST_DEFS}

    print(">>> Loading validation data + golden SPEFs...")
    for d_name in val_df['design_name'].unique():
        d_def_path = def_map.get(d_name)
        if d_def_path:
            d_spef = oracle.generate_golden_spef(d_name, d_def_path)
            val_buffer.add_design(d_name, val_df[val_df['design_name'] == d_name],
                                  d_spef)
    val_loader = val_buffer.get_dataloader()
    print(f">>> Val tiles loaded: {len(val_buffer.all_data)}")

    # Build a net_name → design_name lookup from the buffer's all_data.
    name_to_design: dict[str, str] = {}
    for _, row in val_buffer.all_data.iterrows():
        name_to_design[str(row['net_name']).replace('\\', '')] = str(row['design_name'])

    # Forward pass with comprehensive capture
    POWER_NETS = {'vss', 'vdd', 'vcc', 'gnd', 'vssx', 'vccx'}
    MAX_AGGR_PAD = int(getattr(cfg, 'MAX_AGGR_BUDGET', 512))

    all_pred_total, all_pred_gnd, all_pred_cpl = [], [], []
    all_y_total, all_y_gnd, all_y_cpl, all_valid_aggr = [], [], [], []
    all_design_names: list[str] = []
    all_net_size: list[int] = []
    all_net_z_mean: list[float] = []
    all_net_target_name: list[str] = []
    all_cuboid_gnd: list[np.ndarray] = []
    all_cuboid_z: list[np.ndarray] = []
    all_cuboid_to_net: list[np.ndarray] = []
    cur_net_offset = 0

    print(">>> Running validation forward...")
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            cuboids, mask, labels_dict, meta_dict = batch
            cuboids = cuboids.to(DEVICE)
            mask = mask.to(DEVICE)

            A_tgt = labels_dict['A_tgt'].to(DEVICE)
            Y_total = labels_dict['Y_total'].to(DEVICE)
            Y_gnd = labels_dict['Y_gnd'].to(DEVICE)
            A_aggr = labels_dict['A_aggr'].to(DEVICE)
            Y_cpl = labels_dict['Y_cpl'].to(DEVICE)
            valid_aggr_mask = labels_dict['valid_aggr_mask'].to(DEVICE)
            core_ratios = labels_dict['core_ratios'].to(DEVICE)
            batch_net_ids = labels_dict['batch_net_ids'].to(DEVICE)
            num_nets = labels_dict['num_unique_nets']
            frw_matrix = labels_dict.get('frw_ratio_matrix', None)
            if frw_matrix is not None:
                frw_matrix = frw_matrix.to(DEVICE)
            n_tiles = meta_dict.get('n_tiles', None)
            endpoint_prox = meta_dict.get('endpoint_prox', None)
            if isinstance(n_tiles, torch.Tensor):
                n_tiles = n_tiles.to(DEVICE)
            if isinstance(endpoint_prox, torch.Tensor):
                endpoint_prox = endpoint_prox.to(DEVICE)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                preds = model(cuboids, mask, compute_coupling=True,
                              frw_ratio_matrix=frw_matrix,
                              n_tiles=n_tiles,
                              endpoint_prox=endpoint_prox)
            c_total_phys = preds['c_total_phys'].float()
            c_gnd_seg    = preds['c_gnd_seg'].float()
            sparse_cpl   = preds['sparse_cpl']

            global_pred_total = torch.zeros(num_nets, dtype=torch.float32, device=DEVICE)
            global_pred_total.scatter_add_(0, batch_net_ids,
                torch.sum(c_total_phys * A_tgt * core_ratios, dim=1))
            global_pred_gnd = torch.zeros(num_nets, dtype=torch.float32, device=DEVICE)
            global_pred_gnd.scatter_add_(0, batch_net_ids,
                torch.sum(c_gnd_seg * A_tgt * core_ratios, dim=1))

            B = A_tgt.shape[0]
            MAX_AGGR = Y_cpl.shape[1]
            b_idx, src_idx, dst_idx = sparse_cpl['b_idx'].long(), sparse_cpl['src_idx'].long(), sparse_cpl['dst_idx'].long()

            is_power_mask = torch.zeros((B, cuboids.shape[1]), dtype=torch.bool, device=DEVICE)
            for b in range(B):
                names = meta_dict['cuboid_net_names'][b]
                for i, name in enumerate(names):
                    if str(name).lower() in POWER_NETS:
                        is_power_mask[b, i] = True

            if b_idx.numel() > 0:
                is_dst_power = is_power_mask[b_idx, dst_idx]
                raw_edge_cpl = sparse_cpl['c_cpl'].float() * torch.where(
                    A_tgt[b_idx, src_idx] > 0, core_ratios[b_idx, src_idx], core_ratios[b_idx, dst_idx])
                power_cpl_flux = raw_edge_cpl * is_dst_power.float()
                signal_cpl_flux = raw_edge_cpl * (~is_dst_power).float()
                global_pred_gnd.scatter_add_(0, batch_net_ids[b_idx], power_cpl_flux)

                aggr_mask_E = (A_tgt[b_idx, src_idx].unsqueeze(1) * A_aggr[b_idx, :, dst_idx]
                              + A_tgt[b_idx, dst_idx].unsqueeze(1) * A_aggr[b_idx, :, src_idx])
                tile_cpl = torch.zeros(B, MAX_AGGR, dtype=torch.float32, device=DEVICE)
                tile_cpl.index_add_(0, b_idx, signal_cpl_flux.unsqueeze(1) * aggr_mask_E)
                global_pred_cpl = torch.zeros(num_nets, MAX_AGGR, dtype=torch.float32, device=DEVICE)
                global_pred_cpl.index_add_(0, batch_net_ids, tile_cpl)
            else:
                global_pred_cpl = torch.zeros(num_nets, MAX_AGGR, dtype=torch.float32, device=DEVICE)

            net_indices = torch.arange(num_nets, dtype=torch.long, device=DEVICE)

            # Pad CPL per-aggressor arrays to MAX_AGGR_PAD so we can concatenate
            # across batches with different aggressor budgets.
            def _pad_aggr(arr_np: np.ndarray) -> np.ndarray:
                if arr_np.ndim != 2:
                    return arr_np
                cur = arr_np.shape[1]
                if cur >= MAX_AGGR_PAD:
                    return arr_np[:, :MAX_AGGR_PAD]
                pad = np.zeros((arr_np.shape[0], MAX_AGGR_PAD - cur), dtype=arr_np.dtype)
                return np.concatenate([arr_np, pad], axis=1)

            all_pred_total.append(global_pred_total.cpu().numpy())
            all_pred_gnd.append(global_pred_gnd.cpu().numpy())
            all_pred_cpl.append(_pad_aggr(global_pred_cpl.cpu().numpy()))
            all_y_total.append(Y_total[net_indices].cpu().numpy())
            all_y_gnd.append(Y_gnd[net_indices].cpu().numpy())
            all_y_cpl.append(_pad_aggr(Y_cpl[net_indices].cpu().numpy()))
            all_valid_aggr.append(_pad_aggr(valid_aggr_mask[net_indices].cpu().numpy()))

            # Per-net design name + size + z mean
            for net_id in range(num_nets):
                tile_mask = (batch_net_ids == net_id)
                if tile_mask.any():
                    tile_pos = tile_mask.nonzero(as_tuple=False).flatten()[0].item()
                    target_net = str(meta_dict['target_net_name'][tile_pos]).replace('\\', '')
                    design = name_to_design.get(target_net, 'unknown')
                    all_design_names.append(design)
                    all_net_target_name.append(target_net)

                    tile_indices = tile_mask.nonzero(as_tuple=False).flatten()
                    net_cuboid_count = 0
                    net_z_sum = 0.0
                    for ti in tile_indices.cpu().tolist():
                        v = ~mask[ti]
                        net_cuboid_count += int(v.sum().item())
                        if v.any():
                            net_z_sum += float(cuboids[ti, :, 2][v].sum().item())
                    all_net_size.append(net_cuboid_count)
                    all_net_z_mean.append(net_z_sum / max(net_cuboid_count, 1))

            # Per-cuboid GND for layer breakdown
            for b in range(B):
                v = ~mask[b]
                v_count = int(v.sum().item())
                if v_count == 0:
                    continue
                z_b = cuboids[b, :, 2][v].cpu().numpy()
                gnd_b = c_gnd_seg[b][v].cpu().numpy()
                net_id_b = int(batch_net_ids[b].item()) + cur_net_offset
                all_cuboid_z.append(z_b)
                all_cuboid_gnd.append(gnd_b)
                all_cuboid_to_net.append(np.full(v_count, net_id_b, dtype=np.int32))

            cur_net_offset += num_nets

    pred_total = np.concatenate(all_pred_total)
    pred_gnd   = np.concatenate(all_pred_gnd)
    pred_cpl   = np.concatenate(all_pred_cpl)
    y_total    = np.concatenate(all_y_total)
    y_gnd      = np.concatenate(all_y_gnd)
    y_cpl      = np.concatenate(all_y_cpl)
    valid_aggr = np.concatenate(all_valid_aggr)
    designs    = np.array(all_design_names)
    net_size   = np.array(all_net_size, dtype=np.int32)
    net_z_mean = np.array(all_net_z_mean, dtype=np.float32)
    target_names = np.array(all_net_target_name)
    cuboid_gnd = np.concatenate(all_cuboid_gnd) if all_cuboid_gnd else np.zeros(0, dtype=np.float32)
    cuboid_z   = np.concatenate(all_cuboid_z)   if all_cuboid_z   else np.zeros(0, dtype=np.float32)
    cuboid_to_net = np.concatenate(all_cuboid_to_net) if all_cuboid_to_net else np.zeros(0, dtype=np.int32)

    out_path = al_dir / f"eval_dump{args.out_suffix}.npz"
    np.savez_compressed(
        str(out_path),
        pred_total=pred_total, pred_gnd=pred_gnd, pred_cpl=pred_cpl,
        y_total=y_total, y_gnd=y_gnd, y_cpl=y_cpl,
        valid_aggr=valid_aggr,
        designs=designs, net_size=net_size, net_z_mean=net_z_mean,
        target_names=target_names,
        cuboid_gnd=cuboid_gnd, cuboid_z=cuboid_z, cuboid_to_net=cuboid_to_net,
        model_name=args.model_name,
        use_dspinn=use_dspinn,
    )
    print(f">>> Dumped: {out_path}")
    print(f">>> Summary: {len(y_total)} nets, {cuboid_gnd.shape[0]:,} cuboids, "
          f"MAX_AGGR={y_cpl.shape[1]}, {len(np.unique(designs))} designs")


if __name__ == '__main__':
    main()
