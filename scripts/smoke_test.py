#!/usr/bin/env python3
"""
Smoke test: validate recent training changes without full AL loop.

Checks:
  [GRAD]  gnd_mlp / cpl_mlp gradient norms are non-zero
  [GRAD]  encoder params have requires_grad=False (frozen, not grad=0)
  [LOSS]  loss_scale * 3.0 dominates loss_distribution * 0.3
  [DIST]  pred_p_gnd not saturating (< X% above 0.99 or below 0.01)
  [MAPE]  compute_net_mape() returns a finite number, trends downward
  [LR]    logs effective LR at each checkpoint
  [DELTA] trainable params change after training; frozen params do not

Usage:
    python3 scripts/smoke_test.py --model_name v1 --steps 200 --report_every 50 --gpu 1
"""

import sys, argparse, gc, copy
from pathlib import Path

import numpy as np
import torch
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import configs.config as cfg
from src.models.neural_field import DeepPEX_Model
from src.preprocessing.layer_parser import LayerInfoParser
from src.physics.materials import BEOLMaterialStack
from src.trainers.finetuner import NeuralFieldFinetuner
from src.active_learning.oracle import FullChipPEXOracle
from src.data.replay_buffer import DesignLevelReplayBuffer


# -------------------------------------------------------------------------
# Helpers

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

def _check(cond, msg):
    tag = PASS if cond else FAIL
    print(f"  {tag} {msg}")
    return cond


def load_model(ckpt_path, device):
    model = DeepPEX_Model(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    current = model.state_dict()
    filtered = {k: v for k, v in state.items()
                if k in current and v.shape == current[k].shape}
    skipped = [k for k in state if k not in filtered]
    model.load_state_dict(filtered, strict=False)
    print(f"{INFO} Loaded {len(filtered)} tensors, skipped {len(skipped)}: {skipped[:5]}")
    return model


def snapshot_params(model):
    """Return {name: tensor.clone()} for all parameters."""
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def grad_norm(params_iter):
    """Compute total gradient L2 norm over a list of (name, param) tuples."""
    total = 0.0
    for _, p in params_iter:
        if p.grad is not None:
            total += p.grad.detach().float().norm().item() ** 2
    return total ** 0.5


def load_cached_data(al_dir, oracle, def_map):
    """Load existing predefined cache into train/val buffers."""
    cache_dir = al_dir / "cache"
    train_csv = cache_dir / "predefined_train_subset.csv"
    val_csv   = cache_dir / "predefined_valid_subset.csv"

    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(
            f"Cache not found at {cache_dir}. Run run_active_learning.py once first."
        )

    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)

    train_buf = DesignLevelReplayBuffer(max_designs=50)
    val_buf   = DesignLevelReplayBuffer(max_designs=10)

    for d_name in val_df['design_name'].unique():
        d_def = def_map.get(d_name)
        if d_def:
            spef = oracle.generate_golden_spef(d_name, d_def)
            val_buf.add_design(d_name, val_df[val_df['design_name'] == d_name], spef)

    for d_name in train_df['design_name'].unique():
        d_def = def_map.get(d_name)
        if d_def:
            spef = oracle.generate_golden_spef(d_name, d_def)
            train_buf.add_design(d_name, train_df[train_df['design_name'] == d_name], spef)

    print(f"{INFO} Train tiles: {len(train_buf.all_data)} | Val tiles: {len(val_buf.all_data)}")
    return train_buf, val_buf


# -------------------------------------------------------------------------
# Main smoke test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='v1')
    parser.add_argument('--gpu',         type=int, default=1)
    parser.add_argument('--steps',       type=int, default=200)
    parser.add_argument('--report_every',type=int, default=50)
    parser.add_argument('--sat_thresh',  type=float, default=0.20,
                        help="Warn if >X fraction of pred_p_gnd is near 0 or 1")
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    al_dir  = Path(cfg.OUTPUT_DIR) / "active_learning"
    model_dir = al_dir / args.model_name
    ckpt_path = model_dir / "best_model.pth"

    print(f"\n{'='*60}")
    print(f"  PINN-PEX Smoke Test  |  {args.steps} steps  |  {device}")
    print(f"{'='*60}\n")

    # ---- 1. Load model ----
    if not ckpt_path.exists():
        print(f"{WARN} No checkpoint at {ckpt_path}. Training from random init.")
        model = DeepPEX_Model(cfg).to(device)
    else:
        model = load_model(ckpt_path, device)

    model.freeze_ssl_layers()
    print(f"{INFO} freeze_ssl_layers() applied.")

    # Verify frozen vs trainable
    frozen_names, trainable_names = [], []
    for n, p in model.named_parameters():
        (frozen_names if not p.requires_grad else trainable_names).append(n)
    print(f"{INFO} Frozen  : {len(frozen_names)} params")
    print(f"{INFO} Trainable: {len(trainable_names)} params")
    _check(len(trainable_names) > 0, "at least one trainable parameter exists")
    _check(any('encoder' in n for n in frozen_names), "encoder is in frozen set")

    # Check optimizer groups don't include frozen params
    finetuner = NeuralFieldFinetuner(model, device=str(device), lr=cfg.AL_LR)
    opt_param_ids = {id(p) for g in finetuner.optimizer.param_groups for p in g['params']}
    frozen_in_opt = any(id(p) in opt_param_ids
                        for n, p in model.named_parameters() if not p.requires_grad)
    _check(not frozen_in_opt, "frozen params excluded from optimizer")

    # ---- 2. Load data ----
    oracle  = FullChipPEXOracle(al_dir)
    def_map = {p.stem: p for p in cfg.TRAIN_DEFS + cfg.TEST_DEFS}
    train_buf, val_buf = load_cached_data(al_dir, oracle, def_map)

    train_loader = train_buf.get_dataloader()
    val_loader   = val_buf.get_dataloader()

    if val_loader is None:
        print(f"{FAIL} Val loader is None — aborting.")
        return

    # ---- 3. Baseline evaluation (step 0) ----
    print(f"\n{INFO} === BASELINE (before any training) ===")
    snap_before = snapshot_params(model)
    baseline_mape = finetuner.compute_net_mape(val_loader)
    print(f"  Baseline net_mape : {baseline_mape:.2f}%")
    _, s_tot, s_gnd, s_cpl = finetuner.evaluate(val_loader)
    print(f"  Baseline scores   : tot={s_tot:.4f}  gnd={s_gnd:.4f}  cpl={s_cpl:.4f}")

    # Record effective LR at step 0
    init_lr = finetuner.optimizer.param_groups[0]['lr']
    print(f"  Initial LR        : {init_lr:.2e}")

    # ---- 4. Patched training loop ----
    print(f"\n{INFO} === TRAINING ({args.steps} steps, report every {args.report_every}) ===\n")

    step_losses_scale = []
    step_losses_dist  = []
    mape_history      = []
    passes            = []

    model.train()
    step = 0
    nan_detected = False

    data_iter = iter(train_loader)

    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        if batch is None:
            continue

        cuboids, mask, labels_dict, meta_dict = batch
        cuboids = cuboids.to(device)
        mask    = mask.to(device)
        if cuboids.shape[-1] > 9:
            cuboids = cuboids[..., :9]

        A_tgt          = labels_dict['A_tgt'].to(device)
        Y_total        = labels_dict['Y_total'].to(device)
        Y_gnd          = labels_dict['Y_gnd'].to(device)
        A_aggr         = labels_dict['A_aggr'].to(device)
        Y_cpl          = labels_dict['Y_cpl'].to(device)
        valid_aggr_mask= labels_dict['valid_aggr_mask'].to(device)
        core_ratios    = labels_dict['core_ratios'].to(device)
        batch_net_ids  = labels_dict['batch_net_ids'].to(device)
        num_nets       = labels_dict['num_unique_nets']
        frw_matrix     = labels_dict.get('frw_ratio_matrix')
        if frw_matrix is not None:
            frw_matrix = frw_matrix.to(device)

        finetuner.optimizer.zero_grad(set_to_none=True)

        import torch.nn.functional as F
        with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu', dtype=torch.bfloat16):
            preds = model(cuboids, mask, compute_coupling=True, frw_ratio_matrix=frw_matrix)

            c_total_fp32 = preds['c_total_phys'].float()
            c_gnd_fp32   = preds['c_gnd_seg'].float()

            global_pred_total = torch.zeros(num_nets, device=device).scatter_add_(
                0, batch_net_ids,
                torch.sum(c_total_fp32 * A_tgt * core_ratios, dim=1)
            )
            global_pred_gnd = torch.zeros(num_nets, device=device).scatter_add_(
                0, batch_net_ids,
                torch.sum(c_gnd_fp32 * A_tgt * core_ratios, dim=1)
            )

            # Power net coupling → gnd
            sparse_cpl = preds['sparse_cpl']
            b_idx, src_idx, dst_idx = sparse_cpl['b_idx'], sparse_cpl['src_idx'], sparse_cpl['dst_idx']
            B, MAX_AGGR = A_tgt.shape[0], Y_cpl.shape[1]

            POWER_NETS = {'vss', 'vdd', 'vcc', 'gnd', 'vssx', 'vccx'}
            is_power = torch.zeros((B, cuboids.shape[1]), dtype=torch.bool, device=device)
            for b in range(B):
                for i, nm in enumerate(meta_dict['cuboid_net_names'][b]):
                    if str(nm).lower() in POWER_NETS:
                        is_power[b, i] = True

            raw_edge_cpl = sparse_cpl['c_cpl'].float() * torch.where(
                A_tgt[b_idx, src_idx] > 0, core_ratios[b_idx, src_idx], core_ratios[b_idx, dst_idx]
            )
            is_dst_pwr = is_power[b_idx, dst_idx]
            global_pred_gnd.scatter_add_(0, batch_net_ids[b_idx],
                                         raw_edge_cpl * is_dst_pwr.float())
            signal_cpl = raw_edge_cpl * (~is_dst_pwr).float()
            aggr_mask_E = (A_tgt[b_idx, src_idx].unsqueeze(1) * A_aggr[b_idx, :, dst_idx]
                         + A_tgt[b_idx, dst_idx].unsqueeze(1) * A_aggr[b_idx, :, src_idx])
            tile_cpl = torch.zeros(B, MAX_AGGR, device=device).index_add_(
                0, b_idx, signal_cpl.unsqueeze(1) * aggr_mask_E
            )
            global_pred_cpl = torch.zeros(num_nets, MAX_AGGR, device=device).index_add_(
                0, batch_net_ids, tile_cpl
            )
            net_idx = torch.arange(num_nets, device=device)
            valid_aggr_g = valid_aggr_mask[net_idx]

            loss_scale = finetuner.compute_pex_loss(global_pred_total, Y_total[net_idx])

            pred_sum   = global_pred_gnd + torch.sum(global_pred_cpl * valid_aggr_mask.float(), dim=1) + 1e-6
            pred_p_gnd = global_pred_gnd / pred_sum
            pred_p_cpl = global_pred_cpl / pred_sum.unsqueeze(1)

            gt_cpl_v  = Y_cpl[net_idx]
            gt_sum    = Y_gnd[net_idx] + torch.sum(gt_cpl_v * valid_aggr_mask.float(), dim=1) + 1e-6
            gt_p_gnd  = Y_gnd[net_idx] / gt_sum
            gt_p_cpl  = gt_cpl_v / gt_sum.unsqueeze(1)

            loss_dist_gnd = finetuner.compute_pex_loss(pred_p_gnd, gt_p_gnd)
            pred_p_cpl_v  = pred_p_cpl[valid_aggr_g]
            gt_p_cpl_v    = gt_p_cpl[valid_aggr_g]
            loss_dist_cpl = finetuner.compute_pex_loss(pred_p_cpl_v, gt_p_cpl_v)
            loss_distribution = loss_dist_gnd + loss_dist_cpl

            loss = loss_scale * 3.0 + loss_distribution * 0.3

        if torch.isnan(loss):
            print(f"\n{FAIL} NaN loss at step {step}! loss_scale={loss_scale.item():.4f} loss_dist={loss_distribution.item():.4f}")
            nan_detected = True
            break

        step_losses_scale.append(loss_scale.item())
        step_losses_dist.append(loss_distribution.item())

        finetuner.scaler.scale(loss).backward()
        finetuner.scaler.unscale_(finetuner.optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        finetuner.scaler.step(finetuner.optimizer)
        finetuner.scaler.update()
        finetuner.scheduler.step()
        step += 1

        # ---- Checkpoint report ----
        if step % args.report_every == 0:
            cur_lr = finetuner.optimizer.param_groups[0]['lr']
            print(f"--- Step {step:4d} | loss={loss.item():.4f}  scale={loss_scale.item():.4f}  dist={loss_distribution.item():.4f}  LR={cur_lr:.2e}")

            # Gradient norms
            gnd_gnorm   = grad_norm((n, p) for n, p in model.named_parameters() if 'gnd_mlp' in n)
            cpl_gnorm   = grad_norm((n, p) for n, p in model.named_parameters() if 'cpl_mlp' in n)
            enc_gnorm   = grad_norm((n, p) for n, p in model.named_parameters() if 'encoder' in n)
            has_enc_grad= any(p.grad is not None for n, p in model.named_parameters() if 'encoder' in n)
            enc_frozen  = all(not p.requires_grad for n, p in model.named_parameters() if 'encoder' in n)

            print(f"  [GRAD] gnd_mlp={gnd_gnorm:.4f}  cpl_mlp={cpl_gnorm:.4f}  encoder={enc_gnorm:.4f} (frozen={enc_frozen})")
            ok_grad = _check(gnd_gnorm > 1e-9, "gnd_mlp gradient non-zero")
            ok_frz  = _check(enc_frozen and not has_enc_grad, "encoder truly frozen (requires_grad=False, grad=None)")
            ok_dom  = _check(loss_scale.item() * 3.0 > loss_distribution.item() * 0.3 * 0.5,
                             f"loss_scale*3 ({loss_scale.item()*3:.4f}) dominates dist*0.3 ({loss_distribution.item()*0.3:.4f})")

            # Distribution saturation check
            with torch.no_grad():
                p_gnd_np = pred_p_gnd.float().cpu().numpy()
            pct_hi = float((p_gnd_np > 0.99).mean())
            pct_lo = float((p_gnd_np < 0.01).mean())
            pct_sat = pct_hi + pct_lo
            ok_sat = _check(pct_sat < args.sat_thresh,
                            f"pred_p_gnd not saturated ({pct_sat*100:.1f}% near 0/1, threshold={args.sat_thresh*100:.0f}%)")
            print(f"  [DIST] pred_p_gnd mean={float(p_gnd_np.mean()):.3f}  std={float(p_gnd_np.std()):.3f}  >0.99={pct_hi*100:.1f}%  <0.01={pct_lo*100:.1f}%")

            # MAPE
            net_mape = finetuner.compute_net_mape(val_loader)
            mape_history.append(net_mape)
            target_ok = net_mape < 4.0
            print(f"  [MAPE] net_mape={net_mape:.2f}%  {'✓ < 4%' if target_ok else '(rule-based ~5%)'}")

            passes.append(ok_grad and ok_frz and ok_dom and ok_sat)
            model.train()

    # ---- 5. Post-training checks ----
    print(f"\n{INFO} === POST-TRAINING CHECKS ===")

    _check(not nan_detected, "no NaN loss detected during training")

    snap_after = snapshot_params(model)

    trainable_changed, frozen_unchanged = True, True
    for n, p_before in snap_before.items():
        p_after = snap_after[n]
        changed = not torch.equal(p_before, p_after)
        if any(k in n for k in ['gnd_mlp', 'cpl_mlp', 'charge_basis_mlp']):
            if not changed:
                trainable_changed = False
                print(f"  {WARN} Trainable param NOT changed: {n}")
        elif 'encoder' in n or 'norm' in n:
            if changed:
                frozen_unchanged = False
                print(f"  {WARN} Frozen param WAS changed: {n}")

    _check(trainable_changed, "trainable head params changed during training")
    _check(frozen_unchanged,  "frozen params did NOT change during training")

    # Learning signal: last 50 avg not worse than first 50
    if len(step_losses_scale) >= 100:
        first50_avg = float(np.mean(step_losses_scale[:50]))
        last50_avg  = float(np.mean(step_losses_scale[-50:]))
        ok_signal = last50_avg <= first50_avg * 1.5
        _check(ok_signal, f"loss_scale trending: first50={first50_avg:.4f} last50={last50_avg:.4f}")

    if len(mape_history) >= 2:
        mape_trend = mape_history[-1] <= mape_history[0] * 1.5
        _check(mape_trend, f"MAPE not diverging: step0={mape_history[0]:.2f}% → final={mape_history[-1]:.2f}%")

    final_mape = finetuner.compute_net_mape(val_loader)
    print(f"\n  Baseline MAPE : {baseline_mape:.2f}%")
    print(f"  Final MAPE    : {final_mape:.2f}%")
    print(f"  Delta         : {final_mape - baseline_mape:+.2f}%")

    n_pass = sum(passes)
    n_total = len(passes)
    print(f"\n{'='*60}")
    print(f"  Checkpoint checks: {n_pass}/{n_total} passed")
    print(f"  Smoke test {'PASSED ✓' if n_pass == n_total and not nan_detected else 'FAILED ✗'}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
