# ==========================================================
# FILE: src/trainers/finetuner.py (Vectorized Full-Chip Assembly)
# ==========================================================
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from src.data.samplers import NetGroupedSampler
import os
import csv
import numpy as np
from src.utils.profiler import RuntimeProfiler
import configs.config as cfg

def probe_flux_router_anomalies(preds, step, threshold_ratio=5.0):
    """
    Validation 및 Training 시 CPL 폭발 현상과 GND Cap의 스케일 붕괴를 모니터링합니다.
    """
    print(f"\n[Probe Alert - Step {step}]")
    if 'sparse_cpl' in preds and len(preds['sparse_cpl']['c_cpl']) > 0:
        cpl_data = preds['sparse_cpl']
        w_cpl = cpl_data['w_cpl']
        c_cpl_final = cpl_data['c_cpl']
        
        # 🚨 [수정된 부분] cpl_logits 대신 직관적인 modifier와 residual을 로드합니다.
        cpl_modifier = cpl_data.get('cpl_modifier', torch.ones_like(w_cpl))
        cpl_residual = cpl_data.get('cpl_residual', torch.zeros_like(w_cpl))
        
        amplification_ratio = c_cpl_final / (w_cpl + 1e-9)
        exploded_edges = (amplification_ratio > threshold_ratio).sum().item()
        total_edges = len(c_cpl_final)
        
        print(f"  [가설 A: Exp 증폭] 물리 예측치 대비 {threshold_ratio}배 이상 증폭된 엣지 수: {exploded_edges} / {total_edges}")
        
        # 🚨 [추가된 모니터링] MLP가 실제로 얼마나 물리량을 왜곡(Modify)하고 있는지 추적
        if exploded_edges > 0:
            max_mod = cpl_modifier.max().item()
            mean_mod = cpl_modifier.mean().item()
            max_res = cpl_residual.max().item()
            print(f"    -> 📊 [MLP 개입 지표] 평균 Modifier: {mean_mod:.4f}x | 최대 Modifier: {max_mod:.4f}x")
            print(f"    -> 📊 [Fringing 잔차] 최대 Residual 추가량: {max_res:.6f} fF")

    c_gnd_seg = preds['c_gnd_seg']
    valid_mask = c_gnd_seg > 0
    if valid_mask.sum() > 0:
        valid_gnd = c_gnd_seg[valid_mask]
        cv = valid_gnd.std() / (valid_gnd.mean() + 1e-9)
        print(f"  [가설 C: Scale 텐션] Ground Cap 변동계수(CV): {cv.item():.2f}")
        if cv < 0.1:
            print("  ⚠️ 경고: Ground Cap CV가 너무 낮습니다. 상수 붕괴(Constant Collapse)를 의심하십시오.")


class NeuralFieldFinetuner(nn.Module):
    def __init__(self, model, device='cuda', lr=1e-4):
        super().__init__()
        self.model = model
        self.device = device

        effective_lr = max(lr, 1e-5)
        # 동적 불확실성 가중치(log_var_*)를 폐기하고 순수 모델 파라미터만 최적화
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=effective_lr, 
            weight_decay=1e-4
        )
        
        self.scaler = torch.amp.GradScaler()
        total_al_steps = max(1, int(getattr(cfg, 'AL_TRAIN_STEPS_PER_ITER', 10000)) * int(getattr(cfg, 'AL_FINE_ITERS', 1)))
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_al_steps, eta_min=1e-6)
        self.best_val_score = float('inf')

    def _compute_scale_invariant_mse(self, pred, target, eps=1e-4):
        """
        [Anti-Deception Loss]
        기존의 ((p-t)/t)^2 는 pred=0 일 때 Loss가 1.0으로 고정되는 맹점이 있었습니다.
        이를 (p-t)^2 / (t + eps) 형태로 변경하여, 예측을 포기(0)하면 Target의 크기만큼
        막대한 페널티(Linear growth)를 받도록 강제합니다.
        """
        return torch.mean(((pred - target) ** 2) / (target + eps))

    def _compute_smape(self, pred, target, eps=1e-6):
        if len(pred) == 0: return torch.tensor(0.0, device=pred.device)
        return torch.mean(torch.abs(pred - target) / ((torch.abs(pred) + torch.abs(target)) / 2 + eps)) * 100
    def _compute_rmsle_loss(self, pred, target, eps=1e-3):
        """
        [Forward-Looking] Log-Ratio 기반 손실 함수.
        ((pred-target)/target)^2 의 기울기 폭발을 막고, 6 order of magnitude (fF ~ aF)를
        가지는 정전용량을 안정적으로 매핑합니다.
        """
        log_pred = torch.log1p(pred / eps)
        log_target = torch.log1p(target / eps)
        return torch.mean((log_pred - log_target) ** 2)

    def compute_netlevel_loss(self, pred_cap, target_cap):
        """
        Net-level total-cap loss: log-space smooth-L1 + SymMAPE + zero penalty.
        smooth-L1 in log space grows linearly for large errors (max ~7 per sample)
        vs RMSLE which grows quadratically (max ~49). Safe for single-net batches.
        """
        pos_mask = (target_cap >= 0.005)
        zero_mask = ~pos_mask

        # Log-space smooth-L1: linear growth for large errors — max ≈ log1p(1000) ≈ 7
        log_loss = (
            F.smooth_l1_loss(
                torch.log1p(pred_cap[pos_mask]),
                torch.log1p(target_cap[pos_mask]),
                beta=0.1
            ) if pos_mask.any() else torch.tensor(0.0, device=pred_cap.device)
        )

        # SymMAPE: bounded [0, 2]
        smape = (
            (2.0 * torch.abs(pred_cap[pos_mask] - target_cap[pos_mask]) /
             (torch.abs(pred_cap[pos_mask]) + target_cap[pos_mask].clamp(min=0.005))).mean()
            if pos_mask.any() else torch.tensor(0.0, device=pred_cap.device)
        )

        # True MAPE: direct alignment with the evaluation metric.
        # SymMAPE saturates at ~0.25 when true MAPE is 40%, providing no gradient signal.
        mape_term = (
            (torch.abs(pred_cap[pos_mask] - target_cap[pos_mask]) /
             target_cap[pos_mask].clamp(min=0.005)).mean()
            if pos_mask.any() else torch.tensor(0.0, device=pred_cap.device)
        )

        # Zero penalty
        zero_pen = (
            torch.log1p(pred_cap[zero_mask]).mean()
            if zero_mask.any() else torch.tensor(0.0, device=pred_cap.device)
        )

        return log_loss * 1.5 + smape * 0.3 + mape_term * 0.5 + zero_pen * 0.1
    
    def compute_pex_loss(self,pred_cap, target_cap):
        """
        [CRITICAL FIX] Sparsity-Aware Masked Loss
        - 정답(Golden)이 존재하는 유효 넷(Positive)과 버려진 넷(Zero)의 Loss를 분리합니다.
        """
        # 0.005 fF 이상인 유효한 정답 엣지만 마스킹 (EDA 툴의 일반적인 Cut-off 기준)
        pos_mask = (target_cap >= 0.005).float()
        zero_mask = (target_cap < 0.005).float()
        
        num_pos = torch.clamp(pos_mask.sum(), min=1.0)
        num_zero = torch.clamp(zero_mask.sum(), min=1.0)
        
        # ------------------------------------------------------------------
        # 1. 유효 엣지 (Positive Edges): 가혹하고 정밀하게 학습 (L1 + MAPE + Log)
        # ------------------------------------------------------------------
        loss_pos_l1 = torch.sum(F.l1_loss(pred_cap, target_cap, reduction='none') * pos_mask) / num_pos
        loss_pos_mape = torch.sum((torch.abs(pred_cap - target_cap) / (target_cap + 1e-6)) * pos_mask) / num_pos
        loss_pos_log = torch.sum(F.mse_loss(torch.log1p(pred_cap), torch.log1p(target_cap), reduction='none') * pos_mask) / num_pos
        
        loss_positive = loss_pos_l1 + (loss_pos_mape * 5.0) + (loss_pos_log * 2.0)
        
        # ------------------------------------------------------------------
        # 2. 제로 엣지 (Zero Edges): 아주 부드럽게 눌러주기만 함 (Weak L1)
        # NN이 0을 예측하지 않았다고 해서 가혹하게 패널티를 주지 않음
        # ------------------------------------------------------------------
        loss_zero = torch.sum(F.l1_loss(pred_cap, target_cap, reduction='none') * zero_mask) / num_zero
        
        # Positive에 압도적인 가중치(10배)를 주어 신경망의 관심을 유도
        total_loss = (loss_positive * 10.0) + loss_zero
        
        return total_loss

    @torch.no_grad()
    def evaluate(self, val_loader):
        self.model.eval()        
        tot_smape_sum, gnd_smape_sum, cpl_smape_sum = 0.0, 0.0, 0.0
        batches = 0
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for batch in val_loader:
                if batch is None: continue
                # 평가 단계에서는 메타데이터(meta_dict)가 반드시 필요합니다 (Power Net 판별용)
                cuboids, mask, labels_dict, meta_dict = batch 
                cuboids, mask = cuboids.to(self.device), mask.to(self.device)
                if cuboids.shape[-1] > 9: cuboids = cuboids[..., :9]
                
                A_tgt = labels_dict['A_tgt'].to(self.device)
                Y_total = labels_dict['Y_total'].to(self.device)
                Y_gnd = labels_dict['Y_gnd'].to(self.device)
                A_aggr = labels_dict['A_aggr'].to(self.device)
                Y_cpl = labels_dict['Y_cpl'].to(self.device)
                valid_aggr_mask = labels_dict['valid_aggr_mask'].to(self.device)
                core_ratios = labels_dict['core_ratios'].to(self.device)
                batch_net_ids = labels_dict['batch_net_ids'].to(self.device)
                num_nets = labels_dict['num_unique_nets']
                frw_matrix = labels_dict.get('frw_ratio_matrix', None)
                if frw_matrix is not None: frw_matrix = frw_matrix.to(self.device)

                preds = self.model(cuboids, mask, compute_coupling=True, frw_ratio_matrix=frw_matrix)
                
                # 1. 텐서에서 물리적 절대 정전용량(Physical Farad) 복원
                c_total_phys_fp32 = preds['c_total_phys'].float()
                c_gnd_seg_fp32 = preds['c_gnd_seg'].float()
                
                global_pred_total = torch.zeros(num_nets, dtype=torch.float32, device=self.device).scatter_add_(0, batch_net_ids, torch.sum(c_total_phys_fp32 * A_tgt * core_ratios, dim=1))
                global_pred_gnd = torch.zeros(num_nets, dtype=torch.float32, device=self.device).scatter_add_(0, batch_net_ids, torch.sum(c_gnd_seg_fp32 * A_tgt * core_ratios, dim=1))
                
                # 2. [StarRC 미러링] 파워 넷으로 가는 CPL 플럭스를 GND로 강제 Lumping
                sparse_cpl = preds['sparse_cpl']
                b_idx, src_idx, dst_idx = sparse_cpl['b_idx'], sparse_cpl['src_idx'], sparse_cpl['dst_idx']
                B, MAX_AGGR = A_tgt.shape[0], Y_cpl.shape[1]
                
                POWER_NETS = {'vss', 'vdd', 'vcc', 'gnd', 'vssx', 'vccx'}
                is_power_mask = torch.zeros((B, cuboids.shape[1]), dtype=torch.bool, device=self.device)
                for b in range(B):
                    names = meta_dict['cuboid_net_names'][b]
                    for i, name in enumerate(names):
                        if str(name).lower() in POWER_NETS:
                            is_power_mask[b, i] = True
                            
                is_dst_power = is_power_mask[b_idx, dst_idx]
                raw_edge_cpl = sparse_cpl['c_cpl'].float() * torch.where(A_tgt[b_idx, src_idx] > 0, core_ratios[b_idx, src_idx], core_ratios[b_idx, dst_idx])
                
                power_cpl_flux = raw_edge_cpl * is_dst_power.float()
                signal_cpl_flux = raw_edge_cpl * (~is_dst_power).float()
                
                global_pred_gnd.scatter_add_(0, batch_net_ids[b_idx], power_cpl_flux)
                
                aggr_mask_E = A_tgt[b_idx, src_idx].unsqueeze(1) * A_aggr[b_idx, :, dst_idx] + A_tgt[b_idx, dst_idx].unsqueeze(1) * A_aggr[b_idx, :, src_idx]
                tile_cpl = torch.zeros(B, MAX_AGGR, dtype=torch.float32, device=self.device).index_add_(0, b_idx, signal_cpl_flux.unsqueeze(1) * aggr_mask_E)
                global_pred_cpl = torch.zeros(num_nets, MAX_AGGR, dtype=torch.float32, device=self.device).index_add_(0, batch_net_ids, tile_cpl)

                # =================================================================
                # [SIGN-OFF EVALUATION] 비율이 아닌 순수 물리 단위(Farad)로 SMAPE 검증
                # =================================================================
                net_indices = torch.arange(num_nets, dtype=torch.long, device=self.device)
                
                # Macro Scale 평가
                s_tot = self.compute_pex_loss(global_pred_total, Y_total[net_indices])
                s_gnd = self.compute_pex_loss(global_pred_gnd, Y_gnd[net_indices])
                
                # Net-Level Total CPL 평가 (KCL 관점에서의 CPL 덩어리 합산 비교)
                pred_cpl_sum = torch.sum(global_pred_cpl * valid_aggr_mask.float(), dim=1)
                gt_cpl_sum = torch.sum(Y_cpl[net_indices] * valid_aggr_mask.float(), dim=1)
                
                # 골든 기준 1aF(0.001fF) 이상의 유의미한 CPL을 가진 Net들만 평가 (노이즈 방어)
                valid_net_cpl_mask = gt_cpl_sum > 0.001
                if valid_net_cpl_mask.sum() > 0:
                    s_cpl = self.compute_pex_loss(pred_cpl_sum[valid_net_cpl_mask], gt_cpl_sum[valid_net_cpl_mask])
                else:
                    s_cpl = torch.tensor(0.0, device=self.device)

                tot_smape_sum += s_tot.item() if not torch.isnan(s_tot) else 1000.0
                gnd_smape_sum += s_gnd.item() if not torch.isnan(s_gnd) else 1000.0
                cpl_smape_sum += s_cpl.item() if not torch.isnan(s_cpl) else 1000.0
                batches += 1
                
        self.model.train()
        avg_tot = tot_smape_sum / max(1, batches)
        avg_gnd = gnd_smape_sum / max(1, batches)
        avg_cpl = cpl_smape_sum / max(1, batches)

        # Sign-off 지표로서 가장 중요한 Total Cap에 가장 높은 가중치(40%) 부여
        composite_score = (0.8 * avg_tot) + (0.1 * avg_gnd) + (0.1 * avg_cpl)
        return composite_score, avg_tot, avg_gnd, avg_cpl

    @torch.no_grad()
    def compute_net_mape(self, val_loader):
        """Compute true net-level MAPE for total_cap. Rule-based baseline target: <4%."""
        self.model.eval()
        abs_errors, targets = [], []

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for batch in val_loader:
                if batch is None: continue
                cuboids, mask, labels_dict, meta_dict = batch
                cuboids, mask = cuboids.to(self.device), mask.to(self.device)
                if cuboids.shape[-1] > 9: cuboids = cuboids[..., :9]

                A_tgt = labels_dict['A_tgt'].to(self.device)
                Y_total = labels_dict['Y_total'].to(self.device)
                core_ratios = labels_dict['core_ratios'].to(self.device)
                batch_net_ids = labels_dict['batch_net_ids'].to(self.device)
                num_nets = labels_dict['num_unique_nets']
                frw_matrix = labels_dict.get('frw_ratio_matrix', None)
                if frw_matrix is not None: frw_matrix = frw_matrix.to(self.device)

                preds = self.model(cuboids, mask, compute_coupling=True, frw_ratio_matrix=frw_matrix)
                c_total_phys_fp32 = preds['c_total_phys'].float()

                pred_total = torch.zeros(num_nets, dtype=torch.float32, device=self.device).scatter_add_(
                    0, batch_net_ids,
                    torch.sum(c_total_phys_fp32 * A_tgt * core_ratios, dim=1)
                )
                net_indices = torch.arange(num_nets, dtype=torch.long, device=self.device)
                gt = Y_total[net_indices]
                pos = gt >= 0.005
                if pos.sum() > 0:
                    abs_errors.append(torch.abs(pred_total[pos] - gt[pos]).cpu())
                    targets.append(gt[pos].cpu())

        self.model.train()
        if not targets:
            return float('nan')
        all_err = torch.cat(abs_errors)
        all_tgt = torch.cat(targets)
        mape = (all_err / (all_tgt + 1e-6)).mean().item() * 100.0
        return mape

    def train_steps(self, dataloader, val_loader=None, max_steps=500, save_dir=None, report_every=1000):
        self.model.train()
        step = 0
        REPORT_PER_STEP = report_every
        profiler = RuntimeProfiler(save_dir / "runtime_analysis.csv" if save_dir is not None else "runtime_analysis.csv")
        
        if save_dir is not None:
            best_ckpt = Path(save_dir) / "best_model.pth"
            if best_ckpt.exists():
                state_dict = torch.load(best_ckpt, map_location=self.device, weights_only=True)
                # Shape-filtered load: `strict=False` tolerates missing/extra
                # keys but *not* shape mismatches. When flux_head architecture
                # changes (e.g. cpl_mlp widened, layer_scale_phys_* resized),
                # keep weights that still fit and let the rest init fresh.
                current = self.model.state_dict()
                filtered = {k: v for k, v in state_dict.items()
                            if k in current and v.shape == current[k].shape}
                dropped = [k for k in state_dict if k not in filtered]
                self.model.load_state_dict(filtered, strict=False)
                print(f"Loaded best model from {best_ckpt} "
                      f"({len(filtered)} tensors kept, {len(dropped)} shape-filtered)")
                if val_loader is not None:
                    self.best_val_score = self.compute_net_mape(val_loader)
                    print(f"Current best checkpoint net MAPE: {self.best_val_score:.2f}%")

        while step < max_steps:
            profiler.start("Data_Load")
            for batch in dataloader:
                profiler.stop("Data_Load")
                if batch is None: continue

                profiler.start("H2D_Transfer")
                cuboids, mask, labels_dict, meta_dict = batch
                cuboids = cuboids.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True)
                if cuboids.shape[-1] > 9: cuboids = cuboids[..., :9]
                
                A_tgt = labels_dict['A_tgt'].to(self.device, non_blocking=True)
                Y_total = labels_dict['Y_total'].to(self.device, non_blocking=True)
                Y_gnd = labels_dict['Y_gnd'].to(self.device, non_blocking=True)
                A_aggr = labels_dict['A_aggr'].to(self.device, non_blocking=True)
                Y_cpl = labels_dict['Y_cpl'].to(self.device, non_blocking=True)
                valid_aggr_mask = labels_dict['valid_aggr_mask'].to(self.device, non_blocking=True)
                core_ratios = labels_dict['core_ratios'].to(self.device, non_blocking=True)
                batch_net_ids = labels_dict['batch_net_ids'].to(self.device, non_blocking=True)
                num_nets = labels_dict['num_unique_nets']
                frw_matrix = labels_dict.get('frw_ratio_matrix', None)
                if frw_matrix is not None: frw_matrix = frw_matrix.to(self.device)
                profiler.stop("H2D_Transfer")
                
                self.optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    profiler.start("Model_Forward")
                    preds = self.model(cuboids, mask, compute_coupling=True, frw_ratio_matrix=frw_matrix)
                    profiler.stop("Model_Forward")
                    
                    profiler.start("Loss_Assembly")
                    c_total_phys_fp32 = preds['c_total_phys'].float()
                    c_gnd_seg_fp32 = preds['c_gnd_seg'].float()
                    
                    global_pred_total = torch.zeros(num_nets, dtype=torch.float32, device=self.device).scatter_add_(0, batch_net_ids, torch.sum(c_total_phys_fp32 * A_tgt * core_ratios, dim=1))
                    global_pred_gnd = torch.zeros(num_nets, dtype=torch.float32, device=self.device).scatter_add_(0, batch_net_ids, torch.sum(c_gnd_seg_fp32 * A_tgt * core_ratios, dim=1))
                    
                    sparse_cpl = preds['sparse_cpl']
                    b_idx, src_idx, dst_idx = sparse_cpl['b_idx'], sparse_cpl['src_idx'], sparse_cpl['dst_idx']
                    B, MAX_AGGR = A_tgt.shape[0], Y_cpl.shape[1]
                    
                    raw_edge_cpl = sparse_cpl['c_cpl'].float() * torch.where(A_tgt[b_idx, src_idx] > 0, core_ratios[b_idx, src_idx], core_ratios[b_idx, dst_idx])

                    POWER_NETS = {'vss', 'vdd', 'vcc', 'gnd', 'vssx', 'vccx'}
                    
                    # 1. GPU 상에서 현재 배치의 어떤 Cuboid가 Power Net인지 마스킹
                    is_power_mask = torch.zeros((B, cuboids.shape[1]), dtype=torch.bool, device=self.device)
                    for b in range(B):
                        names = meta_dict['cuboid_net_names'][b]
                        for i, name in enumerate(names):
                            if str(name).lower() in POWER_NETS:
                                is_power_mask[b, i] = True
                                
                    # 2. Edge의 목적지(dst)가 파워 넷인지 판별
                    is_dst_power = is_power_mask[b_idx, dst_idx]
                    
                    # 3. 플럭스 분리: Power로 가는 플럭스 vs 일반 Signal로 가는 플럭스
                    power_cpl_flux = raw_edge_cpl * is_dst_power.float()
                    signal_cpl_flux = raw_edge_cpl * (~is_dst_power).float()
                    
                    # 4. Power Flux를 Ground Cap에 강제 합산 (StarRC Lumping 완벽 구현!)
                    global_pred_gnd.scatter_add_(0, batch_net_ids[b_idx], power_cpl_flux)

                    aggr_mask_E = A_tgt[b_idx, src_idx].unsqueeze(1) * A_aggr[b_idx, :, dst_idx] + A_tgt[b_idx, dst_idx].unsqueeze(1) * A_aggr[b_idx, :, src_idx]
                    tile_cpl = torch.zeros(B, MAX_AGGR, dtype=torch.float32, device=self.device).index_add_(0, b_idx, signal_cpl_flux.unsqueeze(1) * aggr_mask_E)
                    global_pred_cpl = torch.zeros(num_nets, MAX_AGGR, dtype=torch.float32, device=self.device).index_add_(0, batch_net_ids, tile_cpl)
                    net_indices = torch.arange(num_nets, dtype=torch.long, device=self.device)
                    valid_aggr_global = valid_aggr_mask[net_indices]
                    
                    # RMSLE-based net-level loss: bounded regardless of cap magnitude
                    loss_scale = self.compute_netlevel_loss(global_pred_total, Y_total[net_indices])

                    # -----------------------------------------------------------------
                    # 2. MICRO TOPOLOGY DISTRIBUTION: GND와 CPL을 [0, 1] 확률 분포로 변환
                    # -----------------------------------------------------------------
                    gt_cpl_v_all = Y_cpl[net_indices]
                    pred_cpl_sum = torch.sum(global_pred_cpl * valid_aggr_global.float(), dim=1)
                    gt_cpl_sum = torch.sum(gt_cpl_v_all * valid_aggr_global.float(), dim=1)
                    valid_cpl_nets = gt_cpl_sum >= 0.005
                    # Use log-space smooth-L1 for coupling total: bounded + scale-invariant.
                    # compute_netlevel_loss has unbounded weighted_abs that spikes when
                    # valid_cpl_nets is small (e.g. sqrt(500fF) * large_err / 1 net → huge).
                    loss_cpl_total = (
                        F.smooth_l1_loss(
                            torch.log1p(pred_cpl_sum[valid_cpl_nets]),
                            torch.log1p(gt_cpl_sum[valid_cpl_nets]),
                            beta=0.1
                        )
                        if valid_cpl_nets.any() else torch.tensor(0.0, device=self.device)
                    )

                    # [모델의 예측 분포 (Predicted PMF)]
                    pred_sum = global_pred_gnd + pred_cpl_sum
                    pred_sum_safe = pred_sum + 1e-6
                    pred_p_gnd = global_pred_gnd / pred_sum_safe
                    pred_p_cpl = global_pred_cpl / pred_sum_safe.unsqueeze(1)

                    # [Golden SPEF의 정답 분포 (Golden PMF)]
                    # KCL이 깨진 Y_total이 아니라, '실제 할당된 값들의 합'으로 정규화하여 의도만 추출
                    gt_sum = Y_gnd[net_indices] + gt_cpl_sum
                    gt_sum_safe = gt_sum + 1e-6
                    gt_p_gnd = Y_gnd[net_indices] / gt_sum_safe
                    gt_p_cpl = gt_cpl_v_all / gt_sum_safe.unsqueeze(1)

                    # -----------------------------------------------------------------
                    # 3. DISTRIBUTION LOSS: Smooth L1 on proportions [0,1]
                    # compute_pex_loss의 MAPE term이 near-zero 비율(gt_p_gnd<<1)에서
                    # 폭발하는 문제를 막기 위해 Smooth L1만 사용
                    # -----------------------------------------------------------------
                    loss_dist_gnd = F.smooth_l1_loss(pred_p_gnd, gt_p_gnd, beta=0.05)

                    # CPL은 유효한 어그레서(valid_aggr)에 대해서만 비율 검사 수행
                    pred_p_cpl_v = pred_p_cpl[valid_aggr_global]
                    gt_p_cpl_v = gt_p_cpl[valid_aggr_global]
                    loss_dist_cpl = F.smooth_l1_loss(pred_p_cpl_v, gt_p_cpl_v, beta=0.05) if pred_p_cpl_v.numel() > 0 else torch.tensor(0.0, device=self.device)
                    
                    # 4. FINAL COMPOSITE LOSS
                    loss_distribution = loss_dist_gnd + loss_dist_cpl

                    # Direct ground cap loss: magnitude supervision on c_gnd.
                    # Curriculum warm-up (step 500→2000) prevents early large gradients
                    # from disrupting coupling convergence that was already working.
                    GND_WARMUP_START, GND_WARMUP_END, GND_WEIGHT_MAX = 500, 2000, 2.0
                    if step < GND_WARMUP_START:
                        w_gnd = 0.0
                    elif step < GND_WARMUP_END:
                        w_gnd = GND_WEIGHT_MAX * (step - GND_WARMUP_START) / (GND_WARMUP_END - GND_WARMUP_START)
                    else:
                        w_gnd = GND_WEIGHT_MAX
                    loss_gnd_direct = self.compute_netlevel_loss(global_pred_gnd, Y_gnd[net_indices])

                    loss = loss_scale * 3.0 + loss_cpl_total * 1.0 + loss_distribution * 0.3 + loss_gnd_direct * w_gnd
                    profiler.stop("Loss_Assembly")

                if torch.isnan(loss):
                    raise ValueError(f"NaN Loss Detected at step {step}")
                
                profiler.start("Backward_Update")
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                step += 1
                profiler.stop("Backward_Update")
                
                if step % REPORT_PER_STEP == 0:
                    profiler.save_and_reset("FineTuning", f"Step_{step}")
                    print(f">>> [FineTuner] Step {step:04d}: Train Loss = {loss.item():.4f} loss_scale: {loss_scale.item():.4f} loss_cpl_total: {loss_cpl_total.item():.4f} loss_distribution: {loss_distribution.item():.4f} loss_gnd: {loss_gnd_direct.item():.4f} (w={w_gnd:.2f})")
                    if val_loader is not None:
                        val_score, s_tot, s_gnd, s_cpl = self.evaluate(val_loader)
                        net_mape = self.compute_net_mape(val_loader)
                        is_best = "🌟 BEST!" if np.isfinite(net_mape) and net_mape < self.best_val_score else ""
                        target_marker = " ✓ BEAT RULE-BASED" if net_mape < 4.0 else " (rule-based ~5%)"
                        print(f"      - Validation SMAPE [%] -> Tot: {s_tot:.2f} | GND: {s_gnd:.2f} | CPL: {s_cpl:.2f}")
                        print(f"      - Net-level MAPE    : {net_mape:.2f}%{target_marker}")
                        print(f"      - Composite Score: {val_score:.4f} {is_best}")
                        
                        if np.isfinite(net_mape) and net_mape < self.best_val_score and save_dir is not None:
                            self.best_val_score = net_mape
                            torch.save(self.model.state_dict(), Path(save_dir) / "best_model.pth")
                    
                    probe_flux_router_anomalies(preds, step)
                profiler.start("Data_Load")
                if step >= max_steps: break

        return loss.item()
