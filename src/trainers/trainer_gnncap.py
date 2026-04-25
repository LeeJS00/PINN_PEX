# src/trainers/trainer_gnncap.py
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import gc

class GNNCapAblationTrainer(nn.Module):
    def __init__(self, model, mode='A', device='cuda', lr=1e-4):
        super().__init__()
        self.model = model
        self.mode = mode  # 'A', 'B', 'C'
        self.device = device

        # Mode C (Our Full Pipeline) 일 때만 Log-Huber의 Homoscedastic 가중치 사용
        self.log_var_tot = nn.Parameter(torch.zeros(1, device=device))
        self.log_var_gnd = nn.Parameter(torch.zeros(1, device=device))
        self.log_var_cpl = nn.Parameter(torch.zeros(1, device=device))

        self.optimizer = optim.AdamW([
            {'params': filter(lambda p: p.requires_grad, self.model.parameters())},
            {'params': [self.log_var_tot, self.log_var_gnd, self.log_var_cpl], 'lr': 1e-3}
        ], lr=lr, weight_decay=1e-4)
        
        self.scaler = torch.amp.GradScaler()
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10000, eta_min=1e-6)
        
        # [VRAM 방어] Gradient Accumulation Step 설정
        self.accum_steps = 5

    def _compute_loss(self, pred, target):
        if self.mode == 'A':
            # [FIX] 타겟이 0에 가까운 먼지 캡일 때 분모가 0이 되어 폭주하는 현상 방어 (최소 0.01 fF 기준)
            safe_target = torch.clamp(target, min=1e-2)
            
            # [FIX] 극초반 Softplus 예측값으로 인한 무한대 에러 폭발을 막기 위해 최대 Loss 상한선(Max 20.0) 적용
            raw_mare = torch.abs((pred - target) / safe_target)
            return torch.mean(torch.clamp(raw_mare, max=20.0))
        else:
            # [Mode B, C] Our Model: Log-Huber Loss
            log_pred = torch.log1p(torch.clamp(pred, min=0.0))
            log_target = torch.log1p(torch.clamp(target, min=0.0))
            weight = torch.clamp(target / (target.mean() + 1e-6), min=0.5, max=3.0)
            return torch.mean(F.smooth_l1_loss(log_pred, log_target, reduction='none') * weight)

    def train_steps(self, dataloader, max_steps=10000, save_dir=None):
        self.model.train()
        total_loss_sum = 0
        step = 0
        self.optimizer.zero_grad(set_to_none=True)

        print(f"\n🚀 [GNN-Cap Trainer] Starting Mode {self.mode}")
        print(f"  - Assembly (Core Ratios): {'ON' if self.mode in ['B', 'C'] else 'OFF (Smearing Artifacts)'}")
        print(f"  - Faraday Dumping       : {'ON' if self.mode == 'C' else 'OFF (Leakage Ignored)'}")
        print(f"  - Loss Function         : {'Log-Huber' if self.mode in ['B', 'C'] else 'MARE'}")

        while step < max_steps:
            for batch_idx, batch in enumerate(dataloader):
                # [수정] 정확한 스텝에 도달하면 즉시 이중 루프 탈출
                if step >= max_steps: break
                
                if batch is None: continue

                cuboids, mask, labels_dict, _ = batch
                cuboids = cuboids.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True)
                
                A_tgt = labels_dict['A_tgt'].to(self.device, non_blocking=True)
                Y_total = labels_dict['Y_total'].to(self.device, non_blocking=True)
                Y_gnd = labels_dict['Y_gnd'].to(self.device, non_blocking=True)
                A_aggr = labels_dict['A_aggr'].to(self.device, non_blocking=True)
                Y_cpl = labels_dict['Y_cpl'].to(self.device, non_blocking=True)
                valid_aggr_mask = labels_dict['valid_aggr_mask'].to(self.device, non_blocking=True)
                batch_net_ids = labels_dict['batch_net_ids'].to(self.device, non_blocking=True)
                num_nets = labels_dict['num_unique_nets']

                if self.mode in ['B', 'C']:
                    core_ratios = labels_dict['core_ratios'].to(self.device, non_blocking=True)
                else:
                    core_ratios = torch.ones_like(labels_dict['core_ratios']).to(self.device, non_blocking=True)

                with torch.amp.autocast('cuda', dtype=torch.float16):
                    preds = self.model(cuboids, mask, compute_coupling=True)
                    
                    tile_contrib_total = torch.sum(preds['c_total_phys'] * core_ratios, dim=1) 
                    tile_contrib_gnd = torch.sum(preds['c_gnd_seg'] * core_ratios, dim=1)   
                    
                    global_pred_total = torch.zeros(num_nets, device=self.device).scatter_add_(0, batch_net_ids, tile_contrib_total)
                    global_pred_gnd = torch.zeros(num_nets, device=self.device).scatter_add_(0, batch_net_ids, tile_contrib_gnd)
                    
                    sparse_cpl = preds['sparse_cpl']
                    b_idx, src_idx, dst_idx = sparse_cpl['b_idx'], sparse_cpl['src_idx'], sparse_cpl['dst_idx']
                    
                    edge_cpl_filtered = sparse_cpl['c_cpl'] * core_ratios[b_idx, src_idx]
                    tile_cpl_generated_total = torch.zeros(len(batch_net_ids), device=self.device).scatter_add_(0, b_idx, edge_cpl_filtered)
                    
                    aggr_mask_E = A_aggr[b_idx, :, dst_idx] 
                    edge_cpl_expanded = edge_cpl_filtered.unsqueeze(1) * aggr_mask_E
                    tile_cpl = torch.zeros_like(Y_cpl).index_add_(0, b_idx, edge_cpl_expanded)
                    
                    global_pred_cpl = torch.zeros(num_nets, Y_cpl.shape[1], device=self.device)
                    global_pred_cpl.index_add_(0, batch_net_ids, tile_cpl)
                    
                    del edge_cpl_expanded, aggr_mask_E
                    
                    if self.mode == 'C':
                        tile_cpl_valid_total = tile_cpl.sum(dim=1)
                        tile_shadow_leakage = torch.clamp(tile_cpl_generated_total - tile_cpl_valid_total, min=0.0)
                        global_shadow_leakage = torch.zeros(num_nets, device=self.device).scatter_add_(0, batch_net_ids, tile_shadow_leakage)
                        global_pred_gnd_corrected = global_pred_gnd + global_shadow_leakage
                    else:
                        global_pred_gnd_corrected = global_pred_gnd

                    net_indices = torch.zeros(num_nets, dtype=torch.long, device=self.device)
                    net_indices[batch_net_ids] = torch.arange(len(batch_net_ids), device=self.device)
                    
                    global_gt_total = Y_total[net_indices]
                    global_gt_gnd = Y_gnd[net_indices]
                    global_gt_cpl = Y_cpl[net_indices]
                    valid_aggr_global = valid_aggr_mask[net_indices]

                    loss_total = self._compute_loss(global_pred_total, global_gt_total)
                    loss_gnd = self._compute_loss(global_pred_gnd_corrected, global_gt_gnd)
                    
                    pred_cpl_v = global_pred_cpl[valid_aggr_global]
                    gt_cpl_v = global_gt_cpl[valid_aggr_global]
                    loss_cpl = self._compute_loss(pred_cpl_v, gt_cpl_v) if len(gt_cpl_v) > 0 else torch.tensor(0.0, device=self.device)
                    
                    if self.mode == 'C':
                        loss = (loss_total * torch.exp(-self.log_var_tot) + self.log_var_tot) + \
                            (loss_gnd * torch.exp(-self.log_var_gnd) + self.log_var_gnd) + \
                            (loss_cpl * torch.exp(-self.log_var_cpl) + self.log_var_cpl)
                    else:
                        loss = loss_total + loss_gnd + loss_cpl
                    
                    loss = loss / self.accum_steps # Gradient Accumulation

                if torch.isnan(loss):
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                # [복구 1] 가장 중요한 역전파(Backward) 코드 복원
                self.scaler.scale(loss).backward()
                total_loss_sum += (loss.item() * self.accum_steps)

                # [복구 2] Accumulation Step에만 가중치를 업데이트 하도록 들여쓰기 원복
                if (batch_idx + 1) % self.accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    
                    scale_before = self.scaler.get_scale()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    scale_after = self.scaler.get_scale()
                    
                    if scale_before <= scale_after:
                        self.scheduler.step()
                        
                    self.optimizer.zero_grad(set_to_none=True)

                step += 1

                # [수정 3] 로그 출력 및 가비지 컬렉터 주기 조절
                if step % 100 == 0:
                    torch.save(self.model.state_dict(), Path(save_dir) / f"gnncap_mode_{self.mode}_step_{step}.pth")
                    print(f"   [Step {step:05d}/{max_steps}] Loss: {loss.item()*self.accum_steps:.4f} (Mode {self.mode})")
                
                # 가비지 컬렉터는 매 배치가 아니라 일정 주기마다 실행해야 속도 저하가 없습니다.
                if step % 500 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

        if save_dir is not None:
            torch.save(self.model.state_dict(), Path(save_dir) / f"gnncap_mode_{self.mode}_final.pth")
            
        return total_loss_sum / max(1, step)