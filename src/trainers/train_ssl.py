# src/trainers/train_ssl.py
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import gc
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))  # 프로젝트 루트 경로 추가
from src.data.datasets import NeuralFieldSSLDataset, robust_collate
from src.models.neural_field import DeepPEX_Model
from src.utils.gpu_sampler import PhysicsSampler
from src.physics.materials import BEOLMaterialStack
from src.preprocessing.layer_parser import LayerInfoParser
import configs.config as cfg

def train_ssl():
    # [CRITICAL SPEEDUP] 행렬 곱셈 하드웨어 가속 켬
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    RUN_NAME = cfg.RUN_NAME
    DEVICE = f"cuda:{cfg.GPU_ID}" if torch.cuda.is_available() else "cpu"
    SAVE_DIR = Path(cfg.OUTPUT_DIR) / "checkpoints" / RUN_NAME
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    layer_map = LayerInfoParser(cfg.LAYERS_INFO_PATH).parse()
    mat_stack = BEOLMaterialStack(layer_map)
    sampler = PhysicsSampler(mat_stack, device=DEVICE)

    manifest_path = Path(cfg.PROCESSED_DIR) / "dataset_manifest.csv"
    train_df = pd.read_csv(manifest_path)
    
    # only for train_defs
    train_def_names = [p.stem for p in cfg.TRAIN_DEFS]
    train_df = train_df[train_df['design_name'].isin(train_def_names)]
    dataset = NeuralFieldSSLDataset(cfg.PROCESSED_DIR, train_df)
    
    # 메모리 파편화 방지(drop_last) 및 좀비 워커 방지(persistent_workers=False)
    # loader = DataLoader(dataset, batch_size=cfg.SSL_BATCH_SIZE, shuffle=True, drop_last=True,
    #                     collate_fn=robust_collate, num_workers=cfg.SSL_NUM_WORKERS, prefetch_factor=4, persistent_workers=False, pin_memory=True)
    EFFECTIVE_BATCH = cfg.SSL_BATCH_SIZE # 1024
    MINI_BATCH = min(1024, EFFECTIVE_BATCH)
    ACCUM_STEPS = max(1, EFFECTIVE_BATCH // MINI_BATCH)
   
    loader = DataLoader(dataset, batch_size=MINI_BATCH, shuffle=True, drop_last=True,
                        collate_fn=robust_collate, num_workers=cfg.SSL_NUM_WORKERS, pin_memory=True)
    model = DeepPEX_Model(cfg).to(DEVICE)
    # SSL은 동적 샘플링 때문에 torch.compile을 끕니다. (Eager 모드가 훨씬 빠름)
        
    optimizer = optim.AdamW(model.parameters(), lr=cfg.SSL_LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.SSL_EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler()
    
    start_epoch = 0
    if hasattr(cfg, 'SSL_RESUME_CHECKPOINT') and cfg.SSL_RESUME_CHECKPOINT:
        if Path(cfg.SSL_RESUME_CHECKPOINT).exists():
            print(f">>> Resuming SSL Training from {cfg.SSL_RESUME_CHECKPOINT}")
            checkpoint = torch.load(cfg.SSL_RESUME_CHECKPOINT, map_location=DEVICE, weights_only=False)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
            else:
                # Fallback for old checkpoints that only saved model weights
                model.load_state_dict(checkpoint)
        else:
            print(f"Warning: Checkpoint {cfg.SSL_RESUME_CHECKPOINT} not found.")
            
    print(f">>> Start BEM-SSL Training (Epochs: {cfg.SSL_EPOCHS})")
    
    for epoch in range(start_epoch, cfg.SSL_EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.SSL_EPOCHS}")
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(pbar):
            if batch is None: continue
            
            cuboids, mask, _ = batch
            
            # [MEMORY FIX] SSL은 국소적 정전기장만 학습하므로 N=1024로 절단
            MAX_SSL_N = 1024
            if cuboids.shape[1] > MAX_SSL_N:
                cuboids = cuboids[:, :MAX_SSL_N, :]
                mask = mask[:, :MAX_SSL_N]
                
            cuboids = cuboids.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)
            
            # 샘플링
            coords, types, eps_env = sampler.sample_for_ssl(cuboids, mask, num_samples=512)
            eps_env = torch.clamp(eps_env, min=1.0)
            
            
            with torch.amp.autocast('cuda', dtype=torch.float16):
                preds = model(cuboids, mask, compute_coupling=False)
                q_raw = preds['q_raw']
                equiv_r = preds['equiv_radius']
                
                centers = cuboids[..., :3].clone()
                centers[mask] = 1e5
                equiv_r = equiv_r.unsqueeze(1)
                
                dist = torch.cdist(coords, centers)
                # BEM 자기 전위 왜곡 방지
                soft_dist = torch.clamp(dist, min=equiv_r)
                
                V_pred = torch.sum((8.987 * q_raw.unsqueeze(1)) / (eps_env * soft_dist), dim=2)
                
                mask_tgt = (types == 1.0).float()
                mask_aggr = (types == 0.0).float()
                mask_far = (types == -1.0).float()
                
                loss_tgt = F.smooth_l1_loss(V_pred * mask_tgt, mask_tgt, reduction='sum') / torch.clamp(mask_tgt.sum(), min=1.0)
                loss_aggr = F.smooth_l1_loss(V_pred * mask_aggr, torch.zeros_like(V_pred), reduction='sum') / torch.clamp(mask_aggr.sum(), min=1.0)
                loss_far = F.smooth_l1_loss(V_pred * mask_far, torch.zeros_like(V_pred), reduction='sum') / torch.clamp(mask_far.sum(), min=1.0)
                
                loss = (loss_tgt + loss_aggr + loss_far) / ACCUM_STEPS

            if torch.isnan(loss):
                optimizer.zero_grad(set_to_none=True)
                continue
                
            scaler.scale(loss).backward()
            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            total_loss += loss.item() * ACCUM_STEPS
            pbar.set_postfix({'Loss': f"{loss.item() * ACCUM_STEPS:.4f}"})
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            # scaler.step(optimizer)
            # scaler.update()
            
            # total_loss += loss.item()
            # pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        print(f"Epoch {epoch+1} | Avg Loss: {total_loss/max(1, len(loader)):.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        scheduler.step()
        
        if (epoch+1) % 1 == 0:
            torch.save(model.state_dict(), SAVE_DIR / f"bem_ssl_ep{epoch+1}.pth")

        # [MEMORY FIX] GPU 동기화를 유발하는 empty_cache()는 에폭 종료 후에만 딱 1번 호출!
        gc.collect()
        torch.cuda.empty_cache()

    torch.save(model.state_dict(), SAVE_DIR / "bem_ssl_final.pth")
    print(">>> Neural BEM SSL Completed.")

if __name__ == "__main__":
    train_ssl()