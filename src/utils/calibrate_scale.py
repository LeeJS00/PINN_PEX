# calibrate_scale.py
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

import configs.config as cfg
from src.models.neural_field import DeepPEX_Model
from src.preprocessing.layer_parser import LayerInfoParser
from src.physics.materials import BEOLMaterialStack
from src.utils.gpu_sampler import sample_points_for_finetune
from src.data.datasets import NeuralFieldFinetuneDataset, robust_collate

def calibrate():
    DEVICE = f"cuda:{cfg.GPU_ID}" if torch.cuda.is_available() else "cpu"
    DATA_DIR = Path(cfg.PROCESSED_DIR)
    LABEL_PATH = Path(cfg.OUTPUT_DIR) / "active_learning/labeled_dataset.csv"
    
    if not LABEL_PATH.exists():
        print("No labeled data found to calibrate.")
        return

    # 1. Load Data
    df = pd.read_csv(LABEL_PATH)
    # Filter valid
    df = df[(df['capacitance'] > 0.01) & (df['capacitance'] < 1000)]
    print(f"Loaded {len(df)} samples for calibration.")
    
    dataset = NeuralFieldFinetuneDataset(DATA_DIR, df)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, collate_fn=robust_collate)
    
    # 2. Load Model
    class ModelConfig:
        MODEL_DIM = 128
        BASIS_LAYERS = 4
        CORR_LAYERS = 3
        WINDOW_SIZE = cfg.WINDOW_SIZE[0]
        INPUT_DIM = 9

    model = DeepPEX_Model(ModelConfig()).to(DEVICE)
    # Load Basis (Optional, but good for context)
    basis_ckpt = Path(cfg.OUTPUT_DIR) / "checkpoints/ssl_basis_v1/basis_final.pth"
    if basis_ckpt.exists():
        model.basis_model.load_state_dict(torch.load(basis_ckpt))
    
    model.eval()
    
    # 3. Measure Raw Charge
    ratios = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Measuring"):
            if batch is None: continue
            cuboids, mask, labels = batch
            cuboids = cuboids.to(DEVICE)
            mask = mask.to(DEVICE)
            labels = labels.to(DEVICE)
            
            if cuboids.shape[-1] > 9: cuboids = cuboids[..., :9]
            
            # Physics Head의 scale_factor를 1.0으로 임시 설정하여 Raw Charge 계산
            model.physics_head.scale_factor.fill_(1.0)
            
            surf_data = sample_points_for_finetune(cuboids, num_samples=2048)
            t_pts = surf_data['target_points'].to(DEVICE).requires_grad_(True)
            t_nrms = surf_data['target_normals'].to(DEVICE)
            t_areas = surf_data['target_areas'].to(DEVICE)
            
            # Forward (Gradient 계산을 위해 enable_grad context 필요)
            with torch.enable_grad():
                pred_raw, _ = model(cuboids, t_pts, mask, t_nrms, t_areas)
            
            # Ratio = GT / Pred_Raw
            # Pred_Raw가 0이면 제외
            valid = (pred_raw > 1e-20)
            batch_ratios = labels[valid] / pred_raw[valid]
            ratios.extend(batch_ratios.cpu().numpy())
            
    ratios = np.array(ratios)
    median_scale = np.median(ratios)
    mean_scale = np.mean(ratios)
    
    print("\n=== Calibration Result ===")
    print(f"Median Scale Factor: {median_scale:.4e}")
    print(f"Mean Scale Factor  : {mean_scale:.4e}")
    print(f"Current Config Val : {1e15:.4e} (Assumed)")
    
    suggested = median_scale
    print(f"\n>>> SUGGESTION: Set scale_factor = {suggested:.4e}")
    
    # 파일로 저장해서 나중에 로드하게 할 수도 있음
    with open(Path(cfg.OUTPUT_DIR) / "calibrated_scale.txt", "w") as f:
        f.write(str(suggested))

if __name__ == "__main__":
    calibrate()