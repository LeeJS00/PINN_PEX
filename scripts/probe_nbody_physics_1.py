# scripts/probe_nbody_physics.py
import sys
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

sys.path.append(str(Path(__file__).parent.parent))
import configs.config as cfg
from src.data.datasets import NeuralFieldFinetuneDataset

def probe_eps_gnd_physics():
    print("="*60)
    print("🔬 [Data Probing] Analyzing N-Body Screening vs Solid Angle")
    print("="*60)
    
    # -----------------------------------------------------------------
    # [CRITICAL FIX] 라벨이 없는 dataset_manifest 대신 Finetune 전용 CSV를 찾습니다.
    # -----------------------------------------------------------------
    manifest_candidates = ["predefined_valid_subset.csv"]
    manifest_path = None
    
    for cand in manifest_candidates:
        p = Path("/home/jslee/projects/PEX_SSL/output_intel22/active_learning/cache") / cand
        if p.exists():
            manifest_path = p
            break
            
    if manifest_path is None:
        print(f"❌ 파인튜닝용 매니페스트(finetune_manifest.csv 등)를 {cfg.PROCESSED_DIR}에서 찾을 수 없습니다.")
        return

    print(f"✅ Loading Labels from: {manifest_path.name}")
    df = pd.read_csv(manifest_path)
    
    # dropna 에러를 피하기 위해 먼저 capacitance가 존재하는 유효 행만 샘플링
    valid_df = df.dropna(subset=['capacitance']).reset_index(drop=True)
    sample_df = valid_df.sample(n=min(300, len(valid_df)), random_state=42).reset_index(drop=True)
    
    dataset = NeuralFieldFinetuneDataset(cfg.PROCESSED_DIR, sample_df)
    cpl_records = []
    gnd_records = []
    
    for idx in tqdm(range(len(dataset)), desc="Calculating Eps & Gnd Physics"):
        result = dataset[idx]
        if result is None: continue
            
        tensor, mask, labels_dict, meta_dict = result
        cpl_caps = labels_dict.get('coupled_caps', {})
        
        # for k,v in labels_dict.items():
        #     print(k, v)

        # print(a)
        # Ground Label 추출 (스칼라 값으로 가정)
        gnd_cap_tensor = labels_dict.get('ground_cap')
        gnd_cap_val = None
        if gnd_cap_tensor is not None:
            gnd_cap_val = gnd_cap_tensor[0].item() if isinstance(gnd_cap_tensor, torch.Tensor) else float(gnd_cap_tensor)
            # print(gnd_cap_val)
            
        cuboids = tensor[~mask]
        core_ratios = torch.tensor(meta_dict['core_ratios'])[~mask].float()
        
        names_arr = np.array(meta_dict['cuboid_net_names'])[~mask.numpy()]
        target_name = meta_dict['target_net_name'].replace('\\', '')
        
        is_target = (names_arr == target_name)
        if not is_target.any(): continue
            
        # 🚨 [새로운 추가] 유전율(Epsilon) 추출 (tensor의 8번째 채널)
        epsilons = cuboids[:, 8]
        
        centers = cuboids[:, :3]
        sizes = cuboids[:, 3:6]
        mins = centers - sizes / 2.0
        maxs = centers + sizes / 2.0
        
        tgt_mins = mins[is_target]
        tgt_maxs = maxs[is_target]
        tgt_cores = core_ratios[is_target]
        tgt_eps = epsilons[is_target]
        
        # 타겟의 전체 표면적 (유효 코어 비율 적용)
        raw_areas = 2.0 * (sizes[:,0]*sizes[:,1] + sizes[:,1]*sizes[:,2] + sizes[:,2]*sizes[:,0])
        tgt_total_area = torch.sum(tgt_cores * raw_areas[is_target]).item()
        tgt_total_eps_area = torch.sum(tgt_cores * raw_areas[is_target] * tgt_eps).item()
        
        # 3축 투영 면적 연산 보조
        def calc_overlap(ox, oy, oz):
            return (ox * oy) + (oy * oz) + (oz * ox)
            
        dx = torch.clamp(torch.maximum(tgt_mins[:, 0].unsqueeze(1), mins[:, 0].unsqueeze(0)) - torch.minimum(tgt_maxs[:, 0].unsqueeze(1), maxs[:, 0].unsqueeze(0)), min=0.0)
        dy = torch.clamp(torch.maximum(tgt_mins[:, 1].unsqueeze(1), mins[:, 1].unsqueeze(0)) - torch.minimum(tgt_maxs[:, 1].unsqueeze(1), maxs[:, 1].unsqueeze(0)), min=0.0)
        dz = torch.clamp(torch.maximum(tgt_mins[:, 2].unsqueeze(1), mins[:, 2].unsqueeze(0)) - torch.minimum(tgt_maxs[:, 2].unsqueeze(1), maxs[:, 2].unsqueeze(0)), min=0.0)
        
        surf_dist = torch.clamp(torch.sqrt(dx**2 + dy**2 + dz**2), min=1e-4)
        
        ox = torch.clamp(torch.minimum(tgt_maxs[:, 0].unsqueeze(1), maxs[:, 0].unsqueeze(0)) - torch.maximum(tgt_mins[:, 0].unsqueeze(1), mins[:, 0].unsqueeze(0)), min=0.0)
        oy = torch.clamp(torch.minimum(tgt_maxs[:, 1].unsqueeze(1), maxs[:, 1].unsqueeze(0)) - torch.maximum(tgt_mins[:, 1].unsqueeze(1), mins[:, 1].unsqueeze(0)), min=0.0)
        oz = torch.clamp(torch.minimum(tgt_maxs[:, 2].unsqueeze(1), maxs[:, 2].unsqueeze(0)) - torch.maximum(tgt_mins[:, 2].unsqueeze(1), mins[:, 2].unsqueeze(0)), min=0.0)
        
        overlap_area = calc_overlap(ox, oy, oz)
        overlap_peri = 2.0 * (ox + oy + oz)
        
        tile_total_overlap = 0.0 # Ground 계산용 (신호선들에게 뺏긴 면적 합산)
        
        for aggr_name, cpl_val in cpl_caps.items():
            aggr_name = aggr_name.replace('\\', '')
            aggr_mask = (names_arr == aggr_name)
            if not aggr_mask.any(): continue
                
            D_surf = surf_dist[:, aggr_mask]
            A_over = overlap_area[:, aggr_mask]
            P_over = overlap_peri[:, aggr_mask]
            
            # 🚨 [핵심] 타겟과 어그레서 사이의 평균 유전율 계산
            aggr_eps = epsilons[aggr_mask]
            eps_pair = (tgt_eps.unsqueeze(1) + aggr_eps.unsqueeze(0)) / 2.0
            
            # 기존 기하학 피처에 유전율(Eps) 반영
            val_eps_parallel = torch.sum(tgt_cores.unsqueeze(1) * eps_pair * (A_over / D_surf)).item()
            val_eps_fringe = torch.sum(tgt_cores.unsqueeze(1) * eps_pair * (P_over / torch.log1p(D_surf))).item()
            
            cpl_records.append({
                'Target_Net': target_name,
                'Aggr_Net': aggr_name,
                'Tile_Eps_Parallel': val_eps_parallel,
                'Tile_Eps_Fringe': val_eps_fringe,
                'Golden_Cap': cpl_val
            })
            
            # 다른 신호선과 마주본 면적 누적
            tile_total_overlap += torch.sum(tgt_cores.unsqueeze(1) * A_over).item()
            
        # Ground Feature 저장
        if gnd_cap_val is not None:
            # 타겟의 전체 면적 중 신호선들과 마주보지 않고 빈 공간(Ground)으로 노출된 면적
            exposed_area = max(0.0, tgt_total_area - tile_total_overlap)
            
            gnd_records.append({
                'Target_Net': target_name,
                'Tile_Total_Area': tgt_total_area,
                'Tile_Eps_Total_Area': tgt_total_eps_area,
                'Tile_Exposed_Area': exposed_area,
                'Golden_Gnd_Cap': gnd_cap_val
            })

    # 1. Coupling (CPL) 풀칩 조립 및 평가
    print("\n📦 [1] CPL Physics Assembly...")
    if cpl_records:
        cpl_df = pd.DataFrame(cpl_records).groupby(['Target_Net', 'Aggr_Net']).sum().reset_index()
        for col in ['Tile_Eps_Parallel', 'Tile_Eps_Fringe']:
            cpl_df = cpl_df[cpl_df[col] < cpl_df[col].quantile(0.99)]
            
        print("🏆 [CPL 유전율 보정] Pearson & Spearman 상관계수")
        for col in ['Tile_Eps_Parallel', 'Tile_Eps_Fringe']:
            p_corr, _ = pearsonr(cpl_df[col], cpl_df['Golden_Cap'])
            s_corr, _ = spearmanr(cpl_df[col], cpl_df['Golden_Cap'])
            print(f" - {col:<20} : Spearman {s_corr:.4f} / Pearson {p_corr:.4f}")
            
    # 2. Ground (GND) 풀칩 조립 및 평가
    print("\n📦 [2] GND Physics Assembly...")
    if gnd_records:
        gnd_df = pd.DataFrame(gnd_records).groupby('Target_Net').agg({
            'Tile_Total_Area': 'sum',
            'Tile_Eps_Total_Area': 'sum',
            'Tile_Exposed_Area': 'sum',
            'Golden_Gnd_Cap': 'first'
        }).reset_index()
        
        for col in ['Tile_Total_Area', 'Tile_Eps_Total_Area', 'Tile_Exposed_Area']:
            gnd_df = gnd_df[gnd_df[col] < gnd_df[col].quantile(0.99)]
            
        print("🏆 [GND 노출 면적] Pearson & Spearman 상관계수")
        for col in ['Tile_Total_Area', 'Tile_Eps_Total_Area', 'Tile_Exposed_Area']:
            p_corr, _ = pearsonr(gnd_df[col], gnd_df['Golden_Gnd_Cap'])
            s_corr, _ = spearmanr(gnd_df[col], gnd_df['Golden_Gnd_Cap'])
            print(f" - {col:<20} : Spearman {s_corr:.4f} / Pearson {p_corr:.4f}")

if __name__ == "__main__":
    probe_eps_gnd_physics()