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

def probe_heuristic_physics():
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
    
    feature_dict = {
        'Target_Net': [],              # [ADD] GroupBy를 위한 Key 추가
        'Aggr_Net': [],
        'F11_Rank_Discounted_Ad': [],  # 기존 F11
        'F12_Z_Penalized_Ad': [],      # 기존 F12
        'F13_Max_Normalized_Ad': [],   # 기존 F13
        'F14_Soft_Rank_Fringing': [],  # 기존 F14
        'F15_Voronoi_Weight_Ad': [],   # 기존 F15
        'F16_Closest_Ad': [],          # 가장 가까운 타겟 중심점 거리 기반 A/D
        'F17_Closest_Pd': [],          # 가장 가까운 타겟 중심점 거리 기반 P/D (Fringing)
        'F18_Solid_Angle_Approx': [],  # 입체각 유사: A / closest_D^2
        'F19_Edge_Ad': [],             # Edge-to-Edge 거리 기반 A/D
        'F20_Edge_Pd': [],             # Edge-to-Edge 거리 기반 P/D
        'F21_XY_Coupling': [],         # 측면 면적 대비 수평(XY) 거리
        'F22_Z_Coupling': [],          # 상하 면적 대비 수직(Z) 거리
        'F23_Edge_Rank_Discounted_Ad': [],  # Edge 거리 기반으로 계산된 차폐 랭크 페널티
        'F24_Vol_EdgeD3': [],          # 체적 / 거리^3
        'F25_Log_EdgeAd': [],          # log(A/D_edge) 커패시턴스의 비선형성 반영
        'F26_Inverse_EdgeD': [],       # 1 / D_edge 면적 무관 순수 거리의 역수
        'F27_Fringing_Log': [],        # P / log(1 + E_D) 프린징 필드 비선형 근사
        'F28_Harmonic_Distance': [],   # A / (C_D + E_D) 거리 조화 평균
        'F29_Projected_Solid_Angle': [], # (XY_Area / Z_D^2) + (Z_Area / XY_D^2) 방향성 투영 입체각
        'F30_Exp_Decay_Edge': [],      # A * exp(-E_D) 지수 감소형 차폐 반영 거리
        'F31_Log_A_over_D': [],        # log(1 + A/E_D)
        'F32_Area_Weighted_D_inv': [], # (A^2) / E_D
        'F41_Mutual_Ad_Ranked': [],    # Mutual Area / D * (0.5 ** per_cuboid_rank)
        'F42_Edge_Ad_Ranked': [],      # A / E_D * (0.5 ** per_cuboid_rank)
        'F43_Edge_Ad_ExpRank': [],     # A / E_D * exp(-per_cuboid_rank)
        'F44_Mutual_Ad_ExpD': [],      # Mutual Area * exp(-E_D * 5.0)
        'F45_SolidAngle_Ranked': [],   # Mutual Area / E_D^2 * (0.5 ** per_cuboid_rank)
        'F46_Immediate_Z_Coupling': [], # Mutual_Area_Z / DZ (only if DZ < 0.5)
        'F47_Immediate_XY_Coupling': [],# Mutual_Area_XY / XY_D (only if XY_D < 0.5 and DZ < 0.2)
        'F48_Threshold_Ad': [],        # A / E_D (only if E_D < 0.5)
        'F49_Threshold_Mutual_Ad': [], # M_Area_Total / E_D (only if E_D < 0.5)
        'F50_Strong_Coupling_Z': [],   # M_Area_Z / DZ^2 (only if DZ < 0.5)
        'F51_Strong_Coupling_XY': [],  # M_Area_XY / XY_D^2 (only if XY_D < 0.5 and DZ < 0.2)
        'F52_Raycast_Shielding_Flux': [], # Exact Blockage T * (A / E_D)
        'F53_Advanced_Fringing_Flux': [], # Fringing field with Bleed
        'F54_Maxwell_Total_Flux': [],  # Total analytical flux (Parallel + Fringing) * Shielding
        'F55_Debye_Density_Shielding': [], # Exponential decay based on local metal density
        'F56_Solid_Angle_Shadowing': [],   # Shadowing based on angular overlap
        'F57_Graph_Laplacian_Conductance': [], # Algebraic N-body shielding via graph Laplacian pseudo-inverse
        'C1_Maxwell_Shadow': [],           # Parallel + Fringing * Solid Angle Shadowing
        'C2_Vol_Inverse_Dist_Rank': [],    # Volumetric BEM approximation with Rank Shielding
        'C3_Voronoi_Shadow': [],           # Voronoi-weighted Area with Solid Angle Shadowing
        'C4_XY_Dominant_Ranked': [],       # Strong XY coupling strictly gated by Rank
        'C5_Ultimate_Physics': [],         # Maxwell Flux * Rank Penalty * Solid Angle Shadowing
        'D1_Strict_Immediate_Rank0': [],   # (Mutual_Area / E_D) ONLY for Rank == 0
        'D2_Strict_Immediate_Rank0_Z': [], # (Mutual_Area_Z / DZ) ONLY for Rank == 0
        'D3_Strict_Immediate_Rank0_XY': [],# (Mutual_Area_XY / DXY) ONLY for Rank == 0
        'D4_Soft_Rank_Mutual_Area': [],    # (Mutual_Area / E_D) * (0.5 ** Rank_i)
        'D5_Voronoi_Mutual_Area': [],      # (Mutual_Area / E_D) / (ambient_density + 1.0)
        'D6_Closest_Only_Cap': [],         # max(Mutual_Area / E_D)
        'N1_BEM_Inverse_Cap': [],          # Simplified BEM (Inverse Potential Matrix)
        'N2_Markov_Random_Walk_Absorb': [],# FRW Markov Chain 3-hop Absorption Probability
        'N3_Charge_Diffusion': [],         # Graph Laplacian Charge Diffusion
        'F33_Mutual_Area_X': [],
        'F34_Mutual_Area_Y': [],
        'F35_Mutual_Area_Z': [],
        'F36_Mutual_Ad_X': [],
        'F37_Mutual_Ad_Y': [],
        'F38_Mutual_Ad_Z': [],
        'F39_Total_Mutual_Ad': [],
        'F40_Mutual_Ad_Edge': [],
        'Ambient_Density': [],
        'Golden_Cap': []
    }
    
    for idx in tqdm(range(len(dataset)), desc="Computing Heuristic Features"):
        result = dataset[idx]
        if result is None: continue
            
        tensor, mask, labels_dict, meta_dict = result
        cpl_caps = labels_dict.get('coupled_caps', {})
        if not cpl_caps: continue
            
        cuboids = tensor[~mask]
        if len(cuboids) < 2: continue
        
        names_arr = np.array(meta_dict['cuboid_net_names'])
        valid_names = names_arr[~mask.numpy()]
        target_name = meta_dict['target_net_name'].replace('\\', '')
        
        is_target = (valid_names == target_name)
        if not is_target.any(): continue
            
        target_center = cuboids[is_target][:, :3].mean(dim=0)
        centers = cuboids[:, :3]
        sizes = cuboids[:, 3:6]
        
        areas = 2.0 * (sizes[:,0]*sizes[:,1] + sizes[:,1]*sizes[:,2] + sizes[:,2]*sizes[:,0])
        perimeters = 4.0 * (sizes[:,0] + sizes[:,1] + sizes[:,2])
        volumes = sizes[:,0] * sizes[:,1] * sizes[:,2]
        
        # 1. 뼈대 피처 세팅 (mean target center 기준 기존 로직)
        delta = centers - target_center.unsqueeze(0)
        dist_sq = torch.sum(delta**2, dim=-1)
        dist_sq[is_target] = float('inf')
        dists = torch.sqrt(torch.clamp(dist_sq, min=1e-4))
        
        z_penalized_dists = torch.sqrt(delta[:,0]**2 + delta[:,1]**2 + (delta[:,2]*3.0)**2)
        z_penalized_dists[is_target] = float('inf')
        
        # 2. 새로운 피처 세팅 (Closest & Edge-to-edge)
        target_centers = cuboids[is_target][:, :3]
        # 거리 행렬 [N_cuboids, N_target_cuboids]
        delta_matrix = centers.unsqueeze(1) - target_centers.unsqueeze(0)
        dist_matrix = torch.norm(delta_matrix, dim=-1)
        
        # Center-to-center closest
        closest_dists = torch.min(dist_matrix, dim=1)[0]
        closest_dists[is_target] = float('inf')
        closest_dists = torch.clamp(closest_dists, min=1e-4)
        
        # Edge-to-edge 근사
        closest_target_idx = torch.argmin(dist_matrix, dim=1)
        delta_to_closest = delta_matrix[torch.arange(len(centers)), closest_target_idx]
        
        dx = torch.abs(delta_to_closest[:, 0])
        dy = torch.abs(delta_to_closest[:, 1])
        dz = torch.abs(delta_to_closest[:, 2])
        
        target_sizes_closest = cuboids[is_target][closest_target_idx][:, 3:6]
        rx = torch.clamp(dx - (sizes[:, 0] + target_sizes_closest[:, 0]) / 2.0, min=0.0)
        ry = torch.clamp(dy - (sizes[:, 1] + target_sizes_closest[:, 1]) / 2.0, min=0.0)
        rz = torch.clamp(dz - (sizes[:, 2] + target_sizes_closest[:, 2]) / 2.0, min=0.0)
        
        edge_dist_sq = rx**2 + ry**2 + rz**2
        edge_dists = torch.sqrt(torch.clamp(edge_dist_sq, min=1e-6))
        edge_dists[is_target] = float('inf')
        
        # 방향성 (Directional) 분해 (XY 평면 vs Z축)
        xy_dists = torch.sqrt(torch.clamp(rx**2 + ry**2, min=1e-6))
        z_dists = torch.clamp(rz, min=1e-6)
        
        # True Mutual Facing Area 계산 (좌표 기반 겹침)
        target_centers_closest = cuboids[is_target][closest_target_idx][:, :3]
        
        # Y축 겹침 길이
        y1_min = centers[:, 1] - sizes[:, 1] / 2
        y1_max = centers[:, 1] + sizes[:, 1] / 2
        y2_min = target_centers_closest[:, 1] - target_sizes_closest[:, 1] / 2
        y2_max = target_centers_closest[:, 1] + target_sizes_closest[:, 1] / 2
        overlap_y = torch.clamp(torch.minimum(y1_max, y2_max) - torch.maximum(y1_min, y2_min), min=0.0)
        
        # Z축 겹침 길이
        z1_min = centers[:, 2] - sizes[:, 2] / 2
        z1_max = centers[:, 2] + sizes[:, 2] / 2
        z2_min = target_centers_closest[:, 2] - target_sizes_closest[:, 2] / 2
        z2_max = target_centers_closest[:, 2] + target_sizes_closest[:, 2] / 2
        overlap_z = torch.clamp(torch.minimum(z1_max, z2_max) - torch.maximum(z1_min, z2_min), min=0.0)
        
        # X축 겹침 길이
        x1_min = centers[:, 0] - sizes[:, 0] / 2
        x1_max = centers[:, 0] + sizes[:, 0] / 2
        x2_min = target_centers_closest[:, 0] - target_sizes_closest[:, 0] / 2
        x2_max = target_centers_closest[:, 0] + target_sizes_closest[:, 0] / 2
        overlap_x = torch.clamp(torch.minimum(x1_max, x2_max) - torch.maximum(x1_min, x2_min), min=0.0)
        
        mutual_area_x = overlap_y * overlap_z
        mutual_area_y = overlap_x * overlap_z
        mutual_area_z = overlap_x * overlap_y
        
        mutual_area_total = mutual_area_x + mutual_area_y + mutual_area_z
        
        # Per-cuboid Occlusion Rank (현재 큐보이드보다 Target에 더 가까운 Non-target 큐보이드 개수)
        valid_edge_dists = edge_dists.clone()
        valid_edge_dists[is_target] = float('inf')
        per_cuboid_rank = torch.sum(valid_edge_dists.unsqueeze(0) < valid_edge_dists.unsqueeze(1), dim=0).float()
        
        # --- [ADVANCED MAXWELL PHYSICS: Ray-Casting & Shielding] ---
        # 타겟과 커플링되는 각 큐보이드 간의 1:1 대응(가장 가까운 큐보이드)을 엣지로 설정
        N_all = len(centers)
        device = centers.device
        sizes = cuboids[:, 3:6]
        mins = centers - sizes / 2.0
        maxs = centers + sizes / 2.0
        surf_areas = 2 * (sizes[:,0]*sizes[:,1] + sizes[:,1]*sizes[:,2] + sizes[:,2]*sizes[:,0])
        
        tgt_indices = closest_target_idx
        cpl_indices = torch.arange(N_all, device=device)
        
        # === [NOVEL GLOBAL N-BODY MATRIX METHODS] ===
        dist_mat_all = torch.norm(centers.unsqueeze(1) - centers.unsqueeze(0), dim=-1)
        
        # 1. Simplified BEM (Inverse Potential Matrix)
        P_mat = 1.0 / torch.clamp(dist_mat_all, min=1e-3)
        R_eq = torch.sqrt(surf_areas / (4.0 * 3.14159))
        P_mat.fill_diagonal_(0.0)
        P_mat = P_mat + torch.diag(1.0 / torch.clamp(R_eq, min=1e-3))
        P_mat = P_mat + torch.eye(N_all, device=device) * 1e-6 # Ridge for stability
        try:
            C_mat = torch.linalg.inv(P_mat)
        except Exception:
            C_mat = torch.zeros((N_all, N_all), device=device)
        bem_capacitance = -C_mat[tgt_indices, cpl_indices] # (N_all,)
        
        # 2. Markov Random Walk Absorption
        T_mat = surf_areas.unsqueeze(0) / torch.clamp(dist_mat_all**2, min=1e-4)
        T_mat.fill_diagonal_(0.0)
        row_sums = torch.sum(T_mat, dim=1, keepdim=True)
        T_mat = T_mat / torch.clamp(row_sums, min=1e-6)
        M_mat = torch.matmul(torch.matmul(T_mat, T_mat), T_mat) # 3-hop
        markov_absorption = M_mat[tgt_indices, cpl_indices]
        
        # 3. Laplacian Charge Diffusion
        Adj_L = surf_areas.unsqueeze(0) / torch.clamp(dist_mat_all**2, min=1e-4)
        Adj_L.fill_diagonal_(0.0)
        D_L = torch.diag(torch.sum(Adj_L, dim=1))
        Lap = D_L - Adj_L
        alpha = 1.0 / torch.clamp(torch.max(torch.diagonal(D_L)), min=1e-3)
        Diff_Op = torch.eye(N_all, device=device) - alpha * Lap
        Diff_Op3 = torch.matmul(torch.matmul(Diff_Op, Diff_Op), Diff_Op)
        charge_diffusion = Diff_Op3[tgt_indices, cpl_indices]
        # ============================================
        
        c_min = torch.min(mins[tgt_indices], mins[cpl_indices])
        c_max = torch.max(maxs[tgt_indices], maxs[cpl_indices])
        c_dim = torch.clamp(c_max - c_min, min=1e-6)
        
        # Blockage (제3의 메탈) 교집합 계산
        k_mins, k_maxs = mins.unsqueeze(0), maxs.unsqueeze(0)
        inter_min = torch.max(c_min.unsqueeze(1), k_mins)
        inter_max = torch.min(c_max.unsqueeze(1), k_maxs)
        overlap = torch.clamp(inter_max - inter_min, min=0.0)
        
        k_idx = torch.arange(N_all, device=device).view(1, N_all).expand(N_all, N_all)
        mask_k = (k_idx != tgt_indices.unsqueeze(1)) & (k_idx != cpl_indices.unsqueeze(1))
        overlap = overlap * mask_k.unsqueeze(-1).float()
        
        r = overlap / c_dim.unsqueeze(1)
        A_xy, A_yz, A_zx = r[..., 0] * r[..., 1], r[..., 1] * r[..., 2], r[..., 2] * r[..., 0]
        eps = 1e-6
        A_xy = torch.where(A_xy < 1e-5, torch.zeros_like(A_xy), A_xy)
        A_yz = torch.where(A_yz < 1e-5, torch.zeros_like(A_yz), A_yz)
        A_zx = torch.where(A_zx < 1e-5, torch.zeros_like(A_zx), A_zx)
        
        unshielded_xy = torch.exp(torch.sum(torch.log(1.0 - A_xy + eps), dim=1))
        unshielded_yz = torch.exp(torch.sum(torch.log(1.0 - A_yz + eps), dim=1))
        unshielded_zx = torch.exp(torch.sum(torch.log(1.0 - A_zx + eps), dim=1))
        
        S_xy, S_yz, S_zx = torch.clamp(1.0 - unshielded_xy, 0.0, 1.0), torch.clamp(1.0 - unshielded_yz, 0.0, 1.0), torch.clamp(1.0 - unshielded_zx, 0.0, 1.0)
        
        src_centers = centers[tgt_indices]
        dst_centers = centers[cpl_indices]
        delta = torch.abs(dst_centers - src_centers)
        W = delta / torch.clamp(torch.sum(delta, dim=1, keepdim=True), min=1e-6)
        S_total = W[:, 0] * S_yz + W[:, 1] * S_zx + W[:, 2] * S_xy
        
        # Fringing Bleed
        dist_mag = torch.clamp(torch.norm(delta, dim=1), min=1e-6)
        corridor_max_width = torch.clamp(torch.max(c_dim, dim=1)[0], min=1e-6)
        aspect_ratio_corridor = dist_mag / corridor_max_width
        fringing_bleed = 1.0 - (1.0 / (1.0 + 0.5 * torch.log1p(aspect_ratio_corridor)))
        
        ray_vec = dst_centers - src_centers
        ray_len = torch.clamp(torch.norm(ray_vec, dim=-1), min=1e-9)
        inv_dir = 1.0 / ((ray_vec / ray_len.unsqueeze(-1)) + 1e-9)
        ro, rd_inv = src_centers.unsqueeze(1), inv_dir.unsqueeze(1)
        t0, t1 = (mins - ro) * rd_inv, (maxs - ro) * rd_inv
        tmin_k, tmax_k = torch.max(torch.minimum(t0, t1), dim=-1)[0], torch.min(torch.maximum(t0, t1), dim=-1)[0]
        strict_mask = (tmax_k > 0) & (tmin_k < tmax_k) & (tmin_k < ray_len.unsqueeze(-1)) & mask_k
        min_dist_to_block = torch.min(torch.where(strict_mask, torch.clamp(tmin_k, min=0.0), torch.tensor(1e5, device=device)), dim=-1)[0]
        solid_angle_confidence = torch.exp(-min_dist_to_block / dist_mag)
        
        final_bleed = torch.clamp(fringing_bleed + (1.0 - solid_angle_confidence) * 0.2, min=0.01, max=0.95)
        effective_shielding = 1.0 - S_total * (1.0 - final_bleed) * 0.8
        transmission = torch.clamp(1.0 - effective_shielding, min=0.0)
        
        # --- [NOVEL METHOD 1: Debye Density Shielding] ---
        # 큐보이드 체적 밀도를 통해 동적 Debye Length를 계산
        k_vol = (sizes[:, 0] * sizes[:, 1] * sizes[:, 2]).unsqueeze(0) # (1, N)
        k_centers = centers.unsqueeze(0) # (1, N, 3)
        # Corridor 안에 존재하는 큐보이드들의 체적 합산
        is_in_corridor = (k_centers[:, :, 0] >= inter_min[:, :, 0]) & (k_centers[:, :, 0] <= inter_max[:, :, 0]) & \
                         (k_centers[:, :, 1] >= inter_min[:, :, 1]) & (k_centers[:, :, 1] <= inter_max[:, :, 1]) & \
                         (k_centers[:, :, 2] >= inter_min[:, :, 2]) & (k_centers[:, :, 2] <= inter_max[:, :, 2])
        corridor_vol_metal = torch.sum(k_vol * (is_in_corridor & mask_k).float(), dim=1) # (E)
        corridor_vol_total = c_dim[:, 0] * c_dim[:, 1] * c_dim[:, 2] # (E)
        local_density = corridor_vol_metal / torch.clamp(corridor_vol_total, min=1e-6)
        debye_length = 1.0 / (local_density * 10.0 + 1e-3)
        debye_transmission = torch.exp(-ray_len / debye_length)
        
        # --- [NOVEL METHOD 2: Directional Solid Angle Shadowing] ---
        # 타겟에서 볼 때 앵글에 겹치는 블로커들의 입체각 누적
        vec_sa = ray_vec # (E, 3)
        vec_sk = k_centers - src_centers.unsqueeze(1) # (E, N, 3)
        norm_sk = torch.clamp(torch.norm(vec_sk, dim=-1), min=1e-6) # (E, N)
        norm_sa = ray_len.unsqueeze(-1) # (E, 1)
        cos_theta = torch.sum(vec_sa.unsqueeze(1) * vec_sk, dim=-1) / (norm_sa * norm_sk) # (E, N)
        
        # 블로커의 대략적인 입체각 (Area / dist^2)
        k_area_approx = k_vol.squeeze(0)**(2/3) # (N)
        omega_k = k_area_approx.unsqueeze(0) / (norm_sk**2) # (E, N)
        
        # 타겟과 어그레서 사이에 있고(cos_theta > 0), 어그레서보다 가까운 경우만 섀도우로 인정
        is_front = (cos_theta > 0.8) & (norm_sk < norm_sa) & mask_k # (E, N)
        total_shadow = torch.sum(omega_k * cos_theta * is_front.float(), dim=1) # (E)
        shadow_transmission = torch.exp(-total_shadow * 2.0)
        
        # --- [NOVEL METHOD 3: Graph Laplacian Pseudo-inverse (Algebraic N-body)] ---
        # N이 너무 크면 역행렬 계산이 불가능하므로, 전체 N_all에 대한 Adjacency 근사 계산 (O(N^2))
        # Adjacency A_ij = Area_approx / Dist_ij^2
        dist_mat_all = torch.norm(centers.unsqueeze(1) - centers.unsqueeze(0), dim=-1)
        A_mat = (sizes[:, 0]*sizes[:, 1]*sizes[:, 2])**(2/3)
        A_mat = A_mat.unsqueeze(0) * A_mat.unsqueeze(1) # (N, N)
        Adj = A_mat / torch.clamp(dist_mat_all**2, min=1e-4)
        Adj.fill_diagonal_(0.0)
        
        # Laplacian L = D - A
        D_mat = torch.diag(torch.sum(Adj, dim=1))
        L_mat = D_mat - Adj
        
        # Pseudo-inverse 계산 (CPU/GPU)
        try:
            L_pinv = torch.linalg.pinv(L_mat, hermitian=True)
            # Effective Resistance R_eff = L_ii + L_jj - 2*L_ij
            L_ii = torch.diagonal(L_pinv) # (N)
            L_tgt = L_ii[tgt_indices] # (E)
            L_cpl = L_ii[cpl_indices] # (E)
            L_cross = L_pinv[tgt_indices, cpl_indices] # (E)
            R_eff = L_tgt + L_cpl - 2.0 * L_cross
            graph_conductance = 1.0 / torch.clamp(R_eff, min=1e-6) # (E)
        except Exception:
            graph_conductance = torch.zeros_like(ray_len)
        
        # 전체 큐보이드의 A/d 계산 (Max Normalization 및 Rank 계산용)
        all_Ad = areas / dists
        all_Ad[is_target] = 0.0
        max_Ad_global = torch.max(all_Ad).item()
        
        ambient_density = torch.sum(all_Ad).item()
        
        for aggr_name, cpl_val in cpl_caps.items():
            aggr_name = aggr_name.replace('\\', '')
            
            if abs(cpl_val) < 0.005: continue
                
            aggr_mask = (valid_names == aggr_name)
            if not aggr_mask.any(): continue
                
            A = areas[aggr_mask]
            D = dists[aggr_mask]
            Z_D = z_penalized_dists[aggr_mask]
            P = perimeters[aggr_mask]
            V = volumes[aggr_mask]
            
            C_D = closest_dists[aggr_mask]
            E_D = edge_dists[aggr_mask]
            XY_D = xy_dists[aggr_mask]
            Z_D_edge = z_dists[aggr_mask]
            
            DX = torch.clamp(rx[aggr_mask], min=1e-4)
            DY = torch.clamp(ry[aggr_mask], min=1e-4)
            DZ = torch.clamp(rz[aggr_mask], min=1e-4)
            
            M_Area_X = mutual_area_x[aggr_mask]
            M_Area_Y = mutual_area_y[aggr_mask]
            M_Area_Z = mutual_area_z[aggr_mask]
            M_Area_Total = mutual_area_total[aggr_mask]
            Rank_i = per_cuboid_rank[aggr_mask]
            
            # --- 기존 피처 (Ours 원래 방식) ---
            min_D_old = torch.min(D)
            occlusion_rank = torch.sum((dists < min_D_old) & (~is_target)).item()
            
            val_F11 = torch.sum((A / D) * (0.7 ** occlusion_rank)).item()
            val_F12 = torch.sum(A / torch.clamp(Z_D, min=1e-4)).item()
            val_F13 = torch.sum((A / D) / max(max_Ad_global, 1e-6)).item()
            val_F14 = torch.sum((P / D) * (0.9 ** occlusion_rank)).item()
            val_F15 = torch.sum((A / (D**2)) * np.exp(-occlusion_rank / 5.0)).item()
            
            # --- 신규 피처 ---
            min_E_D = torch.min(E_D)
            edge_occlusion_rank = torch.sum((edge_dists < min_E_D) & (~is_target)).item()
            
            val_F16 = torch.sum(A / C_D).item()
            val_F17 = torch.sum(P / C_D).item()
            val_F18 = torch.sum(A / (C_D**2)).item()
            
            val_F19 = torch.sum(A / E_D).item()
            val_F20 = torch.sum(P / E_D).item()
            
            xy_areas = sizes[aggr_mask, 0] * sizes[aggr_mask, 1]
            z_areas = sizes[aggr_mask, 0] * sizes[aggr_mask, 2] + sizes[aggr_mask, 1] * sizes[aggr_mask, 2]
            
            # 방향성에 맞춘 단면적/거리 (수평거리엔 수직단면적 Z면적, 수직거리엔 수평단면적 XY면적이 핵심)
            val_F21 = torch.sum(z_areas / torch.clamp(XY_D, min=1e-4)).item()
            val_F22 = torch.sum(xy_areas / torch.clamp(Z_D_edge, min=1e-4)).item()
            
            val_F23 = torch.sum((A / E_D) * (0.7 ** edge_occlusion_rank)).item()
            val_F24 = torch.sum(V / (E_D**3)).item()
            val_F25 = torch.sum(torch.log1p(A / E_D)).item()
            val_F26 = torch.sum(1.0 / E_D).item()
            
            val_F27 = torch.sum(P / torch.log1p(E_D)).item()
            val_F28 = torch.sum(A / (C_D + E_D)).item()
            val_F29 = torch.sum(z_areas / (torch.clamp(XY_D, min=1e-4)**2) + xy_areas / (torch.clamp(Z_D_edge, min=1e-4)**2)).item()
            val_F30 = torch.sum(A * torch.exp(-E_D)).item()
            val_F31 = torch.sum(torch.log1p(A) / torch.clamp(E_D, min=1e-4)).item()
            val_F32 = torch.sum((A**2) / E_D).item()
            
            val_F33 = torch.sum(M_Area_X).item()
            val_F34 = torch.sum(M_Area_Y).item()
            val_F35 = torch.sum(M_Area_Z).item()
            
            val_F36 = torch.sum(M_Area_X / DX).item()
            val_F37 = torch.sum(M_Area_Y / DY).item()
            val_F38 = torch.sum(M_Area_Z / DZ).item()
            val_F39 = val_F36 + val_F37 + val_F38
            val_F40 = torch.sum(M_Area_Total / E_D).item()
            
            val_F41 = torch.sum((M_Area_Total / E_D) * (0.5 ** Rank_i)).item()
            val_F42 = torch.sum((A / E_D) * (0.5 ** Rank_i)).item()
            val_F43 = torch.sum((A / E_D) * torch.exp(-Rank_i)).item()
            val_F44 = torch.sum(M_Area_Total * torch.exp(-E_D * 5.0)).item()
            val_F45 = torch.sum((M_Area_Total / (E_D**2)) * (0.5 ** Rank_i)).item()
            
            # Thresholded Immediate Features
            mask_Z = (DZ < 0.5)
            mask_XY = (XY_D < 0.5) & (DZ < 0.2)
            mask_E = (E_D < 0.5)
            
            val_F46 = torch.sum((M_Area_Z / DZ)[mask_Z]).item() if mask_Z.any() else 0.0
            val_F47 = torch.sum((M_Area_X / DX)[mask_XY] + (M_Area_Y / DY)[mask_XY]).item() if mask_XY.any() else 0.0
            val_F48 = torch.sum((A / E_D)[mask_E]).item() if mask_E.any() else 0.0
            val_F49 = torch.sum((M_Area_Total / E_D)[mask_E]).item() if mask_E.any() else 0.0
            val_F50 = torch.sum((M_Area_Z / (DZ**2))[mask_Z]).item() if mask_Z.any() else 0.0
            val_F51 = torch.sum((M_Area_X / (DX**2))[mask_XY] + (M_Area_Y / (DY**2))[mask_XY]).item() if mask_XY.any() else 0.0
            
            # --- Advanced Maxwell Features ---
            T_i = transmission[aggr_mask]
            eps_avg = (cuboids[tgt_indices[aggr_mask], 8] + cuboids[aggr_mask, 8]) / 2.0
            val_F52 = torch.sum((A / E_D) * T_i * eps_avg).item()
            
            PI = 3.14159
            L_eff = torch.max(torch.max(M_Area_X, M_Area_Y), M_Area_Z) ** 0.5
            w_i = cuboids[aggr_mask, 3]
            cpl_fringe_wire = (2.0 * PI * eps_avg * L_eff) / torch.log1p(E_D / torch.clamp(w_i, min=1e-3))
            val_F53 = torch.sum(cpl_fringe_wire * T_i).item()
            
            cpl_parallel = ((M_Area_X * eps_avg) / DX) + ((M_Area_Y * eps_avg) / DY) + ((M_Area_Z * eps_avg) / DZ)
            val_F54 = torch.sum((cpl_parallel + cpl_fringe_wire) * T_i).item()
            
            # --- Novel Shielding Features ---
            T_debye = debye_transmission[aggr_mask]
            val_F55 = torch.sum((A / E_D) * T_debye * eps_avg).item()
            
            T_shadow = shadow_transmission[aggr_mask]
            val_F56 = torch.sum((A / E_D) * T_shadow * eps_avg).item()
            
            G_eff = graph_conductance[aggr_mask]
            val_F57 = torch.sum(G_eff * eps_avg).item()
            
            # --- [High-Order Composition Features (C1~C5)] ---
            val_C1 = torch.sum((cpl_parallel + cpl_fringe_wire) * T_shadow * eps_avg).item()
            val_C2 = torch.sum((A / (E_D ** 2)) * (0.5 ** Rank_i)).item()
            val_C3 = torch.sum(((A / E_D) / (ambient_density + 1.0)) * T_shadow).item()
            val_C4 = torch.sum(((M_Area_Y / DY) + (M_Area_X / DX)) * (0.5 ** Rank_i)).item()
            val_C5 = torch.sum((cpl_parallel + cpl_fringe_wire) * (0.7 ** Rank_i) * T_shadow * eps_avg).item()
            
            # --- [Dense-Optimized Features (D1~D6)] ---
            mask_r0 = (Rank_i == 0)
            val_D1 = torch.sum((M_Area_Total / E_D)[mask_r0] * eps_avg).item() if mask_r0.any() else 0.0
            val_D2 = torch.sum((M_Area_Z / DZ)[mask_r0] * eps_avg).item() if mask_r0.any() else 0.0
            val_D3 = torch.sum(((M_Area_X / DX) + (M_Area_Y / DY))[mask_r0] * eps_avg).item() if mask_r0.any() else 0.0
            val_D4 = torch.sum((M_Area_Total / E_D) * (0.5 ** Rank_i) * eps_avg).item()
            val_D5 = torch.sum(((M_Area_Total / E_D) / (ambient_density + 1.0)) * eps_avg).item()
            val_D6 = torch.max((M_Area_Total / E_D) * eps_avg).item() if len(M_Area_Total) > 0 else 0.0
            
            # --- [Global N-Body Features (N1~N3)] ---
            val_N1 = torch.sum(bem_capacitance[aggr_mask] * eps_avg).item()
            val_N2 = torch.sum(markov_absorption[aggr_mask] * eps_avg).item()
            val_N3 = torch.sum(charge_diffusion[aggr_mask] * eps_avg).item()
            
            feature_dict['Target_Net'].append(target_name)
            feature_dict['Aggr_Net'].append(aggr_name)
            feature_dict['F11_Rank_Discounted_Ad'].append(val_F11)
            feature_dict['F12_Z_Penalized_Ad'].append(val_F12)
            feature_dict['F13_Max_Normalized_Ad'].append(val_F13)
            feature_dict['F14_Soft_Rank_Fringing'].append(val_F14)
            feature_dict['F15_Voronoi_Weight_Ad'].append(val_F15)
            
            feature_dict['F16_Closest_Ad'].append(val_F16)
            feature_dict['F17_Closest_Pd'].append(val_F17)
            feature_dict['F18_Solid_Angle_Approx'].append(val_F18)
            feature_dict['F19_Edge_Ad'].append(val_F19)
            feature_dict['F20_Edge_Pd'].append(val_F20)
            feature_dict['F21_XY_Coupling'].append(val_F21)
            feature_dict['F22_Z_Coupling'].append(val_F22)
            feature_dict['F23_Edge_Rank_Discounted_Ad'].append(val_F23)
            feature_dict['F24_Vol_EdgeD3'].append(val_F24)
            feature_dict['F25_Log_EdgeAd'].append(val_F25)
            feature_dict['F26_Inverse_EdgeD'].append(val_F26)
            feature_dict['F27_Fringing_Log'].append(val_F27)
            feature_dict['F28_Harmonic_Distance'].append(val_F28)
            feature_dict['F29_Projected_Solid_Angle'].append(val_F29)
            feature_dict['F30_Exp_Decay_Edge'].append(val_F30)
            feature_dict['F31_Log_A_over_D'].append(val_F31)
            feature_dict['F32_Area_Weighted_D_inv'].append(val_F32)
            
            feature_dict['F33_Mutual_Area_X'].append(val_F33)
            feature_dict['F34_Mutual_Area_Y'].append(val_F34)
            feature_dict['F35_Mutual_Area_Z'].append(val_F35)
            feature_dict['F36_Mutual_Ad_X'].append(val_F36)
            feature_dict['F37_Mutual_Ad_Y'].append(val_F37)
            feature_dict['F38_Mutual_Ad_Z'].append(val_F38)
            feature_dict['F39_Total_Mutual_Ad'].append(val_F39)
            feature_dict['F40_Mutual_Ad_Edge'].append(val_F40)
            
            feature_dict['F41_Mutual_Ad_Ranked'].append(val_F41)
            feature_dict['F42_Edge_Ad_Ranked'].append(val_F42)
            feature_dict['F43_Edge_Ad_ExpRank'].append(val_F43)
            feature_dict['F44_Mutual_Ad_ExpD'].append(val_F44)
            feature_dict['F45_SolidAngle_Ranked'].append(val_F45)
            
            feature_dict['F46_Immediate_Z_Coupling'].append(val_F46)
            feature_dict['F47_Immediate_XY_Coupling'].append(val_F47)
            feature_dict['F48_Threshold_Ad'].append(val_F48)
            feature_dict['F49_Threshold_Mutual_Ad'].append(val_F49)
            feature_dict['F50_Strong_Coupling_Z'].append(val_F50)
            feature_dict['F51_Strong_Coupling_XY'].append(val_F51)
            
            feature_dict['F52_Raycast_Shielding_Flux'].append(val_F52)
            feature_dict['F53_Advanced_Fringing_Flux'].append(val_F53)
            feature_dict['F54_Maxwell_Total_Flux'].append(val_F54)
            feature_dict['F55_Debye_Density_Shielding'].append(val_F55)
            feature_dict['F56_Solid_Angle_Shadowing'].append(val_F56)
            feature_dict['F57_Graph_Laplacian_Conductance'].append(val_F57)
            
            feature_dict['C1_Maxwell_Shadow'].append(val_C1)
            feature_dict['C2_Vol_Inverse_Dist_Rank'].append(val_C2)
            feature_dict['C3_Voronoi_Shadow'].append(val_C3)
            feature_dict['C4_XY_Dominant_Ranked'].append(val_C4)
            feature_dict['C5_Ultimate_Physics'].append(val_C5)
            
            feature_dict['D1_Strict_Immediate_Rank0'].append(val_D1)
            feature_dict['D2_Strict_Immediate_Rank0_Z'].append(val_D2)
            feature_dict['D3_Strict_Immediate_Rank0_XY'].append(val_D3)
            feature_dict['D4_Soft_Rank_Mutual_Area'].append(val_D4)
            feature_dict['D5_Voronoi_Mutual_Area'].append(val_D5)
            feature_dict['D6_Closest_Only_Cap'].append(val_D6)
            
            feature_dict['N1_BEM_Inverse_Cap'].append(val_N1)
            feature_dict['N2_Markov_Random_Walk_Absorb'].append(val_N2)
            feature_dict['N3_Charge_Diffusion'].append(val_N3)
            
            feature_dict['Ambient_Density'].append(ambient_density)
            feature_dict['Golden_Cap'].append(cpl_val)

    if len(feature_dict['Golden_Cap']) == 0:
        return

    raw_df = pd.DataFrame(feature_dict)
    
    # 1. GroupBy를 위한 동적 agg_dict 생성 (모든 F 피처는 sum)
    agg_dict = {
        'Ambient_Density': 'mean',       # 주변 거시 밀도는 평균값으로 대표
        'Golden_Cap': 'first'            # 정답지(Golden Cap)는 넷 단위로 동일하므로 첫 번째 값만 취함
    }
    for col in raw_df.columns:
        if col.startswith('F') or col.startswith('C') or col.startswith('D') or col.startswith('N'):
            agg_dict[col] = 'sum'
            
    # 2. 동일한 넷 쌍(Target, Aggr)에 대해 여러 타일에서 계산된 피처 값 합산
    res_df = raw_df.groupby(['Target_Net', 'Aggr_Net']).agg(agg_dict).reset_index()    
    
    # 3. 이상치 제거 (그룹핑 후에 수행)
    for col in res_df.columns:
        if col.startswith('F') or col.startswith('C') or col.startswith('D') or col.startswith('N'):
            q99 = res_df[col].quantile(0.99)
            if q99 > 0:
                res_df = res_df[res_df[col] <= q99]
            
    print(f"\n📊 [분석 완료] 총 {len(res_df)} 엣지 (타일 병합 후)\n")
    
    results = []
    for col in res_df.columns:
        if col in ['Golden_Cap', 'Ambient_Density', 'Density_Bin', 'Target_Net', 'Aggr_Net']: continue
        # 로그 변환 (0 방지를 위해 log1p 사용)
        feat_vals = np.log1p(res_df[col].clip(lower=0))
        cap_vals = np.log1p(res_df['Golden_Cap'])
        
        if feat_vals.std() < 1e-6 or cap_vals.std() < 1e-6:
            p_corr = 0.0
        else:
            p_corr, _ = pearsonr(feat_vals, cap_vals)
            
        if res_df[col].std() < 1e-6 or res_df['Golden_Cap'].std() < 1e-6:
            s_corr = 0.0
        else:
            s_corr, _ = spearmanr(res_df[col], res_df['Golden_Cap'])
        
        results.append((col, s_corr, p_corr))
        
    results.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print("🏆 [Feature 별 전체 상관계수 랭킹 (Spearman 기준)]")
    for rank, (name, s_corr, p_corr) in enumerate(results, 1):
        print(f" {rank:2d}위 | {name:<32} : Spearman {s_corr:.4f} / Log-Pearson {p_corr:.4f}")
        
    res_df['Density_Bin'] = pd.qcut(res_df['Ambient_Density'], q=3, labels=['Sparse', 'Medium', 'Dense'])
    dense_df = res_df[res_df['Density_Bin'] == 'Dense']
    
    print("\n🔥 [가장 악랄한 Dense 구역(밀집 지역) 생존력 랭킹]")
    dense_results = []
    for col in res_df.columns:
        if col in ['Golden_Cap', 'Ambient_Density', 'Density_Bin', 'Target_Net', 'Aggr_Net']: continue
        
        if dense_df[col].std() < 1e-6 or dense_df['Golden_Cap'].std() < 1e-6:
            s_corr = 0.0
        else:
            s_corr, _ = spearmanr(dense_df[col], dense_df['Golden_Cap'])
            
        dense_results.append((col, s_corr))
        
    dense_results.sort(key=lambda x: abs(x[1]), reverse=True)
    for rank, (name, corr) in enumerate(dense_results, 1):
        survival = "✅ (유의미한 생존!)" if abs(corr) > 0.4 else "⚠️ (약함)" if abs(corr) > 0.2 else "❌ (붕괴)"
        print(f" {rank:2d}위 | {name:<32} : Spearman {corr:.4f} {survival}")

if __name__ == "__main__":
    probe_heuristic_physics()