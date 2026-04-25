# src/models/compute_sheilding.py
import torch

@torch.no_grad()
@torch.compiler.disable 
def compute_sparse_shield_factor(cuboids, b_idx, src_idx, dst_idx, chunk_size=8192):
    """
    [Topology-Aware Shielding] 
    방향성 플럭스 가중치(Directional Weighting)와 확률적 누적(Probabilistic Union)을 결합하여
    Ray-Casting 수준의 차폐 정확도를 O(E) 텐서 연산만으로 달성합니다.
    """
    E = len(b_idx)
    if E == 0: return torch.empty(0, device=cuboids.device)
    B, N, _ = cuboids.shape
    dynamic_chunk_size = max(512, int(8192 * (256 / max(N, 1))))
    mins = cuboids[..., :3] - cuboids[..., 3:6] / 2.0
    maxs = cuboids[..., :3] + cuboids[..., 3:6] / 2.0
    
    total_shield_factor = torch.zeros(E, device=cuboids.device)
    
    # VRAM 폭발 방지를 위한 Chunking 루프
    for i in range(0, E, dynamic_chunk_size):
        end = min(i + dynamic_chunk_size, E)
        
        b_c = b_idx[i:end]
        src_c = src_idx[i:end]
        dst_c = dst_idx[i:end]
        
        src_mins, src_maxs = mins[b_c, src_c], maxs[b_c, src_c]
        dst_mins, dst_maxs = mins[b_c, dst_c], maxs[b_c, dst_c]
        
        # 1. 3D Corridor (복도) 설정
        c_min = torch.min(src_mins, dst_mins)
        c_max = torch.max(src_maxs, dst_maxs)
        c_dim = torch.clamp(c_max - c_min, min=1e-6) # (E_chunk, 3)
        
        # 2. Blockage (제3의 메탈) 교집합 계산
        k_mins, k_maxs = mins[b_c], maxs[b_c] # (E_chunk, N, 3)
        inter_min = torch.max(c_min.unsqueeze(1), k_mins)
        inter_max = torch.min(c_max.unsqueeze(1), k_maxs)
        overlap = torch.clamp(inter_max - inter_min, min=0.0) # (E_chunk, N, 3)
        
        # 자기 자신(Target, Aggr) 제외
        k_idx = torch.arange(N, device=cuboids.device).view(1, N).expand(end - i, N)
        mask_k = (k_idx != src_c.unsqueeze(1)) & (k_idx != dst_c.unsqueeze(1))
        overlap = overlap * mask_k.unsqueeze(-1).float()
        
        # 축별 침범 비율 (0.0 ~ 1.0)
        r = overlap / c_dim.unsqueeze(1) # (E_chunk, N, 3)
        
        # -------------------------------------------------------------
        # [TOPOLOGY FIX 1] 면적 기반 방향성 차폐 (Cross-sectional Blockage)
        # -------------------------------------------------------------
        A_xy = r[..., 0] * r[..., 1] # Z방향 플럭스를 막는 면적
        A_yz = r[..., 1] * r[..., 2] # X방향 플럭스를 막는 면적
        A_zx = r[..., 2] * r[..., 0] # Y방향 플럭스를 막는 면적
        
        # -------------------------------------------------------------
        # [TOPOLOGY FIX 2] 확률적 누적 (Probabilistic Accumulation)
        # S = 1 - prod(1 - A) 
        # 여러 메탈의 차폐율을 더하지 않고, 광학적 불투명도로 누적시킵니다.
        # -------------------------------------------------------------
        eps = 1e-6
        A_xy = torch.where(A_xy < 1e-5, torch.zeros_like(A_xy), A_xy)
        A_yz = torch.where(A_yz < 1e-5, torch.zeros_like(A_yz), A_yz)
        A_zx = torch.where(A_zx < 1e-5, torch.zeros_like(A_zx), A_zx)
        # exp(sum(log(x))) 트릭을 사용하여 prod 연산의 그라디언트 및 수치적 안정성 확보
        unshielded_xy = torch.exp(torch.sum(torch.log(1.0 - A_xy + eps), dim=1))
        unshielded_yz = torch.exp(torch.sum(torch.log(1.0 - A_yz + eps), dim=1))
        unshielded_zx = torch.exp(torch.sum(torch.log(1.0 - A_zx + eps), dim=1))
        
        S_xy = torch.clamp(1.0 - unshielded_xy, 0.0, 1.0) # (E_chunk,)
        S_yz = torch.clamp(1.0 - unshielded_yz, 0.0, 1.0)
        S_zx = torch.clamp(1.0 - unshielded_zx, 0.0, 1.0)
        
        # -------------------------------------------------------------
        # [TOPOLOGY FIX 3] 플럭스 방향 가중치 (Directional Flux Weighting)
        # 플럭스가 주로 흐르는 방향(축)을 계산하여, 해당 방향을 막는 벽면만 차폐로 인정합니다.
        # -------------------------------------------------------------
        src_centers = (src_mins + src_maxs) / 2.0
        dst_centers = (dst_mins + dst_maxs) / 2.0
        delta = torch.abs(dst_centers - src_centers)
        
        # 타겟과 어그레서 사이의 거리를 정규화하여 방향 벡터 생성
        W = delta / torch.clamp(torch.sum(delta, dim=1, keepdim=True), min=1e-6) # (E_chunk, 3)
        
        # X방향 가중치(W_x)는 YZ 평면 차폐(S_yz)와 곱해짐
        S_total = W[:, 0] * S_yz + W[:, 1] * S_zx + W[:, 2] * S_xy
        
        # -------------------------------------------------------------
        # [NEW EXPERT FIX] Fringing Bleed-through (전기장 굴절 효과)
        # -------------------------------------------------------------
        # 두 메탈 사이의 거리(delta)가 멀수록, 전기력선이 우회할 수 있는 공간이 많아집니다.
        # 거리에 비례하여 차폐망을 뚫고 지나가는 최소한의 누수율(Bleed)을 계산합니다.
        dist_mag = torch.clamp(torch.norm(delta, dim=1), min=1e-6)
        corridor_max_width = torch.clamp(torch.max(c_dim, dim=1)[0], min=1e-6)
        aspect_ratio = dist_mag / corridor_max_width
        
        # 거리가 멀어질수록 누수율이 로그 스케일로 포화되도록 물리적 감쇠(Decay) 적용
        fringing_bleed = 1.0 - (1.0 / (1.0 + 0.5 * torch.log1p(aspect_ratio)))
        
        # 2. 3D 입체각(Solid Angle) 기반 거리 가중치 (Distance Inverse-Square)
        # 차폐물이 Source(타겟)에 가까울수록 시야를 더 많이 가리므로 차폐 효과가 증폭됩니다.
        k_centers = (k_mins + k_maxs) / 2.0 # (E_chunk, N, 3)
        
        # 가장 가까운 유효 차폐물까지의 거리 추출
        # (mask_k를 이용하여 타겟/어그레서 자신과의 거리는 무한대로 처리)
        ray_vec = dst_centers - src_centers  # (E_chunk, 3)
        ray_len = torch.clamp(torch.norm(ray_vec, dim=-1), min=1e-9) # (E_chunk,)
        
        # 광선 방향 및 역방향 벡터 (0으로 나누기 방지)
        ray_dir = ray_vec / ray_len.unsqueeze(-1)  # (E_chunk, 3)
        inv_dir = 1.0 / (ray_dir + 1e-9)  # (E_chunk, 3)
        
        ro = src_centers.unsqueeze(1) # (E_chunk, 1, 3)
        rd_inv = inv_dir.unsqueeze(1) # (E_chunk, 1, 3)
        
        # Slabs Method: X, Y, Z 각 축에 대해 큐보이드 평면에 도달하는 거리(t) 계산
        t0 = (k_mins - ro) * rd_inv  # (E_chunk, N, 3)
        t1 = (k_maxs - ro) * rd_inv  # (E_chunk, N, 3)
        
        # 진입점(tmin)과 진출점(tmax) 계산
        tmin_k = torch.max(torch.minimum(t0, t1), dim=-1)[0] # (E_chunk, N)
        tmax_k = torch.min(torch.maximum(t0, t1), dim=-1)[0] # (E_chunk, N)
        
        # 차폐 조건:
        # 1. 광선이 큐보이드를 관통하는가? (tmin < tmax)
        # 2. 장애물이 타겟 앞에 있는가? (tmax > 0)
        # 3. 장애물이 어그레서보다 앞에 있는가? (tmin < ray_len)
        strict_mask = (tmax_k > 0) & (tmin_k < tmax_k) & (tmin_k < ray_len.unsqueeze(-1)) & mask_k
        
        # 차폐물까지의 거리는 진입점(tmin)을 사용 (0 미만은 0으로 클램핑)
        dist_to_block_masked = torch.where(strict_mask, torch.clamp(tmin_k, min=0.0), torch.tensor(1e5, device=cuboids.device))
        min_dist_to_block = torch.min(dist_to_block_masked, dim=-1)[0]
        # 입체각 투영 신뢰도: 차폐물이 두 넷 사이의 거리에 비해 얼마나 가까이 붙어있는가?
        solid_angle_confidence = torch.exp(-min_dist_to_block / dist_mag)
        
        # 최종 Bleed = 기본 굴절 누수 + 차폐물이 멀리 떨어져 있을 때 발생하는 시야(FOV) 누수
        final_bleed = torch.clamp(fringing_bleed + (1.0 - solid_angle_confidence) * 0.2, min=0.01, max=0.95)
        
        # 완벽히 가려졌을 때(S_total=1.0)라도, 물리적 final_bleed 만큼의 프린지 플럭스는 살아남음
        effective_shielding = 1.0 - S_total * (1.0 - final_bleed)*0.8
        
        total_shield_factor[i:end] = effective_shielding
        
    return total_shield_factor