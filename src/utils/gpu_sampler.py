# src/utils/gpu_sampler.py
import torch
import torch.nn.functional as F
import numpy as np

class PhysicsSampler:
    """
    Physics-Informed Neural Field를 위한 공간 샘플링 엔진.
    Target Surface(BC용)와 Volume Space(Energy Loss용) 점들을 추출합니다.
    """
    def __init__(self, material_stack, device='cuda'):
        self.mat_stack = material_stack
        self.device = device
        self._init_gpu_materials()

    def _init_gpu_materials(self):
        # [NEW] Numpy 연산을 걷어내기 위해 Z-boundary와 Epsilon을 GPU 텐서로 캐싱
        # material_stack 내부 구조에 따라 다르겠지만, 일반적인 유전체 층 Z 경계를 추출합니다.
        # (아래는 예시이며, 실제 mat_stack.get_permittivity_bulk 로직을 Torch로 포팅한 것입니다)
        z_vals = np.linspace(-10.0, 50.0, 2000)
        eps_vals = self.mat_stack.get_permittivity_bulk(z_vals)
        eps_vals = np.nan_to_num(eps_vals, nan=1.0)
        eps_vals[eps_vals < 1.0] = 1.0 # 우주에 유전율이 0인 곳은 없습니다.
        
        self.z_bins = torch.tensor(z_vals, dtype=torch.float32, device=self.device)
        self.eps_bins = torch.tensor(eps_vals, dtype=torch.float32, device=self.device)


    def sample_for_ssl(self, cuboids, mask, num_samples=1024):
        B, N, _ = cuboids.shape
        
        n_tgt = int(num_samples * 0.4)
        n_aggr = int(num_samples * 0.4)
        n_far = num_samples - n_tgt - n_aggr
        
        is_target = (cuboids[..., 7] == 1.0)
        is_aggr = (cuboids[..., 7] == 0.0)
        
        # [CRITICAL FIX] Volume 대신 진짜 Aggressor 표면을 샘플링!
        # coords_tgt, _, _, _, valid_tgt  = self._sample_surfaces_with_info(cuboids, is_target, n_tgt)
        # coords_aggr, _, _, _, valid_aggr = self._sample_surfaces_with_info(cuboids, is_aggr, n_aggr)
        tgt_mask = is_target & ~mask
        aggr_mask = is_aggr & ~mask
        
        coords_tgt, _, _, _, valid_tgt  = self._sample_surfaces_with_info(cuboids, tgt_mask, n_tgt)
        coords_aggr, _, _, _, valid_aggr = self._sample_surfaces_with_info(cuboids, aggr_mask, n_aggr)
        coords_far = self._sample_far_field(cuboids, n_far)
        
        coords = torch.cat([coords_tgt, coords_aggr, coords_far], dim=1) 
        
        # Types: 1.0(Target), 0.0(Aggressor), -1.0(Far-field)
        # 이제 0.0은 진짜 0V가 되어야 하는 물리적 도체 표면입니다.
        t_tgt = torch.ones(B, n_tgt, device=self.device)
        t_tgt[~valid_tgt] = -2.0 
        
        t_aggr = torch.zeros(B, n_aggr, device=self.device)
        t_aggr[~valid_aggr] = -2.0 
        
        t_far = torch.full((B, n_far), -1.0, device=self.device)
        
        types = torch.cat([t_tgt, t_aggr, t_far], dim=1)
        
        flat_z = coords[:, :, 2].contiguous() 
        indices = torch.bucketize(flat_z, self.z_bins).clamp(0, len(self.eps_bins)-1)
        epsilon = self.eps_bins[indices].unsqueeze(-1) 
        
        return coords, types, epsilon

    def sample_points_for_finetune(self, cuboids, num_samples=1024):
        """
        [NEW] Finetuning 및 Inference용 (Finetuner에서 호출).
        Gauss Law 적분을 위해 Surface Point, Normal, Area를 모두 반환합니다.
        Target Net의 표면에서만 샘플링합니다.
        
        Returns: Dictionary
            - target_points: (B, N, 3)
            - target_normals: (B, N, 3)
            - target_areas: (B, N, 1) -> 각 점이 대표하는 면적 (TotalArea / N)
        """
        is_target = (cuboids[..., 7] == 1.0)
        
        # Surface Sampling with Normals & Areas
        coords, normals, total_areas, cuboid_indices, _ = self._sample_surfaces_with_info(cuboids, is_target, num_samples)
        
        # 각 포인트의 미소 면적(dA) = 전체 면적 / 샘플 수
        # total_areas: (B, 1) -> (B, N, 1)
        dA = total_areas.unsqueeze(1).expand(-1, num_samples, 1) / num_samples
        
        # Epsilon Lookup
        flat_coords = coords.reshape(-1, 3)
        eps_vals = self.mat_stack.get_permittivity_bulk(flat_coords[:, 2].detach().cpu().numpy())
        epsilon = torch.from_numpy(eps_vals).to(self.device).reshape(coords.shape[0], num_samples, 1)
        
        return {
            'target_points': coords,
            'target_normals': normals,
            'target_areas': dA,
            'target_eps': epsilon,
            'point_to_cuboid_idx': cuboid_indices
        }

    def _sample_surfaces_with_info(self, cuboids, mask, n_points):
        """
        Target Cuboid 표면 위 점, 법선 벡터, 총 면적 계산
        """
        B, N, _ = cuboids.shape
        device = cuboids.device
        
        # Masking: Target이 아닌 Cuboid는 사이즈를 0으로 만들어 선택되지 않게 함
        masked_cuboids = cuboids.clone()
        # mask shape: (B, N) -> (B, N, 9)로 확장해서 곱하거나 인덱싱
        # 간단히 w, h, d (idx 3,4,5)를 0으로 만듦
        mask_expanded = mask.unsqueeze(-1).expand_as(masked_cuboids)
        # 마스크가 False인 곳의 w,h,d를 0으로
        masked_cuboids[~mask_expanded] = 0.0
        
        centers = masked_cuboids[..., :3]
        sizes = masked_cuboids[..., 3:6]
        
        # 표면적 계산: 2*(wh + wd + hd)
        # (B, N)
        face_areas = 2 * (sizes[..., 0]*sizes[..., 1] + sizes[..., 0]*sizes[..., 2] + sizes[..., 1]*sizes[..., 2])
        
        # 배치별 총 표면적 (B,)
        total_batch_area = face_areas.sum(dim=1)
        
        # 샘플링 확률 분포 (B, N)
        # total_batch_area가 0인 경우(Target 없음) 방어
        valid_batch = (total_batch_area > 1e-9)
        safe_total = total_batch_area.clone()
        safe_total[~valid_batch] = 1.0
        probs = face_areas / safe_total.unsqueeze(1)
        probs[~valid_batch, 0] = 1.0 
        
        cuboid_indices = torch.multinomial(probs, n_points, replacement=True)
        
        # Gather Selected Info
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, n_points)
        sel_centers = centers[batch_idx, cuboid_indices] # (B, n, 3)
        sel_sizes = sizes[batch_idx, cuboid_indices]     # (B, n, 3)
        edge_margin = 0.02
        # 1. Random Point in [-0.5, 0.5]^3
        rand_pts = (torch.rand(B, n_points, 3, device=device) - 0.5) * (1.0 - 2 * edge_margin)
        
        # 2. Project to nearest face to get Position & Normal
        # 절대값이 가장 큰 축 찾기 (0=x, 1=y, 2=z)
        abs_vals = torch.abs(rand_pts)
        max_vals, max_idx = torch.max(abs_vals, dim=-1, keepdim=True)
        
        # 해당 축의 부호 (+1 or -1)
        signs = torch.sign(rand_pts.gather(-1, max_idx))
        
        # Mask for the max axis (One-hot)
        mask_axis = torch.zeros_like(rand_pts).scatter_(-1, max_idx, 1.0)
        
        # Project Position: (Original) * (1-Mask) + (Sign*0.5) * Mask
        # 선택된 축은 표면(0.5)으로 이동, 나머지는 유지
        proj_pts = rand_pts * (1 - mask_axis) + (signs * 0.5) * mask_axis
        
        # Calculate Real World Coordinates
        final_pts = sel_centers + proj_pts * sel_sizes        
        # Calculate Normals
        # 큐보이드 표면의 법선은 축 방향 단위 벡터임 (e.g., [1, 0, 0])
        # mask_axis가 축을 나타내고, signs가 방향을 나타냄
        normals = mask_axis * signs
        
        inflation_delta = 1e-4  # 0.0001 um = 0.1 nm
        final_pts = final_pts + normals * inflation_delta
        
        # [반환값 추가] 이 배치가 유효한지(valid_batch)를 리턴
        return final_pts, normals, total_batch_area.unsqueeze(1), cuboid_indices, valid_batch

    def _sample_near_volume(self, surf_pts, n_points):
        """Surface Points에 노이즈를 주어 Near Volume 생성"""
        # surf_pts가 (B, N_surf, 3)일 수 있음. n_points에 맞춰 리샘플링 혹은 슬라이싱
        if surf_pts.shape[1] >= n_points:
            base_pts = surf_pts[:, :n_points, :]
        else:
            # 부족하면 반복
            repeat_cnt = (n_points // surf_pts.shape[1]) + 1
            base_pts = surf_pts.repeat(1, repeat_cnt, 1)[:, :n_points, :]
            
        noise_scale = torch.rand(base_pts.shape, device=self.device) * 0.5 # 0 ~ 0.5um
        noise = torch.randn_like(base_pts) * noise_scale
        return base_pts + noise

    def _sample_far_field(self, cuboids, n_points):
        """윈도우 전체 범위 내 무작위 샘플링"""
        B = cuboids.shape[0]
        # Window Size 가정이 필요함. (보통 +/- 2.0 um)
        win_size = 2.0 
        return (torch.rand(B, n_points, 3, device=cuboids.device) * 2 - 1) * win_size

# Standalone function wrapper for external import
def sample_points_for_finetune(cuboids, num_samples=1024):
    """
    Module-level wrapper compatible with Finetuner.
    Instantiates a temporary sampler (assuming MaterialStack handled inside or not needed for Normals).
    Note: Epsilon is not needed for Normal/Area calculation, only for PhysicsHead which handles it.
    But PhysicsSampler needs MaterialStack for init.
    """
    # 임시 MaterialStack (Dummy) - Finetuning시 샘플링엔 유전율 필요 없음 (PhysicsHead가 처리)
    class DummyStack:
        def get_permittivity_bulk(self, z): return np.ones_like(z) * 3.9
        
    sampler = PhysicsSampler(DummyStack(), device=cuboids.device)
    return sampler.sample_points_for_finetune(cuboids, num_samples)