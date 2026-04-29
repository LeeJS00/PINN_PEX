# src/physics/materials.py
import torch
import torch.nn as nn
import numpy as np
from utils.parse_layer_info import parse_layer_info
import config as cfg

class BEOLMaterialStack(nn.Module):
    """
    layers.info 파일을 읽어 Z축 위치에 따른 유전율(Permittivity)을 반환하는 모듈.
    Neural Field 학습 시, 공간상의 점이 어떤 유전체 안에 있는지 판별하는 데 사용됨.
    """
    def __init__(self, layer_info_path=None, device='cpu'):
        super().__init__()
        self.device = device
        
        # 경로가 없으면 config에서 가져옴
        if layer_info_path is None:
            layer_info_path = cfg.LAYER_INFO_PATH
            
        # 1. 파일 파싱
        raw_info = parse_layer_info(layer_info_path)
        
        # 2. 유전체(Dielectric, type='D') 레이어만 추출하여 리스트 구성
        # (Start Z, End Z, Epsilon)
        stack_list = []
        
        # 딕셔너리를 Z_pos 순으로 정렬
        sorted_layers = sorted(raw_info.items(), key=lambda x: x[1]['z_pos'])
        
        for name, info in sorted_layers:
            # Type이 'D'인 것만 배경 유전율로 사용
            # (Conductor 'C'는 도체 내부이므로 전위가 등전위가 되어 E=0, 
            #  하지만 주변부 계산을 위해 해당 높이의 fill 유전체 정보가 중요함)
            if info['type'] == 'D':
                stack_list.append((info['z_pos'], info['top_z'], info['epsilon']))
                
        if not stack_list:
            print("WARNING: No dielectric layers found in info file. Defaulting to Vacuum.")
            stack_list.append((-100.0, 100.0, 1.0))

        # 3. Tensor 변환 (Broadcasting을 위해 shape 맞춤)
        # Shape: (Num_Layers,)
        self.z_starts = torch.tensor([l[0] for l in stack_list], device=device, dtype=torch.float32)
        self.z_ends = torch.tensor([l[1] for l in stack_list], device=device, dtype=torch.float32)
        self.eps_vals = torch.tensor([l[2] for l in stack_list], device=device, dtype=torch.float32)
        
        # 디버깅용 출력
        # print(f"Initialized BEOL Stack with {len(stack_list)} dielectric layers on {device}.")

    def get_permittivity(self, points):
        """
        Input:
            points: (B, N, 3) or (N, 3) - (x, y, z) coordinates
            
        Output:
            epsilon: (B, N, 1) or (N, 1) - Relative permittivity at those points
        """
        # 차원 정리
        original_shape = points.shape[:-1] # (B, N)
        z_coords = points[..., 2].reshape(-1) # (Total_Points, )
        
        # 결과 텐서 (기본값: 진공 = 1.0)
        epsilon = torch.ones_like(z_coords)
        
        # Vectorized Interval Search
        # 레이어 개수가 적으므로(보통 <20개), Loop가 Binary Search보다 GPU에서 빠를 수 있음.
        for zs, ze, eps in zip(self.z_starts, self.z_ends, self.eps_vals):
            # 해당 레이어 범위 안에 있는 포인트 마스킹
            mask = (z_coords >= zs) & (z_coords < ze)
            epsilon[mask] = eps
            
        return epsilon.reshape(*original_shape, 1)

    def to(self, device):
        # device 이동 지원
        self.z_starts = self.z_starts.to(device)
        self.z_ends = self.z_ends.to(device)
        self.eps_vals = self.eps_vals.to(device)
        self.device = device
        return self