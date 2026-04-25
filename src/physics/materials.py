# src/physics/materials.py
import numpy as np
import torch
from typing import Dict, Union

class BEOLMaterialStack:
    """
    Back-End-Of-Line (BEOL) Layer Stackup 정보를 관리합니다.
    LayerInfoParser에서 파싱된 딕셔너리를 입력받아,
    Z좌표 기반으로 유전율(Permittivity)을 조회할 수 있게 합니다.
    """
    def __init__(self, layer_map: Dict):
        self.layer_map = layer_map
        # Interval List: [(bottom_z, top_z, epsilon), ...]
        self.intervals = []
        for name, info in layer_map.items():                
            bot = info['z_pos']
            top = info.get('top_z', bot + info['thickness'])
            eps = info.get('epsilon', 3.9) 
            self.intervals.append((bot, top, eps))
            
        self.intervals.sort(key=lambda x: x[0])
        self.np_intervals = np.array(self.intervals, dtype=np.float32)

    def get_permittivity_bulk(self, z_values: Union[np.ndarray, float]) -> np.ndarray:
        if np.isscalar(z_values):
            z_values = np.array([z_values], dtype=np.float32)
            return_scalar = True
        else:
            return_scalar = False
            
        z_values = np.asarray(z_values, dtype=np.float32)
        eps_values = np.full_like(z_values, 3.9) # Default SiO2
        
        z_col = z_values[:, None]
        bots = self.np_intervals[:, 0]
        tops = self.np_intervals[:, 1]
        epss = self.np_intervals[:, 2]
        
        # bottom <= z < top
        mask = (z_col >= bots) & (z_col < tops)
        
        has_match = mask.any(axis=1)
        match_idx = mask.argmax(axis=1)
        
        eps_values[has_match] = epss[match_idx[has_match]]
        
        if return_scalar:
            return eps_values[0]
        return eps_values