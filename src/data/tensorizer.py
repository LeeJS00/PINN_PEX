# src/data/tensorizer.py
import numpy as np
from src.physics.materials import BEOLMaterialStack

class FeatureTensorizer:
    """
    Raw Geometry(Cuboids)를 모델 입력용 (N, 9) Tensor로 변환합니다.
    """
    def __init__(self, material_stack: BEOLMaterialStack):
        self.mat_stack = material_stack

    def process(self, cuboids: np.ndarray, type_ids: np.ndarray, window_center: np.ndarray) -> np.ndarray:
        """
        Returns:
            tensor: (N, 9) 
            [0:3] x_rel, y_rel, z_abs
            [3:6] w, h, d
            [6]   semantic_type (1.0=Wire, 0.5=Pin)
            [7]   logic_flag (1.0=Target, 0.0=Aggressor)
            [8]   epsilon
        """
        if len(cuboids) == 0:
            return np.zeros((0, 9), dtype=np.float32)

        N = len(cuboids)
        tensor = np.zeros((N, 9), dtype=np.float32)

        # 1. Geometry
        tensor[:, 0] = cuboids[:, 0] - window_center[0] # Rel X
        tensor[:, 1] = cuboids[:, 1] - window_center[1] # Rel Y
        tensor[:, 2] = cuboids[:, 2]                    # Abs Z
        tensor[:, 3:6] = cuboids[:, 3:6]                # W, H, D

        # 2. Semantic & Logic Flags
        # type_ids mapping from build_dataset:
        # 1: Target Wire, 2: Target Pin, 3: Aggr Wire, 4: Aggr Pin
        
        is_wire = (type_ids == 1) | (type_ids == 3)
        is_pin  = (type_ids == 2) | (type_ids == 4)
        is_target = (type_ids == 1) | (type_ids == 2)
        
        tensor[is_wire, 6] = 1.0  # Semantic: Wire
        tensor[is_pin, 6]  = 0.5  # Semantic: Pin
        tensor[is_target, 7] = 1.0 # Logic: Target
        # Aggressor logic flag remains 0.0

        # 3. Material Property (Epsilon)
        z_centers = tensor[:, 2] 
        tensor[:, 8] = self.mat_stack.get_permittivity_bulk(z_centers)
        
        return tensor