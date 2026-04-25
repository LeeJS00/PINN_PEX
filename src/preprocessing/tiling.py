# src/preprocessing/tiling.py
import numpy as np
from typing import List, Tuple, Dict

class NetTiler:
    def __init__(self, window_size: Tuple[float, float, float], overlap: float = 0.0):
        """
        Robust Tiling (Partition of Unity)
        - 파라미터로 들어오는 overlap은 무시하고, 항상 50% Overlap (Triangle Weighting)을 적용합니다.
        - 이는 물리량 보존(Sum=1.0)을 위한 필수 조건입니다.
        """
        self.win_w, self.win_h, self.win_d = window_size
        
        # 50% Overlap 강제 설정 (Bartlett Windowing)
        # Stride = Window / 2
        self.stride_x = self.win_w / 2.0
        self.stride_y = self.win_h / 2.0
        
        # 가중치가 0이 되는 거리 (Radius) = Window / 2
        # 즉, 윈도우 끝에서 정확히 0이 됨.
        self.radius_x = self.win_w / 2.0
        self.radius_y = self.win_h / 2.0

    def get_tiling_windows(self, target_cuboids: np.ndarray) -> List[np.ndarray]:
        """
        Target Net의 형상을 분석하여, 전체를 커버할 수 있는 최소한의 윈도우 중심점들을 반환합니다.
        
        Args:
            target_cuboids: (N, 6) Array [cx, cy, cz, sx, sy, sz]
        Returns:
            List of center points [(x, y, z), ...]
        """
        if len(target_cuboids) == 0:
            return []

        # 1. 전체 Bounding Box 계산
        mins = target_cuboids[:, :3] - target_cuboids[:, 3:6] / 2
        maxs = target_cuboids[:, :3] + target_cuboids[:, 3:6] / 2

        net_min = np.min(mins, axis=0)
        net_max = np.max(maxs, axis=0)
        
        # 1.1 Adaptive Single Tiling Check
        net_span = net_max - net_min
        # 안전 마진(epsilon)을 둬서 부동소수점 오차 방지
        if net_span[0] <= self.win_w - 1e-3 and net_span[1] <= self.win_h - 1e-3:
            # Net이 Window보다 작음 -> Center에 하나만 배치
            center = (net_min + net_max) / 2.0
            return [center]

        # 2. Grid Generation Helper
        def get_grid_points(min_val, max_val, stride):
            span = max_val - min_val
            # A. Net 크기가 Stride보다 작으면 -> 정중앙에 하나만 배치 (가중치 손실 방지)
            if span < stride:
                return [(min_val + max_val) / 2.0]
            
            # B. Net 크기가 크면 -> min부터 시작해서 max를 덮을 때까지 확장
            # Bartlett Window 특성상 마지막 윈도우 중심이 max보다 뒤에 있어야 끝부분 Sum=1.0이 유지됨
            points = []
            curr = min_val
            while curr <= max_val + 1e-9: # max를 포함하거나 넘어설 때까지
                points.append(curr)
                curr += stride
            
            # 만약 마지막 포인트가 max보다 작아서 커버가 안됐다면 하나 추가
            # (위 while문 조건상 max보다 작을 때 루프를 돌므로, 마지막 points[-1]은 max 근처거나 클 것임)
            # 안전장치: 마지막 윈도우가 Net의 끝을 커버하지 못하는 경우 (거의 없겠지만) 추가
            if points[-1] < max_val:
                points.append(points[-1] + stride)
                
            return points

        # X축, Y축 Grid 생성
        xs = get_grid_points(net_min[0], net_max[0], self.stride_x)
        ys = get_grid_points(net_min[1], net_max[1], self.stride_y)

        # 3. Z축은 Tiling하지 않음 (Full Stack)
        center_z = (net_min[2] + net_max[2]) / 2.0
        
        windows = []
        # Create Grid Points
        for cx in xs:
            for cy in ys:
                # 4. Validity Check (Empty Window 제거)
                win_min_x, win_max_x = cx - self.radius_x, cx + self.radius_x
                win_min_y, win_max_y = cy - self.radius_y, cy + self.radius_y
                
                # Check intersection with Net bbox (Simple AABB)
                if (win_min_x < net_max[0] and win_max_x > net_min[0] and
                    win_min_y < net_max[1] and win_max_y > net_min[1]):
                     
                     # Refined Check with Cuboids
                     has_overlap = False
                     for i in range(len(target_cuboids)):
                         cmins = mins[i]
                         cmaxs = maxs[i]
                         if (cmins[0] < win_max_x and cmaxs[0] > win_min_x and
                            cmins[1] < win_max_y and cmaxs[1] > win_min_y):
                            has_overlap = True
                            break
                     
                     if has_overlap:
                        windows.append(np.array([cx, cy, center_z]))

        return windows

    def get_overlapping_instances(self, all_instances: Dict, center: np.ndarray) -> Dict:
        """
        윈도우 영역과 겹치는 Instance 목록을 반환합니다.
        """
        local_insts = {}
        
        # Window Global Box
        win_half_w = self.win_w / 2.0
        win_half_h = self.win_h / 2.0
        
        wx1 = center[0] - win_half_w
        wy1 = center[1] - win_half_h
        wx2 = center[0] + win_half_w
        wy2 = center[1] + win_half_h
        # print(f" Checking overlapping instances for window box: ({wx1}, {wy1}) - ({wx2}, {wy2})")
        for inst_name, inst in all_instances.items():
            ix1, iy1, ix2, iy2 = inst['rect']
            if not (ix2 <= wx1 or ix1 >= wx2 or iy2 <= wy1 or iy1 >= wy2):
                # deep copy and adjust to local coords
                local_insts[inst_name] = inst.copy()
                local_insts[inst_name]['x'] -= center[0]
                local_insts[inst_name]['y'] -= center[1]
        
        return local_insts

    def clip_cuboids(self, all_cuboids: np.ndarray, all_types: np.ndarray, center: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # (기존 코드와 동일)
        win_half = np.array([self.win_w, self.win_h, self.win_d]) / 2.0
        win_min = center - win_half
        win_max = center + win_half
        c_mins = all_cuboids[:, :3] - all_cuboids[:, 3:6] / 2
        c_maxs = all_cuboids[:, :3] + all_cuboids[:, 3:6] / 2
        inter_min = np.maximum(c_mins, win_min)
        inter_max = np.minimum(c_maxs, win_max)
        dims = inter_max - inter_min
        valid_mask = np.all(dims > 1e-9, axis=1)
        if not np.any(valid_mask): return None, None
        valid_dims = dims[valid_mask]
        valid_mins = inter_min[valid_mask]
        valid_types = all_types[valid_mask]
        new_centers = valid_mins + valid_dims / 2
        new_sizes = valid_dims
        result_cuboids = np.concatenate([new_centers, new_sizes], axis=1)
        return result_cuboids, valid_types
    
    def calculate_stitching_weight(self, local_segments: List[Dict]) -> float:
        return 1.0
        """
        현재 윈도우(Tile)에 포함된 Target Net Segments의 '유효 기여도'를 계산합니다.
        
        Weight = (Weighted Length Sum) / (Total Length Sum)
        
        이 값은 나중에 Inference 시 PredCap * Weight 로 사용됩니다.
        """
        total_len = 0.0
        weighted_len = 0.0
        # Assume effective via length to cover short wire but has many vias
        VIA_EFF_LEN = 0.1
        
        for seg in local_segments:
            if seg.get('tag') != 'target': continue
            segments_to_process = []

            if seg['type'] == 'WIRE':
                # 원본 좌표 (Relative to Window Center)
                p1 = np.array(seg['start'])
                p2 = np.array(seg['end'])
                
                # 1. Split at X=0 (Y-axis crossing)
                # X좌표의 부호가 다르면 교차
                if (p1[0] < 0 < p2[0]) or (p2[0] < 0 < p1[0]):
                    # 교차점 계산 (Linear Interpolation)
                    t = (0 - p1[0]) / (p2[0] - p1[0])
                    y_at_0 = p1[1] + t * (p2[1] - p1[1])
                    mid = np.array([0.0, y_at_0])
                    segments_to_process.append((p1, mid))
                    segments_to_process.append((mid, p2))
                else:
                    segments_to_process.append((p1, p2))
                
                # 2. Split at Y=0 (X-axis crossing)
                final_pieces = []
                for sp1, sp2 in segments_to_process:
                    if (sp1[1] < 0 < sp2[1]) or (sp2[1] < 0 < sp1[1]):
                        # 교차점 계산
                        t = (0 - sp1[1]) / (sp2[1] - sp1[1])
                        x_at_0 = sp1[0] + t * (sp2[0] - sp1[0])
                        mid = np.array([x_at_0, 0.0])
                        final_pieces.append((sp1, mid))
                        final_pieces.append((mid, sp2))
                    else:
                        final_pieces.append((sp1, sp2))
                
                # 3. Calculate Weight for each piece
                for sp1, sp2 in final_pieces:
                    length = np.linalg.norm(sp2 - sp1)
                    if length < 1e-9: continue
                    
                    # 이제 각 조각은 0을 건너뛰지 않으므로(Linear Slope), 중점 근사가 정확함
                    mid = (sp1 + sp2) / 2.0
                    w = self._get_spatial_weight(mid[0], mid[1])
                    
                    total_len += length
                    weighted_len += length * w
                
            elif seg['type'] == 'RECT':
                # RECT는 Effective Length로 환산 (Max Dim)
                # RECT가 작으므로 Splitting 없이 중심 가중치 사용 (속도/복잡도 Trade-off)
                r = seg['rect']
                w_rect = abs(r[2] - r[0])
                h_rect = abs(r[3] - r[1])
                eff_len = max(w_rect, h_rect)
                if eff_len < 1e-9: continue
                
                cx, cy = (r[0]+r[2])/2.0, (r[1]+r[3])/2.0
                w = self._get_spatial_weight(cx, cy)
                
                total_len += eff_len
                weighted_len += eff_len * w
            
            elif seg['type'] == 'VIA':
                # VIA는 Effective Length로 환산 (Max Dim)
                vx, vy = seg['pos']
                w = self._get_spatial_weight(vx, vy)
                total_len += VIA_EFF_LEN
                weighted_len += VIA_EFF_LEN * w
                
        if total_len == 0: return 1.0
        return weighted_len / total_len
    
    def _get_spatial_weight(self, x, y):
        """
        [CRITICAL FIX] Triangle Kernel 폐기 -> Voronoi Cell Box Kernel 도입
        - 경계선에서 가중치가 0으로 수렴하여 전하(Capacitance)가 증발하는 현상 방지.
        - Stride의 절반 영역(Tile Core) 안에 들어온 엣지는 무조건 가중치 1.0 할당.
        - 영역을 벗어난 엣지는 0.0 처리 (인접 타일이 1.0으로 독점하도록 위임).
        """
        # self.stride_x = win_w / 2.0 (50% overlap 기준)
        # 독점 영역(Ownership Cell)은 [-stride/2, +stride/2]
        core_x = self.stride_x / 2.0
        core_y = self.stride_y / 2.0
        
        if (-core_x <= x < core_x) and (-core_y <= y < core_y):
            return 1.0
        return 0.0