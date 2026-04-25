# src/preprocessing/layer_parser.py
import re
from typing import Dict

class LayerInfoParser:
    def __init__(self, info_path):
        self.info_path = info_path
        self.layer_map = {} 

    def parse(self) -> Dict:
        try:
            with open(self.info_path, 'r') as f:
                for line in f:
                    parts = line.split()
                    if not parts: continue
                    
                    if line.startswith("GRD"):
                        if len(parts) < 12: continue
                        name = parts[1].lower()
                        l_type = parts[2].upper()
                        
                        try:
                            # 임시값 저장 (나중에 DB에서 덮어씀)
                            bottom_z = float(parts[10])
                            top_z = float(parts[11])
                            epsilon = float(parts[12]) if len(parts) > 12 else 3.9
                            thickness = top_z - bottom_z
                            
                            if name not in self.layer_map: self.layer_map[name] = {}
                            self.layer_map[name].update({
                                'z_pos': bottom_z, 'thickness': thickness,
                                'top_z': top_z, 'type': l_type, 'epsilon': epsilon
                            })
                        except ValueError: continue
                            
                    elif line.startswith("DB"):
                        name = parts[1].lower()
                        target_name = parts[-1].lower()
                        
                        try:
                            # [CRITICAL FIX] DB 섹션의 4번째 값(Index 3)이 진짜 $lvl 입니다!
                            true_lvl_idx = int(parts[3])
                            res_val = float(parts[6])
                            
                            if name not in self.layer_map: self.layer_map[name] = {}
                            self.layer_map[name].update({
                                'resistance': res_val,
                                'lvl_idx': true_lvl_idx # 진짜 레벨 인덱스
                            })
                            
                            if target_name in self.layer_map and target_name != name:
                                for k, v in self.layer_map[target_name].items():
                                    if k not in self.layer_map[name]:
                                        self.layer_map[name][k] = v
                                # [FIX] 반대로 name(v1)에만 있는 정보(lvl_idx 등)를 target(via1)에도 복사
                                for k, v in self.layer_map[name].items():
                                    if k not in self.layer_map[target_name]:
                                        self.layer_map[target_name][k] = v

                        except (ValueError, IndexError):
                            continue

                for name, info in self.layer_map.items():
                    # 도체이거나 유전율이 0.0으로 파싱된 경우
                    if info.get('type') in ['C', 'V'] or info.get('epsilon', 0.0) <= 0.0:
                        c_z_mid = (info.get('z_pos', 0.0) + info.get('top_z', 0.0)) / 2.0
                        matched_eps = 3.9 # 진공/SiO2 Fallback
                        
                        for d_name, d_info in self.layer_map.items():
                            if d_info.get('type') == 'D' and d_info.get('epsilon', 0.0) > 0.0:
                                # Z 높이 중심점이 절연체의 영역 안에 들어오는지 확인
                                if d_info.get('z_pos', 0.0) - 1e-3 <= c_z_mid <= d_info.get('top_z', 0.0) + 1e-3:
                                    matched_eps = d_info.get('epsilon')
                                    # 'm1'이 'fill_m1'에 포함되는 등 이름적 연관성이 있으면 최우선 확정
                                    if name in d_name:
                                        break
                        self.layer_map[name]['epsilon'] = matched_eps

            return self.layer_map
        except FileNotFoundError: pass