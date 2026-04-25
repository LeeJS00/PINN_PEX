# src/active_learning/super_def_builder.py
import numpy as np
from collections import defaultdict
import re

from src.preprocessing.def_parser import DefStreamParser
from src.utils.naming import sanitize_name

class SpatialDefIndex:
    """원본 Full-Chip DEF의 모든 라우팅과 인스턴스를 2D Grid에 캐싱하는 초고속 공간 검색 엔진"""
    def __init__(self, def_path, layer_info, tech_lef, cell_lib, bin_size=20.0):
        self.bin_size = bin_size
        self.grid = defaultdict(list)
        self.instances = {}
        self.dbu = 2000.0
        self.cell_lib = cell_lib
        self.def_vias = {}

        print(f">>> [Spatial Index] Parsing Full DEF to build spatial routing map...")
        parser = DefStreamParser(def_path, layer_info, tech_lef, cell_lib)
        self.dbu = parser.dbu
        
        for net_name, _, segments in parser.parse():
            for seg in segments:
                self._add_to_grid(net_name, seg)
                
        self.instances = parser.instances
        self.def_vias = parser.def_vias
        print(f">>> [Spatial Index] Cached {len(self.grid)} spatial bins and {len(self.instances)} instances.")

    def _add_to_grid(self, net_name, seg):
        stype = seg['type']
        if stype == 'WIRE':
            min_x, max_x = min(seg['start'][0], seg['end'][0]), max(seg['start'][0], seg['end'][0])
            min_y, max_y = min(seg['start'][1], seg['end'][1]), max(seg['start'][1], seg['end'][1])
        elif stype == 'VIA':
            min_x = max_x = seg['pos'][0]
            min_y = max_y = seg['pos'][1]
        else:
            return 
            
        min_bx, max_bx = int(min_x // self.bin_size), int(max_x // self.bin_size)
        min_by, max_by = int(min_y // self.bin_size), int(max_y // self.bin_size)
        
        # 딕셔너리 카피(.copy()) 대신 net_name만 주입하여 참조 포인터만 넘김 (메모리 최적화)
        seg['net_name'] = net_name
        
        for bx in range(min_bx, max_bx + 1):
            for by in range(min_by, max_by + 1):
                self.grid[(bx, by)].append(seg)

    def query_window(self, min_x, max_x, min_y, max_y):
        """윈도우 안의 선분들을 고속 Clipping하여 반환 (Zero Length Path 완벽 차단)"""
        min_bx, max_bx = int(min_x // self.bin_size), int(max_x // self.bin_size)
        min_by, max_by = int(min_y // self.bin_size), int(max_y // self.bin_size)
        
        raw_segs = []
        seen = set()
        
        # 1. O(1) 해시 그리드 고속 추출
        for bx in range(min_bx, max_bx + 1):
            for by in range(min_by, max_by + 1):
                if (bx, by) in self.grid:
                    for seg in self.grid[(bx, by)]:
                        sid = id(seg)
                        if sid not in seen:
                            seen.add(sid)
                            raw_segs.append(seg)
                        
        clipped_segs = []
        
        # 2. [SPEED FIX] Deepcopy 제거, 경량화된 Dict 직접 생성
        for seg in raw_segs:
            stype = seg['type']
            if stype == 'WIRE':
                x1, y1 = seg['start']
                x2, y2 = seg['end']
                
                if x1 > x2: x1, x2 = x2, x1
                if y1 > y2: y1, y2 = y2, y1
                
                if x2 < min_x or x1 > max_x or y2 < min_y or y1 > max_y:
                    continue
                    
                cx1, cx2 = max(x1, min_x), min(x2, max_x)
                cy1, cy2 = max(y1, min_y), min(y2, max_y)
                
                if cx2 - cx1 < 1e-4 and cy2 - cy1 < 1e-4:
                    continue
                    
                clipped_segs.append({
                    'type': 'WIRE', 'layer': seg.get('layer'), 'net_name': seg['net_name'],
                    'start': (cx1, cy1), 'end': (cx2, cy2), 'width': seg.get('width')
                })
                
            elif stype == 'VIA':
                vx, vy = seg['pos']
                if min_x <= vx <= max_x and min_y <= vy <= max_y:
                    clipped_segs.append(seg) 
                    
        # 인스턴스 추출
        clipped_insts = {}
        for iname, inst in self.instances.items():
            ctype = inst['cell_type']
            w, h = 1.0, 1.0 
            if self.cell_lib and ctype in self.cell_lib:
                w, h = self.cell_lib[ctype]['size']
                
            if inst['orient'] in ['E', 'W', 'FE', 'FW']:
                w, h = h, w
                
            ix_min, ix_max = inst['x'], inst['x'] + w
            iy_min, iy_max = inst['y'], inst['y'] + h
            
            if ix_max >= min_x and ix_min <= max_x and iy_max >= min_y and iy_min <= max_y:
                clipped_insts[iname] = inst
                
        return clipped_segs, clipped_insts


class SuperDefAssembler:
    """[HYPER-OPTIMIZED] String Buffer를 활용한 무지성 메모리 폭발 억제 어셈블러"""
    def __init__(self, design_name, dbu=2000, def_vias=None):
        self.design_name = design_name
        self.dbu = dbu
        self.def_vias = def_vias if def_vias else {}
        self.power_keywords = {'vss', 'vdd', 'vcc', 'gnd', 'vssx', 'vccx'}
        
        # [SPEED FIX] Dictionary 대신 렌더링이 완료된 String Line을 즉시 캐싱합니다.
        self.instance_lines = []
        self.pin_lines = []
        self.special_nets = defaultdict(list)
        self.regular_nets = defaultdict(list)

    def add_tile(self, tile_idx, target_net, clipped_segs, clipped_insts, shift_x, shift_y):
        pfx = f"s{tile_idx}_"
        
        # 부동소수점 연산 O(N) 최적화: 시프트 배율 사전 계산
        shift_x_dbu = int(round(shift_x * self.dbu))
        shift_y_dbu = int(round(shift_y * self.dbu))
        
        # 1. 인스턴스 렌더링
        for iname, inst in clipped_insts.items():
            ix = int(round(inst['x'] * self.dbu)) + shift_x_dbu
            iy = int(round(inst['y'] * self.dbu)) + shift_y_dbu
            self.instance_lines.append(f"- {pfx}{iname} {inst['cell_type']} + PLACED ( {ix} {iy} ) {inst['orient']} ;")
            
        # 2. 라우팅 렌더링 및 넷 분류
        net_coords = {}
        for seg in clipped_segs:
            orig_net = seg['net_name']
            m_net = f"{pfx}{sanitize_name(orig_net)}"
            
            layer = seg.get('layer') or seg.get('bot_layer') or 'm1'
            is_special = any(k in orig_net.lower() for k in self.power_keywords)
            
            if seg['type'] == 'WIRE':
                x1 = int(round(seg['start'][0] * self.dbu)) + shift_x_dbu
                y1 = int(round(seg['start'][1] * self.dbu)) + shift_y_dbu
                x2 = int(round(seg['end'][0] * self.dbu)) + shift_x_dbu
                y2 = int(round(seg['end'][1] * self.dbu)) + shift_y_dbu
                w_dbu = int(round(seg.get('width', 0.044) * self.dbu))
                
                if is_special:
                    self.special_nets[m_net].append(f"  + ROUTED {layer} {w_dbu} + SHAPE STRIPE ( {x1} {y1} ) ( {x2} {y2} )")
                else:
                    wx2, wy2 = ('*' if x1 == x2 else x2), ('*' if y1 == y2 else y2)
                    self.regular_nets[m_net].append(('WIRE', layer, x1, y1, wx2, wy2))
                    
                if is_special and m_net not in net_coords:
                    net_coords[m_net] = (x1, y1, layer)
                    
            elif seg['type'] == 'VIA':
                vx = int(round(seg['pos'][0] * self.dbu)) + shift_x_dbu
                vy = int(round(seg['pos'][1] * self.dbu)) + shift_y_dbu
                w_dbu = int(round(seg.get('width', 0.044) * self.dbu))
                
                if is_special:
                    self.special_nets[m_net].append(f"  + ROUTED {layer} {w_dbu} + SHAPE STRIPE ( {vx} {vy} ) {seg['name']}")
                else:
                    self.regular_nets[m_net].append(('VIA', layer, vx, vy, seg['name']))
                    
                if is_special and m_net not in net_coords:
                    net_coords[m_net] = (vx, vy, layer)
                    
        # 3. [CRITICAL FIX] The Faraday Cage Demolition (가짜 핀 제거)
        # 오직 Power Net만 핀(Port)을 생성하여 StarRC Ground 덤핑 폭주를 원천 차단합니다.
        for m_net, (px, py, layer) in net_coords.items():
            self.pin_lines.append(
                f"- port_{m_net} + NET {m_net} + DIRECTION INOUT + USE POWER\n"
                f"  + LAYER {layer} ( -1 -1 ) ( 1 1 )\n"
                f"  + PLACED ( {px} {py} ) N ;"
            )

    def stream_def(self, file_path, width, height):
        """[SPEED FIX] OS I/O 병목을 제거한 Chunked String Buffer 스트리밍"""
        with open(file_path, 'w') as f:
            buffer = []
            def flush():
                if buffer:
                    f.write('\n'.join(buffer) + '\n')
                    buffer.clear()

            # Header & DieArea
            buffer.extend([
                "VERSION 5.8 ;", "DIVIDERCHAR \"/\" ;", "BUSBITCHARS \"[]\" ;",
                f"DESIGN {self.design_name} ;",
                f"UNITS DISTANCE MICRONS {int(self.dbu)} ;",
                f"DIEAREA ( 0 0 ) ( {int(width * self.dbu)} {int(height * self.dbu)} ) ;\n"
            ])
            
            # Vias
            if self.def_vias:
                buffer.append(f"VIAS {len(self.def_vias)} ;")
                for v_name, v_data in self.def_vias.items():
                    buffer.append(f"- {v_name}")
                    buffer.extend([f"  {line}" for line in v_data['raw_lines']])
                    buffer.append(" ;")
                buffer.append("END VIAS\n")
            flush()

            # Components
            buffer.append(f"COMPONENTS {len(self.instance_lines)} ;")
            buffer.extend(self.instance_lines)
            buffer.append("END COMPONENTS\n")
            flush()
            
            # Pins
            buffer.append(f"PINS {len(self.pin_lines)} ;")
            buffer.extend(self.pin_lines)
            buffer.append("END PINS\n")
            flush()
            
            # Special Nets
            if self.special_nets:
                buffer.append(f"SPECIALNETS {len(self.special_nets)} ;")
                for net, lines in self.special_nets.items():
                    buffer.append(f"- {net}")
                    buffer.extend(lines)
                    buffer.append(" + USE POWER ;")
                buffer.append("END SPECIALNETS\n")
            flush()

            # Regular Nets
            if self.regular_nets:
                buffer.append(f"NETS {len(self.regular_nets)} ;")
                for net, segs in self.regular_nets.items():
                    buffer.append(f"- {net}")
                    first_seg = True
                    prev_pt = None
                    
                    for seg in segs:
                        if seg[0] == 'WIRE':
                            _, layer, x1, y1, wx2, wy2 = seg
                            pfx = "  + ROUTED" if first_seg else "    NEW"
                            buffer.append(f"{pfx} {layer} ( {x1} {y1} ) ( {wx2} {wy2} )")
                            first_seg = False
                            prev_pt = (wx2 if wx2 != '*' else x1, wy2 if wy2 != '*' else y1)
                        else:
                            _, layer, vx, vy, vname = seg
                            if not first_seg and prev_pt == (vx, vy):
                                buffer[-1] += f" {vname}"
                            else:
                                pfx = "  + ROUTED" if first_seg else "    NEW"
                                buffer.append(f"{pfx} {layer} ( {vx} {vy} ) {vname}")
                                first_seg = False
                            prev_pt = (vx, vy)
                            
                    buffer.append(" ;")
                    if len(buffer) > 10000: flush()
                    
                buffer.append("END NETS\n")
            
            flush()
            buffer.append("END DESIGN")
            flush()