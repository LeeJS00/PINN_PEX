# src/utils/mini_def_writer.py
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

from src.utils.naming import sanitize_name


class MiniDefWriter:
    def __init__(self, layer_info: Dict, tech_lef: Dict = None, cell_lib: Dict = None, inst_net_map: Dict = None, dbu: int = 2000, window_size: Tuple = (4.0, 4.0)):
        self.layer_info = layer_info
        self.tech_lef = tech_lef
        self.cell_lib = cell_lib
        self.inst_net_map = inst_net_map
        self.net_pin_map = defaultdict(set)  # net_name -> Set of pin names
        self.net_conn_map = defaultdict(set) # net_name -> Set of connected instance names
        self.dbu = dbu
        self.win_w, self.win_h = window_size[:2]

        # Z-map (Layer Name Mapping)
        self.z_map = {}
        for name, info in self.layer_info.items():
            center_z = info['z_pos'] + info['thickness'] / 2.0
            k = round(center_z, 4)
            self.z_map[k] = name

    def _is_special_net(self, net_name: str) -> bool:
        """이름을 기반으로 Special Net(Power/Ground) 여부 판단"""
        name_upper = net_name.upper()
        # VSS, VCC, VDD, GND, POWER, GROUND 등이 포함되면 Special로 간주
        keywords = ['VSS', 'VCC', 'VDD', 'GND', 'AVSS', 'AVDD']
        return any(k in name_upper for k in keywords)
    
    def _get_use_type(self, net_name: str) -> str:
        """NET의 용도(USE) 결정"""
        name_upper = net_name.upper()
        if 'VSS' in name_upper or 'GND' in name_upper: return "GROUND"
        if 'VCC' in name_upper or 'VDD' in name_upper: return "POWER"
        return "SIGNAL"

    def write(self, segments_data: List[Dict], instances_data: Dict, target_net_name: str, context_size=None) -> str:
        """
        segments_data: List of dicts. Now includes 'net_name' for aggressors.
        """
        self.net_pin_map.clear()
        self.net_conn_map.clear()

        # 1. Header & DIEAREA
        if context_size is not None:
            die_w, die_h = context_size[:2]
        else:
            die_w, die_h = self.win_w, self.win_h
            
        die_w_dbu = int(round(die_w * self.dbu))
        die_h_dbu = int(round(die_h * self.dbu))
        
        offset_x = die_w / 2.0
        offset_y = die_h / 2.0
        safe_name = sanitize_name(target_net_name)
        design_name = f"mini_{safe_name}"[:100]
        # A. Header
        lines = [
            f"VERSION 5.8 ;",
            f"DIVIDERCHAR \"/\" ;",
            f"BUSBITCHARS \"[]\" ;",
            f"DESIGN {design_name} ;",
            f"UNITS DISTANCE MICRONS {self.dbu} ;",
            f"DIEAREA ( 0 0 ) ( {die_w_dbu} {die_h_dbu} ) ;",
            ""
        ]
        
        # 2. COMPONENTS Section
        self.instances_data = instances_data
        if self.instances_data:
            lines.append(f"COMPONENTS {len(self.instances_data)} ;")
            for inst_name, inst in self.instances_data.items():
                x = inst['x'] + offset_x
                y = inst['y'] + offset_y
                cell_name = inst['cell_type']
                orient = inst['orient']
                lines.append(f"- {inst_name} {cell_name} + PLACED ( {int(round(x * self.dbu))} {int(round(y * self.dbu))} ) {orient} ;")
            lines.append("END COMPONENTS")
        else:
            lines.append(f"COMPONENTS 0 ;")
            lines.append("END COMPONENTS")
        lines.append("")

        # 3. Analyze Nets (Classification)
        target_segs = []
        aggr_segs_by_name = defaultdict(list)
        
        for seg in segments_data:
            if seg['type'] in ['PIN', 'INST_PORT']: continue
            if seg.get('tag') == 'target':
                target_segs.append(seg)
            else:
                name = seg.get('net_name', 'AGGR_UNKNOWN')
                aggr_segs_by_name[name].append(seg)

        special_nets = {}
        signal_nets = {}
        for name, segs in aggr_segs_by_name.items():
            if not name or name == target_net_name: continue
            if self._is_special_net(name):
                special_nets[name] = segs
            else:
                signal_nets[name] = segs

        # ---------------------------------------------------------------------
        # [NEW] Logical Connectivity Recovery
        # 물리적 세그먼트(INST_PORT)가 잘려나갔어도, 
        # Instance가 윈도우 안에 있다면 논리적 연결(inst_net_map)을 복원해야 함.
        # ---------------------------------------------------------------------
        relevant_nets = set(signal_nets.keys())
        relevant_nets.add(target_net_name)
        if special_nets:
            relevant_nets.update(special_nets.keys())
        
        # placed된 모든 인스턴스에 대해 핀 연결 검사
        for inst_name, inst_info in self.instances_data.items():
            cell_type = inst_info['cell_type']
            if cell_type not in self.cell_lib: continue
            
            # 이 Cell이 가진 모든 Pin에 대해
            for pin_name in self.cell_lib[cell_type]['pins']:
                # Global Map에서 연결된 Net 확인
                key = (inst_name, pin_name)
                if key in self.inst_net_map:
                    connected_net = self.inst_net_map[key]
                    
                    # 현재 Mini-DEF에 존재하는 Net이라면 연결 추가
                    if connected_net in relevant_nets:
                        conn_str = f"( {inst_name} {pin_name} )"
                        self.net_conn_map[connected_net].add(conn_str)
        # ---------------------------------------------------------------------

        # 4. PINS Section (Boundary/IO Pins)
        pin_lines, total_pins = self._generate_pins(segments_data, offset_x, offset_y)
        lines.append(f"PINS {total_pins} ;")
        lines.extend(pin_lines)
        # for p in vcc_ports: lines.append(p)
        # if vcc_ports: lines.append(" ;")
        # for p in vss_ports: lines.append(p)
        # if vss_ports: lines.append(" ;")
        lines.append("END PINS")
        lines.append("")
        # 5. SPECIALNETS
        if special_nets:
            lines.append(f"SPECIALNETS {len(special_nets)} ;")
            for name, segs in special_nets.items():
                use_type = self._get_use_type(name)
                lines.append(f"- {name} ( * {name} ) ")
                # Add Instance Connections
                for conn in sorted(list(self.net_conn_map[name])):
                    lines[-1] += f" {conn}"
                
                path_lines = self._generate_routing_lines(segs, offset_x, offset_y, is_special=True)
                lines.extend(path_lines)
                lines.append(f' + USE {use_type}')
                lines.append(" ;")
            lines.append("END SPECIALNETS")
            lines.append("")

        # 6. NETS Section 작성 (Target + Signal Aggressors)
        # Target은 무조건 Signal Net
        total_regular_nets = 1 + len(signal_nets)
        lines.append(f"NETS {total_regular_nets} ;")
        
        # 6.1 Target Net
        safe_target = sanitize_name(target_net_name)
        lines.append(f"- {safe_target}")
        if target_net_name in self.net_conn_map:
            for conn in set(self.net_conn_map[target_net_name]): # Unique
                lines.append(conn)

        # Target Pin 연결 구문 (Optional but good for completeness)
        connections = self._generate_net_pin_connections(safe_target)
        lines.extend(connections)
        path_lines = self._generate_routing_lines(target_segs, offset_x, offset_y, is_special=False)
        lines.extend(path_lines)
        lines.append(" ;")
        
        # 6.2 Signal Aggressors
        for name, segs in signal_nets.items():
            safe_name = sanitize_name(name)
            if len(safe_name) > 100: safe_name = safe_name[:100]
            lines.append(f"- {safe_name}") # + USE SIGNAL (Default)
            if name in self.net_conn_map:
                for conn in set(self.net_conn_map[name]): # Unique
                    lines.append(conn)

            connections = self._generate_net_pin_connections(safe_name)            
            lines.extend(connections)
            path_lines = self._generate_routing_lines(segs, offset_x, offset_y, is_special=False)
            lines.extend(path_lines)
            lines.append(" ;")

        # print(round(f"Total NETS written: {total_regular_nets}")
        lines.append("END NETS")
        lines.append("")        
        lines.append("END DESIGN")
        
        return "\n".join(lines)
    
    def _generate_net_pin_connections(self, net_name: str) -> List[str]:
        connections = []
        pin_names = self.net_pin_map.get(net_name, [])
        # One line two pins
        for pin_name in pin_names:
            prefix = 'INST_PORT_'
            if pin_name.startswith(prefix):
                _p_name = pin_name.replace(prefix, '').split('_')
                inst_name = '_'.join(_p_name[:-1])
                inst_pin_name = _p_name[-1]
                connections.append(f"DEBUG ( {inst_name} {inst_pin_name} )")
                if inst_name not in self.instances_data:
                    continue
                connections.append(f" ( {inst_name} {inst_pin_name} )")
            # Pin
            else:
                connections.append(f" ( PIN {pin_name} )")
        return connections

    def _generate_routing_lines(self, segments, off_x, off_y, is_special=False):
        lines = []
        if not segments: return lines
        prev_x, prev_y = None, None
        init_pin = True
        for seg in segments:
            # 첫 번째 라인은 + ROUTED, 나머지는 NEW
            # (모든 넷에 대해 동일 규칙 적용)
            prefix = "  + ROUTED" if init_pin else "    NEW"
            width_str = ""
            shape_str = ""

            if is_special:
                if seg.get('width'):
                    w_dbu = int(round(seg['width'] * self.dbu))
                    width_str = f" {w_dbu}"
                    shape_str = " + SHAPE STRIPE"
                else:
                    width_str = f" 0"
                    shape_str = " + SHAPE STRIPE"

            layer = seg.get('layer', 'm1')

            if seg['type'] == 'WIRE':
                x1, y1 = seg['start'][0] + off_x, seg['start'][1] + off_y
                x2, y2 = seg['end'][0] + off_x, seg['end'][1] + off_y
                ix1, iy1 = int(round(x1 * self.dbu)), int(round(y1 * self.dbu))
                ix2, iy2 = int(round(x2 * self.dbu)), int(round(y2 * self.dbu))
                if ix1 == ix2 and iy1 == iy2: continue
                prev_x, prev_y = x2, y2

                if ix1 == ix2: ix2 = '*'
                if iy1 == iy2: iy2 = '*'
                lines.append(f"{prefix} {layer}{width_str}{shape_str} ( {ix1} {iy1} ) ( {ix2} {iy2} )")
                init_pin = False

            elif seg['type'] == 'RECT':
                layer = seg['layer']
                r = seg['rect']
                # Apply Offset to coords
                x1, y1 = r[0] + off_x, r[1] + off_y
                x2, y2 = r[2] + off_x, r[3] + off_y
                prev_x, prev_y = x2, y2
                
                if is_special:
                    # [CRITICAL FIX 1] SPECIALNETS는 RECT를 지원하지 않음 -> 중심선 WIRE(STRIPE)로 수학적 등가 변환
                    w, h = abs(x2 - x1), abs(y2 - y1)
                    if w > h:
                        route_w, sx, sy, ex, ey = h, x1 + h/2.0, (y1+y2)/2.0, x2 - h/2.0, (y1+y2)/2.0
                    else:
                        route_w, sx, sy, ex, ey = w, (x1+x2)/2.0, y1 + w/2.0, (x1+x2)/2.0, y2 - w/2.0
                        
                    isx, isy = int(round(sx * self.dbu)), int(round(sy * self.dbu))
                    iex, iey = int(round(ex * self.dbu)), int(round(ey * self.dbu))
                    route_w_dbu = int(round(route_w * self.dbu))
                    
                    lines.append(f"{prefix} {layer} {route_w_dbu} + SHAPE STRIPE ( {isx} {isy} ) ( {iex} {iey} )")
                    init_pin = False
                    prev_x, prev_y = ex, ey
                else:
                    # 일반 SIGNAL 넷의 정상적인 RECT 문법
                    ix1, iy1 = int(round(x1 * self.dbu)), int(round(y1 * self.dbu))
                    w_dbu = int(round(abs(x2 - x1) * self.dbu))
                    h_dbu = int(round(abs(y2 - y1) * self.dbu))
                    
                    lines.append(f"{prefix} {layer} ( {ix1} {iy1} ) RECT ( 0 0 {w_dbu} {h_dbu} )")
                    init_pin = False
                
            elif seg['type'] == 'VIA':
                via_name = seg['name']
                vx, vy = seg['pos'][0] + off_x, seg['pos'][1] + off_y
                if vx == prev_x and vy == prev_y:
                    lines[-1] += f" {via_name}"

                else:
                    ivx, ivy = int(round(vx * self.dbu)), int(round(vy * self.dbu))
                    lines.append(f"{prefix} {layer}{width_str}{shape_str} ( {ivx} {ivy} ) {via_name}")

                init_pin = False

        return lines

    def _generate_pins(self, segments, off_x, off_y):
        pins = []
        vcc_ports = [f"- vcc + NET vcc + SPECIAL + DIRECTION INOUT + USE POWER"]
        vss_ports = [f"- vssx + NET vssx + SPECIAL + DIRECTION INOUT + USE GROUND"]
        total_pins = 0
        
        tgt_min_x = off_x - (self.win_w / 2.0) + 0.01
        tgt_max_x = off_x + (self.win_w / 2.0) - 0.01
        tgt_min_y = off_y - (self.win_h / 2.0) + 0.01
        tgt_max_y = off_y + (self.win_h / 2.0) - 0.01
        
        for seg in segments:
            if seg['type'] == 'WIRE':
                layer = seg.get('layer', 'm1')
                start = seg['start']
                end = seg['end']
                
                px, py = None, None
                
                if start[0] == end[0]: # Vertical
                    tx = start[0] + off_x
                    ly, ry = start[1] + off_y, end[1] + off_y
                    if ly <= tgt_min_y or ry >= tgt_max_y:
                        py = tgt_min_y if ly <= tgt_min_y else tgt_max_y
                        px = tx

                elif start[1] == end[1]: # Horizontal
                    ty = start[1] + off_y
                    lx, rx = start[0] + off_x, end[0] + off_x
                    if lx <= tgt_min_x or rx >= tgt_max_x:
                        px = tgt_min_x if lx <= tgt_min_x else tgt_max_x
                        py = ty
                
                if px is not None and py is not None:
                    net_name = sanitize_name(seg['net_name'].lower())
                    if not self._is_special_net(net_name):
                        pin_name = f"PIN_{total_pins}"
                        total_pins += 1
                        self.net_pin_map[net_name].add(pin_name)
                        
                        # [THE USER'S GENIUS FIX: Micro-Pin]
                        # DEF 파서 에러를 피하기 위한 최소 면적(1x1 DBU = 0.5x0.5 nm)만 부여.
                        # 파라데이 새장(Faraday Cage) 이펙트를 0으로 수렴시킴.
                        pins.append(f"- {pin_name} + NET {net_name} + DIRECTION INOUT + USE SIGNAL")       
                        pins.append(f"  + LAYER {layer} ( -1 -1 ) ( 1 1 )")
                        pins.append(f"  + PLACED ( {int(round(px * self.dbu))} {int(round(py * self.dbu))} ) N ;")

        # [CRITICAL FIX 3] VCC/VSS 핀이 실제로 존재할 때만 리스트에 편입시키고 개수 증가
        if len(vcc_ports) > 1:
            vcc_ports[-1] += " ;"
            total_pins += 1
            pins.extend(vcc_ports)
            
        if len(vss_ports) > 1:
            vss_ports[-1] += " ;"
            total_pins += 1
            pins.extend(vss_ports)
            
        return pins, total_pins