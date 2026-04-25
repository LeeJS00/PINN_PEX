# src/preprocessing/def_parser.py
import re
import numpy as np
from typing import Dict, List, Tuple, Generator

from collections import defaultdict

class DefStreamParser:
    def __init__(self, def_path: str, layer_info: Dict, tech_lef: Dict = None, cell_lib: Dict = None):
        self.def_path = def_path
        self.layer_info = layer_info
        self.tech_lef = tech_lef if tech_lef else {'vias': {}}
        self.cell_lib = cell_lib if cell_lib else {}
        self.instances = {} # Instance 정보 저장용
        self.pins = {}
        self.inst_net_map = {}
        self.def_vias = {}
        self.used_pins = set()
        self.dbu = 2000
        
    def parse(self) -> Generator[Tuple[str, np.ndarray, List[Dict]], None, None]:
        """
        Yields (net_name, cuboids_array, def_segments)
        - cuboids_array: ML 입력용 기하학 (N, 6)
        - def_segments: DEF 복원용 메타데이터 리스트
        """

        current_net_name = None
        current_cuboids = []
        current_segments = []
        net_tokens = []
        in_vias_section = False
        current_via_name = None
        via_temp_data = {}
        
        re_units = re.compile(r'UNITS DISTANCE MICRONS\s+(\d+)')
        re_net_start = re.compile(r'^\s*-\s+(\S+)') 
        re_comp_detail = re.compile(r'^\s*-\s+(\S+)\s+(\S+).*\+\s+(PLACED|FIXED|COVER)\s+\(\s*([-\d\.]+)\s+([-\d\.]+)\s*\)\s+(\S+)')
        re_end_nets = re.compile(r'^\s*END (?:NETS|SPECIALNETS)')
        re_pin_start = re.compile(r'^\s*-\s+(\S+)\s+\+\s+NET\s+(\S+)\s\+\sDIRECTION+\s+(\S+)\s\+\sUSE\s(\S+)')
        re_port_layer = re.compile(r'\+\s+LAYER\s+(\S+)\s+\(\s*([-\d\.]+)\s+([-\d\.]+)\s*\)\s+\(\s*([-\d\.]+)\s+([-\d\.]+)\s*\)')
        re_port_placed = re.compile(r'\+\s+PLACED\s+\(\s*([-\d\.]+)\s+([-\d\.]+)\s*\)\s+(\S+)')
        re_fixed = re.compile(r'\+\s+FIXED\s+\(\s*([-\d\.]+)\s+([-\d\.]+)\s*\)\s+(\S+)')
        re_net_pins = re.compile(r'\(\s(PIN|\S+)\s(\S+)\s\)')
        is_component_section = False
        is_pin_section = False
        is_in_pin_section = False
        in_nets_section = False
        ports_list = []
        in_pin_count = 0
        is_special_net = False

        with open(self.def_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                if line.startswith("UNITS DISTANCE"):
                    match = re_units.search(line)
                    if match: self.dbu = int(match.group(1))
                    continue
                if line.startswith("VIAS") and len(line.split()) > 1 and line.split()[1].isdigit():
                    in_vias_section = True
                    continue
                if line.startswith("END VIAS"):
                    in_vias_section = False
                    # 파싱 완료된 DEF 비아들을 tech_lef 딕셔너리에 병합하여 ML 모델이 인식하게 함
                    if 'vias' not in self.tech_lef: self.tech_lef['vias'] = {}
                    for v_name, v_data in self.def_vias.items():
                        self.tech_lef['vias'][v_name] = {'layers': v_data['layers']}
                    continue
                    
                if in_vias_section:
                    if line.startswith("- "):
                        current_via_name = line.split()[1]
                        self.def_vias[current_via_name] = {'layers': defaultdict(list), 'raw_lines':[]}
                        via_temp_data = {}
                    
                    elif current_via_name:
                        if line == ';': continue
                        self.def_vias[current_via_name]['raw_lines'].append(line)
                        
                        if line.startswith("+ RECT"):
                            # + RECT M6 ( -1000 -80 ) ( 1000 80 )
                            parts = line.replace('(', ' ').replace(')', ' ').split()
                            layer = parts[2].lower()
                            x1, y1, x2, y2 = float(parts[3])/self.dbu, float(parts[4])/self.dbu, float(parts[5])/self.dbu, float(parts[6])/self.dbu
                            self.def_vias[current_via_name]['layers'][layer].append((x1, y1, x2, y2))
                        
                        # Rule-based Vias (M4_M3widePWR0p864_1 등) 파싱
                        elif line.startswith("+ CUTSIZE"):
                            parts = line.split()
                            via_temp_data['cutsize'] = (float(parts[2])/self.dbu, float(parts[3])/self.dbu)
                        elif line.startswith("+ CUTSPACING"):
                            parts = line.split()
                            via_temp_data['spacing'] = (float(parts[2])/self.dbu, float(parts[3])/self.dbu)
                        elif line.startswith("+ ROWCOL"):
                            parts = line.split()
                            via_temp_data['rowcol'] = (int(parts[2]), int(parts[3])) # rows, cols
                        elif line.startswith("+ LAYERS"):
                            parts = line.split()
                            via_temp_data['layers'] = [parts[2].lower(), parts[3].lower(), parts[4].lower()] # bot, cut, top
                       
                        # VIARULE 정보가 모두 모였으면 수학적으로 RECT들을 생성
                        if all(k in via_temp_data for k in['cutsize', 'spacing', 'rowcol', 'layers']):
                            rows, cols = via_temp_data['rowcol']
                            cw, ch = via_temp_data['cutsize']
                            sx, sy = via_temp_data['spacing']
                           
                            # Array 중심을 (0,0)으로 하는 Cut Rect 생성 (ML Voxelization 용도)
                            # (자세한 Enclosure 메탈 크기는 모델의 맥락 파악에 큰 영향이 없으므로 Cut Box만 생성해도 충분함)
                            cut_layer = via_temp_data['layers'][1]
                            self.def_vias[current_via_name]['layers'][cut_layer].append((-cw/2, -ch/2, cw/2, ch/2))
                            via_temp_data = {} # Reset
                    continue

                if line.startswith("COMPONENTS"):
                    is_component_section = True
                    continue
                if line.startswith("PINS"):
                    is_component_section = False
                    is_pin_section = True
                    continue

                if line.startswith("NETS") or line.startswith("SPECIALNETS"):
                    yield "", np.array(current_cuboids, dtype=np.float32), current_segments
                    is_component_section = False
                    is_pin_section = False
                    in_nets_section = True
                    continue
                
                if re_end_nets.match(line):
                    if current_net_name:
                        yield current_net_name, np.array(current_cuboids, dtype=np.float32), current_segments
                    current_net_name = None
                    current_cuboids = []
                    current_segments = []
                    if line.startswith("END COMPONENTS"): is_component_section = False
                    if line.startswith("END DESIGN"): in_nets_section = False
                    continue

                if is_component_section:
                    if line.startswith("-"):
                        match = re_comp_detail.search(line)
                        if match:
                            inst_name, cell_type, status, _x, _y, orient = match.groups()
                            # 좌표 변환 (DBU -> Micron)
                            size = self.cell_lib[cell_type]['size']
                            x = float(_x)/self.dbu
                            y = float(_y)/self.dbu
                            w, h = size
                            self.instances[inst_name] = {
                                'cell_type': cell_type,
                                'x': x,
                                'y': y,
                                'rect': (x, y, x+w, y+h), 
                                'orient': orient,
                                'status': status
                            }
                            # Add instance pin metal
                            pin_cubs, pin_segs = self._create_INST_PORT(inst_name)
                            for pin_cub, pin_seg in zip(pin_cubs, pin_segs):
                                name = pin_seg['name']
                                yield f"INST_PORT_{name}", np.array([pin_cub], dtype=np.float32), [pin_seg]

                if is_pin_section:
                    if line.startswith("-"):
                        #[FIX 1] 정규표현식 대신 안전한 토큰 분리 방식 사용
                        parts = line.split()
                        if len(parts) >= 2:
                            pin_name = parts[1]
                            net_name = pin_name
                            direction = 'INOUT'
                            use = 'SIGNAL'
                            
                            if '+ NET' in line:
                                idx = parts.index('NET') if 'NET' in parts else -1
                                if idx != -1 and idx + 1 < len(parts): net_name = parts[idx+1]
                            
                            if '+ DIRECTION' in line:
                                idx = parts.index('DIRECTION') if 'DIRECTION' in parts else -1
                                if idx != -1 and idx + 1 < len(parts): direction = parts[idx+1]
                                
                            ports_list = []
                            self.pins[pin_name] = {
                                'net_name': net_name,
                                'direction': direction,
                                'use': use,
                                'status': 'PLACED'
                            }
                            is_in_pin_section = True
                            in_pin_count = 0
                            continue

                    if is_in_pin_section and line.startswith("+"):
                        # PORT 레이어 및 위치 정보 파싱
                        layer_match = re_port_layer.search(line)
                        if layer_match:
                            layer_name = layer_match.group(1).lower()
                            x1 = float(layer_match.group(2)) / self.dbu
                            y1 = float(layer_match.group(3)) / self.dbu
                            x2 = float(layer_match.group(4)) / self.dbu
                            y2 = float(layer_match.group(5)) / self.dbu
                            pin_temp_metal = (layer_name, x1, y1, x2, y2)
                            continue
                        
                        placed_match = re_port_placed.search(line)   
                        if placed_match:
                            px = float(placed_match.group(1)) / self.dbu
                            py = float(placed_match.group(2)) / self.dbu
                            orient = placed_match.group(3)
                            # PIN 위치 및 레이어 정보 저장
                            ports_list.append({
                                'layer': layer_name,
                                'rect': (pin_temp_metal[1], pin_temp_metal[2], pin_temp_metal[3], pin_temp_metal[4]),
                                'placed': (px, py),
                                'orient': orient,
                                'in_pin_index': in_pin_count
                            })
                            in_pin_count += 1
                            continue

                        fixed_match = re_fixed.search(line)
                        if fixed_match:
                            fx = float(fixed_match.group(1)) / self.dbu
                            fy = float(fixed_match.group(2)) / self.dbu
                            orient = fixed_match.group(3)
                            # PIN 위치 및 레이어 정보 저장
                            ports_list.append({
                                'layer': layer_name,
                                'rect': (pin_temp_metal[1], pin_temp_metal[2], pin_temp_metal[3], pin_temp_metal[4]),
                                'placed': (fx, fy),
                                'orient': orient,
                                'in_pin_index': in_pin_count
                            })
                            in_pin_count += 1
                            continue

                    if is_in_pin_section and ';' in line:
                        self.pins[pin_name]['ports'] = ports_list
                        pin_cubs, pin_segs = self._create_pin(pin_name)
                        for pin_cub, pin_seg in zip(pin_cubs, pin_segs):
                            name = pin_seg['name']
                            yield f"PIN_{name}", np.array([pin_cub], dtype=np.float32), [pin_seg]
                        is_in_pin_section = False
                        continue
                    

                if in_nets_section:                     
                    # New Net Start
                    if line.startswith("-"):
                        # 이전 넷 토큰들이 모여있다면 파싱 수행
                        if current_net_name and net_tokens:
                            cubs, segs = self._parse_net_tokens(net_tokens, current_net_name)
                            if cubs:
                                yield current_net_name, np.array(cubs, dtype=np.float32), segs
                        
                        # 새로운 넷 시작
                        parts = line.split()
                        current_net_name = parts[1]
                        net_tokens = parts[2:] # '-' 와 '이름' 제외한 나머지 토큰 저장
                    elif current_net_name:
                        # 줄바꿈된 데이터 계속 누적
                        net_tokens.extend(line.split())
                        
                        # 세미콜론(;)을 만나면 해당 넷의 정의가 끝난 것임!
                        if ';' in line:
                            cubs, segs = self._parse_net_tokens(net_tokens, current_net_name)
                            if cubs:
                                yield current_net_name, np.array(cubs, dtype=np.float32), segs
                            
                            current_net_name = None
                            net_tokens = []

                    # Routing Parsing
                    if current_net_name and any(x in line for x in ["ROUTED", "NEW", "SHAPE", "STRIPE"]):
                        clean_line = line.replace(';', '')
                        new_cubs, new_segs = self._parse_routing_line(clean_line, current_net_name)
                        if new_cubs: current_cuboids.extend(new_cubs)
                        if new_segs: current_segments.extend(new_segs)

            if current_net_name:
                yield current_net_name, np.array(current_cuboids, dtype=np.float32), current_segments
                
    def _parse_net_tokens(self, tokens: List[str], net_name: str) -> Tuple[List, List]:
        """
        [NEW] 세미콜론(;)까지 누적된 모든 토큰을 한 번에 정밀 파싱하여
        단 하나의 세그먼트나 비아도 유실되지 않도록 보장합니다.
        """
        cuboids = []
        segments =[]
        i = 0
        
        # 1. 포트 연결 ( inst pin ) 파싱
        while i < len(tokens):
            t = tokens[i]
            if t in['+', 'ROUTED', 'FIXED', 'COVER', 'NEW']:
                break # 라우팅 섹션 시작
            if t == '(':
                # ( U1 A ) 형태
                if i + 3 < len(tokens) and tokens[i+3] == ')':
                    comp, pin = tokens[i+1], tokens[i+2]
                    if comp == 'PIN':
                        if pin in self.pins:
                            self.pins[pin]['net_name'] = net_name
                            self.used_pins.add(pin)
                    else:
                        if comp in self.instances:
                            self.inst_net_map[(comp, pin)] = net_name
                            p_name = f"{comp}_{pin}"
                            if p_name not in self.pins: self.pins[p_name] = {}
                            self.pins[p_name]['net_name'] = net_name
                            self.used_pins.add(p_name)
                    i += 4
                else:
                    i += 1
            else:
                i += 1

        # 2. 라우팅 경로 파싱
        routing_started = False
        current_layer = None
        current_width = None
        current_pt = None
        
        while i < len(tokens):
            t = tokens[i]
            if t == ';': break
            
            if t in ['ROUTED', 'FIXED', 'COVER', 'NEW']:
                routing_started = True
                i += 1
                if i < len(tokens) and tokens[i].lower() in self.layer_info:
                    current_layer = tokens[i].lower()
                    i += 1
                    # Width가 명시된 경우
                    if i < len(tokens) and tokens[i][0].isdigit() and tokens[i] != '0':
                        current_width = float(tokens[i]) / self.dbu
                        i += 1
                    else:
                        current_width = None
                    current_pt = None # 새 경로이므로 이전 점 초기화
                continue

            if not routing_started:
                i += 1
                continue

            # 좌표 ( x y ) 파싱
            if t == '(':
                try:
                    x_str = tokens[i+1]
                    y_str = tokens[i+2]
                    
                    # ')' 닫는 괄호 찾기
                    j = i + 3
                    while tokens[j] != ')': j += 1
                    
                    # 와일드카드(*) 처리
                    if x_str == '*': new_x = current_pt[0] if current_pt else 0.0
                    else: new_x = float(x_str) / self.dbu
                    
                    if y_str == '*': new_y = current_pt[1] if current_pt else 0.0
                    else: new_y = float(y_str) / self.dbu
                    
                    new_pt = (new_x, new_y)
                    
                    # 선분 생성
                    if current_pt is not None and current_layer:
                        if current_pt != new_pt:
                            actual_width = current_width if current_width is not None else self._get_default_width(current_layer)
                            c_cubs = self._create_wire_cuboid(current_layer, current_pt, new_pt, actual_width)
                            if c_cubs: cuboids.append(c_cubs)
                            
                            segments.append({
                                'type': 'WIRE', 'layer': current_layer,
                                'start': current_pt, 'end': new_pt,
                                'width': actual_width, 'net_name': net_name
                            })
                            
                    current_pt = new_pt
                    i = j + 1
                except:
                    i += 1
                continue
                
            # RECT 파싱
            elif t == 'RECT':
                # RECT ( dx1 dy1 dx2 dy2 ) 또는 RECT dx1 dy1 dx2 dy2 지원
                pts =[]
                j = i + 1
                while len(pts) < 4 and j < len(tokens):
                    if tokens[j] not in ['(', ')']:
                        pts.append(float(tokens[j]) / self.dbu)
                    j += 1
                
                if len(pts) == 4 and current_pt and current_layer:
                    dx1, dy1, dx2, dy2 = pts
                    r_cub = self._create_rect_cuboid(current_layer, current_pt, dx1, dy1, dx2, dy2)
                    if r_cub: cuboids.append(r_cub)
                    
                    abs_rect = (current_pt[0]+dx1, current_pt[1]+dy1, current_pt[0]+dx2, current_pt[1]+dy2)
                    segments.append({
                        'type': 'RECT', 'layer': current_layer, 'rect': abs_rect,
                        'ref_point': current_pt, 'net_name': net_name
                    })
                i = j
                continue
                
            # VIA 파싱
            elif t.upper().startswith('VIA') or (t in self.tech_lef.get('vias', {})):
                if current_pt:
                    via_cubs = self._create_via_cuboids(t, current_pt)
                    if via_cubs: cuboids.extend(via_cubs)
                    
                    via_info = self.tech_lef.get('vias', {}).get(t, {})
                    metal_layers = via_info.get('metal_layers', [])
                    if len(metal_layers) >= 2:
                        bot_layer, top_layer = metal_layers[0], metal_layers[1]
                    else:
                        # LEF 정보가 부족할 경우 이름에서 숫자(Level)를 직접 추출
                        v_match = re.search(r'VIA(\d+)', t, re.IGNORECASE)
                        if v_match:
                            v_lvl = int(v_match.group(1))
                            bot_layer = f'm{v_lvl}'
                            top_layer = f'm{v_lvl + 1}'
                        else:
                            # 최후의 보루: 현재 라우팅 레이어를 기준으로 삼음
                            bot_layer = current_layer
                            top_layer = current_layer
                    
                    actual_width = current_width if current_width is not None else self._get_default_width(current_layer)
                    
                    segments.append({
                        'type': 'VIA', 'name': t, 'pos': current_pt,
                        'bot_layer': bot_layer, 'top_layer': top_layer,
                        'width': actual_width, 'net_name': net_name
                    })
                i += 1
                continue
                
            i += 1

        return cuboids, segments

    def _create_INST_PORT(self, inst_name: str) -> Tuple[List, List]:
        """
        Instance Pin의 형상을 절대 좌표 Cuboid로 변환합니다.
        이를 통해 ML 입력에서 Pin 형상을 물리적으로 재현합니다.
        """
        if inst_name not in self.instances: return ([], [])
        cuboids = []
        segments = []
        comp_info = self.instances[inst_name]
        cell_type = comp_info['cell_type']
        if cell_type not in self.cell_lib: return ([], [])
        origin_x, origin_y = comp_info['x'], comp_info['y']
        orient = comp_info['orient']
        cell_data = self.cell_lib[cell_type]            
        cell_w, cell_h = cell_data['size']
        for pin_name, pin_data in cell_data['pins'].items():
            # pin_name = f'{cell_name}:{_pin_name}'
            for metals in pin_data:
                layer = metals['layer']
                # Local Rect
                rx1, ry1, rx2, ry2 = metals['rect']
                
                # Orientation Transform (Rotation/Flip)
                pts = [
                    self._transform(rx1, ry1, orient, cell_w, cell_h),
                    self._transform(rx2, ry2, orient, cell_w, cell_h),
                    self._transform(rx1, ry2, orient, cell_w, cell_h),
                    self._transform(rx2, ry1, orient, cell_w, cell_h)
                ]
                
                tx_coords = [p[0] for p in pts]
                ty_coords = [p[1] for p in pts]
                
                # Absolute Coordinates
                abs_x1 = min(tx_coords) + origin_x
                abs_x2 = max(tx_coords) + origin_x
                abs_y1 = min(ty_coords) + origin_y
                abs_y2 = max(ty_coords) + origin_y
                
                # Cuboid 추가
                if layer in self.layer_info:
                    l_info = self.layer_info[layer]
                    cz = l_info['z_pos'] + l_info['thickness'] / 2
                    sz = l_info['thickness']
                    cuboids.append([
                        (abs_x1 + abs_x2) / 2,
                        (abs_y1 + abs_y2) / 2,
                        cz,
                        abs(abs_x2 - abs_x1),
                        abs(abs_y2 - abs_y1),
                        sz
                    ])
                    # print(self.cell_lib[cell_type]['pins'][pin_name][0]['direction'])
                    segments.append({
                        'type': 'INST_PORT',
                        'name': f'INST_PORT_{inst_name}_{pin_name}',
                        'layer': layer,
                        'pos': (abs_x1, abs_y1, abs_x2, abs_y2),
                        'cell_type': cell_type,
                        'from_inst': inst_name,
                        'pin_name': pin_name,
                        'direction': self.cell_lib[cell_type]['pins'][pin_name][0]['direction']
                    })
                    # print(inst_name, pin_name, layer, (abs_x1, abs_y1, abs_x2, abs_y2))
                                 
        return cuboids, segments
        
    def _create_pin(self, pin_name: str) -> Tuple[List, List]:
        if pin_name not in self.pins:
            return ([], [])
        cuboids = []
        segments = []
        for port in self.pins[pin_name]['ports']:
            in_pin_index = port['in_pin_index']
            layer = port['layer']
            rx1, ry1, rx2, ry2 = port['rect']
            orient = port['orient']
            # Orientation Transform
            pts = [
                self._transform(rx1, ry1, orient, 0, 0),
                self._transform(rx2, ry2, orient, 0, 0),
                self._transform(rx1, ry2, orient, 0, 0),
                self._transform(rx2, ry1, orient, 0, 0)
            ]
            tx_coords = [p[0] for p in pts]
            ty_coords = [p[1] for p in pts]
            abs_x1 = min(tx_coords)
            abs_x2 = max(tx_coords)
            abs_y1 = min(ty_coords)
            abs_y2 = max(ty_coords)
            
            # Cuboid 추가
            if layer in self.layer_info:
                l_info = self.layer_info[layer]
                cz = l_info['z_pos'] + l_info['thickness'] / 2
                sz = l_info['thickness']
                cuboids.append([
                    (abs_x1 + abs_x2) / 2,
                    (abs_y1 + abs_y2) / 2,
                    cz,
                    abs(abs_x2 - abs_x1),
                    abs(abs_y2 - abs_y1),
                    sz
                ])
                segments.append({
                    'type': 'PIN',
                    'name': f'PIN_{pin_name}_{in_pin_index}',
                    'layer': layer,
                    'pos': (abs_x1, abs_y1, abs_x2, abs_y2),
                    'pin_name': pin_name,
                    'in_pin_index': in_pin_index,
                    'direction': self.pins[pin_name]['direction'],
                })

        return cuboids, segments

    def _parse_routing_line(self, line: str, net_name: str) -> Tuple[List, List]:
        cuboids = []
        segments = []
        
        tokens = line.replace('(', ' ( ').replace(')', ' ) ').split()
        
        current_layer = None
        current_pt = None 
        current_width = None 
        
        i = 0
        while i < len(tokens):
            t = tokens[i]
            
            if t in ['+', 'ROUTED', 'NEW', 'SHAPE', 'STRIPE']:
                i += 1
                continue
                
            # Layer Name
            if t.lower() in self.layer_info:
                current_layer = t.lower() # 소문자로 통일
                i += 1; 
                # 다음 토큰이 숫자라면 Width임 (SPECIALNETS)
                if i < len(tokens) and tokens[i][0].isdigit():
                    current_width = float(tokens[i])/self.dbu
                    i += 1
                else:
                    current_width = None # Regular NETS use LEF width
                continue

            # Point: ( x y )
            # Point: ( x y ) or ( x y z )
            if t == '(':
                coords = []
                j = i + 1
                while j < len(tokens) and tokens[j] != ')':
                    coords.append(tokens[j])
                    j += 1
                
                if len(coords) < 2:
                    raise ValueError(f"[StrictError] Invalid coordinate format in Net {net_name}: {tokens[i:j+1]}")

                x_str, y_str = coords[0], coords[1]
                
                # 와일드카드(*) 처리
                if x_str == '*':
                    if current_pt is None: raise ValueError(f"[StrictError] Wildcard * used for X but no previous point exists in Net {net_name}.")
                    new_x = current_pt[0]
                else:
                    new_x = float(x_str) / self.dbu
                    
                if y_str == '*':
                    if current_pt is None: raise ValueError(f"[StrictError] Wildcard * used for Y but no previous point exists in Net {net_name}.")
                    new_y = current_pt[1]
                else:
                    new_y = float(y_str) / self.dbu
                    
                new_pt = (new_x, new_y)

                if current_pt is not None and current_layer:
                    if current_pt[0] != new_pt[0] or current_pt[1] != new_pt[1]:
                        
                        actual_width = current_width if current_width is not None else self._get_default_width(current_layer)
                        
                        seg_cuboid = self._create_wire_cuboid(current_layer, current_pt, new_pt, actual_width)
                        cuboids.append(seg_cuboid) # None 방어 해제. 생성 실패시 위에서 에러남
                        
                        segments.append({
                            'type': 'WIRE',
                            'layer': current_layer,
                            'start': current_pt,
                            'end': new_pt,
                            'width': actual_width,
                            'net_name': net_name
                        })
                
                current_pt = new_pt
                i = j + 1
                continue
                
            # RECT
            if t == 'RECT':
                dx1 = float(tokens[i+2]) / self.dbu
                dy1 = float(tokens[i+3]) / self.dbu
                dx2 = float(tokens[i+4]) / self.dbu
                dy2 = float(tokens[i+5]) / self.dbu
                
                if current_pt and current_layer:
                    # 1. Cuboid
                    rect_cuboid = self._create_rect_cuboid(current_layer, current_pt, dx1, dy1, dx2, dy2)
                    if rect_cuboid: cuboids.append(rect_cuboid)
                    
                    # 2. Segment Info
                    # 절대 좌표 RECT로 변환하여 저장
                    abs_rect = (current_pt[0]+dx1, current_pt[1]+dy1, current_pt[0]+dx2, current_pt[1]+dy2)
                    segments.append({
                        'type': 'RECT',
                        'layer': current_layer,
                        'rect': abs_rect,
                        'ref_point': current_pt,
                        'net_name': net_name
                    })
                    
                i += 7
                continue
            
            # VIA
            is_via = t.upper().startswith('VIA') or (t in self.tech_lef.get('vias', {}))
            if is_via:
                if current_pt:
                    via_cubs = self._create_via_cuboids(t, current_pt)
                    if not via_cubs:
                        raise ValueError(f"[StrictError] Failed to create cuboids for VIA '{t}'")
                    cuboids.extend(via_cubs)
                    # [NEW] LEF 정보를 통해 이 비아가 연결하는 상/하부 레이어 가져오기
                    via_info = self.tech_lef.get('vias', {}).get(t, {})
                    metal_layers = via_info.get('metal_layers', [])
                    if len(metal_layers) >= 2:
                        bot_layer, top_layer = metal_layers[0], metal_layers[1]
                    else:
                        # LEF 정보가 부족할 경우 이름에서 숫자(Level)를 직접 추출
                        v_match = re.search(r'VIA(\d+)', t, re.IGNORECASE)
                        if v_match:
                            v_lvl = int(v_match.group(1))
                            bot_layer = f'm{v_lvl}'
                            top_layer = f'm{v_lvl + 1}'
                        else:
                            # 최후의 보루: 현재 라우팅 레이어를 기준으로 삼음
                            bot_layer = current_layer
                            top_layer = current_layer
                    actual_width = current_width if current_width is not None else self._get_default_width(current_layer)
                    segments.append({
                        'type': 'VIA',
                        'name': t,
                        'pos': current_pt,
                        'bot_layer': bot_layer, 
                        'top_layer': top_layer,
                        'layer': current_layer,
                        'net_name': net_name,
                        'width':actual_width
                    })

                i += 1
                continue
                
            i += 1
            
        return cuboids, segments

    def _create_wire_cuboid(self, layer_name, p1, p2, actual_width):
        # [STRICT] 레이어 매핑 누락 허용 안 함
        if layer_name not in self.layer_info:
            raise KeyError(f"[StrictError] Layer '{layer_name}' not found in layers.info")
            
        info = self.layer_info[layer_name]
        
        # ... (이하 동일 계산) ...
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        if abs(x2-x1) > abs(y2-y1): # Horizontal
            sx, sy = length, actual_width
        else:
            sx, sy = actual_width, length
            
        cz = info['z_pos'] + info['thickness'] / 2
        sz = info['thickness']
        return[cx, cy, cz, sx, sy, sz]


    def _create_rect_cuboid(self, layer_name, ref_pt, dx1, dy1, dx2, dy2):
        if layer_name not in self.layer_info: return None
        info = self.layer_info[layer_name]
        
        rx, ry = ref_pt
        abs_x1, abs_y1 = rx + dx1, ry + dy1
        abs_x2, abs_y2 = rx + dx2, ry + dy2
        
        cx, cy = (abs_x1 + abs_x2) / 2, (abs_y1 + abs_y2) / 2
        sx, sy = abs(abs_x2 - abs_x1), abs(abs_y2 - abs_y1)
        
        cz = info['z_pos'] + info['thickness'] / 2
        sz = info['thickness']
        
        return [cx, cy, cz, sx, sy, sz]

    def _create_via_cuboids(self, via_name, pt):
        cuboids = []
        cx, cy = pt
        if via_name.lower() in self.tech_lef.get('vias', {}):
            via_data = self.tech_lef['vias'][via_name.lower()]
            for layer_name, rects in via_data.get('layers', {}).items():
                # print(layer_name, rects)
                lname = layer_name.lower()
                if lname not in self.layer_info: continue
                l_info = self.layer_info[lname]
                z_pos = l_info['z_pos']
                thickness = l_info['thickness']
                for r in rects:
                    rx1, ry1, rx2, ry2 = r
                    acx = cx + (rx1 + rx2) / 2
                    acy = cy + (ry1 + ry2) / 2
                    asx = abs(rx2 - rx1)
                    asy = abs(ry2 - ry1)
                    acz = z_pos + thickness / 2
                    asz = thickness
                    cuboids.append([acx, acy, acz, asx, asy, asz])
        return cuboids
    
        
    def _transform(self, x, y, orient, w, h):
        """좌표 변환 헬퍼 (N, S, FN, FS 지원)"""
        # 간단화된 로직. 실제로는 8방향 모두 지원해야 안전함.
        if orient == 'N': return x, y
        elif orient == 'S': return w - x, h - y
        elif orient == 'FN': return w - x, y
        elif orient == 'FS': return x, h - y
        # FW, FE 등은 x, y Swap 필요 (여기선 생략, 필요시 추가)
        return x, y

    def _get_default_width(self, layer_name: str) -> float:
        """LEF 정보에서 해당 레이어의 Default Width를 가져옵니다. (대소문자 완벽 무시)"""            
        layer_lower = layer_name.lower()
        matched_key = None
        for k in self.tech_lef['layers'].keys():
            if k.lower() == layer_lower:
                matched_key = k
                break
                
        if not matched_key:
            raise KeyError(f"[StrictError] Routing Layer '{layer_name}' not found in technology.lef")
            
        width = self.tech_lef['layers'][matched_key].get('width')
        if width is None:
            raise ValueError(f"[StrictError] Layer '{layer_name}' has no WIDTH defined in technology.lef.")
        return width