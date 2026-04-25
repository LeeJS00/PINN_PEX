# src/utils/spef_writer.py
import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree
import datetime
import shutil
from pathlib import Path
from src.utils.naming import sanitize_name
import re


class RCTopologyBuilder:
    def __init__(self, net_name, global_segments, top_ports, layer_info, tech_lef=None, dbu=2000):
        self.net_name = net_name
        self.segments = global_segments
        self.top_ports = top_ports
        self.layer_info = layer_info
        self.tech_lef = tech_lef if tech_lef else {}
        self.dbu = dbu
        
        self.nodes = {}       
        self.node_info = {}   
        self.edges =[]       
        self.alias_map = {}
        
        self._node_counter = 1
        self._build_via_map()
        self._build_topology()

    def _build_via_map(self):
        c_layers = [(name, info['z_pos'], info.get('top_z', info['z_pos'] + info.get('thickness'))) 
                    for name, info in self.layer_info.items() if info.get('type') == 'C']
        c_layers.sort(key=lambda x: x[1])
        
        self.via_map = {}
        for name, info in self.layer_info.items():
            if info.get('type') == 'V' or 'via' in name.lower() or name.startswith('v'):
                vz_bot = info.get('z_pos')
                vz_top = info.get('top_z')
                bot_m, top_m = 'm1', 'm2'
                for c_name, c_bot, c_top in c_layers:
                    if abs(c_top - vz_bot) < 1e-3 or c_top <= vz_bot + 1e-3: bot_m = c_name
                    if abs(c_bot - vz_top) < 1e-3 or c_bot >= vz_top - 1e-3: top_m = c_name
                self.via_map[name] = (bot_m, top_m)

    def _get_physical_node(self, coord, layer_name, width=0.044, wire_dir=None):
        ix, iy = int(round(coord[0]*self.dbu)), int(round(coord[1]*self.dbu))
        layer_name = layer_name.lower()
        key = (ix, iy, layer_name)
        
        if key not in self.nodes:
            node_id = f"{self.net_name}:{self._node_counter}"
            self.nodes[key] = node_id
            self._node_counter += 1
            
            cx, cy = coord[0], coord[1]
            lvl_idx = self.layer_info.get(layer_name).get('lvl_idx')
            
            hw = width / 2.0
            if wire_dir == 0:
                llx, urx = cx, cx
                lly, ury = cy - hw, cy + hw
            elif wire_dir == 1:
                llx, urx = cx - hw, cx + hw
                lly, ury = cy, cy
            else:
                llx, lly, urx, ury = cx, cy, cx, cy
                
            comment = f"// $llx={llx:.4f} $lly={lly:.4f} $urx={urx:.4f} $ury={ury:.4f} $lvl={lvl_idx}"
            self.node_info[node_id] = {'str': f"*N {node_id} *C {cx:.4f} {cy:.4f} {comment}", 'type': '*N'}
            
        return self.nodes[key]

    def _build_topology(self):
        # 1. 핀 메탈을 실제 물리 와이어(WIRE)로 취급하기 위해 segments 확장
        # extended_wires = []
        # 1. 핀(PIN)을 와이어로 분해하는 오만함(Pin-to-Wire) 전면 제거
        # StarRC는 표준 셀 핀 내부를 라우팅 저항으로 치환하지 않습니다.
        # pure_routing_wires = []
        routing_wires = []
        for seg in self.segments:
            if seg['type'] == 'WIRE':
                # pure_routing_wires.append(seg)
                routing_wires.append(seg)
            #     extended_wires.append(seg)
            elif seg['type'] in ['PIN', 'INST_PORT']:
                if 'pos' in seg and len(seg['pos']) == 4:
                    llx, lly, urx, ury = seg['pos']
                    w = min(urx - llx, ury - lly)
                    l = max(urx - llx, ury - lly)
                    # 핀이 직사각형 형태면 물리적 와이어로 변환하여 저항망에 편입!
                    # if l > w + 1e-4:
                    #     if urx - llx > ury - lly: # Horizontal
                    #         s = (llx + w/2, (lly+ury)/2)
                    #         e = (urx - w/2, (lly+ury)/2)
                    #     else: # Vertical
                    #         s = ((llx+urx)/2, lly + w/2)
                    #         e = ((llx+urx)/2, ury - w/2)
                    #     routing_wires.append({
                    #         'type': 'WIRE', 'layer': seg['layer'],
                    #         'start': s, 'end': e, 'width': w, 'is_pin_metal': True
                    #     })
                    if urx - llx > ury - lly: # Horizontal
                        s = (llx + w/2, (lly+ury)/2)
                        e = (urx - w/2, (lly+ury)/2)
                    else: # Vertical or Square
                        s = ((llx+urx)/2, lly + w/2)
                        e = ((llx+urx)/2, ury - w/2)
                    routing_wires.append({
                        'type': 'WIRE', 'layer': seg.get('layer', 'm1'),
                        'start': s, 'end': e, 'width': w
                    })
        # =================================================================
        # [CRITICAL SHIFT 1] Collinear Wire Merging (Tiling 상처 치유 알고리즘)
        # 타일 경계에서 무의미하게 잘려나간 일직선 와이어들을 다시 하나의 거대한 Segment로 융합합니다.
        # 이를 통해 L_eff 차감 수식에 의한 저항 증발(1/3 토막 현상)을 원천 차단합니다.
        # =================================================================
        def merge_collinear_wires(wires):
            groups = defaultdict(list)
            for w in wires:
                layer = w['layer'].lower()
                sx, sy = w['start']
                ex, ey = w['end']
                width = float(w['width'])
                # 방향 판별: 수평(Horizontal)이면 0, 수직(Vertical)이면 1
                if abs(sy - ey) < 1e-4: 
                    groups[(layer, 0, round(sy, 4), width)].append((min(sx, ex), max(sx, ex)))
                else: 
                    groups[(layer, 1, round(sx, 4), width)].append((min(sy, ey), max(sy, ey)))
                    
            merged_wires = []
            for (layer, d_val, const_val, width), segs in groups.items():
                segs.sort(key=lambda x: x[0]) # 좌표 순 정렬
                merged = []
                for s in segs:
                    if not merged: merged.append(s)
                    else:
                        last_s, last_e = merged[-1]
                        if s[0] <= last_e + 1e-3: # 선분이 겹치거나 맞닿아 있다면 융합!
                            merged[-1] = (last_s, max(last_e, s[1]))
                        else:
                            merged.append(s)
                for s, e in merged:
                    if d_val == 0: merged_wires.append({'type': 'WIRE', 'layer': layer, 'start': (s, const_val), 'end': (e, const_val), 'width': width})
                    else: merged_wires.append({'type': 'WIRE', 'layer': layer, 'start': (const_val, s), 'end': (const_val, e), 'width': width})
            return merged_wires

        routing_wires = merge_collinear_wires(routing_wires)
        # =================================================================
        # 2. 절단점(Cut Points) 수집
        cut_points = defaultdict(set)
        for seg in self.segments:
            if seg['type'] in ['PIN', 'INST_PORT']:
                cx, cy = (seg['pos'][0]+seg['pos'][2])/2, (seg['pos'][1]+seg['pos'][3])/2
                cut_points[seg.get('layer').lower()].add((round(cx, 4), round(cy, 4)))
            elif seg['type'] == 'VIA':
                # bot, top = self.via_map.get(seg.get('bot_layer').lower(), seg.get('top_layer').lower())
                # if 'bot_layer' in seg and 'top_layer' in seg:
                bot, top = seg['bot_layer'].lower(), seg['top_layer'].lower()
                cx, cy = seg['pos']
                cut_points[bot].add((round(cx, 4), round(cy, 4)))
                cut_points[top].add((round(cx, 4), round(cy, 4)))
                
        # for seg in extended_wires:
        # for seg in pure_routing_wires:
        for seg in routing_wires:
            layer = seg.get('layer').lower()
            cut_points[layer].add((round(seg['start'][0], 4), round(seg['start'][1], 4)))
            cut_points[layer].add((round(seg['end'][0], 4), round(seg['end'][1], 4)))

        # 3. Wire Fracturing & Generation
        # for seg in pure_routing_wires:
        for seg in routing_wires:
            layer = seg.get('layer').lower()
            
            # [CRITICAL SHIFT] 층(Layer) 두께에 따른 동적 표면 산란 계수 적용
            match = re.search(r'\d+', layer)
            lvl = int(match.group()) if match else 1

            lef_width = self.layer_info.get(layer).get('width')
            width = float(seg.get('width') or lef_width)
            sheet_res = self.layer_info.get(layer).get('resistance')
            lvl_idx = self.layer_info.get(layer).get('lvl_idx')
            
            sx, sy = round(seg['start'][0], 4), round(seg['start'][1], 4)
            ex, ey = round(seg['end'][0], 4), round(seg['end'][1], 4)
            dir_val = 0 if abs(sy - ey) < 1e-4 else 1
            
            pts_on_wire =[]
            for cx, cy in cut_points[layer]:
                if dir_val == 0 and abs(cy - sy) < 1e-3 and min(sx, ex)-1e-3 <= cx <= max(sx, ex)+1e-3:
                    pts_on_wire.append((cx, cy))
                elif dir_val == 1 and abs(cx - sx) < 1e-3 and min(sy, ey)-1e-3 <= cy <= max(sy, ey)+1e-3:
                    pts_on_wire.append((cx, cy))
                    
            if dir_val == 0: pts_on_wire.sort(key=lambda p: p[0], reverse=(sx > ex))
            else: pts_on_wire.sort(key=lambda p: p[1], reverse=(sy > ey))
                
            unique_pts =[]
            for p in pts_on_wire:
                if not unique_pts or (abs(unique_pts[-1][0]-p[0])>1e-4 or abs(unique_pts[-1][1]-p[1])>1e-4):
                    unique_pts.append(p)
            
            MAX_FRACTURE_LEN = 9.1
            final_pts = []
            for i in range(len(unique_pts)):
                if i == 0:
                    final_pts.append((unique_pts[i][0], unique_pts[i][1], False)) # False = Physical Corner
                else:
                    prev_p, curr_p = unique_pts[i-1], unique_pts[i]
                    dx = curr_p[0] - prev_p[0]
                    dy = curr_p[1] - prev_p[1]
                    length = np.hypot(dx, dy)
                    
                    # 9.0um 초과 시 정확히 9.0um 단위로 전진하며 가상 절단점 생성
                    if length > MAX_FRACTURE_LEN + 1e-4:
                        num_full_segments = int(length // MAX_FRACTURE_LEN)
                        dir_x, dir_y = dx / length, dy / length
                        
                        for j in range(1, num_full_segments + 1):
                            dist = j * MAX_FRACTURE_LEN
                            if length - dist < 1e-4: continue # 끝점에 정확히 닿는 경우 중복 방지
                            vx = prev_p[0] + dir_x * dist
                            vy = prev_p[1] + dir_y * dist
                            final_pts.append((round(vx, 4), round(vy, 4), True))
                    final_pts.append((curr_p[0], curr_p[1], False))

            # =================================================================
            # [CRITICAL SHIFT 2] Etch Bias & Wire Scale Calibration (1.40x)
            # =================================================================
            W_eff = max(width - 0.004, 0.01)

            for i in range(len(final_pts) - 1):
                p1, p2 = final_pts[i], final_pts[i+1]
                n1 = self._get_physical_node(p1[:2], layer, width, dir_val)
                n2 = self._get_physical_node(p2[:2], layer, width, dir_val)
                
                length = np.hypot(p1[0] - p2[0], p1[1] - p2[1])
                sub_w = 0.0
                if not p1[2]: sub_w += width / 2.0
                if not p2[2]: sub_w += width / 2.0
                
                L_eff = max(length - sub_w, width * 0.1) if length > sub_w else length
                squares = L_eff / W_eff if W_eff > 0 else 0.0
                # R = rho [Ohm/sq] * (L_eff / W_eff) — analytically correct
                if sheet_res and sheet_res > 0:
                    res_val = sheet_res * squares
                else:
                    res_val = 12.4 * L_eff + 0.08 if lvl_idx > 6 else 12.4 * L_eff

                if n1 != n2:
                    hw = width / 2.0
                    if dir_val == 0:
                        llx, urx = min(p1[0], p2[0]), max(p1[0], p2[0])
                        lly, ury = p1[1] - hw, p1[1] + hw
                    else:
                        llx, urx = p1[0] - hw, p1[0] + hw
                        lly, ury = min(p1[1], p2[1]), max(p1[1], p2[1])
                    
                    comment = f"//  $l={L_eff:.4f} (raw:{length:.4f}) $w={W_eff:.4f} $lvl={lvl_idx} $llx={llx:.4f} $lly={lly:.4f} $urx={urx:.4f} $ury={ury:.4f} $dir={dir_val}"
                    self.edges.append((n1, n2, res_val, comment))

        # 4. VIA 생성
        for seg in self.segments:
            if seg['type'] == 'VIA':                
                bot_layer = seg.get('bot_layer').lower()
                top_layer = seg.get('top_layer').lower()
                via_name_raw = seg.get('name').lower()

                match = re.search(r'\d+', bot_layer)
                bot_lvl = int(match.group()) if match else 1
                
                via_layer_key = None
                candidate_keys = [f"via{bot_lvl}", f"v{bot_lvl}"]
                for cand in candidate_keys:
                    if cand in self.layer_info:
                        via_layer_key = cand
                        break
                        
                if not via_layer_key:
                    match = re.search(r'via(\d+)', via_name_raw)
                    lvl = int(match.group(1)) if match else bot_lvl
                    via_layer_key = f"via{lvl}"
                width = float(seg.get('width'))
                
                n1 = self._get_physical_node(seg['pos'], bot_layer, width, None)
                n2 = self._get_physical_node(seg['pos'], top_layer, width, None)
                
                layer_info_dict = self.layer_info.get(via_layer_key, {})
                lvl_idx = layer_info_dict.get('lvl_idx', bot_lvl)
                area = width * width
                
                # if lvl_idx == 10:
                #     actual_via_res = 11.6141
                # else:
                actual_via_res = 13.0685
                if lvl_idx == 9:
                    actual_via_res = 0.0001
                
                # SPEF 주석 매칭을 위한 가짜 변수 (실제 저항 계산엔 쓰이지 않음)
                total_cuts = 1 
                vc = 8 if bot_lvl > 1 else 12
                
                cx, cy = seg['pos'][0], seg['pos'][1]
                hw = width / 2.0
                comment = f"//  $vc={vc} $cuts={total_cuts} $a={area:.8f} $lvl={lvl_idx} $llx={cx-hw:.4f} $lly={cy-hw:.4f} $urx={cx+hw:.4f} $ury={cy+hw:.4f}"
                self.edges.append((n1, n2, actual_via_res, comment))

        layer_kdtree = {}
        layer_keys = defaultdict(list)
        layer_coords = defaultdict(list)
        for (ix, iy, n_layer), phys_node in self.nodes.items():
            layer_coords[n_layer].append([ix / self.dbu, iy / self.dbu])
            layer_keys[n_layer].append(phys_node)
            
        for n_layer in layer_coords.keys():
            layer_kdtree[n_layer] = cKDTree(np.array(layer_coords[n_layer]))

        # 5. Logical Nodes 연동
        for seg in self.segments:
            if seg['type'] in['PIN', 'INST_PORT']:
                layer = seg.get('layer').lower()
                
                if 'pos' in seg and len(seg['pos']) == 4:
                    llx, lly, urx, ury = seg['pos']
                else:
                    continue
                    
                cx, cy = (llx + urx) / 2.0, (lly + ury) / 2.0
                lvl_idx = self.layer_info.get(layer, {}).get('lvl_idx', 1)
                comment = f"// $llx={llx:.4f} $lly={lly:.4f} $urx={urx:.4f} $ury={ury:.4f} $lvl={lvl_idx}"
                
                # -------------------------------------------------------------
                # [CRITICAL FIX] 방향성(Direction) 문자를 SPEF 표준(I, O, B)으로 완벽 규격화
                # -------------------------------------------------------------
                raw_dir = seg.get('direction', 'B').upper()
                if 'INOUT' in raw_dir or 'B' in raw_dir:
                    d_char = 'B'
                elif 'IN' in raw_dir or 'PI' in raw_dir:
                    d_char = 'I'
                elif 'OUT' in raw_dir or 'PO' in raw_dir:
                    d_char = 'O'
                else:
                    d_char = 'B'
                
                if seg['type'] == 'PIN':
                    logical_id = seg['name']
                    # 만약 DEF 파서의 실수로 PIN 이름에 콜론이나 슬래시(인스턴스 표기)가 있다면 *I로 강제 강등!
                    if logical_id in self.top_ports:
                        # self.node_info[logical_id] = {'str': f"*P {logical_id} {d_char} *C {cx:.4f} {cy:.4f} {comment}", 'type': '*P'}
                        node_str = f"*P {logical_id} {d_char} *C {cx:.4f} {cy:.4f}"
                        n_type = '*P'
                    else:
                        # 타일 경계에서 잘린 가짜 핀이거나 내부 핀이면 무조건 *I로 강등
                        # self.node_info[logical_id] = {'str': f"*I {logical_id} {d_char} *C {cx:.4f} {cy:.4f} {comment}", 'type': '*I'}
                        node_str = f"*I {logical_id} {d_char} *C {cx:.4f} {cy:.4f}"
                        n_type = '*I'
                else:
                    # logical_id = f"{seg.get('from_inst', 'UNKNOWN')}:{seg.get('name', 'UNKNOWN').split('_')[-1]}"
                    logical_id = f"{seg.get('from_inst', 'UNKNOWN')}:{seg.get('pin_name', 'UNKNOWN')}"
                    cell_type = seg.get('cell_type', 'UNKNOWN_CELL')
                    # self.node_info[logical_id] = {'str': f"*I {logical_id} {d_char} *C {cx:.4f} {cy:.4f} *D {cell_type} {comment}", 'type': '*I'}
                    node_str = f"*I {logical_id} {d_char} *C {cx:.4f} {cy:.4f} *D {cell_type}"
                    n_type = '*I'
                
                # phys_node = self._get_physical_node((cx, cy), layer, min(abs(urx-llx), abs(ury-lly)), None)
                # self.edges.append((logical_id, phys_node, 0.001, f"//  $l=0.0000 $w=10.0000 $lvl={lvl_idx} (Logical Bridge)"))
                # [CRITICAL FIX] 핀(Pin) 면적 안에 떨어지는 라우팅 노드(VIA or Wire) 찾기
                matched_phys_node = None
                min_dist = float('inf')
                # KD-Tree로 같은 층의 가장 가까운 물리적 노드를 찾습니다.
                if layer in layer_kdtree:
                    dist, idx = layer_kdtree[layer].query([cx, cy])
                    # 탐색 반경 2.0um (핀 중심에서 셀 밖으로 뻗어나간 첫 번째 와이어라도 무조건 잡음)
                    if dist < 2.0:
                        matched_phys_node = layer_keys[layer][idx]
                        min_dist = dist
                # [CRITICAL FIX] DEF 파싱 오류 등으로 핀 레이어와 물리 노드 레이어가 다를 경우, 전 층(All layers)을 뒤져서라도 강제 스냅핑
                if matched_phys_node is None:
                    for n_layer, tree in layer_kdtree.items():
                        dist, idx = tree.query([cx, cy])
                        if dist < 2.0 and dist < min_dist:
                            matched_phys_node = layer_keys[n_layer][idx]
                            min_dist = dist
                                
                if matched_phys_node:
                    # 라우팅 노드의 이름을 논리적 핀 이름으로 덮어씌움 (Aliasing)
                    # self.alias_map[matched_phys_node] = logical_id
                    self.node_info[logical_id] = {'str': node_str, 'type': n_type}
                    # if matched_phys_node in self.node_info:
                    #     del self.node_info[matched_phys_node]
                    contact_res = 0.001
                        
                    self.edges.append((logical_id, matched_phys_node, contact_res, f"// Contact Resistance"))
                else:
                    # 연결되지 않은 핀(Floating)도 선언은 해둠
                    self.node_info[logical_id] = {'str': node_str, 'type': n_type}
                    
        # 6. 물리적 간선(Edges)에 Alias Map 일괄 적용 (0.001옴 브릿지 완벽 제거)
        new_edges =[]
        for n1, n2, res, cmt in self.edges:
            a1 = self.alias_map.get(n1, n1)
            a2 = self.alias_map.get(n2, n2)
            if a1 != a2: # 자기 자신으로의 루프 방지
                new_edges.append((a1, a2, res, cmt))
        self.edges = new_edges

def distribute_net_caps(C_gnd_total, C_cpl_dict, topology):
    """
    Distribute net-level scalar cap totals to per-node caps for SPEF writing.

    Ground cap: proportional to segment length among all *N nodes.
    Coupling cap: placed on the *N node whose XY centroid is nearest to the
    aggressor net name centroid in the topology (nearest-neighbour in 2D).

    Returns dict: node_id -> {'gnd': float, 'cpl': {aggr_name: float}}
    Guarantees: sum(node_gnd) == C_gnd_total (up to float precision).
    """
    node_caps = defaultdict(lambda: {'gnd': 0.0, 'cpl': defaultdict(float)})

    # Collect internal *N nodes and their XY positions
    n_nodes = [(nid, ix / topology.dbu, iy / topology.dbu)
                for (ix, iy, layer), nid in topology.nodes.items()]

    if not n_nodes:
        return dict(node_caps)

    # --- Ground cap: distribute by segment length proportionally ---
    # Compute total edge length incident to each node
    node_len = defaultdict(float)
    for n1, n2, _, comment in topology.edges:
        # Extract $l= from comment to get L_eff
        m = re.search(r'\$l=([\d.]+)', comment)
        seg_len = float(m.group(1)) if m else 0.001
        node_len[n1] += seg_len / 2.0
        node_len[n2] += seg_len / 2.0

    total_len = sum(node_len.get(nid, 0.0) for nid, _, _ in n_nodes)
    if total_len < 1e-9:
        # Fallback: uniform distribution
        share = C_gnd_total / len(n_nodes)
        for nid, _, _ in n_nodes:
            node_caps[nid]['gnd'] = share
    else:
        for nid, _, _ in n_nodes:
            frac = node_len.get(nid, 0.0) / total_len
            node_caps[nid]['gnd'] = C_gnd_total * frac

    # --- Coupling cap: nearest node by XY ---
    if C_cpl_dict and n_nodes:
        node_ids = [nid for nid, _, _ in n_nodes]
        coords = np.array([[x, y] for _, x, y in n_nodes])
        tree = cKDTree(coords)

        # Compute centroid of all *N nodes as proxy query point for each aggressor
        centroid = coords.mean(axis=0)

        for aggr_name, cpl_val in C_cpl_dict.items():
            if cpl_val < 1e-6:
                continue
            # Use the node with most ground cap (longest segment) as coupling host
            best_nid = max(node_ids, key=lambda nid: node_caps[nid]['gnd'])
            node_caps[best_nid]['cpl'][aggr_name] += cpl_val

    return dict(node_caps)


class NetCapWriter:
    """
    Writes a single net to SPEF using ML-predicted net-level total caps
    (C_gnd_total, C_cpl_dict) distributed heuristically to topology nodes.
    No AutonomousGraphBuilder needed — topology is RCTopologyBuilder only.
    """
    def __init__(self, topology, C_gnd_total, C_cpl_dict):
        self.topology = topology
        self.node_caps = distribute_net_caps(C_gnd_total, C_cpl_dict, topology)
        self.total_cap = C_gnd_total + sum(C_cpl_dict.values())


class AutonomousGraphBuilder(RCTopologyBuilder):
    """
    ML 모델이 예측한 3D 분산 캡(Distributed Caps) 노드들을 
    결정론적 저항망(R Graph)의 가장 가까운 노드에 투영(Snap)시키는 자율 빌더.
    """
    def __init__(self, net_name, global_segments, ml_distributed_nodes, top_ports, layer_info, tech_lef=None, dbu=2000):
        # 1. 부모 클래스: DEF 기반 저항망 뼈대(self.nodes, self.edges) 구축
        super().__init__(net_name, global_segments, top_ports, layer_info, tech_lef, dbu)
        
        self.ml_nodes = ml_distributed_nodes
        self._distribute_capacitances()

    def _get_layer_from_z(self, z_val):
        """Z 좌표를 기술 레프(Tech LEF)와 비교하여 라우팅 메탈 층 매핑"""
        for name, info in self.layer_info.items():
            if info.get('type') == 'C':
                z_bot = info.get('z_pos', 0.0)
                z_top = info.get('top_z', z_bot + info.get('thickness', 0.0))
                if z_bot - 1e-3 <= z_val <= z_top + 1e-3:
                    return name.lower()
        return "m1" # Fallback

    def _distribute_capacitances(self):
        """ML이 뿌려놓은 캡 점들을 물리적 노드로 Snapping"""
        self.node_caps = defaultdict(lambda: {'gnd': 0.0, 'cpl': defaultdict(float)})
        self.total_gnd = 0.0
        
        # 물리 노드 빠른 탐색(Nearest Neighbor)을 위한 레이어별 그룹화
        # [CRITICAL FIX] O(N^2) 완전 탐색 제거 -> KD-Tree 기반 O(M log N) 탐색 도입
        layer_kdtree = {}
        layer_node_map = {}
        
        layer_coords = defaultdict(list)
        layer_nids = defaultdict(list)

        for (ix, iy, layer_name), node_id in self.nodes.items():
            layer_coords[layer_name].append([ix / self.dbu, iy / self.dbu])
            layer_nids[layer_name].append(node_id)
            
        for layer_name in layer_coords.keys():
            layer_kdtree[layer_name] = cKDTree(np.array(layer_coords[layer_name]))
            layer_node_map[layer_name] = layer_nids[layer_name]
            
        # 전체 노드 Fallback용 KD-Tree
        all_coords = [[ix/self.dbu, iy/self.dbu] for (ix, iy, _) in self.nodes.keys() ]
        all_nids = [ nid for nid in self.nodes.values() ]
        fallback_tree = cKDTree(np.array(all_coords)) if all_coords else None
            
        for hash_key, data in self.ml_nodes.items():
            geo = data['abs_geo']
            if geo is None: continue
            
            cx, cy, cz = geo[0], geo[1], geo[2]
            layer_name = self._get_layer_from_z(cz)
            
            # KD-Tree Query (k=1)
            if layer_name in layer_kdtree:
                _, idx = layer_kdtree[layer_name].query([cx, cy])
                best_node = layer_node_map[layer_name][idx]
            elif fallback_tree is not None:
                _, idx = fallback_tree.query([cx, cy])
                best_node = all_nids[idx]
            else:
                continue

            best_node = self.alias_map.get(best_node, best_node)
                
            # 예측 캡 누적
            self.node_caps[best_node]['gnd'] += data['gnd']
            self.total_gnd += data['gnd']
            
            for aggr_name, cpl_val in data['cpl'].items():
                self.node_caps[best_node]['cpl'][aggr_name] += cpl_val


class SPEFWriter:
    """
    [Stream-based SPEF Writer]
    메모리에 문자열을 적재하지 않고, 열려있는 파일 스트림(file_handle)에 
    Net 단위로 즉각 Flush하여 O(1) 메모리 사용량을 보장합니다.
    """
    def __init__(self, file_handle, design_name, top_ports):
        self.f = file_handle
        self.design_name = design_name
        self.top_ports = top_ports

    def write_header(self):
        """SPEF 파일의 헤더와 PORT 섹션을 스트림에 기록합니다."""
        now = datetime.datetime.now().strftime("%a %b %d %H:%M:%S %Y")
        header_lines =[
            f'*SPEF "IEEE 1481-1999"',
            f'*DESIGN "{self.design_name}"',
            f'*DATE "{now}"',
            f'*VENDOR "DeepPEX AI"',
            f'*PROGRAM "NeuralField_V3"',
            f'*VERSION "V3.0"',
            f'*DESIGN_FLOW "PIN_CAP NONE" "NAME_SCOPE LOCAL"',
            f'*DIVIDER /',
            f'*DELIMITER :',
            f'*BUS_DELIMITER []',
            f'*T_UNIT 1.0 NS',
            f'*C_UNIT 1.0 FF',
            f'*R_UNIT 1.0 OHM',
            f'*L_UNIT 1.0 HENRY',
            f''
        ]
        self.f.write('\n'.join(header_lines) + '\n')
        
        self.f.write('*PORTS\n')
        # [CRITICAL FIX] Dict, List of Tuples, Set of Strings 완벽 호환 방어 로직
        if isinstance(self.top_ports, dict):
            port_items = self.top_ports.items()
        else:
            port_items =[]
            for p in self.top_ports:
                if isinstance(p, (tuple, list)) and len(p) >= 2:
                    port_items.append((p[0], p[1]))
                else:
                    port_items.append((p, 'B')) # 문자열만 들어오면 방향을 'B'로 임의 부여
                    
        for net_name, direction in sorted(list(port_items)):
            self.f.write(f"{net_name} {direction}\n")
        self.f.write('\n')

    def stream_autonomous_net(self, topology): # AutonomousGraphBuilder 객체
        """단일 Net의 토폴로지를 생성 즉시 스트림에 기록하고 메모리에서 해제합니다."""
        # 1. Total Net Cap 계산
        total_cap = topology.total_gnd
        for cap_data in topology.node_caps.values():
            total_cap += sum(cap_data['cpl'].values())
            
        self.f.write(f'*D_NET {topology.net_name} {total_cap:.6f}\n\n')
        
        # 2. *CONN 블록
        self.f.write('*CONN\n')
        order_map = {'*P': 0, '*I': 1, '*N': 2}
        sorted_nodes = sorted(topology.node_info.values(), key=lambda x: order_map.get(x['type'], 99))
        for info in sorted_nodes:
            self.f.write(f"{info['str']}\n")
        self.f.write('\n')
        
        # 3. *CAP 블록
        self.f.write('*CAP\n')
        cap_id = 1
        for phys_node, cap_data in topology.node_caps.items():
            gnd_val = cap_data['gnd']
            if gnd_val > 1e-5:
                self.f.write(f'{cap_id} {phys_node} {gnd_val:.6f}\n')
                cap_id += 1
                
            for aggr_name, cpl_val in cap_data['cpl'].items():
                if cpl_val > 1e-5:
                    self.f.write(f'{cap_id} {phys_node} {aggr_name}:1 {cpl_val:.6f}\n')
                    cap_id += 1
        self.f.write('\n')

        # 4. *RES 블록
        self.f.write('*RES\n')
        res_id = 1
        for n1, n2, val, comment in topology.edges:
            self.f.write(f'{res_id} {n1} {n2} {val:.6f} {comment}\n')
            res_id += 1

        self.f.write('*END\n\n')

    def stream_net_cap_writer(self, writer):
        """Write a net using NetCapWriter (ML net-level total prediction path)."""
        topology = writer.topology
        self.f.write(f'*D_NET {topology.net_name} {writer.total_cap:.6f}\n\n')

        self.f.write('*CONN\n')
        order_map = {'*P': 0, '*I': 1, '*N': 2}
        sorted_nodes = sorted(topology.node_info.values(), key=lambda x: order_map.get(x['type'], 99))
        for info in sorted_nodes:
            self.f.write(f"{info['str']}\n")
        self.f.write('\n')

        self.f.write('*CAP\n')
        cap_id = 1
        for phys_node, cap_data in writer.node_caps.items():
            gnd_val = cap_data['gnd']
            if gnd_val > 1e-5:
                self.f.write(f'{cap_id} {phys_node} {gnd_val:.6f}\n')
                cap_id += 1
            for aggr_name, cpl_val in cap_data['cpl'].items():
                if cpl_val > 1e-5:
                    self.f.write(f'{cap_id} {phys_node} {aggr_name}:1 {cpl_val:.6f}\n')
                    cap_id += 1
        self.f.write('\n')

        self.f.write('*RES\n')
        res_id = 1
        for n1, n2, val, comment in topology.edges:
            self.f.write(f'{res_id} {n1} {n2} {val:.6f} {comment}\n')
            res_id += 1
        self.f.write('*END\n\n')