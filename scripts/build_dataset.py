# scripts/build_dataset.py
import sys
import argparse
import numpy as np
import pickle
import gzip
import torch
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import multiprocessing as mp
import pandas as pd

# Project Root Setup
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing.def_parser import DefStreamParser
from src.preprocessing.layer_parser import LayerInfoParser
from src.preprocessing.lef_parser import LefParser
from src.preprocessing.cell_parser import CellLibParser
from src.preprocessing.tiling import NetTiler
from src.physics.materials import BEOLMaterialStack
from src.data.tensorizer import FeatureTensorizer
from src.utils.naming import NameRegistry
import configs.config as cfg


# (A) 토폴로지 저장 병렬 수행 함수 (내부 정의)
def _save_topology_worker(task):
    path, data = task
    with gzip.open(path, 'wb', compresslevel=3) as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    # torch.save(data, pt_path)

def transform_point(x, y, orient, w, h):
    if orient == 'N': return x, y
    elif orient == 'S': return w - x, h - y
    elif orient == 'FN': return x, h - y 
    elif orient == 'FS': return w - x, y
    return x, y

def get_inst_pin_rects(instances, cell_lib, origin):
    ox, oy = origin[0], origin[1]
    pin_rects_by_layer = {}
    for _, inst in instances.items():
        cell_type = inst['cell_type']
        if cell_type not in cell_lib: continue
        inst_x, inst_y = inst['x'], inst['y']
        orient = inst['orient']
        cell_w, cell_h = cell_lib[cell_type]['size']
        for pin_name, pin_data in cell_lib[cell_type]['pins'].items():
            for metal in pin_data:
                layer = metal['layer']
                rx1, ry1, rx2, ry2 = metal['rect']
                pts = [transform_point(rx1, ry1, orient, cell_w, cell_h),
                       transform_point(rx2, ry2, orient, cell_w, cell_h),
                       transform_point(rx1, ry2, orient, cell_w, cell_h),
                       transform_point(rx2, ry1, orient, cell_w, cell_h)]
                txs = [p[0] for p in pts]
                tys = [p[1] for p in pts]
                abs_x1, abs_x2 = min(txs) + inst_x, max(txs) + inst_x
                abs_y1, abs_y2 = min(tys) + inst_y, max(tys) + inst_y
                
                if layer not in pin_rects_by_layer: pin_rects_by_layer[layer] = []
                pin_rects_by_layer[layer].append((abs_x1 - ox, abs_y1 - oy, abs_x2 - ox, abs_y2 - oy))
    return pin_rects_by_layer

def clip_line_by_rect(p1, p2, rect):
    x1, y1 = p1
    x2, y2 = p2
    rx1, ry1, rx2, ry2 = rect
    if max(x1, x2) < rx1 or min(x1, x2) > rx2 or max(y1, y2) < ry1 or min(y1, y2) > ry2:
        return p1, p2, False
    if abs(y1 - y2) < 1e-9: # H
        if y1 < ry1 or y1 > ry2: return p1, p2, False
        if rx1 <= x1 <= rx2: x1 = rx2 if x2 > x1 else rx1
        if rx1 <= x2 <= rx2: x2 = rx2 if x1 > x2 else rx1
    elif abs(x1 - x2) < 1e-9: # V
        if x1 < rx1 or x1 > rx2: return p1, p2, False
        if ry1 <= y1 <= ry2: y1 = ry2 if y2 > y1 else ry1
        if ry1 <= y2 <= ry2: y2 = ry2 if y1 > y2 else ry1
    return (x1, y1), (x2, y2), True

def clip_segments_by_inst_pins(segments, pin_rects_by_layer):
    processed = []
    for seg in segments:
        if seg['type'] != 'WIRE':
            processed.append(seg)
            continue
        layer = seg.get('layer')
        if layer not in pin_rects_by_layer:
            processed.append(seg)
            continue
        p1, p2 = seg['start'], seg['end']
        valid_seg = True
        for rect in pin_rects_by_layer[layer]:
            p1, p2, _ = clip_line_by_rect(p1, p2, rect)
            if (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 < 1e-9:
                valid_seg = False; break
        if valid_seg:
            new_seg = seg.copy()
            new_seg['start'], new_seg['end'] = p1, p2
            processed.append(new_seg)
    return processed

def clip_and_shift_segment(seg, win_min, win_max, origin):
    # (이전 코드와 동일 - 생략시 에러나므로 필수 구현 필요)
    # 지면 관계상 핵심 로직만 유지
    ox, oy = origin[0], origin[1]
    xmin, ymin = win_min
    xmax, ymax = win_max
    new_seg = seg.copy()
    if seg['type'] == 'WIRE':
        x1, y1 = seg['start']
        x2, y2 = seg['end']
        x1_c = max(xmin, min(xmax, x1))
        x2_c = max(xmin, min(xmax, x2))
        y1_c = max(ymin, min(ymax, y1))
        y2_c = max(ymin, min(ymax, y2))
        
        if abs(x1-x2) > 1e-9: 
             if abs(x1_c - x2_c) < 1e-9: return None
             if y1 < ymin or y1 > ymax: return None
             y1_c = y2_c = y1
        elif abs(y1-y2) > 1e-9:
             if abs(y1_c - y2_c) < 1e-9: return None
             if x1 < xmin or x1 > xmax: return None
             x1_c = x2_c = x1
        else:
             if not (xmin <= x1 <= xmax and ymin <= y1 <= ymax): return None
             x1_c, y1_c = x1, y1; x2_c, y2_c = x2, y2
        new_seg['start'] = (x1_c - ox, y1_c - oy)
        new_seg['end'] = (x2_c - ox, y2_c - oy)
    elif seg['type'] == 'RECT':
        r = seg['rect']
        rx1, ry1, rx2, ry2 = max(xmin, r[0]), max(ymin, r[1]), min(xmax, r[2]), min(ymax, r[3])
        if rx1 >= rx2 or ry1 >= ry2: return None
        new_seg['rect'] = (rx1 - ox, ry1 - oy, rx2 - ox, ry2 - oy)
    elif seg['type'] == 'VIA':
        vx, vy = seg['pos']
        if not (xmin <= vx <= xmax and ymin <= vy <= ymax): return None
        new_seg['pos'] = (vx - ox, vy - oy)
    elif seg['type'] in ['PIN', 'INST_PORT']:
        r = seg['pos']
        rx1, ry1, rx2, ry2 = max(xmin, r[0]), max(ymin, r[1]), min(xmax, r[2]), min(ymax, r[3])
        if rx1 >= rx2 or ry1 >= ry2: return None
        new_seg['pos'] = (rx1 - ox, ry1 - oy, rx2 - ox, ry2 - oy)
    return new_seg

def filter_short_segments(segments, min_len=0.001):
    filtered = []
    for seg in segments:
        # VIA, PIN, INST_PORT 등은 기하학적 길이에 상관없이 무조건 보존하여 
        # 타겟 넷 주변의 커플링 노드가 통째로 날아가는 현상 차단
        if seg['type'] not in ['WIRE', 'RECT']:
            filtered.append(seg)
            continue
        
        if seg['type'] == 'WIRE':
            l = np.sqrt((seg['start'][0]-seg['end'][0])**2 + (seg['start'][1]-seg['end'][1])**2)
            # 커플링에 참여하는 핵심 짧은 라우팅 보존을 위해 기준을 1nm 수준으로 대폭 완화
            if l >= 1e-5: filtered.append(seg) 
        elif seg['type'] == 'RECT':
            if abs(seg['rect'][2]-seg['rect'][0]) >= 1e-5 and abs(seg['rect'][3]-seg['rect'][1]) >= 1e-5:
                filtered.append(seg)
    return filtered

class SpatialGrid:
    def __init__(self, bin_size_x, bin_size_y):
        self.bx = bin_size_x
        self.by = bin_size_y
        self.grid = defaultdict(list)

    def build_from_cuboids(self, global_geo):
        """O(N) 속도로 칩 전체의 메탈을 해시 버킷(Grid)에 던져 넣습니다."""
        mins = global_geo[:, :2] - global_geo[:, 3:5] / 2
        maxs = global_geo[:, :2] + global_geo[:, 3:5] / 2
        
        min_idx = np.floor(mins / [self.bx, self.by]).astype(np.int32)
        max_idx = np.floor(maxs / [self.bx, self.by]).astype(np.int32)
        
        for i in range(len(global_geo)):
            for x in range(min_idx[i, 0], max_idx[i, 0] + 1):
                for y in range(min_idx[i, 1], max_idx[i, 1] + 1):
                    self.grid[(x, y)].append(i)

    def query_window(self, center, window_size):
        """O(1) 해시 매핑으로 특정 타일 윈도우 주변의 큐보이드 인덱스만 쏙 빼옵니다."""
        cx, cy = center[0], center[1]
        wx, wy = window_size[0], window_size[1]
        
        min_x, max_x = cx - wx/2, cx + wx/2
        min_y, max_y = cy - wy/2, cy + wy/2
        
        min_idx_x, min_idx_y = int(np.floor(min_x / self.bx)), int(np.floor(min_y / self.by))
        max_idx_x, max_idx_y = int(np.floor(max_x / self.bx)), int(np.floor(max_y / self.by))
        
        indices = set()
        for x in range(min_idx_x, max_idx_x + 1):
            for y in range(min_idx_y, max_idx_y + 1):
                if (x, y) in self.grid:
                    indices.update(self.grid[(x, y)])
        return list(indices)

# --- Worker Logic (Global Scope for Pickling) ---
# 전역 변수로 데이터 공유 (Linux Fork 활용)
_global_geo = None
_net_data = None
_pin_data = None
_tiler = None
_tensorizer = None
_out_path = None
_pt_out_path = None
_def_name = None
_name_registry = None
_spatial_grid = None


def init_worker(global_geo, net_data, pin_data, tiler, tensorizer, out_path, pt_out_path, def_name, name_registry, spatial_grid):
    global _global_geo, _net_data, _pin_data, _tiler, _tensorizer, _out_path, _pt_out_path, _def_name, _name_registry, _spatial_grid
    _global_geo = global_geo
    _net_data = net_data
    _pin_data = pin_data
    _tiler = tiler
    _tensorizer = tensorizer
    _out_path = out_path
    _pt_out_path = pt_out_path
    _def_name = def_name
    _name_registry = name_registry
    _spatial_grid = spatial_grid

def process_window_job(job_data):
    """
    Worker Process (수십억 번의 파이썬 for 루프가 완벽히 소멸된 버전)
    """
    nid, w_idx, center, context_size, win_size, net_name, save_name = job_data
    
    # 1. Spatial Grid Query (O(1) 속도)
    nearby_indices = _spatial_grid.query_window(center, context_size)
    if not nearby_indices: return None
    
    nearby_geo = _global_geo[nearby_indices]
    nearby_ids = nearby_geo[:, 6]
    
    # 2. Logic Classification (Target / Aggressor / Pin)
    types = np.zeros(len(nearby_geo), dtype=np.int32)
    margin = 1.5
    core_min_x, core_max_x = center[0] - win_size[0]/2.0 + margin, center[0] + win_size[0]/2.0 - margin
    core_min_y, core_max_y = center[1] - win_size[1]/2.0 + margin, center[1] + win_size[1]/2.0 - margin

    # 큐보이드의 Bounding Box 계산
    box_min_x = nearby_geo[:, 0] - nearby_geo[:, 3] / 2.0
    box_max_x = nearby_geo[:, 0] + nearby_geo[:, 3] / 2.0
    box_min_y = nearby_geo[:, 1] - nearby_geo[:, 4] / 2.0
    box_max_y = nearby_geo[:, 1] + nearby_geo[:, 4] / 2.0

    # 중심점이 아니라, BBox가 단 1nm라도 코어와 겹치면(Intersection) True!
    is_in_core = (box_max_x > core_min_x) & (box_min_x < core_max_x) & \
                    (box_max_y > core_min_y) & (box_min_y < core_max_y)

    # 1. 1차 분류: 모든 Node를 Target과 Aggressor로 나눔
    is_tgt_mask = np.zeros(len(nearby_geo), dtype=bool)
    is_tgt_mask[(nearby_ids == nid)] = True
    for i, uid in enumerate(nearby_ids):
        if uid < 0:
            pid = int(uid)
            if pid in _pin_data and _pin_data[pid]['net_name'] == net_name:
                is_tgt_mask[i] = True

    # 2. 2차 분류: 최종 Type 할당
    types = np.zeros(len(nearby_geo), dtype=np.int32)
    types[is_tgt_mask & is_in_core] = 1      # 드디어 거대 넷이 type=1 로 살아남습니다!
    types[is_tgt_mask & ~is_in_core] = 3     # 문맥용 타겟
    types[~is_tgt_mask] = 3                  # 모든 Aggressor
    
    for i, uid in enumerate(nearby_ids):
        if uid < 0:
            pid = int(uid)
            if pid in _pin_data:
                is_tgt = (_pin_data[pid]['net_name'] == net_name)
                types[i] = 2 if is_tgt else 4 # Pin

    tensor_mask = (types > 0)
    if not np.any(tensor_mask): return None
    
    masked_geo = nearby_geo[tensor_mask]
    masked_types = types[tensor_mask]
    masked_ids_filtered = nearby_ids[tensor_mask]
    
    # 3. NumPy Vectorized Clipping & Tensorizing
    clipped_geo, clipped_types = _tiler.clip_cuboids(masked_geo[:, :6], masked_types, center)
    if clipped_geo is None or not np.any(clipped_types == 1): return None
    
    final_tensor = _tensorizer.process(clipped_geo, clipped_types, center).astype(np.float32)
    
    # 4. ML Node Mapping Information (결정론적 저항망 구축용)
    abs_geometries = clipped_geo.copy().astype(np.float32)
    ctx_min, ctx_max = center - context_size / 2.0, center + context_size / 2.0
        
    c_mins = masked_geo[:, :3] - masked_geo[:, 3:6] / 2
    c_maxs = masked_geo[:, :3] + masked_geo[:, 3:6] / 2
    inter_min = np.maximum(c_mins, ctx_min)
    inter_max = np.minimum(c_maxs, ctx_max)
    dims = inter_max - inter_min
    
    # 교차 부피가 존재하는 유효 큐보이드만 100% 보존 (Halo 포함)
    valid_mask = np.all(dims > 1e-9, axis=1)
    if not np.any(valid_mask): return None
    
    clipped_geo = masked_geo[valid_mask].copy()
    clipped_types = masked_types[valid_mask]
    valid_ids = masked_ids_filtered[valid_mask]
    
    if not np.any(clipped_types == 1): return None
    
    # 클리핑된 크기와 중심으로 큐보이드 좌표 정밀 업데이트
    clipped_geo[:, 3:6] = dims[valid_mask]
    clipped_geo[:, :3] = inter_min[valid_mask] + dims[valid_mask] / 2.0
    
    # Tensorizer에는 6차원(x,y,z,w,h,d)만 전달
    final_tensor = _tensorizer.process(clipped_geo[:, :6], clipped_types, center).astype(np.float32)
    abs_geometries = clipped_geo[:, :6].copy().astype(np.float32)
    
    cuboid_net_names = []
    for uid in valid_ids:
        if uid == nid:
            cuboid_net_names.append(net_name)
        elif uid >= 0:
            cuboid_net_names.append(_net_data[int(uid)]['name'])
        else:
            pid = int(uid)
            cuboid_net_names.append(_pin_data[pid]['net_name'] if pid in _pin_data else "UNKNOWN_PIN")

    # 5. Fast I/O Save (무거운 Segment 정보 완전 배제)
    safe_net_name = _name_registry.register(net_name)
    save_data = {
        'cuboids': final_tensor,
        'net_name': net_name,
        'def_name': _def_name,
        'origin': center,
        'window_idx': w_idx,
        'abs_geometries': abs_geometries,
        'cuboid_net_names': cuboid_net_names
    }
    
    with gzip.open(_out_path / f"{save_name}", 'wb', compresslevel=3) as f:
        pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # pt_save_name = save_name.replace('.pkl.gz', '.pt')
    # pt_save_data = save_data.copy()
    # pt_save_data['cuboids'] = torch.from_numpy(pt_save_data['cuboids']).float()
    # pt_save_data['abs_geometries'] = torch.from_numpy(pt_save_data['abs_geometries']).float()
    # torch.save(pt_save_data, _pt_out_path / pt_save_name)
        
    return {
        'sample_filename': save_name,
        'def_name': _def_name,
        'net_name': net_name,
        'tile_idx': w_idx
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--def_path', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default=str(cfg.PROCESSED_DIR))
    parser.add_argument('--pt_out_dir', type=str, default=str(cfg.PT_DIR))
    parser.add_argument('--num_workers', type=int, default=16, help="Number of parallel workers")
    args = parser.parse_args()
    
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    pt_out_path = Path(args.pt_out_dir)
    pt_out_path.mkdir(parents=True, exist_ok=True)
    def_path = Path(args.def_path)

    # 1. Parsing
    print(">>>[1/4] Parsing Tech & DEF...")
    layer_map = LayerInfoParser(cfg.LAYERS_INFO_PATH).parse()
    tech_lef = LefParser(cfg.TECH_LEF_PATH).parse()
    cell_lib = CellLibParser(cfg.CELL_LEF_PATH).parse()
    mat_stack = BEOLMaterialStack(layer_map)
    tensorizer = FeatureTensorizer(mat_stack)
    
    def_parser = DefStreamParser(args.def_path, layer_map, tech_lef, cell_lib)
    
    all_cuboids_list =[]
    net_data = {}
    _pin_data, pin_data = {}, {}
    net_id_counter = 0
    pin_id_counter = 1
    name_registry = NameRegistry()
    
    for net_name, cuboids, segments in tqdm(def_parser.parse(), desc="Parsing Geometry"):
        if not cuboids.size > 0: continue
        is_pin = 'PIN' in net_name.upper() or 'INST_PORT' in net_name.upper()
        
        if is_pin:
            pid = int(-pin_id_counter)
            pin_id_counter += 1
            _pin_data[pid] = {'name': net_name, 'segments': segments, 'net_name': None, 'cuboids': cuboids}
            ids = np.full((len(cuboids), 1), pid, dtype=np.float32)
        else:
            nid = net_id_counter
            net_id_counter += 1
            # [최적화] 나중에 검색하지 않도록 파싱 시점에 cuboids 텐서를 통째로 메모리에 쥠
            net_data[nid] = {'name': net_name, 'segments': segments, 'cuboids': cuboids}
            ids = np.full((len(cuboids), 1), nid, dtype=np.float32)
            
        all_cuboids_list.append(np.hstack([cuboids, ids]))

    if not all_cuboids_list: return
    
    # Post-process Pin Data
    for pid, p_info in _pin_data.items():
        p_name = p_info['name']
        if p_name.startswith("PIN_"):
            real = p_name.replace("PIN_", "").split('_')[:-1]
            real = '_'.join(real)
            if real in def_parser.pins:
                p_info['net_name'] = def_parser.pins[real]['net_name']
                pin_data[pid] = p_info
        elif p_name.startswith("INST_PORT_"):
            if p_info['segments']:
                seg = p_info['segments'][0]
                if 'from_inst' in seg:
                    key = (seg['from_inst'], seg['name'].replace("INST_PORT_", "").split('_')[-1])
                    if key in def_parser.inst_net_map:
                        p_info['net_name'] = def_parser.inst_net_map[key]
                        pin_data[pid] = p_info

    top_ports =[
        (name, 'I' if 'IN' in info.get('direction', '').upper() else 'O' if 'OUT' in info.get('direction', '').upper() else 'B')
        for name, info in getattr(def_parser, 'pins', {}).items() if info.get('type', 'PIN') == 'PIN'
    ]

    with gzip.open(out_path / f"{def_path.stem}_inst_net_map.pkl.gz", 'wb') as f:
        pickle.dump(def_parser.inst_net_map, f)
        
    global_geometry = np.vstack(all_cuboids_list)
    print(f"    Total Cuboids: {len(global_geometry)}")

    # 2. O(N) Spatial Hashing
    print(">>>[2/4] Building Spatial Grid Hash Map...")
    spatial_grid = SpatialGrid(bin_size_x=cfg.WINDOW_SIZE[0], bin_size_y=cfg.WINDOW_SIZE[1])
    spatial_grid.build_from_cuboids(global_geometry)

    # 3. Job Generation (O(1) 속도 최적화의 끝판왕)
    print(">>> [3/4] Generating Jobs...")
    halo_margin = 1.5
    tiler = NetTiler(cfg.WINDOW_SIZE, 2.0 * halo_margin) # Stride = Window - 2*Margin
    context_margin = 2.0
    context_size = np.array([cfg.WINDOW_SIZE[0]+2*context_margin, cfg.WINDOW_SIZE[1]+2*context_margin, cfg.WINDOW_SIZE[2]])
    win_size = np.array(cfg.WINDOW_SIZE)
    
    # -----------------------------------------------------------------
    # [CRITICAL FIX 1] OS 디스크 I/O 병목 제거
    # 디스크를 100만 번 찌르지 않고, 폴더 내 파일 목록을 메모리(Set)에 1번만 로드
    # -----------------------------------------------------------------
    print("    - Scanning existing files in memory...")
    existing_tiles = set(f.name for f in out_path.glob("*.pkl.gz"))
    
    topo_out_path = out_path / "topology"
    topo_out_path.mkdir(parents=True, exist_ok=True)
    existing_topos = set(f.name for f in topo_out_path.glob("*.pkl.gz"))
    pt_topo_out_path = pt_out_path / "topology"
    pt_topo_out_path.mkdir(parents=True, exist_ok=True)
    # -----------------------------------------------------------------
    # [CRITICAL FIX 2] 500억 번의 파이썬 Loop 제거
    # Pin 데이터를 Net Name을 키(Key)로 하는 O(1) 딕셔너리로 사전 Grouping
    # -----------------------------------------------------------------
    pins_by_net = defaultdict(list)
    for pid, p_info in pin_data.items():
        n_name = p_info.get('net_name')
        if n_name:
            pins_by_net[n_name].append(p_info)

    jobs = []
    topo_jobs =[] # 토폴로지 저장용 병렬 큐
    generated_map =[]
    skip_count = 0
    
    target_nets = {nid: info for nid, info in net_data.items() 
                   if info['name'].lower() not in['vss', 'vdd', 'vcc', 'gnd', 'vssx', 'vccx']}
    def_stem = def_path.stem

    for nid, info in tqdm(target_nets.items(), desc="Job Generation"):
        net_name = info['name']
        safe_net_name = name_registry.register(net_name)
        
        # O(1) 토폴로지 구성 및 Job 큐 삽입
        topo_save_name = f"{def_stem}___topo_{safe_net_name}.pkl.gz"
        if topo_save_name not in existing_topos:
            global_segments = info['segments'].copy()
            # 50만 번의 루프가 O(1) 딕셔너리 참조로 교체됨!
            for p_info in pins_by_net.get(net_name, []):
                for s in p_info['segments']:
                    if 'direction' in p_info: s['direction'] = p_info['direction']
                    global_segments.append(s)
            
            # 무거운 압축 저장은 메인 스레드에서 하지 않고 모아둠
            topo_data = {'top_ports': top_ports, 'global_segments': global_segments}
            topo_jobs.append((topo_out_path / topo_save_name, topo_data))
            # pt_topo_save_name = topo_save_name.replace('.pkl.gz', '.pt')
            # topo_jobs.append((topo_out_path / topo_save_name, pt_topo_out_path / pt_topo_save_name, topo_data))

        # O(1) 타일 생성 및 Job 큐 삽입
        target_cuboids_list = [info['cuboids']]
        
        # 넷과 연결된 PIN/PORT 데이터 병합 (pins_by_net 또는 _pin_data 딕셔너리 구조에 맞게 순회)
        for p_info in pins_by_net.get(net_name, []): 
            if 'cuboids' in p_info and len(p_info['cuboids']) > 0:
                target_cuboids_list.append(p_info['cuboids'])
                
        # 완벽해진 기하학 정보로 타일 생성
        target_cuboids = np.vstack(target_cuboids_list)
        window_centers = tiler.get_tiling_windows(target_cuboids[:, :6])
        
        for w_idx, raw_center in enumerate(window_centers):
            save_name = f"{def_stem}__{safe_net_name}_tile{w_idx}.pkl.gz"
            
            # [FIX 1 적용] 메모리 Set 검색이므로 OS 락(Lock) 없이 0.000001초 만에 확인
            if save_name in existing_tiles:
                generated_map.append({'sample_filename': save_name, 'def_name': def_path.name, 'net_name': net_name, 'tile_idx': w_idx})
                skip_count += 1
                continue
                
            center = np.round(raw_center * def_parser.dbu) / def_parser.dbu
            jobs.append((nid, w_idx, center, context_size, win_size, net_name, save_name))

    print(f"    Total Tiles to Process: {len(jobs)} (Skipped: {skip_count})")
    print(f"    Total Topos to Save   : {len(topo_jobs)}")

    # =========================================================================
    # 4. Parallel Processing (Topology & Tiles)
    # =========================================================================
    if topo_jobs or jobs:
        print(f">>> [4/4] Parallel Processing with {args.num_workers} workers...")
        
        # -----------------------------------------------------------------
        # [CRITICAL FIX 2] Multiprocessing 워커 풀 완전 분리 및 안전한 초기화
        # starmap 꼼수를 버리고, 타일 워커 풀(Pool)을 생성할 때 initializer 로 
        # 전역 변수를 OS 레벨에서 완벽하게 주입(Bind)합니다.
        # -----------------------------------------------------------------
        
        # 4-1. 토폴로지 저장 (초기화 불필요)
        if topo_jobs:
            with mp.Pool(processes=args.num_workers) as pool_topo:
                list(tqdm(pool_topo.imap_unordered(_save_topology_worker, topo_jobs, chunksize=100), 
                          total=len(topo_jobs), desc="Saving Topologies"))
        
        # 4-2. 메인 타일 생성 (전역 변수 초기화 필수)
        if jobs:
            init_args = (global_geometry, net_data, pin_data, tiler, tensorizer, out_path, pt_out_path, def_path.name, name_registry, spatial_grid)
            
            # Pool 생성 시점에 init_worker를 묶어서, 어떤 워커가 스폰되든 무조건 전역 변수가 세팅되게 강제함
            with mp.Pool(processes=args.num_workers, initializer=init_worker, initargs=init_args) as pool_tiles:
                for result in tqdm(pool_tiles.imap_unordered(process_window_job, jobs, chunksize=50), 
                                   total=len(jobs), desc="Processing Tiles"):
                    if result:
                        generated_map.append(result)

    # 5. Finalize
    name_registry.save_csv(out_path.parent / f"{def_path.stem}_net_mapping.csv")
    if generated_map:
        df = pd.DataFrame(generated_map).sort_values(by=['net_name', 'tile_idx'])
        df.to_csv(out_path.parent / f"{out_path.name}_map.csv", index=False)
        print(f"\n>>> Done! Total Valid Samples: {len(df)}.")

if __name__ == "__main__":
    mp.set_start_method('fork', force=True) 
    main()