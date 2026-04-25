# src/preprocessing/cell_parser.py
import re
from typing import Dict, Any

class CellLibParser:
    """
    Standard Cell Library (LEF)를 파싱하여, 
    Cell별 Pin의 기하학적 정보(Layer, Rect)를 추출합니다.
    """
    def __init__(self, lef_path):
        self.lef_path = lef_path
        self.cell_lib = {} # { 'CELL_NAME': { 'size': (w, h), 'pins': { 'PIN_NAME': [ {layer, rect}, ... ] } } }

    def parse(self) -> Dict[str, Any]:
        print(f"Parsing Cell LEF: {self.lef_path}")
        
        current_macro = None
        current_pin = None
        current_size = (0.0, 0.0)
        
        re_macro = re.compile(r'^\s*MACRO\s+(\S+)')
        re_pin = re.compile(r'^\s*PIN\s+(\S+)')
        re_direction = re.compile(r'^\s*DIRECTION\s+(\S+)')
        re_size = re.compile(r'^\s*SIZE\s+([\d\.]+)\s+BY\s+([\d\.]+)')
        re_layer = re.compile(r'^\s*LAYER\s+(\S+)')
        re_rect = re.compile(r'^\s*RECT\s+([\d\.\-]+)\s+([\d\.\-]+)\s+([\d\.\-]+)\s+([\d\.\-]+)')
        re_end = re.compile(r'^\s*END\s+(\S+)')

        current_layer = None

        try:
            with open(self.lef_path, 'r', errors='ignore') as f:
                for line in f:
                    line = line.split('#')[0].strip() # 주석 제거
                    if not line: continue

                    # 1. MACRO Start
                    match_macro = re_macro.match(line)
                    if match_macro:
                        current_macro = match_macro.group(1)
                        self.cell_lib[current_macro] = {'pins': {}, 'size': (0,0)}
                        continue
                    
                    # 2. SIZE
                    match_size = re_size.match(line)
                    if match_size and current_macro:
                        self.cell_lib[current_macro]['size'] = (float(match_size.group(1)), float(match_size.group(2)))
                        continue

                    # 3. PIN Start
                    match_pin = re_pin.match(line)
                    if match_pin and current_macro:
                        current_pin = match_pin.group(1)
                        self.cell_lib[current_macro]['pins'][current_pin] = []
                        continue

                    # 4. Layer & Rect inside Pin
                    if current_pin:
                        match_direction = re_direction.match(line)
                        if match_direction:
                            current_direction = match_direction.group(1).upper()
                            continue
                        
                        match_layer = re_layer.match(line)
                        if match_layer:
                            current_layer = match_layer.group(1).lower()
                            continue
                        
                        match_rect = re_rect.match(line)
                        if match_rect and current_layer:
                            # x1, y1, x2, y2
                            coords = [float(g) for g in match_rect.groups()]
                            self.cell_lib[current_macro]['pins'][current_pin].append({
                                'layer': current_layer,
                                'rect': coords,
                                'direction': current_direction
                            })
                            continue

                    # 5. END
                    match_end = re_end.match(line)
                    if match_end:
                        token = match_end.group(1)
                        if token == current_pin:
                            current_pin = None
                            current_layer = None
                        elif token == current_macro:
                            current_macro = None

        except FileNotFoundError:
            raise FileNotFoundError(f"Cell LEF not found: {self.lef_path}")
            
        print(f"Loaded {len(self.cell_lib)} cells from library.")
        return self.cell_lib