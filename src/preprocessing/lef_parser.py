# src/preprocessing/lef_parser.py
import re

class LefParser:
    def __init__(self, lef_path, verbose=False):
        self.lef_path = lef_path
        self.tech_info = {'layers': {}, 'vias': {}}
        self.verbose = verbose

    def parse(self):
        try:
            with open(self.lef_path, 'r', errors='ignore') as f:
                in_layer = False
                current_layer = None
                layer_data = {}

                in_via = False
                current_via = None
                via_layers = {}
                via_metal_layers =[]
                via_cut_layer = None
                current_via_layer = None
                via_res = 1.0
                
                in_macro = False
                current_macro = None
                
                for line in f:
                    line = line.split('#')[0].strip() # 주석 제거
                    if not line: continue
                    # [안전장치] Tech LEF 내부에 간혹 있는 MACRO(Boundary Cell 등) 블록 진입 시 LAYER 파싱 무시
                    if line.startswith('MACRO '):
                        parts = line.split()
                        if len(parts) >= 2:
                            in_macro = True
                            current_macro = parts[1]
                        continue
                        
                    if in_macro:
                        if line.startswith('END '):
                            parts = line.split()
                            if len(parts) >= 2 and parts[1] == current_macro:
                                in_macro = False
                        continue

                    # 1. LAYER Parsing State
                    if line.startswith('LAYER ') and not in_layer and not in_via:
                        parts = line.split()
                        if len(parts) == 2:
                            current_layer = parts[1]
                            in_layer = True
                            layer_data = {}
                        continue

                    if in_layer:
                        if line.startswith('WIDTH '):
                            parts = line.split()
                            if len(parts) >= 2:
                                layer_data['width'] = float(parts[1])
                        elif line.startswith('END '):
                            parts = line.split()
                            # print(">>> [Lef Parser] ", parts[1], current_layer)
                            if len(parts) >= 2 and parts[1] == current_layer:
                                self.tech_info['layers'][current_layer.lower()] = layer_data
                                in_layer = False
                                current_layer = None
                                # print(layer_data)
                        continue

                    # 2. VIA Parsing State
                    if line.startswith('VIA ') and not in_via and not in_layer:
                        parts = line.split()
                        if len(parts) >= 2:
                            current_via = parts[1]
                            in_via = True
                            via_layers = {}
                            via_metal_layers =[]
                            current_via_layer = None
                            via_res = 1.0
                        continue

                    if in_via:
                        if line.startswith('RESISTANCE '):
                            parts = line.split()
                            if len(parts) >= 2:
                                via_res = float(parts[1])
                        elif line.startswith('LAYER '):
                            parts = line.split()
                            if len(parts) >= 2:
                                current_via_layer = parts[1].lower()
                                if current_via_layer not in via_layers:
                                    via_layers[current_via_layer] = []
                                # 'v'나 'via'가 안 붙은 레이어는 도체(Metal)로 간주
                                if not (current_via_layer.startswith('v') or 'via' in current_via_layer):
                                    if current_via_layer not in via_metal_layers:
                                        via_metal_layers.append(current_via_layer)
                                else:
                                    via_cut_layer = current_via_layer

                        elif line.startswith('RECT ') and current_via_layer:
                            parts = line.split()
                            if len(parts) >= 5:
                                try:
                                    rect = [float(x) for x in parts[1:5]]
                                    via_layers[current_via_layer].append(rect)
                                except ValueError: pass
                        elif line.startswith('END '):
                            parts = line.split()
                            if len(parts) >= 2 and parts[1] == current_via:
                                self.tech_info['vias'][current_via.lower()] = {
                                     'layers': via_layers,
                                     'metal_layers': via_metal_layers,
                                     'cut_layer': via_cut_layer,
                                     'resistance': via_res
                                 }
                                in_via = False
                                current_via = None
                                current_via_layer = None
                        continue

            if self.verbose:
                print(f"[LefParser] Successfully parsed {self.lef_path}")
                print(f"[LefParser] Layers: {list(self.tech_info['layers'].keys())}")
                print(f"[LefParser] Vias: {list(self.tech_info['vias'].keys())}")
                print(f"[LefParser] Via {list(self.tech_info['vias'].keys())[0]}: {self.tech_info['vias'][list(self.tech_info['vias'].keys())[0]]}")
            return self.tech_info
            
        except FileNotFoundError:
            raise FileNotFoundError(f"LEF file not found: {self.lef_path}")