"""Phase 4 — Richer feature set v2.

Adds (vs v1):
  - sum_L_M{i}     : Σ L_seg per layer (μm; not divided by W)
  - max_L_M{i}     : longest single segment per layer
  - sum_W_M{i}     : Σ W_seg (proxy for total contact perimeter)
  - n_seg_M{i}     : segment count per layer (each adds boundary R)
  - n_segments     : total segment count
  - n_zero_l_wire  : number of zero-length wires (pin contacts)
  - one              : constant 1 (per-net intercept)

This lets the regression learn:
  - layer-resistance per square (nsq_M{i})
  - per-segment fixed cost (n_seg_M{i})
  - per-zero-length contact cost (n_zero_l_wire)
  - per-net residual offset (one)
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path("/home/jslee/projects/PINNPEX")
sys.path.insert(0, str(PROJ))

from configs import config as cfg  # noqa: E402
from src.preprocessing.def_parser   import DefStreamParser   # noqa: E402
from src.preprocessing.layer_parser import LayerInfoParser   # noqa: E402
from src.preprocessing.lef_parser   import LefParser         # noqa: E402
from src.preprocessing.cell_parser  import CellLibParser     # noqa: E402

import importlib.util
_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parent
_spec = importlib.util.spec_from_file_location(
    "compare_spef_pinpex",
    str(PROJ / "src" / "evaluation" / "compare_spef.py"),
)
_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
parse_spef = _mod.parse_spef_with_coordinates

DEF_DIR  = Path("/home/jslee/projects/PEX_SSL/data/raw/def/intel22")
SPEF_DIR = Path("/home/jslee/projects/PEX_SSL/golden_data/spef_data/intel22")

DESIGNS = [
    "intel22_aes_cipher_top_f3", "intel22_gcd_f3", "intel22_ibex_core_f3",
    "intel22_ldpc_decoder_802_3an_f3",
    "intel22_mc_top_f3", "intel22_spi_top_f3", "intel22_usbf_top_f3",
    "intel22_vga_enh_top_f3", "intel22_wb_conmax_top_f3", "intel22_nova_f3",
    "intel22_tv80s_f3",
]

VIA_NUM_RE = re.compile(r"^VIA(\d+)_", re.IGNORECASE)
LAYER_M_RE = re.compile(r"^m(\d+)$", re.IGNORECASE)


def parser_resources():
    return (LayerInfoParser(cfg.LAYERS_INFO_PATH).parse(),
            LefParser(cfg.TECH_LEF_PATH).parse(),
            CellLibParser(cfg.CELL_LEF_PATH).parse())


def features_for_design(def_path, parser_res):
    layer_map, tech_lef, cell_lib = parser_res
    parser = DefStreamParser(str(def_path), layer_map, tech_lef, cell_lib)
    rows = []
    for net_name, _, segs in parser.parse():
        if not segs:
            continue
        nsq = defaultdict(float)
        sumL = defaultdict(float)
        maxL = defaultdict(float)
        sumW = defaultdict(float)
        nseg_layer = defaultdict(int)
        rsq = defaultdict(float)
        sum_rect_area = defaultdict(float)
        nrect_layer = defaultdict(int)
        via_name_count = defaultdict(int)
        n_segments = 0
        n_zero_l   = 0

        for s in segs:
            t = s["type"]
            if t == "WIRE":
                m = LAYER_M_RE.match(s["layer"].lower())
                if not m:
                    continue
                lvl = int(m.group(1))
                p1, p2 = s["start"], s["end"]
                L = abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])
                W = max(float(s["width"]), 1e-6)
                if L == 0:
                    n_zero_l += 1
                    continue
                nsq[lvl] += L / W
                sumL[lvl] += L
                maxL[lvl] = max(maxL[lvl], L)
                sumW[lvl] += W
                nseg_layer[lvl] += 1
                n_segments += 1
            elif t == "RECT":
                m = LAYER_M_RE.match(s["layer"].lower())
                if not m:
                    continue
                lvl = int(m.group(1))
                x1,y1,x2,y2 = s["rect"]
                dx, dy = abs(x2-x1), abs(y2-y1)
                if dx <= 0 or dy <= 0:
                    continue
                rsq[lvl]  += max(dx, dy) / max(min(dx, dy), 1e-6)
                sum_rect_area[lvl] += dx * dy
                nrect_layer[lvl] += 1
            elif t == "VIA":
                vn = s["name"]
                via_name_count[vn] += 1

        row = {"net_name": net_name, "n_segments": n_segments,
               "n_zero_l_wire": n_zero_l, "one": 1.0}
        for k, v in nsq.items():
            row[f"nsq_M{k}"] = float(v)
        for k, v in sumL.items():
            row[f"sumL_M{k}"] = float(v)
        for k, v in maxL.items():
            row[f"maxL_M{k}"] = float(v)
        for k, v in sumW.items():
            row[f"sumW_M{k}"] = float(v)
        for k, v in nseg_layer.items():
            row[f"nseg_M{k}"] = int(v)
        for k, v in rsq.items():
            row[f"rsq_M{k}"] = float(v)
        for k, v in sum_rect_area.items():
            row[f"rect_area_M{k}"] = float(v)
        for k, v in nrect_layer.items():
            row[f"nrect_M{k}"] = int(v)
        for k, v in via_name_count.items():
            row[f"nvian_{k}"] = int(v)
        rows.append(row)
    return pd.DataFrame(rows).fillna(0.0)


def golden_R_for(design):
    g = parse_spef(SPEF_DIR / f"{design}_starrc.spef")
    return pd.DataFrame([{"net_name": n, "R_gold": float(info["total_res"])}
                         for n, info in g.items()])


def main():
    pres = parser_resources()
    cache = _V3 / "cache"
    for d in DESIGNS:
        out = cache / f"feat_v2_{d}.parquet"
        if out.exists():
            print(f"  [cache hit] {d}")
            continue
        print(f"  parsing {d} ...", flush=True)
        df = features_for_design(DEF_DIR / f"{d}.def", pres)
        try:
            df = df.merge(golden_R_for(d), on="net_name", how="left")
        except Exception as e:
            print(f"    [warn] golden missing for {d}: {e}")
        df["design"] = d
        df.to_parquet(out)
        print(f"    n_nets={len(df)}  cols={len(df.columns)}")


if __name__ == "__main__":
    main()
