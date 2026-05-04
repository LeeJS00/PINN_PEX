"""Phase 1 — Build per-net "physics features" suitable for linear R regression.

For each net we extract:
  - n_squares_M{i}  : Σ L_seg / W_seg over WIRE segments on layer M{i}
                      (i.e. number of squares; sheet_R times this gives the
                       wire-R contribution exactly when sheet_R is constant.)
  - rect_squares_M{i}: Σ max(dx,dy)/min(dx,dy) over RECT patches on M{i}
                       (kept separate so the regression can decide its weight.)
  - n_via_<NAME>     : count of each unique DEF VIA token
  - n_via_lvl{j}     : aggregated count by parsed VIA<j> level
  - n_segments       : total number of WIRE segments
  - n_pin_contacts   : count of zero-length pin contacts (proxy)
  - design           : design name (str)
  - net_name         : net name

Optionally we also include the GROUND-TRUTH y from golden RES (R_gold) when
available (training/eval).

We use PINNPEX's `DefStreamParser` for DEF parsing and `parse_spef_with_coordinates`
for SPEF — same parsers used elsewhere, no re-implementation.
"""
from __future__ import annotations

import importlib.util
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

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parent
_WS = _V3.parent      # experiments/cross_design_tv80s_2026_05_02

_spec = importlib.util.spec_from_file_location(
    "compare_spef_pinpex",
    str(PROJ / "src" / "evaluation" / "compare_spef.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
parse_spef = _mod.parse_spef_with_coordinates

DEF_DIR  = Path("/home/jslee/projects/PEX_SSL/data/raw/def/intel22")
SPEF_DIR = Path("/home/jslee/projects/PEX_SSL/golden_data/spef_data/intel22")

DESIGNS_TRAIN = [
    "intel22_aes_cipher_top_f3", "intel22_gcd_f3", "intel22_ibex_core_f3",
    "intel22_ldpc_decoder_802_3an_f3",
    "intel22_mc_top_f3", "intel22_spi_top_f3", "intel22_usbf_top_f3",
    "intel22_vga_enh_top_f3", "intel22_wb_conmax_top_f3",
]
DESIGNS_TEST = ["intel22_nova_f3", "intel22_tv80s_f3"]
DESIGN_TEST = "intel22_tv80s_f3"   # legacy name — keep but use DESIGNS_TEST

VIA_NUM_RE = re.compile(r"^VIA(\d+)_", re.IGNORECASE)
LAYER_M_RE = re.compile(r"^m(\d+)$", re.IGNORECASE)


def _parser_resources():
    layer_map = LayerInfoParser(cfg.LAYERS_INFO_PATH).parse()
    tech_lef  = LefParser(cfg.TECH_LEF_PATH).parse()
    cell_lib  = CellLibParser(cfg.CELL_LEF_PATH).parse()
    return layer_map, tech_lef, cell_lib


def features_for_design(def_path: Path, parser_res) -> pd.DataFrame:
    layer_map, tech_lef, cell_lib = parser_res
    parser = DefStreamParser(str(def_path), layer_map, tech_lef, cell_lib)
    rows = []
    for net_name, _, segs in parser.parse():
        if not segs:
            continue
        nsq = defaultdict(float)         # n_squares per metal layer
        rsq = defaultdict(float)         # rect-squares per metal layer
        via_name_count = defaultdict(int)  # per VIA name
        via_lvl_count  = defaultdict(int)  # aggregated v{j}
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
                L = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
                W = max(float(s["width"]), 1e-6)
                if L == 0:
                    n_zero_l += 1
                    continue
                nsq[lvl] += L / W
                n_segments += 1
            elif t == "RECT":
                m = LAYER_M_RE.match(s["layer"].lower())
                if not m:
                    continue
                lvl = int(m.group(1))
                x1, y1, x2, y2 = s["rect"]
                dx = abs(x2 - x1); dy = abs(y2 - y1)
                if dx <= 0 or dy <= 0:
                    continue
                rsq[lvl] += max(dx, dy) / max(min(dx, dy), 1e-6)
            elif t == "VIA":
                vn = s["name"]
                via_name_count[vn] += 1
                m = VIA_NUM_RE.match(vn)
                if m:
                    via_lvl_count[int(m.group(1))] += 1
        row = {"net_name": net_name, "n_segments": n_segments,
               "n_zero_l_wire": n_zero_l}
        for k, v in nsq.items():
            row[f"nsq_M{k}"] = float(v)
        for k, v in rsq.items():
            row[f"rsq_M{k}"] = float(v)
        for k, v in via_lvl_count.items():
            row[f"nvia_lvl{k}"] = int(v)
        for k, v in via_name_count.items():
            # column-safe key
            row[f"nvian_{k}"] = int(v)
        rows.append(row)
    return pd.DataFrame(rows).fillna(0.0)


def golden_R_for(design: str) -> pd.DataFrame:
    spef = SPEF_DIR / f"{design}_starrc.spef"
    g = parse_spef(spef)
    return pd.DataFrame([{"net_name": n, "R_gold": float(info["total_res"])}
                         for n, info in g.items()])


def main():
    cache_dir = _V3 / "cache"
    parser_res = _parser_resources()

    for d in DESIGNS_TRAIN + DESIGNS_TEST:
        out = cache_dir / f"feat_{d}.parquet"
        if out.exists():
            print(f"  [cache hit] {d}")
            continue
        def_path = DEF_DIR / f"{d}.def"
        if not def_path.exists():
            print(f"  [skip] {d}: DEF missing")
            continue
        print(f"  parsing {d} ...", flush=True)
        df = features_for_design(def_path, parser_res)
        try:
            gR = golden_R_for(d)
            df = df.merge(gR, on="net_name", how="left")
        except Exception as e:
            print(f"    [warn] golden SPEF for {d} unavailable: {e}")
        df["design"] = d
        df.to_parquet(out)
        print(f"    n_nets={len(df)}  cols={len(df.columns)}  -> {out.name}")

    # quick aggregate summary
    all_cols = set()
    for d in DESIGNS_TRAIN + DESIGNS_TEST:
        p = cache_dir / f"feat_{d}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        all_cols.update(df.columns)
    print(f"\nfeature columns across all designs: {len(all_cols)}")
    via_cols = sorted([c for c in all_cols if c.startswith("nvian_")])
    print(f"unique DEF VIA names: {len(via_cols)}")
    for c in via_cols[:20]:
        print(f"  {c}")
    if len(via_cols) > 20:
        print(f"  ... +{len(via_cols)-20} more")


if __name__ == "__main__":
    main()
