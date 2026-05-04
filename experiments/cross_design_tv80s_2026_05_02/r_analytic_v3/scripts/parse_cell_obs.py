"""Phase 11 — Parse cell LEF OBS section to capture internal cell routing.

PINNPEX's `CellLibParser` extracts only PIN geometry. The OBS (obstruction)
section that follows the PINs in each MACRO is the cell's *internal* metal
routing on M1/M2 plus its v0/v1 vias — exactly the 30 squares/net the
DEF parser was missing.

For each cell we compute aggregated structural features:
  cell_obs_nsq_M{i}  : sum L/W of all OBS RECTs on metal layer M{i}
  cell_obs_n_via_v{i}: count of OBS RECTs on via layer v{i}
  cell_obs_M{i}_count: count of M{i} OBS RECTs (proxy for fragmentation)

These are per-cell-type constants. At inference, for each net we sum them
over the cells connected (via DEF's pin tuples).
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path("/home/jslee/projects/PINNPEX")
sys.path.insert(0, str(PROJ))

from configs import config as cfg  # noqa: E402

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parent

CELL_LEF = cfg.CELL_LEF_PATH

LAYER_M_RE = re.compile(r"^m(\d+)$", re.IGNORECASE)
LAYER_V_RE = re.compile(r"^v(\d+)$", re.IGNORECASE)
RE_MACRO = re.compile(r"^\s*MACRO\s+(\S+)")
RE_OBS_START = re.compile(r"^\s*OBS\s*$")
RE_LAYER = re.compile(r"^\s*LAYER\s+(\S+)\s*;")
RE_RECT  = re.compile(r"^\s*RECT\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)\s*;")
RE_END   = re.compile(r"^\s*END\s*(\S+)?\s*$")


def parse_cell_obs(lef_path):
    """Parse OBS sections per MACRO. Returns dict cell -> features."""
    out = {}  # cell -> {nsq_M{i}, n_via_v{i}, ...}
    cur_macro = None
    in_obs = False
    cur_layer = None
    cur_feat = None
    with open(lef_path, errors="ignore") as f:
        for line in f:
            line = line.split("#")[0].rstrip()
            if not line:
                continue
            m = RE_MACRO.match(line)
            if m:
                cur_macro = m.group(1)
                cur_feat = defaultdict(float)
                out[cur_macro] = cur_feat
                in_obs = False
                cur_layer = None
                continue
            if cur_macro is None:
                continue
            if RE_OBS_START.match(line):
                in_obs = True
                cur_layer = None
                continue
            if in_obs:
                m_layer = RE_LAYER.match(line)
                if m_layer:
                    cur_layer = m_layer.group(1).lower()
                    continue
                m_rect = RE_RECT.match(line)
                if m_rect and cur_layer:
                    x1, y1, x2, y2 = (float(g) for g in m_rect.groups())
                    dx, dy = abs(x2 - x1), abs(y2 - y1)
                    if dx <= 0 or dy <= 0:
                        continue
                    if (mm := LAYER_M_RE.match(cur_layer)):
                        lvl = int(mm.group(1))
                        cur_feat[f"obs_nsq_M{lvl}"] += max(dx, dy) / max(min(dx, dy), 1e-6)
                        cur_feat[f"obs_area_M{lvl}"] += dx * dy
                        cur_feat[f"obs_count_M{lvl}"] += 1
                    elif (vv := LAYER_V_RE.match(cur_layer)):
                        vlvl = int(vv.group(1))
                        cur_feat[f"obs_n_via_v{vlvl}"] += 1
                    continue
                if RE_END.match(line):
                    # could be END of OBS or END of MACRO/PIN
                    in_obs = False
                    cur_layer = None
            else:
                # outside OBS — only watch for MACRO end
                pass
            # END of macro
            m_end = RE_END.match(line)
            if m_end and m_end.group(1) == cur_macro:
                cur_macro = None
                in_obs = False
    return {k: dict(v) for k, v in out.items()}


def main():
    print(f"Parsing cell LEF OBS: {CELL_LEF}", flush=True)
    obs = parse_cell_obs(CELL_LEF)
    print(f"  parsed {len(obs)} cells", flush=True)

    # Show summary
    keys = set()
    for v in obs.values():
        keys.update(v.keys())
    keys = sorted(keys)
    print(f"  feature keys: {keys}")

    # convert to dataframe
    rows = []
    for cell, feat in obs.items():
        row = {"cell_type": cell}
        for k in keys:
            row[k] = float(feat.get(k, 0.0))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_parquet(_V3 / "cache" / "cell_obs_features.parquet")
    print(f"\nSaved: r_analytic_v3/cache/cell_obs_features.parquet")
    print(df.describe().T.to_string())


if __name__ == "__main__":
    main()
