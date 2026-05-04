"""Phase 12 — Per-cell-type SIZE + signal-pin OBS (excluding VCC/VSS power).

For each MACRO in cell LEF:
  - SIZE w BY h  (cell footprint)
  - Power pin names (VCC/VSS/VDD)
  - Signal pin OBS contribution: OBS RECTs not overlapping power pin ports

This filters out the power rail M1 plates that pollute v5's OBS aggregation.

Output: cache/cell_sizes_pins.parquet with one row per cell type:
  cell_type, size_w, size_h, area, n_pins_signal, n_pins_power,
  obs_signal_nsq_M1, obs_signal_nsq_M2  (OBS minus power-rail bands)
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

from configs import config as cfg

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parent

CELL_LEF = cfg.CELL_LEF_PATH

POWER_PIN_NAMES = {"vcc", "vss", "vdd", "vssx", "vccx", "vddx", "gnd", "gndx"}

RE_MACRO = re.compile(r"^\s*MACRO\s+(\S+)")
RE_SIZE = re.compile(r"^\s*SIZE\s+([\d\.]+)\s+BY\s+([\d\.]+)")
RE_PIN  = re.compile(r"^\s*PIN\s+(\S+)")
RE_DIR  = re.compile(r"^\s*DIRECTION\s+(\S+)")
RE_LAYER = re.compile(r"^\s*LAYER\s+(\S+)\s*;")
RE_RECT  = re.compile(r"^\s*RECT\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)\s*;")
RE_OBS   = re.compile(r"^\s*OBS\s*$")
RE_END   = re.compile(r"^\s*END\s*(\S+)?\s*$")
RE_PORT  = re.compile(r"^\s*PORT\s*$")

LAYER_M = re.compile(r"^m(\d+)$")
LAYER_V = re.compile(r"^v(\d+)$")


def parse_cell_lef(lef_path):
    cells = {}
    cur_macro = None
    cur_size = None
    cur_pin = None
    cur_pin_is_power = False
    cur_pin_dir = None
    in_pin_port = False
    in_obs = False
    cur_layer = None
    pin_data = None
    obs_data = None
    pins = None  # list of pin records

    with open(lef_path, errors="ignore") as f:
        for line in f:
            line = line.split("#")[0].rstrip()
            if not line: continue
            stripped = line.lstrip()

            m = RE_MACRO.match(line)
            if m:
                cur_macro = m.group(1)
                cells[cur_macro] = {"size": (0.0, 0.0), "pins": [], "obs": []}
                pins = cells[cur_macro]["pins"]
                obs_data = cells[cur_macro]["obs"]
                cur_pin = None; cur_layer = None; in_obs = False; in_pin_port = False
                continue

            if cur_macro is None:
                continue

            m = RE_SIZE.match(line)
            if m:
                cells[cur_macro]["size"] = (float(m.group(1)), float(m.group(2)))
                continue

            m = RE_OBS.match(line)
            if m:
                in_obs = True; cur_layer = None; cur_pin = None; in_pin_port = False
                continue

            m = RE_PIN.match(line)
            if m:
                cur_pin = m.group(1)
                cur_pin_is_power = cur_pin.lower() in POWER_PIN_NAMES
                cur_pin_dir = None
                in_obs = False
                in_pin_port = False
                cur_layer = None
                pin_data = {"name": cur_pin, "is_power": cur_pin_is_power,
                              "direction": None, "rects": []}
                pins.append(pin_data)
                continue

            if cur_pin is not None:
                m = RE_DIR.match(line)
                if m:
                    cur_pin_dir = m.group(1).upper()
                    pin_data["direction"] = cur_pin_dir
                    continue
                if RE_PORT.match(line):
                    in_pin_port = True; cur_layer = None
                    continue

            if in_pin_port or in_obs:
                m = RE_LAYER.match(line)
                if m:
                    cur_layer = m.group(1).lower()
                    continue
                m = RE_RECT.match(line)
                if m and cur_layer:
                    coords = tuple(float(g) for g in m.groups())
                    rec = {"layer": cur_layer, "rect": coords}
                    if in_obs:
                        obs_data.append(rec)
                    elif in_pin_port and pin_data is not None:
                        pin_data["rects"].append(rec)
                    continue

            m = RE_END.match(line)
            if m:
                tok = m.group(1)
                if tok == cur_pin:
                    cur_pin = None; in_pin_port = False; cur_layer = None
                elif tok == cur_macro:
                    cur_macro = None; in_obs = False; cur_layer = None
                else:
                    # END of OBS or PORT
                    if in_obs and tok is None:
                        in_obs = False; cur_layer = None
                    if in_pin_port and tok is None:
                        in_pin_port = False; cur_layer = None
    return cells


def main():
    print(f"Parsing {CELL_LEF}", flush=True)
    cells = parse_cell_lef(CELL_LEF)
    print(f"  parsed {len(cells)} cells")

    rows = []
    for ct, info in cells.items():
        w, h = info["size"]
        area = w * h
        signal_pins = [p for p in info["pins"] if not p["is_power"]]
        power_pins  = [p for p in info["pins"] if p["is_power"]]
        # power-pin metal area per layer (subtract this from OBS to estimate signal-internal)
        pwr_area_M = defaultdict(float)
        for p in power_pins:
            for r in p["rects"]:
                m = LAYER_M.match(r["layer"])
                if not m: continue
                lvl = int(m.group(1))
                x1, y1, x2, y2 = r["rect"]
                pwr_area_M[lvl] += abs(x2 - x1) * abs(y2 - y1)

        obs_total = defaultdict(lambda: defaultdict(float))   # layer_lvl -> stat
        for r in info["obs"]:
            m = LAYER_M.match(r["layer"]); v = LAYER_V.match(r["layer"])
            if m:
                lvl = int(m.group(1))
                x1, y1, x2, y2 = r["rect"]
                dx, dy = abs(x2 - x1), abs(y2 - y1)
                if dx <= 0 or dy <= 0: continue
                obs_total[lvl]["nsq"] += max(dx, dy) / max(min(dx, dy), 1e-6)
                obs_total[lvl]["area"] += dx * dy
                obs_total[lvl]["count"] += 1
            elif v:
                vlvl = int(v.group(1))
                obs_total[("v", vlvl)]["count"] += 1

        # signal-side OBS estimate: total area minus power-pin port area on same layer
        # If power-pin port covers most of OBS area on M1, the signal-internal nsq is smaller.
        # Approximate via area scaling:
        #   signal_obs_nsq[lvl] ≈ obs_nsq[lvl] × (1 - pwr_area[lvl] / obs_area[lvl])
        # If pwr_area > obs_area, the OBS is just power overlap → signal contribution near 0.

        row = {"cell_type": ct, "size_w": w, "size_h": h, "area": area,
               "n_pins_signal": len(signal_pins), "n_pins_power": len(power_pins),
               "n_pins_input":  sum(1 for p in signal_pins if p.get("direction") == "INPUT"),
               "n_pins_output": sum(1 for p in signal_pins if p.get("direction") == "OUTPUT")}
        for lvl, stats in obs_total.items():
            if isinstance(lvl, tuple):
                row[f"obs_n_via_v{lvl[1]}"] = float(stats["count"])
            else:
                row[f"obs_total_nsq_M{lvl}"] = float(stats["nsq"])
                row[f"obs_total_area_M{lvl}"] = float(stats["area"])
                row[f"obs_total_count_M{lvl}"] = float(stats["count"])
                pwr = pwr_area_M.get(lvl, 0.0)
                obs_a = stats["area"]
                # signal contribution scaling
                if obs_a > 0:
                    sig_frac = max(0.0, 1.0 - pwr / obs_a)
                else:
                    sig_frac = 0.0
                row[f"obs_signal_nsq_M{lvl}"] = float(stats["nsq"] * sig_frac)
                row[f"obs_signal_area_M{lvl}"] = float(stats["area"] * sig_frac)
        rows.append(row)
    df = pd.DataFrame(rows).fillna(0.0)
    out = _V3 / "cache" / "cell_sizes_pins.parquet"
    df.to_parquet(out)
    print(f"  saved: {out}")
    print(df.describe().T.to_string())


if __name__ == "__main__":
    main()
