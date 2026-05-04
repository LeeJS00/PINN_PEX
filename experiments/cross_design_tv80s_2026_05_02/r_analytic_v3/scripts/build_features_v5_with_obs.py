"""Phase 11b — v5 features: aggregate cell OBS into signal nets.

Per (inst, pin) tuple in a signal net, look up the cell type's OBS features
(obs_nsq_M1, obs_n_via_v0, etc.) and sum across pins. This captures the
30-square M1/M2 cell-internal routing the previous v4 couldn't see.

Important: each pin tuple gets the FULL cell OBS — the regression coefficient
will learn the appropriate per-pin fraction (typically 1 / n_pins_per_cell).
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

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parent

DEF_DIR = Path("/home/jslee/projects/PEX_SSL/data/raw/def/intel22")

DESIGNS = [
    "intel22_aes_cipher_top_f3", "intel22_gcd_f3", "intel22_ibex_core_f3",
    "intel22_mc_top_f3", "intel22_spi_top_f3", "intel22_usbf_top_f3",
    "intel22_vga_enh_top_f3", "intel22_wb_conmax_top_f3", "intel22_nova_f3",
    "intel22_tv80s_f3",
]

PIN_RE = re.compile(r"\(\s+(\S+)\s+(\S+)\s+\)")
COMP_RE = re.compile(r"^-\s+(\S+)\s+(\S+)\s+\+")  # "- inst_name cell_type +"


def parse_def_components(def_path):
    """Build inst_name -> cell_type from DEF COMPONENTS section."""
    out = {}
    in_comp = False
    with open(def_path, errors="ignore") as f:
        for line in f:
            s = line.rstrip()
            stripped = s.strip()
            if not in_comp:
                if s.startswith("COMPONENTS "):
                    in_comp = True
                continue
            if s.startswith("END COMPONENTS"):
                break
            m = COMP_RE.match(stripped)
            if m:
                inst_name, cell_type = m.group(1), m.group(2)
                out[inst_name] = cell_type
    return out


def parse_signal_net_pin_lists(def_path):
    """Per signal net, list of (inst, pin) tuples from DEF NETS header."""
    out = {}
    in_nets = False
    cur_net = None
    cur_pins = []
    seen_routed = False
    with open(def_path, errors="ignore") as f:
        for line in f:
            s = line.rstrip()
            if not in_nets:
                if s.startswith("NETS "):
                    in_nets = True
                continue
            if s.startswith("END NETS"):
                if cur_net is not None:
                    out[cur_net] = cur_pins
                break
            stripped = s.strip()
            if stripped.startswith("- "):
                if cur_net is not None:
                    out[cur_net] = cur_pins
                cur_net = stripped[2:].split()[0]
                cur_pins = []
                seen_routed = False
            elif cur_net is not None and not seen_routed:
                if any(stripped.startswith(p) for p in
                        ("+ ROUTED", "+ FREQUENCY", "+ SOURCE", "+ USE")):
                    seen_routed = True
                else:
                    cur_pins.extend(PIN_RE.findall(stripped))
    return out


def main():
    cell_obs = pd.read_parquet(_V3 / "cache" / "cell_obs_features.parquet")
    cell_obs_dict = cell_obs.set_index("cell_type").to_dict(orient="index")
    obs_cols = [c for c in cell_obs.columns if c != "cell_type"]
    print(f"Loaded {len(cell_obs_dict)} cells, OBS columns: {obs_cols}")

    cache = _V3 / "cache"
    for d in DESIGNS:
        out_path = cache / f"feat_v5_obs_{d}.parquet"
        if out_path.exists():
            print(f"  [cache hit] {d}")
            continue
        def_path = DEF_DIR / f"{d}.def"
        print(f"  parsing {d} components + NETS ...", flush=True)
        comp_map = parse_def_components(def_path)
        sig_to_pins = parse_signal_net_pin_lists(def_path)
        print(f"    n_components: {len(comp_map)}, n_nets: {len(sig_to_pins)}")

        rows = []
        unknown_cells = set()
        n_pins_unmatched = 0
        for sig_net, pin_tuples in sig_to_pins.items():
            agg = defaultdict(float)
            n_matched = 0
            for inst, pin in pin_tuples:
                if inst.upper() == "PIN":
                    continue  # top-level pad, no cell OBS
                cell_type = comp_map.get(inst)
                if cell_type is None:
                    n_pins_unmatched += 1
                    continue
                if cell_type not in cell_obs_dict:
                    unknown_cells.add(cell_type)
                    continue
                feat = cell_obs_dict[cell_type]
                for c in obs_cols:
                    agg[c] += feat[c]
                n_matched += 1
            row = {"net_name": sig_net, "n_pins_obs_matched": n_matched}
            for c in obs_cols:
                row[c] = float(agg[c])
            rows.append(row)
        df = pd.DataFrame(rows).fillna(0.0)
        df.to_parquet(out_path)
        print(f"    n_nets={len(df)}  unknown_cells={len(unknown_cells)} (e.g. {list(unknown_cells)[:3]})  "
              f"unmatched_pins={n_pins_unmatched}")
        print(f"    avg obs_nsq_M1/net={df['obs_nsq_M1'].mean():.2f}, "
              f"obs_nsq_M2/net={df['obs_nsq_M2'].mean():.2f}, "
              f"obs_n_via_v0/net={df['obs_n_via_v0'].mean():.2f}")


if __name__ == "__main__":
    main()
