"""Phase 12b — v6 features: aggregate cell SIGNAL OBS (excluding power) + cell SIZE.

Per-net aggregates:
  obs_signal_nsq_M{i}    : Σ (cell signal-OBS nsq M{i}) over pins
  obs_signal_area_M{i}   : Σ (cell signal-OBS area M{i})
  obs_n_via_v{i}_pin     : Σ (cell OBS via{i} count) over pins
  cell_size_w_sum        : Σ cell width over pins (proxy for total internal route)
  cell_size_h_sum        : Σ cell height
  cell_area_sum          : Σ cell area
  n_input_pins           : count of INPUT pins from cell library lookup
  n_output_pins          : count of OUTPUT pins
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

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parent

DEF_DIR = Path("/home/jslee/projects/PEX_SSL/data/raw/def/intel22")

DESIGNS = [
    "intel22_aes_cipher_top_f3", "intel22_gcd_f3", "intel22_ibex_core_f3",
    "intel22_ldpc_decoder_802_3an_f3",
    "intel22_mc_top_f3", "intel22_spi_top_f3", "intel22_usbf_top_f3",
    "intel22_vga_enh_top_f3", "intel22_wb_conmax_top_f3", "intel22_nova_f3",
    "intel22_tv80s_f3",
]

PIN_RE = re.compile(r"\(\s+(\S+)\s+(\S+)\s+\)")
COMP_RE = re.compile(r"^-\s+(\S+)\s+(\S+)\s+\+")


def parse_def_components(def_path):
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
                out[m.group(1)] = m.group(2)
    return out


def parse_signal_net_pin_lists(def_path):
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
    cell_df = pd.read_parquet(_V3 / "cache" / "cell_sizes_pins.parquet")
    cell_dict = cell_df.set_index("cell_type").to_dict(orient="index")
    feature_cols = [c for c in cell_df.columns if c not in
                     ["cell_type", "n_pins_power", "obs_total_nsq_M1", "obs_total_area_M1",
                      "obs_total_count_M1", "obs_total_nsq_M2", "obs_total_area_M2",
                      "obs_total_count_M2"]]
    print(f"Loaded {len(cell_dict)} cells. Features: {feature_cols}")

    cache = _V3 / "cache"
    for d in DESIGNS:
        out_path = cache / f"feat_v6_{d}.parquet"
        if out_path.exists():
            print(f"  [cache hit] {d}")
            continue
        def_path = DEF_DIR / f"{d}.def"
        print(f"  parsing {d} ...", flush=True)
        comp_map = parse_def_components(def_path)
        sig_to_pins = parse_signal_net_pin_lists(def_path)
        rows = []
        for sig_net, pin_tuples in sig_to_pins.items():
            agg = defaultdict(float)
            n_matched = 0
            for inst, pin in pin_tuples:
                if inst.upper() == "PIN":
                    continue
                cell_type = comp_map.get(inst)
                if cell_type is None or cell_type not in cell_dict:
                    continue
                feat = cell_dict[cell_type]
                for c in feature_cols:
                    agg[c] += feat.get(c, 0.0)
                n_matched += 1
            row = {"net_name": sig_net, "n_pins_obs_matched": n_matched}
            for c in feature_cols:
                row[c] = float(agg[c])
            rows.append(row)
        df = pd.DataFrame(rows).fillna(0.0)
        # rename to v6 prefixes
        rename = {}
        for c in feature_cols:
            if c.startswith("obs_signal"):
                rename[c] = f"v6_{c}"
            elif c.startswith("obs_n_via"):
                rename[c] = f"v6_{c}_pin"
            elif c.startswith("size_") or c == "area":
                rename[c] = f"v6_cell_{c}_sum"
            elif c.startswith("n_pins"):
                rename[c] = f"v6_{c}"
        df = df.rename(columns=rename)
        df.to_parquet(out_path)
        print(f"    n_nets={len(df)}  avg signal_M1={df['v6_obs_signal_nsq_M1'].mean():.2f}, "
              f"signal_M2={df['v6_obs_signal_nsq_M2'].mean():.2f}, "
              f"size_h_sum={df['v6_cell_size_h_sum'].mean():.2f}")


if __name__ == "__main__":
    main()
