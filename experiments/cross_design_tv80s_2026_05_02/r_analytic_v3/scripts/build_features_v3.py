"""Phase 4 — v3 feature set: add n_pins per net from DEF header line.

The DEF NETS section has, per net:
    - <net_name>
      ( <inst|PIN> <pin> ) ( <inst|PIN> <pin> ) ...
      + ROUTED ...

n_pins = count of ( ... ) tuples between the net header and "+ ROUTED".
Each pin is a cell-pin attachment that contributes a small fixed R via
an unrouted M1 stub inside the cell — currently invisible to our metric.

Combined with v2 features, gives the regression a chance to learn a per-pin
fixed cost (~few Ω per pin).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path("/home/jslee/projects/PINNPEX")
sys.path.insert(0, str(PROJ))

from configs import config as cfg  # noqa: E402

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parent

DEF_DIR  = Path("/home/jslee/projects/PEX_SSL/data/raw/def/intel22")

DESIGNS = [
    "intel22_aes_cipher_top_f3", "intel22_gcd_f3", "intel22_ibex_core_f3",
    "intel22_ldpc_decoder_802_3an_f3",
    "intel22_mc_top_f3", "intel22_spi_top_f3", "intel22_usbf_top_f3",
    "intel22_vga_enh_top_f3", "intel22_wb_conmax_top_f3", "intel22_nova_f3",
    "intel22_tv80s_f3",
]

PIN_RE = re.compile(r"\(\s+(\S+)\s+(\S+)\s+\)")


def parse_def_pin_counts(def_path):
    """Count pin-tuples per net in DEF NETS section."""
    out = []
    in_nets = False
    cur_net = None
    cur_pins = 0
    cur_pin_pin = 0  # pin connections to top-level PIN
    cur_inst_pin = 0
    seen_routed = False
    with open(def_path, "r", errors="ignore") as f:
        for line in f:
            s = line.rstrip()
            if not in_nets:
                if s.startswith("NETS "):
                    in_nets = True
                continue
            if s.startswith("END NETS"):
                if cur_net is not None:
                    out.append({"net_name": cur_net, "n_pins": cur_pins,
                                  "n_pin_PIN": cur_pin_pin, "n_pin_inst": cur_inst_pin})
                break
            stripped = s.strip()
            if stripped.startswith("- "):
                # flush previous
                if cur_net is not None:
                    out.append({"net_name": cur_net, "n_pins": cur_pins,
                                  "n_pin_PIN": cur_pin_pin, "n_pin_inst": cur_inst_pin})
                cur_net = stripped[2:].split()[0]
                cur_pins = 0; cur_pin_pin = 0; cur_inst_pin = 0
                seen_routed = False
            elif cur_net is not None and not seen_routed:
                # we're between header and first ROUTED
                if "ROUTED" in stripped or "+ FREQUENCY" in stripped or "+ SOURCE" in stripped \
                        or "+ USE" in stripped:
                    seen_routed = True
                else:
                    # pin tuples on this line
                    matches = PIN_RE.findall(stripped)
                    cur_pins += len(matches)
                    for inst, pin in matches:
                        if inst == "PIN":
                            cur_pin_pin += 1
                        else:
                            cur_inst_pin += 1
    return pd.DataFrame(out)


def main():
    cache = _V3 / "cache"
    for d in DESIGNS:
        out = cache / f"pins_{d}.parquet"
        if out.exists():
            print(f"  [cache hit] {d}")
            continue
        def_path = DEF_DIR / f"{d}.def"
        print(f"  parsing pins {d} ...", flush=True)
        df = parse_def_pin_counts(def_path)
        df.to_parquet(out)
        print(f"    n_nets={len(df)}  total_pins={df['n_pins'].sum()}  "
              f"avg_pins/net={df['n_pins'].mean():.2f}")


if __name__ == "__main__":
    main()
