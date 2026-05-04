"""Phase 13a — Build c_gnd ground truth target per net per design.

Parses golden SPEFs and extracts `sum_gnd_cap` per net (sum of all gnd_caps
in the *CAP section). Saves per-design parquets keyed by net_name.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

PROJ = Path("/home/jslee/projects/PINNPEX")
sys.path.insert(0, str(PROJ))

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parent

SPEF_DIR = Path("/home/jslee/projects/PEX_SSL/golden_data/spef_data/intel22")

_spec = importlib.util.spec_from_file_location(
    "compare_spef_pinpex",
    str(PROJ / "src" / "evaluation" / "compare_spef.py"),
)
_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
parse_spef = _mod.parse_spef_with_coordinates

DESIGNS = [
    "intel22_aes_cipher_top_f3", "intel22_gcd_f3", "intel22_ibex_core_f3",
    "intel22_ldpc_decoder_802_3an_f3",
    "intel22_mc_top_f3", "intel22_spi_top_f3", "intel22_usbf_top_f3",
    "intel22_vga_enh_top_f3", "intel22_wb_conmax_top_f3", "intel22_nova_f3",
    "intel22_tv80s_f3",
]


def main():
    cache = _V3 / "cache"
    for d in DESIGNS:
        out = cache / f"cgnd_{d}.parquet"
        if out.exists():
            print(f"  [cache hit] {d}")
            continue
        spef = SPEF_DIR / f"{d}_starrc.spef"
        if not spef.exists():
            print(f"  [skip] {d}: SPEF missing")
            continue
        print(f"  parsing {d} ...", flush=True)
        nets = parse_spef(spef)
        rows = []
        for n, info in nets.items():
            rows.append({
                "net_name":      n,
                "c_gnd_gold":    float(info["sum_gnd_cap"]),
                "c_cpl_gold":    float(info["sum_cpl_cap"]),
                "total_cap_gold": float(info["sum_gnd_cap"]) + float(info["sum_cpl_cap"]),
            })
        df = pd.DataFrame(rows)
        df.to_parquet(out)
        print(f"    n_nets={len(df)}  c_gnd mean={df['c_gnd_gold'].mean():.4f}fF  "
              f"median={df['c_gnd_gold'].median():.4f}fF  max={df['c_gnd_gold'].max():.2f}fF")


if __name__ == "__main__":
    main()
