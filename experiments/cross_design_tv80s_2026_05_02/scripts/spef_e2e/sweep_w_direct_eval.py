"""Evaluate the c_gnd direct/ratio blend sweep on tv80s.

For each `_test_e2e_v7_w{w}/` SPEF, compute c_gnd MAPE + total_cap MAPE
vs golden, and rank by c_gnd MAPE (the metric the blend weight controls).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent.parent

# load PINNPEX SPEF parser
_spec = importlib.util.spec_from_file_location(
    "compare_spef_pinpex",
    str(_WS.parent.parent / "src" / "evaluation" / "compare_spef.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
parse_spef = _mod.parse_spef_with_coordinates


GOLDEN = Path("/home/jslee/projects/PEX_SSL/golden_data/spef_data/intel22/intel22_tv80s_f3_starrc.spef")
SWEEP_DIRS = [
    ("w=0.0",   "_test_e2e_v7_w0.0/tv80s.spef"),
    ("w=0.3",   "_test_e2e_v7_w0.3/tv80s.spef"),
    ("w=0.5",   "_test_e2e_v7_w0.5/tv80s.spef"),
    ("w=0.7",   "_test_e2e_v7_w0.7/tv80s.spef"),
    ("w=1.0",   "_test_e2e_v7_w1.0/tv80s.spef"),
    ("FINAL",   "tv80s_FINAL.spef"),
    ("baseline_v7", "_test_e2e_v7/tv80s_v7.spef"),
]


def metrics(p, g):
    common = sorted(set(p.keys()) & set(g.keys()))
    out = {"n_common": len(common)}
    for label, getter in [
        ("total_cap", lambda x: x["total_cap"]),
        ("c_gnd",     lambda x: x["sum_gnd_cap"]),
        ("c_cpl",     lambda x: x["sum_cpl_cap"]),
        ("total_res", lambda x: x["total_res"]),
    ]:
        gv = np.array([getter(g[n]) for n in common])
        pv = np.array([getter(p[n]) for n in common])
        nz = gv > 1e-6
        ape = 100 * np.abs(pv - gv) / np.maximum(gv, 1e-6)
        out[f"{label}_mape"] = float(ape[nz].mean())
        out[f"{label}_bias"] = float(((pv - gv) / np.maximum(gv, 1e-6))[nz].mean() * 100)
    return out


def main():
    print(f"Parsing golden ...", flush=True)
    g = parse_spef(GOLDEN)
    print(f"  golden nets: {len(g)}\n")

    rows = []
    for label, rel in SWEEP_DIRS:
        path = _WS / "output" / "spef_e2e" / rel
        if not path.exists():
            print(f"[SKIP] {label}: not found ({path})")
            continue
        p = parse_spef(path)
        m = metrics(p, g)
        m["label"] = label
        m["spef"]  = rel
        m["spef_size_MB"] = path.stat().st_size / 2**20
        rows.append(m)
        print(f"{label:>10s}  c_gnd={m['c_gnd_mape']:6.3f}%  total={m['total_cap_mape']:6.3f}%  c_cpl={m['c_cpl_mape']:6.3f}%  R={m['total_res_mape']:6.3f}%  bias_gnd={m['c_gnd_bias']:+5.2f}%  ({rel})")

    df = pd.DataFrame(rows)
    df = df[["label", "n_common", "total_cap_mape", "c_gnd_mape", "c_cpl_mape",
             "total_res_mape", "c_gnd_bias", "total_cap_bias", "c_cpl_bias",
             "total_res_bias", "spef_size_MB", "spef"]]
    out_path = _WS / "reports" / "w_direct_sweep.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print("\n=== Ranking by c_gnd MAPE ===")
    print(df.sort_values("c_gnd_mape").to_string(index=False))


if __name__ == "__main__":
    main()
