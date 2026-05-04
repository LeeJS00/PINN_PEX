"""Extract calibrated per-layer sheet R and per-via R from golden SPEFs.

Strategy:
  golden *RES section contains every wire+via segment with annotations:
    (idx) (node_a) (node_b) (R) // $l=L $w=W $lvl=LAYER ...   <- metal segment
    (idx) (node_a) (node_b) (R) // $vc=VC $savc=N $lvl=VIA_LVL ...  <- via

  For each metal segment: sheet_R[layer] estimate = R * W / L
  For each via:
    R_via_per is approximately R / max(savc, 1) ;
    OR R is the total parallel R for vc-class via cluster.
    We extract median R per (lvl, vc) bucket.

This calibration replaces the hand-coded DEFAULT_SHEET_R_INTEL22 in
pex_pipeline/compute_resistance.py.

Output: reports/sheet_r_calibration.json + per-design CSVs.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent.parent

# Golden SPEFs (intel22, _f3 corner) — same paths used by training pipeline.
GOLDEN_DIR = Path("/home/jslee/projects/PEX_SSL/golden_data/spef_data/intel22")
TRAIN_DESIGNS = [
    "intel22_aes_cipher_top_f3_starrc.spef",
    "intel22_gcd_f3_starrc.spef",
    "intel22_ibex_core_f3_starrc.spef",
    "intel22_mc_top_f3_starrc.spef",
    "intel22_spi_top_f3_starrc.spef",
    "intel22_usbf_top_f3_starrc.spef",
    "intel22_vga_enh_top_f3_starrc.spef",
    "intel22_wb_conmax_top_f3_starrc.spef",
    "intel22_nova_f3_starrc.spef",
]
TEST_DESIGN = "intel22_tv80s_f3_starrc.spef"

# Layer index → name mapping from the SPEF *LAYER_MAP comment block.
# (Reference from tv80s SPEF):
#   *2 m8, *3 m7, *4 m6, *5 m5, *6 m4, *7 m3, *8 m2, *9 m1
#   *10 v1, *11 v2, *12 v3, ..., *17 tv1
# (m9 and friends not visible in tv80s — see if other designs use them.)
LAYER_INDEX_TO_NAME = {
    1: "c4",
    2: "m8", 3: "m7", 4: "m6", 5: "m5", 6: "m4",
    7: "m3", 8: "m2", 9: "m1",
    10: "v1", 11: "v2", 12: "v3", 13: "v4",
    14: "v5", 15: "v6", 16: "v7", 17: "tv1",
}

LINE_RE = re.compile(
    r"^\s*\d+\s+\S+\s+\S+\s+(?P<R>[\d.eE+-]+)\s*//\s*(?P<ann>.+)$"
)
KV_RE = re.compile(r"\$(\w+)=([-+]?[\d.eE]+(?:[A-Za-z][\w]*)?|\d+x\d+|[\w_]+)")


def parse_res_section(spef_path: Path, max_nets: int = None):
    """Yield (R, kv_dict) for each *RES line.

    Uses streaming parse — the giant SPEF (multi-GB) requires it.
    """
    in_res = False
    n_nets = 0
    with open(spef_path, "r", errors="replace") as f:
        for line in f:
            s = line.rstrip()
            if s.startswith("*D_NET"):
                in_res = False
                n_nets += 1
                if max_nets is not None and n_nets > max_nets:
                    break
            elif s.startswith("*RES"):
                in_res = True
            elif s.startswith("*END"):
                in_res = False
            elif in_res and s and not s.startswith("*"):
                m = LINE_RE.match(s)
                if not m:
                    continue
                R = float(m.group("R"))
                ann = m.group("ann")
                kv = dict(KV_RE.findall(ann))
                yield R, kv


def calibrate_one_design(spef_path: Path, max_nets: int = 1500):
    """Walk the *RES of one design and accumulate per-layer stats."""
    metal_rows = []   # (layer_idx, l, w, R, sheet_est)
    via_rows = []     # (layer_idx, vc, savc, R)
    for R, kv in parse_res_section(spef_path, max_nets=max_nets):
        lvl = int(kv.get("lvl", -1))
        if "l" in kv and "w" in kv:
            L = float(kv["l"])
            W = float(kv["w"])
            if L > 0 and W > 0:
                sheet = R * W / L
                metal_rows.append((lvl, L, W, R, sheet))
        elif "vc" in kv:
            vc = int(kv["vc"])
            savc = int(kv.get("savc", 0))
            via_rows.append((lvl, vc, savc, R))
        else:
            # Edge-case "//  $l=0.0000 $w=10.0000" type rows (pin contact). skip.
            pass
    return metal_rows, via_rows


def main():
    ap_metal = defaultdict(list)
    ap_via = defaultdict(list)
    print(f"Designs to process: {len(TRAIN_DESIGNS)} + 1 test (for hold-out comparison)\n")

    per_design_summary = []
    for fn in TRAIN_DESIGNS + [TEST_DESIGN]:
        path = GOLDEN_DIR / fn
        if not path.exists():
            print(f"  [skip] {fn} (not found)")
            continue
        print(f"  -> {fn} ...", flush=True)
        # tv80s has 3,380 nets; aes ~10K; ibex ~20K; ldpc ~200K; mpeg ~600K.
        # Cap each design at 3,000 nets for reasonable runtime + statistical stability.
        # On metal segments alone aes hits ~250K segments easily.
        metal, via = calibrate_one_design(path, max_nets=3000)
        d = pd.DataFrame(metal, columns=["lvl", "L", "W", "R", "sheet"])
        v = pd.DataFrame(via, columns=["lvl", "vc", "savc", "R"])
        per_design_summary.append({
            "design": fn,
            "n_metal": len(d),
            "n_via": len(v),
        })
        # accumulate (only train designs go into final calibration)
        if fn != TEST_DESIGN:
            for lvl, sub in d.groupby("lvl"):
                ap_metal[int(lvl)].extend(sub["sheet"].tolist())
            for (lvl, vc), sub in v.groupby(["lvl", "vc"]):
                ap_via[(int(lvl), int(vc))].extend(sub["R"].tolist())
        # save per-design split
        d.to_csv(_WS / "reports" / f"sheet_r_metal_{fn}.csv", index=False)
        v.to_csv(_WS / "reports" / f"sheet_r_via_{fn}.csv",   index=False)

    # ------------------------------------------------------------------
    # Summarize per-layer sheet R (median over training designs)
    # ------------------------------------------------------------------
    print("\n=== Per-layer sheet R (median over train designs, ohm/sq) ===")
    metal_summary = []
    for lvl in sorted(ap_metal):
        vals = np.array(ap_metal[lvl])
        # robust median; trim outliers (top/bottom 5% — wire contacts $l=0 etc.)
        lo, hi = np.percentile(vals, [5, 95])
        keep = vals[(vals >= lo) & (vals <= hi)]
        med = float(np.median(keep)) if len(keep) else float("nan")
        mn  = float(keep.mean()) if len(keep) else float("nan")
        name = LAYER_INDEX_TO_NAME.get(lvl, f"lvl{lvl}")
        print(f"  lvl={lvl:>2d} ({name:>4s})  n={len(vals):>7d}  median={med:.4f}  mean={mn:.4f}  range[5,95]=[{lo:.3f}, {hi:.3f}]")
        metal_summary.append({
            "lvl": lvl, "name": name, "n": int(len(vals)),
            "sheet_median": med, "sheet_mean": mn,
            "p5": float(lo), "p95": float(hi),
        })

    # ------------------------------------------------------------------
    # Summarize per-via R (median over (lvl, vc))
    # ------------------------------------------------------------------
    print("\n=== Per-via R (median by (lvl, vc) class, ohm) ===")
    via_summary = []
    for (lvl, vc) in sorted(ap_via):
        vals = np.array(ap_via[(lvl, vc)])
        if len(vals) < 5:
            continue
        med = float(np.median(vals)); mn = float(vals.mean())
        name = LAYER_INDEX_TO_NAME.get(lvl, f"lvl{lvl}")
        print(f"  lvl={lvl:>2d} ({name:>4s})  vc={vc:>2d}  n={len(vals):>7d}  median={med:.3f}  mean={mn:.3f}")
        via_summary.append({"lvl": lvl, "name": name, "vc": vc, "n": int(len(vals)),
                              "R_median": med, "R_mean": mn})

    # ------------------------------------------------------------------
    # Aggregate via R per layer (collapsed across vc) — useful when we don't
    # know vc class at inference time.
    # ------------------------------------------------------------------
    print("\n=== Per-via R collapsed by layer (median over all vc classes, weighted by count) ===")
    via_per_layer = []
    for lvl in sorted({k[0] for k in ap_via}):
        vals = []
        for (l2, _vc), v in ap_via.items():
            if l2 == lvl:
                vals.extend(v)
        vals = np.array(vals)
        if len(vals) < 5:
            continue
        name = LAYER_INDEX_TO_NAME.get(lvl, f"lvl{lvl}")
        med = float(np.median(vals)); mn = float(vals.mean())
        print(f"  lvl={lvl:>2d} ({name:>4s})  n={len(vals):>7d}  median={med:.3f}  mean={mn:.3f}")
        via_per_layer.append({"lvl": lvl, "name": name, "n": int(len(vals)),
                                "R_median": med, "R_mean": mn})

    out_path = _WS / "reports" / "sheet_r_calibration.json"
    with open(out_path, "w") as f:
        json.dump({
            "metal_per_layer": metal_summary,
            "via_per_lvl_vc": via_summary,
            "via_per_layer": via_per_layer,
            "per_design_counts": per_design_summary,
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
