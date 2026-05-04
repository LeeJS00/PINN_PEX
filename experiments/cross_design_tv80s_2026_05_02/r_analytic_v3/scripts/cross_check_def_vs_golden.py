"""Phase 5d — Cross-check DEF features vs golden RES counts.

For each tv80s net, compare:
  - n_via_per_lvl from DEF (PINNPEX parser)
  - n_via_per_lvl from golden RES (count of $vc segments)
  - sum L/W for metal layers from DEF
  - sum L/W for metal segments in golden RES (from $l, $w annotations)

This tells us where the DEF→features extraction is incomplete.
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

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parent

GOLDEN = Path("/home/jslee/projects/PEX_SSL/golden_data/spef_data/intel22/intel22_tv80s_f3_starrc.spef")

KV_RE = re.compile(r"\$(\w+)=([-+]?[\d.eE]+(?:[A-Za-z][\w]*)?|\d+x\d+|[\w_]+)")
LINE_RE = re.compile(r"^\s*\d+\s+\S+\s+\S+\s+(?P<R>[\d.eE+-]+)\s*//\s*(?P<ann>.+)$")

LVL_TO_LAYER = {
    7: "M3", 8: "M2", 9: "M1", 6: "M4", 5: "M5", 4: "M6", 3: "M7", 2: "M8",
    10: "v1", 11: "v2", 12: "v3", 13: "v4", 14: "v5", 15: "v6", 16: "v7",
}


def parse_res(spef_path):
    in_res = False
    cur = None
    nets = {}
    with open(spef_path, errors="ignore") as f:
        for line in f:
            s = line.rstrip()
            if s.startswith("*D_NET"):
                cur = s.split()[1]
                nets[cur] = {"metal_nsq": defaultdict(float),
                             "metal_R":  defaultdict(float),
                             "via_count": defaultdict(int),
                             "via_R":    defaultdict(float)}
                in_res = False
            elif s.startswith("*RES"):
                in_res = True
            elif s.startswith("*END"):
                in_res = False
            elif in_res and cur is not None:
                m = LINE_RE.match(s)
                if not m:
                    continue
                R = float(m.group("R"))
                kv = dict(KV_RE.findall(m.group("ann")))
                lvl = int(kv.get("lvl", -1))
                lname = LVL_TO_LAYER.get(lvl)
                if lname is None:
                    continue
                if "l" in kv and "w" in kv:
                    L = float(kv["l"]); W = float(kv["w"])
                    if L > 0 and W > 0:
                        nets[cur]["metal_nsq"][lname] += L / W
                        nets[cur]["metal_R"][lname] += R
                elif "vc" in kv:
                    nets[cur]["via_count"][lname] += 1
                    nets[cur]["via_R"][lname] += R
    return nets


def main():
    print("Parsing golden RES topology ...", flush=True)
    g = parse_res(GOLDEN)
    print(f"  {len(g)} nets parsed")

    # load DEF-derived features
    df_def = pd.read_parquet(_V3 / "cache" / "feat_v2_intel22_tv80s_f3.parquet")
    pins   = pd.read_parquet(_V3 / "cache" / "pins_intel22_tv80s_f3.parquet")
    df = df_def.merge(pins, on="net_name", how="left").fillna(0.0)
    df = df.dropna(subset=["R_gold"])
    df = df[df["R_gold"] > 0.1].reset_index(drop=True).copy()
    print(f"  {len(df)} test nets joined")

    # comparison: per-net DEF nsq_M{i} vs golden metal_nsq[M{i}]
    # also per-via per-lvl
    diff_rows = []
    for _, row in df.iterrows():
        n = row["net_name"]
        if n not in g:
            continue
        gold = g[n]
        rec = {"net_name": n, "R_gold": row["R_gold"]}
        for L in ["M1", "M2", "M3", "M4", "M5"]:
            rec[f"def_nsq_{L}"]  = float(row.get(f"nsq_{L}", 0))
            rec[f"gold_nsq_{L}"] = float(gold["metal_nsq"][L])
            rec[f"diff_nsq_{L}"] = rec[f"gold_nsq_{L}"] - rec[f"def_nsq_{L}"]
        for v in ["v1", "v2", "v3", "v4"]:
            n_def_via = 0
            for col in row.index:
                if col.startswith(f"nvian_VIA{v[1]}_"):
                    n_def_via += int(row[col])
            rec[f"def_nvia_{v}"]  = n_def_via
            rec[f"gold_nvia_{v}"] = gold["via_count"][v]
            rec[f"diff_nvia_{v}"] = rec[f"gold_nvia_{v}"] - rec[f"def_nvia_{v}"]
        rec["n_pins"] = float(row.get("n_pins", 0))
        diff_rows.append(rec)
    cmp = pd.DataFrame(diff_rows)

    print("\n=== Avg per-net deltas (golden RES − DEF features) ===")
    metal_cols = [c for c in cmp.columns if c.startswith("diff_nsq_")]
    via_cols   = [c for c in cmp.columns if c.startswith("diff_nvia_")]
    for c in metal_cols:
        print(f"  {c:<20s}: mean={cmp[c].mean():+8.3f}  median={cmp[c].median():+8.3f}  "
              f"max_abs={cmp[c].abs().max():.2f}")
    for c in via_cols:
        print(f"  {c:<20s}: mean={cmp[c].mean():+8.3f}  median={cmp[c].median():+8.3f}  "
              f"max_abs={cmp[c].abs().max():.2f}")

    # any net with consistent over/under-extraction?
    print("\n=== Top 10 nets where DEF vs golden vias disagree most ===")
    cmp["total_diff_via"] = cmp[via_cols].sum(axis=1).abs()
    print(cmp.nlargest(10, "total_diff_via")[
        ["net_name", "R_gold"] +
        [c for c in cmp.columns if c.startswith("def_nvia_") or c.startswith("gold_nvia_")]
    ].to_string(index=False))

    print("\n=== Top 10 nets where DEF vs golden M2 nsq disagree most ===")
    cmp["abs_diff_M2"] = cmp["diff_nsq_M2"].abs()
    print(cmp.nlargest(10, "abs_diff_M2")[
        ["net_name", "R_gold", "def_nsq_M2", "gold_nsq_M2", "diff_nsq_M2", "n_pins"]
    ].to_string(index=False))

    cmp.to_parquet(_V3 / "outputs" / "def_vs_golden_diffs.parquet")


if __name__ == "__main__":
    main()
