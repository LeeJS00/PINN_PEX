"""Stream DEF NETS section to extract per-net via counts and wirelengths.

Output columns per net:
  net_name
  wirelen_M{i}        : sum of segment lengths on metal layer M{i} (μm)
  n_via_v{i}          : count of VIA{i}_* tokens in net (i ∈ {1..N_via_layers})
  total_wirelen_um
  total_n_via

Used by analytic_r_dressed_test.py to evaluate the analytic R policy on
the full DEF→R chain (no SPEF leakage from RES).

Reference: DEF format §SNET / NETS.
  - ROUTED <metal> ( x1 y1 ) ( x2 y2 ) [ VIA<n>_* ]
  - NEW    <metal> ( x1 y1 ) ( x2 y2 )
  Coordinates in DBU (header default 2000 → 1 μm = 2000).
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


METAL_RE = re.compile(r"^(m\d+)$", re.IGNORECASE)
COORD_RE = re.compile(r"\(\s*([-]?\d+|\*)\s+([-]?\d+|\*)\s*\)")
VIA_RE = re.compile(r"\bVIA(\d+)_[\w]*", re.IGNORECASE)


def stream_def_nets(def_path: Path, dbu: float = 2000.0):
    """Yield dict {net_name, wirelen_M{i}, n_via_v{j}, total_*} per net."""
    in_nets = False
    cur_net = None
    cur = None
    last_x = last_y = None  # tracking '*' continuation
    cur_layer = None

    def parse_pt(x_tok, y_tok):
        nonlocal last_x, last_y
        x = last_x if x_tok == "*" else int(x_tok)
        y = last_y if y_tok == "*" else int(y_tok)
        last_x, last_y = x, y
        return x, y

    def flush():
        if cur_net is not None and cur is not None:
            yield_obj = dict(cur)
            yield_obj["net_name"] = cur_net
            return yield_obj
        return None

    with open(def_path, "r", errors="replace") as f:
        for line in f:
            s = line.rstrip()
            if not in_nets:
                if s.startswith("NETS "):
                    in_nets = True
                continue
            if s.startswith("END NETS"):
                # flush last
                obj = flush()
                if obj is not None:
                    yield obj
                break

            stripped = s.strip()
            # Net header: "- <name>"
            if stripped.startswith("- "):
                obj = flush()
                if obj is not None:
                    yield obj
                cur_net = stripped[2:].strip().split()[0]
                cur = {
                    "wirelen": defaultdict(float),
                    "n_via":   defaultdict(int),
                }
                last_x = last_y = None
                cur_layer = None

            elif cur is not None:
                # Process route rows: "+ ROUTED ..." or "NEW ..." or continuation
                # Strip leading + or NEW
                tokens = stripped.lstrip("+").strip().split()
                if not tokens:
                    continue

                # Identify a layer change; restart the per-segment endpoint chain
                head = tokens[0].upper()
                if head == "ROUTED" or head == "NEW":
                    tokens = tokens[1:]
                    last_x = last_y = None
                    cur_layer = None
                # Now tokens may start with metal-layer name
                if tokens and METAL_RE.match(tokens[0]):
                    cur_layer = tokens[0].lower()
                    # rest of line: coords + optional VIA
                    rest = " ".join(tokens[1:])
                else:
                    rest = " ".join(tokens)

                # Coordinate sequence parsing — track endpoint→endpoint segments
                pts = COORD_RE.findall(rest)
                if cur_layer and METAL_RE.match(cur_layer) and len(pts) >= 1:
                    if last_x is None:
                        x, y = parse_pt(*pts[0])
                        last_x, last_y = x, y
                        pts = pts[1:]
                    for x_tok, y_tok in pts:
                        # snapshot previous endpoint BEFORE parse_pt mutates last_*
                        prev_x, prev_y = last_x, last_y
                        x, y = parse_pt(x_tok, y_tok)
                        dx = abs(x - prev_x); dy = abs(y - prev_y)
                        L_um = (dx + dy) / dbu
                        if L_um > 0:
                            cur["wirelen"][cur_layer] += L_um
                        last_x, last_y = x, y

                # Count vias
                for m in VIA_RE.findall(rest):
                    cur["n_via"][f"v{m}"] += 1


def main():
    if len(sys.argv) > 1:
        defs = [Path(x) for x in sys.argv[1:]]
    else:
        defs = [Path("/home/jslee/projects/PEX_SSL/data/raw/def/intel22/intel22_tv80s_f3.def")]

    for def_path in defs:
        print(f"Parsing: {def_path}", flush=True)
        rows = []
        for obj in stream_def_nets(def_path, dbu=2000.0):
            r = {"net_name": obj["net_name"]}
            for k, v in obj["wirelen"].items():
                r[f"wirelen_{k.upper()}"] = v
            for k, v in obj["n_via"].items():
                r[f"n_via_{k}"] = v
            rows.append(r)

        df = pd.DataFrame(rows).fillna(0.0)
        # Make sure expected columns are present (even if zero)
        for L in ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"]:
            col = f"wirelen_{L}"
            if col not in df.columns:
                df[col] = 0.0
        for v in ["v1", "v2", "v3", "v4", "v5", "v6", "v7"]:
            col = f"n_via_{v}"
            if col not in df.columns:
                df[col] = 0
        out = Path("reports") / f"def_via_counts_{def_path.stem}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"  n_nets: {len(df)}")
        print(f"  total wirelen sum (μm): {df.filter(like='wirelen_').sum().sum():.1f}")
        print(f"  total via count: {df.filter(like='n_via_').sum().sum():.0f}")
        print(f"  per-via-layer counts:")
        for c in sorted(df.columns):
            if c.startswith("n_via_"):
                print(f"    {c}: {df[c].sum():.0f}")
        print(f"  per-metal wirelen totals (μm):")
        for c in sorted(df.columns):
            if c.startswith("wirelen_"):
                print(f"    {c}: {df[c].sum():.1f}")
        print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
