"""
One-pass SPEF parser. Reads a StarRC SPEF and yields per-net targets.

We use a streaming reader (single pass over the file) rather than per-net
seeks. SPEF is line-oriented and the 11 designs total ~3 GB, so this is
~1-2 min per design.

Targets emitted:
    net_name      str
    total_cap_fF  float    — header value from `*D_NET <name> <cap>`
    c_gnd_fF      float    — sum of grounded *CAP entries (3-token form)
    c_cpl_total_fF float   — sum of coupling *CAP entries (4-token form, halved per StarRC convention)
    total_res_ohm float    — sum of *RES entries
    n_aggressors  int      — distinct aggressor net count (from coupling 4-token entries)
    cpl_p95_fF    float    — 95th percentile of coupling-pair caps
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Iterator


_p_dnet = re.compile(r"\*D_NET\s+(\S+)\s+([0-9.eE+\-]+)")
_p_section = re.compile(r"\*(CAP|RES|END)")


def _norm(s: str) -> str:
    return s.replace("\\", "").strip() if s else ""


def _net_of(node: str) -> str:
    return _norm(node.split(":")[0])


def stream_spef(spef_path: Path) -> Iterator[dict]:
    """Yield one record per `*D_NET` block in `spef_path`.

    Each record:
        net_name, total_cap_fF, c_gnd_fF, c_cpl_total_fF, total_res_ohm,
        coupling_pairs: list[(aggressor_name, cap_fF)]
    """
    section = None
    cur = None
    with open(spef_path, "r") as f:
        for line in f:
            s = line.rstrip()
            if not s:
                continue
            m = _p_dnet.match(s)
            if m:
                if cur is not None:
                    yield cur
                cur = {
                    "net_name": _norm(m.group(1)),
                    "total_cap_fF": float(m.group(2)),
                    "c_gnd_fF": 0.0,
                    "c_cpl_total_fF": 0.0,
                    "total_res_ohm": 0.0,
                    "coupling_pairs": [],
                }
                section = None
                continue
            if cur is None:
                continue
            sm = _p_section.match(s.strip())
            if sm:
                tag = sm.group(1)
                if tag == "END":
                    section = None
                else:
                    section = tag
                continue
            tokens = s.split()
            if section == "CAP" and tokens:
                # Forms:
                #   <id> <node> <val>           (grounded)
                #   <id> <node1> <node2> <val>  (coupling)
                if len(tokens) >= 4 and tokens[0].rstrip(":").isdigit():
                    try:
                        v = float(tokens[3])
                    except ValueError:
                        continue
                    target_net = cur["net_name"]
                    n1 = _net_of(tokens[1])
                    n2 = _net_of(tokens[2])
                    aggressor = n1 if n1 != target_net else n2
                    cur["c_cpl_total_fF"] += v
                    cur["coupling_pairs"].append((aggressor, v))
                elif len(tokens) >= 3 and tokens[0].rstrip(":").isdigit():
                    try:
                        v = float(tokens[2])
                    except ValueError:
                        continue
                    cur["c_gnd_fF"] += v
            elif section == "RES" and tokens:
                if len(tokens) >= 4 and tokens[0].rstrip(":").isdigit():
                    val = tokens[3].split("/")[0]
                    try:
                        cur["total_res_ohm"] += float(val)
                    except ValueError:
                        pass
        if cur is not None:
            yield cur


def parse_spef_to_targets(spef_path: Path) -> dict:
    """Parse the entire SPEF into a dict keyed by net_name."""
    out = {}
    for rec in stream_spef(spef_path):
        coupl = rec["coupling_pairs"]
        if coupl:
            caps = [c for _, c in coupl]
            caps.sort()
            p95 = caps[int(0.95 * (len(caps) - 1))]
        else:
            p95 = 0.0
        n_aggr = len({a for a, _ in coupl})
        out[rec["net_name"]] = {
            "total_cap_fF": rec["total_cap_fF"],
            "c_gnd_fF": rec["c_gnd_fF"],
            "c_cpl_total_fF": rec["c_cpl_total_fF"],
            "total_res_ohm": rec["total_res_ohm"],
            "n_aggressors_spef": n_aggr,
            "cpl_p95_fF": p95,
        }
    return out


if __name__ == "__main__":
    import sys
    p = Path(sys.argv[1])
    nets = parse_spef_to_targets(p)
    print(f"{p.name}: {len(nets)} nets")
    items = list(nets.items())[:3]
    for k, v in items:
        print(" ", k, v)
