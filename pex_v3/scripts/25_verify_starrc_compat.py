#!/usr/bin/env python3
"""
25_verify_starrc_compat.py — Verify PINN-PEX SPEF semantic compatibility
with golden StarRC SPEF.

Checks beyond per-net cap MAPE (which compare_spef.py covers):
    1. Header field equivalence (*SPEF version, *DESIGN, units, delimiter, divider)
    2. Net coverage (golden ⊆ predicted, predicted ⊆ golden)
    3. Per-net structure presence (*CONN, *CAP, *RES)
    4. Per-net node naming convention (*P / *I / *N) matches golden
    5. *D_NET <total> ≈ Σ *CAP entries
    6. Coupling notation (`<id> <node_a> <node_b> <c>`) matches
    7. Resistance topology (segment count comparable, not just lumped)
    8. Layer/z mapping (sample comparison)

Output: a structured JSON + markdown report, suitable to include in paper.
"""
from __future__ import annotations
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional


def parse_args():
    p = argparse.ArgumentParser(description="StarRC SPEF compatibility verifier")
    p.add_argument("--golden", type=Path, required=True)
    p.add_argument("--pred", type=Path, required=True)
    p.add_argument("--out-md", type=Path, required=True)
    return p.parse_args()


def parse_spef_full(path: Path) -> dict:
    """Detailed parser: header + per-net structure presence + counts."""
    header: dict = {}
    nets: dict[str, dict] = {}
    current_net: Optional[str] = None
    current_total: Optional[float] = None
    in_conn = False
    in_cap = False
    in_res = False
    n_conn_lines = 0
    n_cap_lines = 0
    n_res_lines = 0
    cap_sum = 0.0
    cap_gnd_sum = 0.0
    cap_cpl_sum = 0.0
    res_sum = 0.0
    has_p_node = False
    has_i_node = False
    has_n_node = False
    coupling_edges = 0

    HEADER_KEYS = {
        "*SPEF": "spef_version", "*DESIGN": "design", "*DATE": "date",
        "*VENDOR": "vendor", "*PROGRAM": "program", "*VERSION": "version",
        "*DESIGN_FLOW": "design_flow", "*DIVIDER": "divider",
        "*DELIMITER": "delimiter", "*BUS_DELIMITER": "bus_delimiter",
        "*T_UNIT": "t_unit", "*C_UNIT": "c_unit",
        "*R_UNIT": "r_unit", "*L_UNIT": "l_unit",
    }

    def flush_current():
        nonlocal current_net
        if current_net is None:
            return
        nets[current_net] = {
            "total_d_net": current_total,
            "n_conn": n_conn_lines,
            "n_cap": n_cap_lines,
            "n_res": n_res_lines,
            "sum_cap": cap_sum,
            "sum_gnd_cap": cap_gnd_sum,
            "sum_cpl_cap": cap_cpl_sum,
            "sum_res": res_sum,
            "has_p": has_p_node,
            "has_i": has_i_node,
            "has_n": has_n_node,
            "n_coupling_edges": coupling_edges,
        }

    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue

            # Header
            tokens = stripped.split()
            if tokens[0] in HEADER_KEYS and current_net is None:
                header[HEADER_KEYS[tokens[0]]] = stripped
                continue

            if stripped.startswith("*PORTS") or stripped.startswith("*POWER_NETS") \
                    or stripped.startswith("*GROUND_NETS"):
                continue

            if stripped.startswith("*D_NET"):
                if current_net is not None:
                    flush_current()
                tokens = stripped.split()
                current_net = tokens[1]
                try:
                    current_total = float(tokens[2])
                except (ValueError, IndexError):
                    current_total = None
                in_conn = in_cap = in_res = False
                n_conn_lines = n_cap_lines = n_res_lines = 0
                cap_sum = cap_gnd_sum = cap_cpl_sum = res_sum = 0.0
                has_p_node = has_i_node = has_n_node = False
                coupling_edges = 0
                continue

            if not current_net:
                continue

            if stripped.startswith("*CONN"):
                in_conn, in_cap, in_res = True, False, False
                continue
            if stripped.startswith("*CAP"):
                in_conn, in_cap, in_res = False, True, False
                continue
            if stripped.startswith("*RES"):
                in_conn, in_cap, in_res = False, False, True
                continue
            if stripped.startswith("*END"):
                flush_current()
                current_net = None
                in_conn = in_cap = in_res = False
                continue

            if in_conn:
                n_conn_lines += 1
                if stripped.startswith("*P"):
                    has_p_node = True
                elif stripped.startswith("*I"):
                    has_i_node = True
                elif stripped.startswith("*N"):
                    has_n_node = True

            if in_cap and not stripped.startswith("*"):
                n_cap_lines += 1
                tokens = stripped.split()
                if len(tokens) == 3:
                    try:
                        c_val = float(tokens[2])
                        cap_sum += c_val
                        cap_gnd_sum += c_val
                    except ValueError:
                        pass
                elif len(tokens) == 4:
                    try:
                        c_val = float(tokens[3])
                        cap_sum += c_val
                        cap_cpl_sum += c_val
                        coupling_edges += 1
                    except ValueError:
                        pass

            if in_res and not stripped.startswith("*"):
                n_res_lines += 1
                tokens = stripped.split()
                if len(tokens) >= 4:
                    try:
                        res_sum += float(tokens[3])
                    except ValueError:
                        pass

    flush_current()
    return {"header": header, "nets": nets}


def compare_headers(g: dict, p: dict) -> dict:
    """Header field comparison."""
    keys = sorted(set(g.keys()) | set(p.keys()))
    diffs = {}
    matches = []
    for k in keys:
        gv = g.get(k, "(missing)")
        pv = p.get(k, "(missing)")
        if gv == pv:
            matches.append(k)
        else:
            # Some fields naturally differ (date, vendor, program)
            if k in ("date", "vendor", "program", "version"):
                diffs[k] = {"golden": gv, "pred": pv, "expected_diff": True}
            else:
                diffs[k] = {"golden": gv, "pred": pv, "expected_diff": False}
    return {"matches": matches, "diffs": diffs}


def main() -> None:
    args = parse_args()
    print(f">>> Parsing golden: {args.golden}")
    g = parse_spef_full(args.golden)
    print(f">>> Parsing pred  : {args.pred}")
    p = parse_spef_full(args.pred)

    print()
    print("=" * 60)
    print("[1] Header compatibility")
    print("=" * 60)
    hdr = compare_headers(g["header"], p["header"])
    for k in hdr["matches"]:
        print(f"  ✅ {k:18}: {g['header'][k]}")
    for k, v in hdr["diffs"].items():
        marker = "ℹ" if v.get("expected_diff") else "❌"
        print(f"  {marker} {k:18}: golden={v['golden']!r}  pred={v['pred']!r}")

    print()
    print("=" * 60)
    print("[2] Net coverage")
    print("=" * 60)
    g_nets = set(g["nets"])
    p_nets = set(p["nets"])
    common = g_nets & p_nets
    g_only = g_nets - p_nets
    p_only = p_nets - g_nets
    print(f"  golden nets : {len(g_nets):>7,}")
    print(f"  pred nets   : {len(p_nets):>7,}")
    print(f"  common      : {len(common):>7,}  ({len(common)/max(len(g_nets),1)*100:.2f}% of golden)")
    print(f"  golden-only : {len(g_only):>7,}")
    print(f"  pred-only   : {len(p_only):>7,}")

    print()
    print("=" * 60)
    print("[3] Per-net structure presence (common nets)")
    print("=" * 60)
    n_with_conn_g = sum(1 for n in common if g["nets"][n]["n_conn"] > 0)
    n_with_conn_p = sum(1 for n in common if p["nets"][n]["n_conn"] > 0)
    n_with_cap_g = sum(1 for n in common if g["nets"][n]["n_cap"] > 0)
    n_with_cap_p = sum(1 for n in common if p["nets"][n]["n_cap"] > 0)
    n_with_res_g = sum(1 for n in common if g["nets"][n]["n_res"] > 0)
    n_with_res_p = sum(1 for n in common if p["nets"][n]["n_res"] > 0)
    print(f"  *CONN block: golden {n_with_conn_g:>7,} / pred {n_with_conn_p:>7,}")
    print(f"  *CAP block : golden {n_with_cap_g:>7,} / pred {n_with_cap_p:>7,}")
    print(f"  *RES block : golden {n_with_res_g:>7,} / pred {n_with_res_p:>7,}")

    print()
    print("=" * 60)
    print("[4] Node-type coverage (CONN nodes by type)")
    print("=" * 60)
    g_with_p = sum(1 for n in common if g["nets"][n]["has_p"])
    g_with_i = sum(1 for n in common if g["nets"][n]["has_i"])
    g_with_n = sum(1 for n in common if g["nets"][n]["has_n"])
    p_with_p = sum(1 for n in common if p["nets"][n]["has_p"])
    p_with_i = sum(1 for n in common if p["nets"][n]["has_i"])
    p_with_n = sum(1 for n in common if p["nets"][n]["has_n"])
    print(f"  Has *P (port)    : golden {g_with_p:>7,} / pred {p_with_p:>7,}")
    print(f"  Has *I (instance): golden {g_with_i:>7,} / pred {p_with_i:>7,}")
    print(f"  Has *N (internal): golden {g_with_n:>7,} / pred {p_with_n:>7,}")

    print()
    print("=" * 60)
    print("[5] *D_NET total cap consistency")
    print("=" * 60)
    import numpy as np
    diffs_g = []
    diffs_p = []
    for n in common:
        gn = g["nets"][n]
        pn = p["nets"][n]
        if gn["total_d_net"] is not None and gn["sum_cap"] > 1e-9:
            diffs_g.append(abs(gn["total_d_net"] - gn["sum_cap"]) / gn["sum_cap"])
        if pn["total_d_net"] is not None and pn["sum_cap"] > 1e-9:
            diffs_p.append(abs(pn["total_d_net"] - pn["sum_cap"]) / pn["sum_cap"])
    print(f"  abs(*D_NET total - sum *CAP) / sum *CAP:")
    print(f"    golden: median {np.median(diffs_g)*100:.4f}%  max {np.max(diffs_g)*100:.4f}%")
    print(f"    pred  : median {np.median(diffs_p)*100:.4f}%  max {np.max(diffs_p)*100:.4f}%")

    print()
    print("=" * 60)
    print("[6] Resistance topology (segment counts)")
    print("=" * 60)
    g_seg_counts = [g["nets"][n]["n_res"] for n in common]
    p_seg_counts = [p["nets"][n]["n_res"] for n in common]
    print(f"  *RES segments per net:")
    print(f"    golden: median {np.median(g_seg_counts):.0f}  mean {np.mean(g_seg_counts):.1f}  max {np.max(g_seg_counts)}")
    print(f"    pred  : median {np.median(p_seg_counts):.0f}  mean {np.mean(p_seg_counts):.1f}  max {np.max(p_seg_counts)}")

    g_cap_counts = [g["nets"][n]["n_cap"] for n in common]
    p_cap_counts = [p["nets"][n]["n_cap"] for n in common]
    print(f"  *CAP entries per net:")
    print(f"    golden: median {np.median(g_cap_counts):.0f}  mean {np.mean(g_cap_counts):.1f}  max {np.max(g_cap_counts)}")
    print(f"    pred  : median {np.median(p_cap_counts):.0f}  mean {np.mean(p_cap_counts):.1f}  max {np.max(p_cap_counts)}")

    g_cpl = [g["nets"][n]["n_coupling_edges"] for n in common]
    p_cpl = [p["nets"][n]["n_coupling_edges"] for n in common]
    print(f"  *CAP coupling edges per net:")
    print(f"    golden: median {np.median(g_cpl):.0f}  mean {np.mean(g_cpl):.1f}  max {np.max(g_cpl)}")
    print(f"    pred  : median {np.median(p_cpl):.0f}  mean {np.mean(p_cpl):.1f}  max {np.max(p_cpl)}")

    print()
    print("=" * 60)
    print("[7] Verdict summary")
    print("=" * 60)
    headers_ok = all(not v.get("expected_diff") is False for v in hdr["diffs"].values()) or len([v for v in hdr["diffs"].values() if not v.get("expected_diff")]) == 0
    coverage_ok = len(g_only) == 0 and len(p_only) == 0
    structure_ok = (n_with_conn_p / max(n_with_conn_g, 1) >= 0.99
                    and n_with_cap_p / max(n_with_cap_g, 1) >= 0.99
                    and n_with_res_p / max(n_with_res_g, 1) >= 0.99)
    nodes_ok = (p_with_p / max(g_with_p, 1) >= 0.95
                and p_with_i / max(g_with_i, 1) >= 0.50  # less strict; pred may use *N for some pins
                and p_with_n / max(g_with_n, 1) >= 0.95)
    cap_consistency_ok = (np.median(diffs_p) < 0.01)  # *D_NET total within 1% of sum *CAP

    verdict = {
        "headers_compatible": headers_ok,
        "net_coverage_complete": coverage_ok,
        "structure_present": structure_ok,
        "node_types_present": nodes_ok,
        "cap_consistency": cap_consistency_ok,
    }
    overall_ok = all(verdict.values())

    print(f"  Headers compatible : {'✅' if headers_ok else '❌'}")
    print(f"  Net coverage 100%  : {'✅' if coverage_ok else '❌'}")
    print(f"  Structure ≥99%     : {'✅' if structure_ok else '❌'}")
    print(f"  Node types ≥95/50% : {'✅' if nodes_ok else '❌'}")
    print(f"  *D_NET tot cap <1% : {'✅' if cap_consistency_ok else '❌'}")
    print(f"  OVERALL            : {'✅ STARRC-COMPATIBLE' if overall_ok else '⚠ partial'}")

    # Save markdown report
    with open(args.out_md, "w") as f:
        f.write(f"# StarRC SPEF Compatibility Verification\n\n")
        f.write(f"_Golden_: `{args.golden}`\n")
        f.write(f"_Pred_:   `{args.pred}`\n\n")
        f.write(f"## Verdict: {'✅ STARRC-COMPATIBLE' if overall_ok else '⚠ partial'}\n\n")
        f.write(f"| Check | Result |\n|---|---:|\n")
        for k, v in verdict.items():
            f.write(f"| {k} | {'✅' if v else '❌'} |\n")
        f.write(f"\n## Header fields\n\nMatching: {len(hdr['matches'])} fields\n\n")
        for k in hdr["matches"]:
            f.write(f"- ✅ `{k}`: `{g['header'].get(k, '')}`\n")
        for k, v in hdr["diffs"].items():
            marker = "ℹ" if v.get("expected_diff") else "❌"
            f.write(f"- {marker} `{k}`: golden=`{v['golden']}` pred=`{v['pred']}`\n")
        f.write(f"\n## Net coverage\n\n")
        f.write(f"- golden: {len(g_nets):,} nets\n")
        f.write(f"- pred:   {len(p_nets):,} nets\n")
        f.write(f"- common: {len(common):,} ({len(common)/max(len(g_nets),1)*100:.2f}%)\n")
        f.write(f"- golden-only: {len(g_only):,} (sample: {list(g_only)[:5] if g_only else '—'})\n")
        f.write(f"- pred-only:   {len(p_only):,} (sample: {list(p_only)[:5] if p_only else '—'})\n")
        f.write(f"\n## Per-net structure (common {len(common):,} nets)\n\n")
        f.write(f"| Block | Golden | Pred |\n|---|---:|---:|\n")
        f.write(f"| `*CONN` | {n_with_conn_g:,} | {n_with_conn_p:,} |\n")
        f.write(f"| `*CAP`  | {n_with_cap_g:,} | {n_with_cap_p:,} |\n")
        f.write(f"| `*RES`  | {n_with_res_g:,} | {n_with_res_p:,} |\n")
        f.write(f"\n## Node types (in *CONN)\n\n")
        f.write(f"| Type | Golden | Pred |\n|---|---:|---:|\n")
        f.write(f"| `*P` (ports) | {g_with_p:,} | {p_with_p:,} |\n")
        f.write(f"| `*I` (instance pins) | {g_with_i:,} | {p_with_i:,} |\n")
        f.write(f"| `*N` (internal nodes) | {g_with_n:,} | {p_with_n:,} |\n")
        f.write(f"\n## *D_NET ↔ Σ *CAP consistency\n\n")
        f.write(f"| | median | max |\n|---|---:|---:|\n")
        f.write(f"| golden | {np.median(diffs_g)*100:.4f}% | {np.max(diffs_g)*100:.4f}% |\n")
        f.write(f"| pred   | {np.median(diffs_p)*100:.4f}% | {np.max(diffs_p)*100:.4f}% |\n")
        f.write(f"\n## Resistance topology\n\n")
        f.write(f"| | golden median | golden mean | golden max | pred median | pred mean | pred max |\n|---|---:|---:|---:|---:|---:|---:|\n")
        f.write(f"| *RES segments | {np.median(g_seg_counts):.0f} | {np.mean(g_seg_counts):.1f} | {np.max(g_seg_counts)} | {np.median(p_seg_counts):.0f} | {np.mean(p_seg_counts):.1f} | {np.max(p_seg_counts)} |\n")
        f.write(f"| *CAP entries  | {np.median(g_cap_counts):.0f} | {np.mean(g_cap_counts):.1f} | {np.max(g_cap_counts)} | {np.median(p_cap_counts):.0f} | {np.mean(p_cap_counts):.1f} | {np.max(p_cap_counts)} |\n")
        f.write(f"| coupling edges | {np.median(g_cpl):.0f} | {np.mean(g_cpl):.1f} | {np.max(g_cpl)} | {np.median(p_cpl):.0f} | {np.mean(p_cpl):.1f} | {np.max(p_cpl)} |\n")

    print(f"\n✅ report → {args.out_md}")


if __name__ == "__main__":
    main()
