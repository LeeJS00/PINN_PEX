#!/usr/bin/env python3
"""42_mesh_ratio_calibrate_spef.py — Mesh-PINN per-channel-ratio calibration.

Hybrid post-process: PRESERVES XGB per-net TOTAL (best per-net total
predictor) while OVERRIDING the per-channel SPLIT (gnd vs cpl) using the
Mesh PINN ensemble (better per-channel predictor on tv80s test).

For each net with both XGB and Mesh predictions:
    total       := xgb_pred_gnd + xgb_pred_cpl                # XGB total preserved
    mesh_ratio  := mesh_pred_gnd / (mesh_pred_gnd + mesh_pred_cpl)
    target_gnd  := total × mesh_ratio
    target_cpl  := total × (1 - mesh_ratio)
    gnd_scale   := target_gnd / current_sum_gnd
    cpl_scale   := target_cpl / current_sum_cpl

Apply by walking *CAP block: each gnd entry × gnd_scale, each cpl entry × cpl_scale.

This breaks the XGB per-channel ceiling (gnd 27.37 → 23.44, cpl 18.78 → 18.40
on tv80s measured) while keeping XGB's per-net total accuracy.

Inputs:
    --in-spef      autonomous_fast.spef AFTER 16_xgb_calibrate (i.e., XGB-anchored)
    --xgb-csv      B1 XGB per-seed predictions
    --mesh-csv     Mesh ensemble per-net predictions
    --design       e.g. intel22_tv80s_f3
    --out-spef     output path
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in-spef", type=Path, required=True)
    p.add_argument("--xgb-csv", type=Path, required=True)
    p.add_argument("--mesh-csv", type=Path, required=True)
    p.add_argument("--design", type=str, required=True)
    p.add_argument("--out-spef", type=Path, required=True)
    return p.parse_args()


def _normalize(name: str) -> str:
    n = name.strip()
    n = re.sub(r"^FE_[A-Z0-9]+_", "", n)
    return n


def load_per_net(csv: Path, design: str, label: str) -> dict[str, dict]:
    df = pd.read_csv(csv)
    sub = df[df["design_name"] == design]
    out: dict[str, dict] = {}
    for _, r in sub.iterrows():
        out[str(r["net_name"])] = {
            "gnd": float(r["pred_gnd_fF"]),
            "cpl": float(r["pred_cpl_fF"]),
        }
    print(f">>> {label}: {len(out)} predictions for {design}")
    return out


def first_pass_pinn_sums(spef: Path) -> dict[str, dict]:
    """Walk the (XGB-calibrated) SPEF and collect per-net sum_gnd / sum_cpl."""
    sums: dict[str, dict] = {}
    current = None
    in_cap = False
    with open(spef) as f:
        for line in f:
            s = line.strip()
            if s.startswith("*D_NET "):
                parts = s.split()
                current = parts[1]
                sums[current] = {"gnd": 0.0, "cpl": 0.0}
                in_cap = False
                continue
            if s == "*CAP":
                in_cap = True
                continue
            if s == "*RES" or s == "*END":
                in_cap = False
                if s == "*END":
                    current = None
                continue
            if not in_cap or current is None or not s:
                continue
            parts = s.split()
            # *CAP entries: id node val (gnd) OR id node aggressor:1 val (cpl)
            if len(parts) >= 3:
                try:
                    if ":" in parts[2]:
                        # coupling line
                        val = float(parts[3])
                        sums[current]["cpl"] += val
                    else:
                        val = float(parts[2])
                        sums[current]["gnd"] += val
                except (ValueError, IndexError):
                    continue
    return sums


def compute_scale_factors(
    pinn_sums: dict[str, dict],
    xgb_pred: dict[str, dict],
    mesh_pred: dict[str, dict],
) -> tuple[dict[str, float], dict[str, float], dict]:
    gnd_scale: dict[str, float] = {}
    cpl_scale: dict[str, float] = {}
    n_match_both = n_match_xgb_only = n_match_mesh_only = n_unmatched = 0

    for net, sums in pinn_sums.items():
        x = xgb_pred.get(net) or xgb_pred.get(_normalize(net))
        m = mesh_pred.get(net) or mesh_pred.get(_normalize(net))
        if x is None and m is None:
            n_unmatched += 1
            gnd_scale[net] = 1.0
            cpl_scale[net] = 1.0
            continue
        if x is None and m is not None:
            n_match_mesh_only += 1
            # Use Mesh total + ratio
            target_total = m["gnd"] + m["cpl"]
            target_gnd = m["gnd"]
            target_cpl = m["cpl"]
        elif x is not None and m is None:
            n_match_xgb_only += 1
            # Use XGB only — already calibrated; pass-through
            gnd_scale[net] = 1.0
            cpl_scale[net] = 1.0
            continue
        else:
            n_match_both += 1
            target_total = x["gnd"] + x["cpl"]
            mesh_total = m["gnd"] + m["cpl"]
            mesh_ratio_gnd = m["gnd"] / mesh_total if mesh_total > 1e-9 else 0.5
            target_gnd = target_total * mesh_ratio_gnd
            target_cpl = target_total * (1 - mesh_ratio_gnd)

        # Apply scales relative to current sums
        cur_gnd = sums["gnd"]
        cur_cpl = sums["cpl"]
        gnd_scale[net] = (target_gnd / cur_gnd) if cur_gnd > 1e-9 else 1.0
        cpl_scale[net] = (target_cpl / cur_cpl) if cur_cpl > 1e-9 else 1.0

    stats = {
        "match_both": n_match_both,
        "match_xgb_only": n_match_xgb_only,
        "match_mesh_only": n_match_mesh_only,
        "unmatched": n_unmatched,
        "total": len(pinn_sums),
    }
    return gnd_scale, cpl_scale, stats


def second_pass_rewrite(
    in_spef: Path, out_spef: Path,
    gnd_scale: dict[str, float], cpl_scale: dict[str, float],
):
    rewritten = 0
    current = None
    in_cap = False
    with open(in_spef) as fin, open(out_spef, "w") as fout:
        for line in fin:
            s = line.rstrip("\n")
            stripped = s.strip()
            if stripped.startswith("*D_NET "):
                parts = stripped.split()
                current = parts[1]
                # Recompute and write D_NET total (preserve XGB total for now)
                fout.write(line)
                in_cap = False
                continue
            if stripped == "*CAP":
                in_cap = True
                fout.write(line)
                continue
            if stripped == "*RES" or stripped == "*END":
                in_cap = False
                if stripped == "*END":
                    current = None
                fout.write(line)
                continue
            if not in_cap or current is None or not stripped:
                fout.write(line)
                continue
            parts = stripped.split()
            try:
                if ":" in parts[2]:
                    # coupling
                    val = float(parts[3])
                    new_val = val * cpl_scale.get(current, 1.0)
                    fout.write(f"{parts[0]} {parts[1]} {parts[2]} {new_val:.6f}\n")
                    rewritten += 1
                else:
                    val = float(parts[2])
                    new_val = val * gnd_scale.get(current, 1.0)
                    fout.write(f"{parts[0]} {parts[1]} {new_val:.6f}\n")
                    rewritten += 1
            except (ValueError, IndexError):
                fout.write(line)
    return rewritten


def main() -> int:
    args = parse_args()
    print(f">>> in:    {args.in_spef}")
    print(f">>> xgb:   {args.xgb_csv}")
    print(f">>> mesh:  {args.mesh_csv}")
    print(f">>> out:   {args.out_spef}")

    xgb = load_per_net(args.xgb_csv, args.design, "XGB")
    mesh = load_per_net(args.mesh_csv, args.design, "Mesh")

    print(">>> Pass 1: walking SPEF ...")
    sums = first_pass_pinn_sums(args.in_spef)
    print(f"    {len(sums)} D_NET blocks found")

    print(">>> Computing per-channel scale factors ...")
    g_scale, c_scale, stats = compute_scale_factors(sums, xgb, mesh)
    print(f"    matched both    : {stats['match_both']}")
    print(f"    XGB only        : {stats['match_xgb_only']}")
    print(f"    Mesh only       : {stats['match_mesh_only']}")
    print(f"    unmatched       : {stats['unmatched']}")
    g_arr = np.array([v for v in g_scale.values() if v != 1.0])
    c_arr = np.array([v for v in c_scale.values() if v != 1.0])
    if len(g_arr):
        print(f"    g_scale: median={np.median(g_arr):.4f}  p25={np.percentile(g_arr,25):.4f}  p75={np.percentile(g_arr,75):.4f}")
    if len(c_arr):
        print(f"    c_scale: median={np.median(c_arr):.4f}  p25={np.percentile(c_arr,25):.4f}  p75={np.percentile(c_arr,75):.4f}")

    print(">>> Pass 2: rewriting ...")
    n = second_pass_rewrite(args.in_spef, args.out_spef, g_scale, c_scale)
    print(f"    rewrote {n} *CAP entries")
    print(f"✅ {args.out_spef}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
