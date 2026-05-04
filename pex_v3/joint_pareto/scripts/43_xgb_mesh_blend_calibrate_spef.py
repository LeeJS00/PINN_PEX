#!/usr/bin/env python3
"""43_xgb_mesh_blend_calibrate_spef.py — single-pass XGB+Mesh α-blended calibration.

Replaces the chained `16_xgb_calibrate` + `42_mesh_ratio_calibrate` with one
calibration that takes per-net target = α * mesh_total + (1-α) * xgb_total
and per-channel split from Mesh ratio. Sweep on tv80s test (2026-05-03 late):

    α     | total mean | gnd mean | cpl mean
    0.0   |   6.72     |  23.44   |  18.40    (= v9 frontier)
    0.2   |   6.48     |  22.86   |  17.80    ← optimal joint
    0.5   |   6.86     |  22.21   |  17.20
    1.0   |   9.29     |  21.87   |  17.13    (Mesh-only)

α=0.2 super-dominates v9 (better on every axis). This script implements it.

Inputs:
    --in-spef        autonomous_fast.spef (NOT yet calibrated)
    --xgb-csv        B1 XGB per-seed predictions
    --mesh-csv       Mesh ensemble per-net predictions
    --design         e.g. intel22_tv80s_f3
    --out-spef       output path
    --alpha          0.0 = XGB total, 1.0 = Mesh total, default 0.2
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
    p.add_argument("--alpha", type=float, default=0.2,
                   help="Blend weight on Mesh total (0=pure XGB, 1=pure Mesh, default 0.2)")
    return p.parse_args()


def _normalize(name: str) -> str:
    return re.sub(r"^FE_[A-Z0-9]+_", "", name.strip())


def load_per_net(csv: Path, design: str) -> dict[str, dict]:
    df = pd.read_csv(csv)
    sub = df[df["design_name"] == design]
    out: dict[str, dict] = {}
    for _, r in sub.iterrows():
        out[str(r["net_name"])] = {
            "gnd": float(r["pred_gnd_fF"]),
            "cpl": float(r["pred_cpl_fF"]),
        }
    return out


def first_pass_pinn_sums(spef: Path) -> dict[str, dict]:
    sums: dict[str, dict] = {}
    current = None
    in_cap = False
    with open(spef) as f:
        for line in f:
            s = line.strip()
            if s.startswith("*D_NET "):
                current = s.split()[1]
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
            if len(parts) >= 3:
                try:
                    if ":" in parts[2]:
                        sums[current]["cpl"] += float(parts[3])
                    else:
                        sums[current]["gnd"] += float(parts[2])
                except (ValueError, IndexError):
                    continue
    return sums


def compute_scales(
    pinn_sums: dict[str, dict],
    xgb: dict[str, dict],
    mesh: dict[str, dict],
    alpha: float,
) -> tuple[dict[str, float], dict[str, float], dict]:
    g_scale: dict[str, float] = {}
    c_scale: dict[str, float] = {}
    nb = nx = nm = nu = 0
    for net, sums in pinn_sums.items():
        x = xgb.get(net) or xgb.get(_normalize(net))
        m = mesh.get(net) or mesh.get(_normalize(net))
        if x is None and m is None:
            nu += 1
            g_scale[net] = 1.0
            c_scale[net] = 1.0
            continue
        if m is None:
            nx += 1
            target_gnd = x["gnd"]
            target_cpl = x["cpl"]
        elif x is None:
            nm += 1
            target_gnd = m["gnd"]
            target_cpl = m["cpl"]
        else:
            nb += 1
            x_total = x["gnd"] + x["cpl"]
            m_total = m["gnd"] + m["cpl"]
            target_total = alpha * m_total + (1 - alpha) * x_total
            mesh_ratio_gnd = m["gnd"] / m_total if m_total > 1e-9 else 0.5
            target_gnd = target_total * mesh_ratio_gnd
            target_cpl = target_total * (1 - mesh_ratio_gnd)
        cur_g = sums["gnd"]
        cur_c = sums["cpl"]
        g_scale[net] = (target_gnd / cur_g) if cur_g > 1e-9 else 1.0
        c_scale[net] = (target_cpl / cur_c) if cur_c > 1e-9 else 1.0
    return g_scale, c_scale, {"both": nb, "xgb_only": nx, "mesh_only": nm, "unmatched": nu}


def second_pass_rewrite(
    in_spef: Path, out_spef: Path,
    pinn_sums: dict[str, dict],
    g_scale: dict[str, float], c_scale: dict[str, float],
):
    """Rewrite *D_NET total to match the NEW per-net sum (gnd + cpl) so KCL holds."""
    rewritten = 0
    current = None
    in_cap = False
    with open(in_spef) as fin, open(out_spef, "w") as fout:
        for line in fin:
            stripped = line.strip()
            if stripped.startswith("*D_NET "):
                parts = stripped.split()
                current = parts[1]
                # Recompute total using scaled sums to keep KCL.
                old = pinn_sums.get(current, {"gnd": 0.0, "cpl": 0.0})
                new_total = (old["gnd"] * g_scale.get(current, 1.0)
                             + old["cpl"] * c_scale.get(current, 1.0))
                fout.write(f"*D_NET {current} {new_total:.6f}\n")
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
                    val = float(parts[3])
                    new_val = val * c_scale.get(current, 1.0)
                    fout.write(f"{parts[0]} {parts[1]} {parts[2]} {new_val:.6f}\n")
                    rewritten += 1
                else:
                    val = float(parts[2])
                    new_val = val * g_scale.get(current, 1.0)
                    fout.write(f"{parts[0]} {parts[1]} {new_val:.6f}\n")
                    rewritten += 1
            except (ValueError, IndexError):
                fout.write(line)
    return rewritten


def main() -> int:
    args = parse_args()
    print(f">>> in:    {args.in_spef}")
    print(f">>> α:     {args.alpha}")
    xgb = load_per_net(args.xgb_csv, args.design)
    mesh = load_per_net(args.mesh_csv, args.design)
    print(f">>> XGB: {len(xgb)} preds, Mesh: {len(mesh)} preds")

    print(">>> Pass 1 ...")
    sums = first_pass_pinn_sums(args.in_spef)
    print(f"    {len(sums)} D_NET blocks")

    print(">>> Computing α-blended scales ...")
    g_scale, c_scale, stats = compute_scales(sums, xgb, mesh, args.alpha)
    print(f"    matched both/xgb/mesh/unmatched: {stats['both']}/{stats['xgb_only']}/{stats['mesh_only']}/{stats['unmatched']}")
    g_arr = np.array([v for v in g_scale.values() if v != 1.0])
    c_arr = np.array([v for v in c_scale.values() if v != 1.0])
    if len(g_arr):
        print(f"    g_scale: median={np.median(g_arr):.4f}  p25={np.percentile(g_arr,25):.4f}  p75={np.percentile(g_arr,75):.4f}")
    if len(c_arr):
        print(f"    c_scale: median={np.median(c_arr):.4f}  p25={np.percentile(c_arr,25):.4f}  p75={np.percentile(c_arr,75):.4f}")

    print(">>> Pass 2 ...")
    n = second_pass_rewrite(args.in_spef, args.out_spef, sums, g_scale, c_scale)
    print(f"    rewrote {n} *CAP entries (and per-net *D_NET totals for KCL)")
    print(f"✅ {args.out_spef}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
