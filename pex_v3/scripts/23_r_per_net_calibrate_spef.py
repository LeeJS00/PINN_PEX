#!/usr/bin/env python3
"""
23_r_per_net_calibrate_spef.py — Per-net R rescaling on PINNPEX SPEF.

Generalization of `20_r_alpha_calibrate_spef.py`: instead of a single
global α scalar, applies a per-net scaling factor derived from an
external R prediction CSV/parquet (e.g., sister session's v3 hybrid
2.21% MAPE results).

Two-pass:
  Pass 1: read predicted SPEF, compute per-net Σ(*RES values).
  Pass 2: rewrite SPEF, scale every *RES line in net N by
          alpha_N = sister_R_pred[N] / our_total_R[N].

Conflict-free (no shared file edits): we only consume sister's parquet
output, no changes to `pex_pipeline/` files.

Sister source: `experiments/.../r_analytic_v3/outputs/test_predictions_v6_s3.parquet`
columns: net_name, R_pred_v6_s3 (per-net predicted R in Ω, lumped sum).
"""
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Per-net R rescaling on SPEF")
    p.add_argument("--in-spef", type=Path, required=True)
    p.add_argument("--out-spef", type=Path, required=True)
    p.add_argument("--r-pred-parquet", type=Path, required=True,
                   help="Parquet with columns net_name + r_pred col.")
    p.add_argument("--r-pred-col", type=str, default="R_pred_v6_s3",
                   help="Column name with predicted per-net total R (Ω).")
    p.add_argument("--net-col", type=str, default="net_name")
    return p.parse_args()


def first_pass_per_net_r_sum(in_spef: Path) -> dict[str, float]:
    """Per-net Σ R from *RES section."""
    sums: dict[str, float] = {}
    current_net: str | None = None
    in_res = False
    with open(in_spef) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue
            if stripped.startswith("*D_NET"):
                tokens = stripped.split()
                current_net = tokens[1]
                sums[current_net] = 0.0
                in_res = False
                continue
            if not current_net:
                continue
            if stripped.startswith("*RES"):
                in_res = True
                continue
            if stripped.startswith("*END") or stripped.startswith("*CONN") \
                    or stripped.startswith("*CAP") or stripped.startswith("*D_NET"):
                in_res = False
                if stripped.startswith("*END"):
                    current_net = None
                continue
            if in_res and not stripped.startswith("*"):
                tokens = stripped.split()
                if len(tokens) < 4:
                    continue
                try:
                    r_val = float(tokens[3])
                except ValueError:
                    continue
                sums[current_net] += r_val
    return sums


def second_pass_rewrite(
    in_spef: Path,
    out_spef: Path,
    alpha_per_net: dict[str, float],
) -> dict:
    out_spef.parent.mkdir(parents=True, exist_ok=True)
    n_res_scaled = 0
    n_unmatched_nets = 0
    current_net: str | None = None
    in_res = False
    with open(in_spef) as fin, open(out_spef, "w") as fout:
        for line in fin:
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                fout.write(line)
                continue
            if stripped.startswith("*D_NET"):
                tokens = stripped.split()
                current_net = tokens[1]
                in_res = False
                fout.write(line)
                continue
            if not current_net:
                fout.write(line)
                continue
            if stripped.startswith("*RES"):
                in_res = True
                fout.write(line)
                continue
            if stripped.startswith("*END") or stripped.startswith("*CONN") \
                    or stripped.startswith("*CAP"):
                in_res = False
                fout.write(line)
                if stripped.startswith("*END"):
                    current_net = None
                continue
            if in_res and not stripped.startswith("*"):
                tokens = stripped.split()
                if len(tokens) < 4:
                    fout.write(line)
                    continue
                try:
                    r_val = float(tokens[3])
                except ValueError:
                    fout.write(line)
                    continue
                alpha = alpha_per_net.get(current_net, 1.0)
                if alpha == 1.0 and current_net not in alpha_per_net:
                    n_unmatched_nets += 1
                new_r = r_val * alpha
                tokens[3] = f"{new_r:.6g}"
                leading = line[: len(line) - len(line.lstrip())]
                rest = " ".join(tokens[4:]) if len(tokens) > 4 else ""
                new_line = leading + " ".join(tokens[:4]) + ((" " + rest) if rest else "") + "\n"
                fout.write(new_line)
                n_res_scaled += 1
            else:
                fout.write(line)
    return {"n_res_scaled": n_res_scaled, "n_unmatched_nets": n_unmatched_nets}


def main() -> None:
    args = parse_args()
    print(f">>> in:    {args.in_spef}")
    print(f">>> out:   {args.out_spef}")
    print(f">>> r_pred parquet: {args.r_pred_parquet}")
    print(f">>> col:   {args.r_pred_col}")

    print()
    print(">>> Loading sister R predictions ...")
    pred_df = pd.read_parquet(args.r_pred_parquet)
    if args.net_col not in pred_df.columns:
        raise SystemExit(f"missing {args.net_col!r} column")
    if args.r_pred_col not in pred_df.columns:
        raise SystemExit(f"missing {args.r_pred_col!r} column")
    pred_map = dict(zip(pred_df[args.net_col].astype(str), pred_df[args.r_pred_col].astype(float)))
    print(f">>> {len(pred_map):,} per-net predictions")

    print()
    print(">>> Pass 1: computing per-net Σ R from input SPEF ...")
    our_sums = first_pass_per_net_r_sum(args.in_spef)
    print(f">>> {len(our_sums):,} nets in SPEF")

    print()
    print(">>> Computing per-net α ...")
    alpha: dict[str, float] = {}
    n_match = 0
    EPS = 1e-9
    MIN_PRED = 1e-3
    for net, our_r in our_sums.items():
        sister_r = pred_map.get(net)
        if sister_r is None:
            alpha[net] = 1.0  # pass-through
            continue
        if our_r > EPS and sister_r > MIN_PRED:
            alpha[net] = sister_r / our_r
            n_match += 1
        else:
            alpha[net] = 1.0
    import numpy as np
    arr = np.array([a for a in alpha.values() if a != 1.0])
    print(f">>> matched + scaled: {n_match:,} / {len(our_sums):,}")
    if len(arr) > 0:
        print(f">>> α distribution: median={np.median(arr):.4f}  "
              f"p25={np.percentile(arr, 25):.4f}  p75={np.percentile(arr, 75):.4f}  "
              f"min={np.min(arr):.4f}  max={np.max(arr):.4f}")

    print()
    print(">>> Pass 2: rewriting SPEF ...")
    res = second_pass_rewrite(args.in_spef, args.out_spef, alpha)
    print(f">>> scaled {res['n_res_scaled']:,} *RES entries")
    if res["n_unmatched_nets"] > 0:
        print(f">>> {res['n_unmatched_nets']} unmatched-net *RES entries (pass-through)")
    print(f"✅ {args.out_spef}")


if __name__ == "__main__":
    main()
