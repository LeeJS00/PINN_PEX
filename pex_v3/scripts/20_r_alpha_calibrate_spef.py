#!/usr/bin/env python3
"""
20_r_alpha_calibrate_spef.py — Resistance global α-scalar SPEF post-processor.

Scales every `*RES` line value in a PINN-PEX autonomous SPEF by a global
α scalar. Preserves all *D_NET, *CAP, *CONN, *END structure.

Origin (충돌 회피):
    Sister session reports `r_analytic_v3/reports/SPEF_RES_WRITE_CHANGE_REPORT_KO.md`
    documents an analytic v2 calibration α=1.4777 fit on 9 train designs
    against golden StarRC SPEF lumped R. PINNPEX core RCTopologyBuilder
    (`src/utils/spef_writer.py`) emits per-segment R using `layers.info`
    sheet R + via R 13.07Ω hardcoded; missing the global α factor.

This script applies that scalar as a SPEF post-process — same pattern
as `16_xgb_calibrate_spef.py` for capacitance — without touching the
sister session's `pex_pipeline/` files.

Quick measurement (tv80s seed0):
    α=1.0    →  R MAPE 28.37%  (PINN-PEX as-is)
    α=1.4777 →  R MAPE 11.78%  (report value, cross-codebase transfer)
    α=1.37   →  R MAPE  9.44%  (tv80s self-fit upper bound)

Pre-req: input SPEF written by PINNPEX `evaluator.py --spef_write`.
"""
from __future__ import annotations
import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="R global α-calibration on SPEF")
    p.add_argument("--in-spef", type=Path, required=True)
    p.add_argument("--out-spef", type=Path, required=True)
    p.add_argument("--alpha", type=float, required=True,
                   help="Global resistance scalar (e.g. 1.4777 from analytic v2 calibration).")
    return p.parse_args()


def rewrite(in_spef: Path, out_spef: Path, alpha: float) -> dict:
    """Walk SPEF, scale *RES line value (token 3) by alpha. Returns counters."""
    out_spef.parent.mkdir(parents=True, exist_ok=True)

    in_res = False
    n_res_scaled = 0
    n_res_skipped = 0
    n_lines = 0

    with open(in_spef) as fin, open(out_spef, "w") as fout:
        for line in fin:
            n_lines += 1
            stripped = line.strip()

            # Section markers
            if stripped.startswith("*RES"):
                in_res = True
                fout.write(line)
                continue
            if stripped.startswith("*END") or stripped.startswith("*D_NET") \
                    or stripped.startswith("*CONN") or stripped.startswith("*CAP"):
                in_res = False
                fout.write(line)
                continue

            if not in_res:
                fout.write(line)
                continue

            # Inside *RES block — comment-only or data line
            if not stripped or stripped.startswith("//") or stripped.startswith("*"):
                fout.write(line)
                continue

            # Standard SPEF *RES line: `<id> <node1> <node2> <r_val>` (4 tokens, 5th+ may be //)
            tokens = stripped.split()
            if len(tokens) < 4:
                fout.write(line)
                n_res_skipped += 1
                continue
            try:
                r_val = float(tokens[3])
            except ValueError:
                fout.write(line)
                n_res_skipped += 1
                continue

            new_r = r_val * alpha
            tokens[3] = f"{new_r:.6g}"
            # Preserve indentation + any trailing comment
            leading = line[: len(line) - len(line.lstrip())]
            # If there are extra tokens (e.g., comments), preserve them
            rest = " ".join(tokens[4:]) if len(tokens) > 4 else ""
            new_line = leading + " ".join(tokens[:4]) + ((" " + rest) if rest else "") + "\n"
            fout.write(new_line)
            n_res_scaled += 1

    return {
        "n_lines": n_lines,
        "n_res_scaled": n_res_scaled,
        "n_res_skipped": n_res_skipped,
        "alpha": alpha,
    }


def main() -> None:
    args = parse_args()
    print(f">>> in:    {args.in_spef}")
    print(f">>> out:   {args.out_spef}")
    print(f">>> alpha: {args.alpha}")
    res = rewrite(args.in_spef, args.out_spef, args.alpha)
    print(f">>> processed {res['n_lines']:,} lines")
    print(f">>> scaled  {res['n_res_scaled']:,} *RES entries (×{res['alpha']:.4f})")
    if res["n_res_skipped"] > 0:
        print(f">>> skipped {res['n_res_skipped']} unparseable *RES lines")
    print(f"✅ {args.out_spef}")


if __name__ == "__main__":
    main()
