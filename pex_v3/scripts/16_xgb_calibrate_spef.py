#!/usr/bin/env python3
"""
16_xgb_calibrate_spef.py — XGBoost-anchored calibration of PINN autonomous SPEF.

Hypothesis (2026-05-03):
    PINN model predicts per-cuboid charge that, after tile→net aggregation
    in the SPEF assembly path, drifts to 47.69% per-net MAPE on tv80s
    (vs 31.08% direct per-net). XGBoost (B1) hits 4.66% per-net (per-net
    direct, no SPEF assembly). Use XGBoost per-net total as a calibration
    anchor, rescaling each PINN-predicted *CAP block so that:
        sum(gnd_caps)         := xgb_pred_gnd
        sum(cpl_caps)         := xgb_pred_cpl_total
    while preserving:
        - relative ratios within ground (per-node distribution)
        - relative ratios within coupling (per-aggressor + per-node)
        - resistance network (R block untouched)
        - structure (D_NET, CONN, RES, END all unmodified)

Output: a new SPEF (`*_xgb_calibrated.spef`) ready for compare_spef.py.

Inputs:
    --in-spef    /path/to/intel22_<design>_autonomous.spef
    --xgb-csv    /path/to/B1_xgboost_real/seed{N}/eval_predictions_test.csv
    --design     intel22_tv80s_f3 (filter applied to xgb-csv)
    --out-spef   /path/to/intel22_<design>_xgb_calibrated.spef
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "pex_v3"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="XGBoost SPEF calibration")
    p.add_argument("--in-spef", type=Path, required=True)
    p.add_argument("--xgb-csv", type=Path, required=True)
    p.add_argument("--design", type=str, required=True)
    p.add_argument("--out-spef", type=Path, required=True)
    p.add_argument(
        "--gnd-only", action="store_true",
        help="If set, calibrate only ground caps (leave coupling untouched). Diagnostic.",
    )
    p.add_argument(
        "--cpl-only", action="store_true",
        help="If set, calibrate only coupling caps (leave ground untouched). Diagnostic.",
    )
    return p.parse_args()


def _normalize_net_name(name: str) -> str:
    """Strip SPEF escape prefix (FE_*) and StarRC fanout-extracted-net suffixes
    so XGBoost's golden net name (driver-side) matches SPEF's escaped name.

    The PINN SPEF writer preserves original DEF net names; XGBoost was trained
    against c_gnd_fF / c_cpl_total_fF aggregated by golden SPEF net name. Both
    should match exactly except for one wrinkle: SPEF writers sometimes prefix
    `FE_OFN25_` or similar scaffolding to escaped net names. Strip those.
    """
    n = name.strip()
    # Remove leading FE_*_ scaffolding (StarRC fanout-extracted)
    n = re.sub(r"^FE_[A-Z0-9]+_", "", n)
    return n


def load_xgb_predictions(xgb_csv: Path, design: str) -> dict[str, dict]:
    """Load XGBoost per-net predictions for a single design.

    Returns: dict net_name -> {pred_gnd, pred_cpl, golden_gnd, golden_cpl}
    """
    df = pd.read_csv(xgb_csv)
    sub = df[df["design_name"] == design].reset_index(drop=True)
    if len(sub) == 0:
        raise SystemExit(f"No rows for design={design} in {xgb_csv}")

    out = {}
    for _, r in sub.iterrows():
        out[str(r["net_name"])] = {
            "pred_gnd": float(r["pred_gnd_fF"]),
            "pred_cpl": float(r["pred_cpl_fF"]),
            "pred_total": float(r["pred_total_fF"]),
            "golden_gnd": float(r["golden_gnd_fF"]),
            "golden_cpl": float(r["golden_cpl_fF"]),
            "golden_total": float(r["golden_total_fF"]),
        }
    return out


def first_pass_compute_pinn_sums(in_spef: Path) -> dict[str, dict]:
    """Walk the SPEF once to collect per-net (sum_gnd, sum_cpl) of PINN outputs.

    Returns: dict net_name -> {pinn_gnd, pinn_cpl, pinn_total}
    """
    sums: dict[str, dict] = {}
    current_net: str | None = None
    in_cap = False

    with open(in_spef) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue

            if stripped.startswith("*D_NET"):
                tokens = stripped.split()
                current_net = tokens[1]
                sums[current_net] = {
                    "pinn_gnd": 0.0,
                    "pinn_cpl": 0.0,
                    "pinn_total": 0.0,
                }
                in_cap = False
                continue
            if not current_net:
                continue

            if stripped.startswith("*CONN"):
                in_cap = False
                continue
            if stripped.startswith("*CAP"):
                in_cap = True
                continue
            if stripped.startswith("*RES") or stripped.startswith("*END"):
                in_cap = False
                continue

            if in_cap and not stripped.startswith("*"):
                tokens = stripped.split()
                if len(tokens) == 3:  # ground cap
                    try:
                        cap_val = float(tokens[2])
                    except ValueError:
                        continue
                    sums[current_net]["pinn_gnd"] += cap_val
                    sums[current_net]["pinn_total"] += cap_val
                elif len(tokens) == 4:  # coupling cap
                    try:
                        cap_val = float(tokens[3])
                    except ValueError:
                        continue
                    sums[current_net]["pinn_cpl"] += cap_val
                    sums[current_net]["pinn_total"] += cap_val
    return sums


def build_scale_factors(
    pinn_sums: dict, xgb_preds: dict, gnd_only: bool, cpl_only: bool
) -> tuple[dict, dict]:
    """Compute per-net scale factors. Missing in XGBoost → no rescale (1.0)."""
    gnd_scale: dict[str, float] = {}
    cpl_scale: dict[str, float] = {}
    matched = 0
    unmatched = []

    EPS = 1e-12
    MIN_PRED = 1e-4  # below this, treat as no-signal and skip rescale (avoid divide-by-zero)

    for net_name, pinn_vals in pinn_sums.items():
        # Try direct match, then normalized match
        xgb = xgb_preds.get(net_name)
        if xgb is None:
            xgb = xgb_preds.get(_normalize_net_name(net_name))
        if xgb is None:
            unmatched.append(net_name)
            gnd_scale[net_name] = 1.0
            cpl_scale[net_name] = 1.0
            continue

        matched += 1
        # Ground rescale
        if cpl_only:
            gnd_scale[net_name] = 1.0
        elif pinn_vals["pinn_gnd"] > EPS and xgb["pred_gnd"] > MIN_PRED:
            gnd_scale[net_name] = xgb["pred_gnd"] / pinn_vals["pinn_gnd"]
        else:
            gnd_scale[net_name] = 1.0

        # Coupling rescale
        if gnd_only:
            cpl_scale[net_name] = 1.0
        elif pinn_vals["pinn_cpl"] > EPS and xgb["pred_cpl"] > MIN_PRED:
            cpl_scale[net_name] = xgb["pred_cpl"] / pinn_vals["pinn_cpl"]
        else:
            cpl_scale[net_name] = 1.0

    print(f">>> XGBoost match: {matched} / {len(pinn_sums)} nets")
    if unmatched:
        print(f"    {len(unmatched)} unmatched (pass-through, no rescale). "
              f"Sample: {unmatched[:5]}")
    print(f">>> Scale factor distribution (gnd):")
    g_arr = np.array([s for s in gnd_scale.values() if s != 1.0])
    if len(g_arr) > 0:
        print(f"    n={len(g_arr)}  median={np.median(g_arr):.3f}  "
              f"p25={np.percentile(g_arr, 25):.3f}  "
              f"p75={np.percentile(g_arr, 75):.3f}  "
              f"min={np.min(g_arr):.3f}  max={np.max(g_arr):.3f}")
    print(f">>> Scale factor distribution (cpl):")
    c_arr = np.array([s for s in cpl_scale.values() if s != 1.0])
    if len(c_arr) > 0:
        print(f"    n={len(c_arr)}  median={np.median(c_arr):.3f}  "
              f"p25={np.percentile(c_arr, 25):.3f}  "
              f"p75={np.percentile(c_arr, 75):.3f}  "
              f"min={np.min(c_arr):.3f}  max={np.max(c_arr):.3f}")
    return gnd_scale, cpl_scale


def second_pass_rewrite(
    in_spef: Path,
    out_spef: Path,
    gnd_scale: dict[str, float],
    cpl_scale: dict[str, float],
) -> None:
    """Rewrite SPEF with per-net cap rescaling. Updates *D_NET total too."""
    out_spef.parent.mkdir(parents=True, exist_ok=True)
    current_net: str | None = None
    in_cap = False
    pending_dnet_lines: list[str] = []  # buffer current net so we can patch *D_NET total
    n_caps_written = 0

    # Compute new totals per net (so *D_NET total reflects calibration)
    pinn_sums = first_pass_compute_pinn_sums(in_spef)
    new_totals: dict[str, float] = {}
    for net, vals in pinn_sums.items():
        new_totals[net] = (
            vals["pinn_gnd"] * gnd_scale.get(net, 1.0)
            + vals["pinn_cpl"] * cpl_scale.get(net, 1.0)
        )

    with open(in_spef) as fin, open(out_spef, "w") as fout:
        for line in fin:
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                fout.write(line)
                continue

            if stripped.startswith("*D_NET"):
                tokens = stripped.split()
                current_net = tokens[1]
                # Replace total cap (token 2) with new calibrated total
                new_total = new_totals.get(current_net, float(tokens[2]) if tokens[2].replace(".", "", 1).isdigit() else 0.0)
                tokens[2] = f"{new_total:.6f}"
                fout.write(" ".join(tokens) + "\n")
                in_cap = False
                continue
            if not current_net:
                fout.write(line)
                continue

            if stripped.startswith("*CONN"):
                in_cap = False
                fout.write(line)
                continue
            if stripped.startswith("*CAP"):
                in_cap = True
                fout.write(line)
                continue
            if stripped.startswith("*RES") or stripped.startswith("*END"):
                in_cap = False
                fout.write(line)
                if stripped.startswith("*END"):
                    current_net = None
                continue

            if in_cap and not stripped.startswith("*"):
                tokens = stripped.split()
                if len(tokens) == 3:  # ground
                    try:
                        cap_val = float(tokens[2])
                    except ValueError:
                        fout.write(line)
                        continue
                    new_val = cap_val * gnd_scale.get(current_net, 1.0)
                    tokens[2] = f"{new_val:.6e}"
                    # Preserve original line indentation
                    leading = line[: len(line) - len(line.lstrip())]
                    fout.write(leading + " ".join(tokens) + "\n")
                    n_caps_written += 1
                elif len(tokens) == 4:  # coupling
                    try:
                        cap_val = float(tokens[3])
                    except ValueError:
                        fout.write(line)
                        continue
                    new_val = cap_val * cpl_scale.get(current_net, 1.0)
                    tokens[3] = f"{new_val:.6e}"
                    leading = line[: len(line) - len(line.lstrip())]
                    fout.write(leading + " ".join(tokens) + "\n")
                    n_caps_written += 1
                else:
                    fout.write(line)
            else:
                fout.write(line)

    print(f">>> Rewrote {n_caps_written} *CAP entries (gnd + cpl)")
    print(f">>> Written: {out_spef}")


def main() -> None:
    args = parse_args()
    print(f">>> in:    {args.in_spef}")
    print(f">>> xgb:   {args.xgb_csv}")
    print(f">>> design:{args.design}")
    print(f">>> out:   {args.out_spef}")

    if args.gnd_only and args.cpl_only:
        raise SystemExit("--gnd-only and --cpl-only are mutually exclusive")

    print()
    print(">>> Pass 1: computing PINN per-net sums from SPEF ...")
    pinn_sums = first_pass_compute_pinn_sums(args.in_spef)
    print(f"    {len(pinn_sums)} nets in SPEF")

    print()
    print(">>> Loading XGBoost predictions ...")
    xgb_preds = load_xgb_predictions(args.xgb_csv, args.design)
    print(f"    {len(xgb_preds)} XGBoost predictions for {args.design}")

    print()
    print(">>> Computing scale factors ...")
    gnd_scale, cpl_scale = build_scale_factors(
        pinn_sums, xgb_preds, args.gnd_only, args.cpl_only
    )

    print()
    print(">>> Pass 2: rewriting SPEF with calibration ...")
    second_pass_rewrite(args.in_spef, args.out_spef, gnd_scale, cpl_scale)
    print()
    print("✅ Calibration complete")


if __name__ == "__main__":
    main()
