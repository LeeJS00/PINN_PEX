"""04_compare_golden.py — STAGE 3 of TreePEX tool: pred SPEF vs StarRC golden.

Parses both SPEF files, aligns by net name, computes per-design accuracy
(tot/gnd/cpl MAPE_med, R²) and per-cap-decile breakdown.

Output: outputs/reports/<design>_report.json
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PREDS = ROOT / "outputs" / "predictions"
REPORTS = ROOT / "outputs" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

GOLDEN_DIR = Path("/home/jslee/projects/PINNPEX/golden_data/spef_data/intel22")
EPS_FF = 1e-3


def parse_spef_per_net_total(path: Path) -> dict:
    """Parse a SPEF file and return {net_name: total_cap_fF} dict.

    Reads *D_NET <name> <total_cap> lines. Handles unit conversion if
    *C_UNIT is in PF (1.0 PF → 1000 fF; 1.0 FF → 1.0).
    Also resolves *NAME_MAP (e.g., *123 = some_net) so numeric net IDs
    in *D_NET are mapped to their actual names.
    """
    text = path.read_text(errors="replace")
    # Detect unit
    m_unit = re.search(r"\*C_UNIT\s+([\d.]+)\s+(PF|FF)", text, re.IGNORECASE)
    unit_mult = 1.0
    if m_unit:
        val = float(m_unit.group(1)); unit = m_unit.group(2).upper()
        unit_mult = (val * 1000.0) if unit == "PF" else val
    else:
        unit_mult = 1.0

    # Build NAME_MAP: pattern like "*123 some_net"
    name_map = {}
    in_namemap = False
    for ln in text.split("\n"):
        s = ln.strip()
        if s.startswith("*NAME_MAP"):
            in_namemap = True; continue
        if in_namemap:
            if not s or s.startswith("*") and not s[1:].split()[0].lstrip("-").isdigit():
                in_namemap = False
                continue
            parts = s.split()
            if len(parts) >= 2 and parts[0].startswith("*"):
                name_map[parts[0]] = parts[1]

    # Parse *D_NET
    out = {}
    for m in re.finditer(r"^\*D_NET\s+(\S+)\s+([\d.eE+-]+)", text, re.MULTILINE):
        net_id, c_str = m.group(1), m.group(2)
        net = name_map.get(net_id, net_id)
        try:
            out[net] = float(c_str) * unit_mult
        except ValueError:
            continue
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--design", type=str, required=True)
    p.add_argument("--golden", type=Path, default=None,
                   help="Override golden SPEF path (default: GOLDEN_DIR/<design>_starrc.spef)")
    return p.parse_args()


def r2(p, g):
    g = np.asarray(g); p = np.asarray(p)
    ss_res = float(((g - p) ** 2).sum())
    ss_tot = float(((g - g.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def mape_med(p, g):
    g = np.asarray(g); p = np.asarray(p)
    return float(np.median(np.abs(p - g) / np.maximum(np.abs(g), EPS_FF) * 100))


def mape_mean(p, g):
    g = np.asarray(g); p = np.asarray(p)
    return float(np.mean(np.abs(p - g) / np.maximum(np.abs(g), EPS_FF) * 100))


def main():
    args = parse_args()
    pred_csv = PREDS / f"{args.design}_pred.csv"
    pred_spef = ROOT / "outputs" / "spef" / f"{args.design}_pred.spef"
    golden_spef = Path(args.golden) if args.golden else GOLDEN_DIR / f"{args.design}_starrc.spef"
    if not pred_csv.exists():
        print(f"[error] missing {pred_csv}; run STAGE 1"); return 1
    if not golden_spef.exists():
        print(f"[error] missing golden {golden_spef}"); return 1

    print(f">>> TreePEX STAGE 3 [compare-golden] design={args.design}")
    t0 = time.time()
    df_pred = pd.read_csv(pred_csv)
    print(f"  pred CSV: {len(df_pred):,} nets")

    pred_caps = dict(zip(df_pred["net_name"].astype(str), df_pred["pred_total"]))
    pred_gnds = dict(zip(df_pred["net_name"].astype(str), df_pred["pred_gnd"]))
    pred_cpls = dict(zip(df_pred["net_name"].astype(str), df_pred["pred_cpl"]))
    gold_caps_g = dict(zip(df_pred["net_name"].astype(str), df_pred["c_gnd_fF"]))
    gold_caps_c = dict(zip(df_pred["net_name"].astype(str), df_pred["c_cpl_total_fF"]))

    # Parse golden SPEF for D_NET total cap (cross-check)
    print(f"  parsing golden SPEF: {golden_spef.name}")
    golden_dnet = parse_spef_per_net_total(golden_spef)
    print(f"  golden parsed: {len(golden_dnet):,} *D_NET entries")

    # Parse our predicted SPEF for cross-validation
    print(f"  parsing predicted SPEF: {pred_spef.name}")
    pred_dnet = parse_spef_per_net_total(pred_spef)
    print(f"  pred parsed: {len(pred_dnet):,} *D_NET entries")

    # Align nets present in both (and in pred_csv)
    common = set(pred_dnet.keys()) & set(golden_dnet.keys()) & set(df_pred["net_name"].astype(str))
    print(f"  common nets (pred SPEF ∩ golden SPEF ∩ pred CSV): {len(common):,}")

    # Per-net comparison
    rows = []
    for net in common:
        rows.append({
            "net_name": net,
            "pred_total_csv":  pred_caps.get(net, float("nan")),
            "pred_total_spef": pred_dnet[net],
            "gold_total_spef": golden_dnet[net],
            "pred_gnd": pred_gnds.get(net, float("nan")),
            "pred_cpl": pred_cpls.get(net, float("nan")),
            "gold_gnd": gold_caps_g.get(net, float("nan")),
            "gold_cpl": gold_caps_c.get(net, float("nan")),
        })
    cmp = pd.DataFrame(rows).dropna()
    print(f"  per-net rows for analysis: {len(cmp):,}")

    # Per-design metrics
    metrics = {
        "design": args.design,
        "n_nets_pred_csv": int(len(df_pred)),
        "n_nets_pred_spef": int(len(pred_dnet)),
        "n_nets_golden_spef": int(len(golden_dnet)),
        "n_nets_compared": int(len(cmp)),
        "MAPE_tot_med": mape_med(cmp["pred_total_csv"], cmp["gold_total_spef"]),
        "MAPE_tot_mean": mape_mean(cmp["pred_total_csv"], cmp["gold_total_spef"]),
        "MAPE_gnd_med": mape_med(cmp["pred_gnd"], cmp["gold_gnd"]),
        "MAPE_cpl_med": mape_med(cmp["pred_cpl"], cmp["gold_cpl"]),
        "R2_tot": r2(cmp["pred_total_csv"], cmp["gold_total_spef"]),
        "R2_gnd": r2(cmp["pred_gnd"], cmp["gold_gnd"]),
        "R2_cpl": r2(cmp["pred_cpl"], cmp["gold_cpl"]),
        "compare_wall_s": time.time() - t0,
        # Cross-check: SPEF round-trip preserved cap values?
        "spef_roundtrip_max_abs_err_fF": float((cmp["pred_total_csv"] - cmp["pred_total_spef"]).abs().max()),
    }
    # Per-cap-decile
    cmp["cap_decile"] = pd.qcut(cmp["gold_total_spef"], 10,
                                  labels=[f"C{i+1}" for i in range(10)], duplicates="drop")
    decile_rows = []
    for cd, sub in cmp.groupby("cap_decile", observed=True):
        decile_rows.append({
            "cap_decile": str(cd), "n": int(len(sub)),
            "cap_mean_fF": float(sub["gold_total_spef"].mean()),
            "MAPE_tot_med": mape_med(sub["pred_total_csv"], sub["gold_total_spef"]),
            "R2_tot": r2(sub["pred_total_csv"], sub["gold_total_spef"]),
            "R2_cpl": r2(sub["pred_cpl"], sub["gold_cpl"]),
        })
    metrics["per_cap_decile"] = decile_rows

    out_json = REPORTS / f"{args.design}_report.json"
    out_json.write_text(json.dumps(metrics, indent=2))
    cmp.to_csv(REPORTS / f"{args.design}_per_net_compare.csv", index=False)

    print(f"\n>>> {args.design} | n_compared={len(cmp):,}")
    print(f"  MAPE_tot_med = {metrics['MAPE_tot_med']:.3f}%  R²_tot = {metrics['R2_tot']:.4f}")
    print(f"  MAPE_gnd_med = {metrics['MAPE_gnd_med']:.3f}%  MAPE_cpl_med = {metrics['MAPE_cpl_med']:.3f}%")
    print(f"  SPEF round-trip max abs err = {metrics['spef_roundtrip_max_abs_err_fF']:.6f} fF (lossless if ~0)")
    print(f"  wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
