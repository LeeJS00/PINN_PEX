"""Produce per-feature runtime + MAE + R² report from two
`dump_features.py` outputs (baseline vs patched).

Output: a markdown table on stdout (+ optional --out file) covering:
  - per-stage / per-block runtime totals (V3 sum, V4 sum)
  - per-feature value drift across nets common to both dumps
    (count, baseline_mean, patched_mean, abs_max, MAE, MAE_pct, R²)

Drift gates (Round 1 acceptance, FEATURE_SPEEDUP_PLAN.md §6):
  - All non-stochastic features should have MAE_pct = 0 (no algorithmic change).
  - Stochastic features (top-K *_score, *_overlap_um2 etc affected by V3-A
    target sub-sampling) should have R² ≥ 0.99 and MAE_pct ≤ 3 %.

Usage:
    python3 TreePEX/scripts/compare_features.py \\
        --baseline TreePEX/outputs/cold_reports/feature_dumps/intel22_nova_f3__baseline.json \\
        --patched  TreePEX/outputs/cold_reports/feature_dumps/intel22_nova_f3__patched.json \\
        --out TreePEX/outputs/cold_reports/diff_intel22_nova_f3.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def load_dump(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def rows_to_df(rows: List[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("net_name")


def feature_drift(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    """Per-column MAE / R² over common-index rows."""
    if df_a.empty or df_b.empty:
        return pd.DataFrame()
    common = df_a.index.intersection(df_b.index)
    a = df_a.loc[common]
    b = df_b.loc[common]
    cols = sorted(set(a.columns) & set(b.columns) - {"_runtime_s",
                                                     "_n_cuboids_raw",
                                                     "_n_tiles"})
    rows = []
    for c in cols:
        try:
            va = pd.to_numeric(a[c], errors="coerce").to_numpy()
            vb = pd.to_numeric(b[c], errors="coerce").to_numpy()
        except Exception:
            continue
        mask = np.isfinite(va) & np.isfinite(vb)
        if not mask.any():
            continue
        va = va[mask]; vb = vb[mask]
        diff = vb - va
        mae = float(np.mean(np.abs(diff)))
        denom = np.maximum(np.abs(va), 1e-9)
        mae_pct = float(np.mean(np.abs(diff) / denom) * 100)
        ss_res = float(((va - vb) ** 2).sum())
        ss_tot = float(((va - va.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12) if ss_tot > 0 else (
            1.0 if mae == 0.0 else float("nan"))
        rows.append({
            "feature": c,
            "n": int(mask.sum()),
            "baseline_mean": float(va.mean()),
            "patched_mean": float(vb.mean()),
            "abs_max_diff": float(np.max(np.abs(diff))) if len(diff) else 0.0,
            "MAE": mae,
            "MAE_pct": mae_pct,
            "R2": r2,
        })
    return pd.DataFrame(rows).sort_values("MAE_pct", ascending=False).reset_index(drop=True)


def runtime_summary(rows_a: List[dict], rows_b: List[dict], label: str) -> dict:
    def stats(rows):
        if not rows:
            return None
        runs = np.asarray([r.get("_runtime_s", 0.0) for r in rows], dtype=float)
        return {
            "n": int(len(runs)),
            "sum_s": float(runs.sum()),
            "mean_s": float(runs.mean()),
            "p50_s": float(np.percentile(runs, 50)),
            "p95_s": float(np.percentile(runs, 95)),
            "max_s": float(runs.max()),
        }
    return {"block": label, "baseline": stats(rows_a), "patched": stats(rows_b)}


def fmt_runtime_table(summaries: List[dict]) -> str:
    out = ["| Block | n | baseline sum (s) | patched sum (s) | speedup | "
           "base mean (s) | patched mean (s) | base p95 (s) | patched p95 (s) |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        b = s.get("baseline"); p = s.get("patched")
        if b is None or p is None:
            out.append(f"| {s['block']} | – | – | – | – | – | – | – | – |")
            continue
        speedup = b["sum_s"] / max(p["sum_s"], 1e-9)
        out.append(f"| {s['block']} | {b['n']} | "
                   f"{b['sum_s']:.3f} | {p['sum_s']:.3f} | {speedup:.2f}× | "
                   f"{b['mean_s']*1000:.2f} ms | {p['mean_s']*1000:.2f} ms | "
                   f"{b['p95_s']*1000:.1f} ms | {p['p95_s']*1000:.1f} ms |")
    return "\n".join(out)


def fmt_feature_table(df: pd.DataFrame, top: int = 80) -> str:
    if df.empty:
        return "_(no features in common)_"
    out = ["| Feature | n | baseline mean | patched mean | abs max diff | "
           "MAE | MAE% | R² |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in df.head(top).iterrows():
        out.append(f"| `{r['feature']}` | {r['n']} | "
                   f"{r['baseline_mean']:.4g} | {r['patched_mean']:.4g} | "
                   f"{r['abs_max_diff']:.4g} | {r['MAE']:.4g} | "
                   f"{r['MAE_pct']:.3f}% | {r['R2']:.5f} |")
    if len(df) > top:
        out.append(f"_({len(df) - top} more features omitted; sorted by MAE%)_")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--patched", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="optional markdown output (else stdout only)")
    ap.add_argument("--top-rows", type=int, default=80,
                    help="max feature rows in table (sorted by MAE%)")
    args = ap.parse_args()

    base = load_dump(args.baseline)
    patch = load_dump(args.patched)

    assert base["design"] == patch["design"], (
        f"design mismatch: {base['design']} vs {patch['design']}")
    common_v3 = (set(r["net_name"] for r in base["v3"])
                 & set(r["net_name"] for r in patch["v3"]))
    common_v4 = (set(r["net_name"] for r in base["v4"])
                 & set(r["net_name"] for r in patch["v4"]))

    v3_a = rows_to_df(base["v3"]); v3_b = rows_to_df(patch["v3"])
    v4_a = rows_to_df(base["v4"]); v4_b = rows_to_df(patch["v4"])

    summaries = [
        runtime_summary(base["v3"], patch["v3"], "V3 (41-D)"),
        runtime_summary(base["v4"], patch["v4"], "V4 H3 (26-D)"),
    ]
    drift_v3 = feature_drift(v3_a, v3_b)
    drift_v4 = feature_drift(v4_a, v4_b)

    md = []
    md.append(f"# Feature comparison: `{base['design']}`")
    md.append(f"")
    md.append(f"- baseline label: `{base['label']}` "
              f"(n_selected={base['n_selected']}, wall={base['wall_total_s']:.1f}s)")
    md.append(f"- patched  label: `{patch['label']}` "
              f"(n_selected={patch['n_selected']}, wall={patch['wall_total_s']:.1f}s)")
    md.append(f"- common nets: V3 {len(common_v3)} / V4 {len(common_v4)}")
    md.append("")
    md.append("## Per-block runtime")
    md.append(fmt_runtime_table(summaries))
    md.append("")
    md.append("## V3 per-feature value drift (sorted by MAE%)")
    md.append(fmt_feature_table(drift_v3, top=args.top_rows))
    md.append("")
    md.append("## V4 H3 per-feature value drift (sorted by MAE%)")
    md.append(fmt_feature_table(drift_v4, top=args.top_rows))
    md.append("")
    text = "\n".join(md)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"\n>>> wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
