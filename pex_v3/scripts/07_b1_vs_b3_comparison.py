#!/usr/bin/env python3
"""
07_b1_vs_b3_comparison.py — Apples-to-apples B1 vs B3 statistical comparison.

Both B1 (XGBoost on hand features) and B3 (PINN legacy on tile data) now have
5-seed results on the SAME v3 valid split (12,594 in-dist nets held out from
training designs).

Outputs:
    pex_v3/output/baselines/B1_vs_B3/
        per_run.csv        — concatenated per-seed rows from both methods
        per_method.csv     — summary stats per method
        mwu_pairs.csv      — Mann-Whitney U + Cohen's d for B1 vs B3
        comparison.md      — paper-grade markdown table for inclusion

Per benchmarking-statistician.md: paired raw n=5 + MWU + Cohen's d + bootstrap CI.
"""
from __future__ import annotations
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "pex_v3"))

import pandas as pd  # noqa: E402

from src.evaluation.seed_aggregator import (  # noqa: E402
    aggregate_per_method,
    aggregate_mwu_pairs,
    bootstrap_median_ci,
)


def main():
    b1_dir = _PROJECT_ROOT / "pex_v3" / "output" / "baselines" / "B1_xgboost_real"
    b3_dir = _PROJECT_ROOT / "pex_v3" / "output" / "baselines" / "B3_pinn_real"
    out_dir = _PROJECT_ROOT / "pex_v3" / "output" / "baselines" / "B1_vs_B3"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load per_run.csv from each method
    b1 = pd.read_csv(b1_dir / "per_run.csv")
    b3 = pd.read_csv(b3_dir / "per_run.csv")

    # Verify both have method labels
    if not (b1["method"] == "B1_xgboost").all():
        b1["method"] = "B1_xgboost"
    if not (b3["method"] == "B3_pinn_baseline").all():
        b3["method"] = "B3_pinn_baseline"

    print(f">>> B1 5 seeds: median MAPE per seed = {b1['cap_mape_median'].tolist()}")
    print(f">>> B3 5 seeds: median MAPE per seed = {b3['cap_mape_median'].tolist()}")

    combined = pd.concat([b1, b3], ignore_index=True, sort=False)
    combined.to_csv(out_dir / "per_run.csv", index=False)

    per_method = aggregate_per_method(combined, metric_col="cap_mape_median")
    per_method.to_csv(out_dir / "per_method.csv", index=False)

    mwu = aggregate_mwu_pairs(combined, metric_col="cap_mape_median")
    mwu.to_csv(out_dir / "mwu_pairs.csv", index=False)

    print()
    print("=== per_method.csv ===")
    print(per_method.to_string(index=False))
    print()
    print("=== mwu_pairs.csv ===")
    print(mwu.to_string(index=False))

    # Markdown summary
    b1_med, b1_lo, b1_hi = bootstrap_median_ci(
        b1["cap_mape_median"].to_numpy(), n_resamples=10000, seed=0
    )
    b3_med, b3_lo, b3_hi = bootstrap_median_ci(
        b3["cap_mape_median"].to_numpy(), n_resamples=10000, seed=1
    )
    md = []
    md.append("# B1 vs B3 — Phase B paired comparison\n")
    md.append("_Eval set: v3 valid split (12,594 in-dist nets held out from training designs)_\n")
    md.append("\n")
    md.append("## Per-method 5-seed summary\n\n")
    md.append("| Method | n | Median | Mean | Stdev | min | max | bootstrap 95% CI on median |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for _, r in per_method.iterrows():
        method = r["method"]
        if method == "B1_xgboost":
            ci_lo, ci_hi = b1_lo, b1_hi
        else:
            ci_lo, ci_hi = b3_lo, b3_hi
        md.append(
            f"| {method} | {int(r['n_seeds'])} | {r['median']*100:.3f}% | "
            f"{r['mean']*100:.3f}% | {r['stdev']*100:.3f}pp | "
            f"{r['min']*100:.3f}% | {r['max']*100:.3f}% | "
            f"[{ci_lo*100:.3f}%, {ci_hi*100:.3f}%] |\n"
        )
    md.append("\n## Pairwise MWU + Cohen's d\n\n")
    md.append("| A | B | n_a | n_b | U | p | Cohen's d | label | support |\n")
    md.append("|---|---|---:|---:|---:|---:|---:|---|---|\n")
    for _, r in mwu.iterrows():
        md.append(
            f"| {r['method_a']} | {r['method_b']} | {int(r['n_a'])} | {int(r['n_b'])} | "
            f"{r['U']:.1f} | {r['p_value']:.4g} | {r['cohens_d']:.2f} | "
            f"{r['cohens_d_label']} | **{r['support']}** |\n"
        )
    md.append("\n## Anti-overclaim sanity\n\n")
    if (mwu["support"] == "supported").all():
        md.append(
            "All pairs marked **supported** (p<0.05 + |d|≥0.5). The "
            "B1-vs-B3 comparison passes the project's own §3 protocol gate.\n"
        )
    else:
        md.append("Some pairs are NOT supported. Inspect mwu_pairs.csv.\n")

    with open(out_dir / "comparison.md", "w") as f:
        f.write("".join(md))
    print()
    print(f"=== Wrote {out_dir / 'comparison.md'} ===")
    print("".join(md))


if __name__ == "__main__":
    main()
