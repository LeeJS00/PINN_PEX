#!/usr/bin/env python3
"""
12_b4_compact_gam_eval.py — Phase 0.5 B4 evaluation on v3 valid + test (OOD).

Three variants of compact + ML residual baseline (per A2 audit estimate
3 days; here ~1 hour because Sakurai features already exist in
NetFeatureVector).

Output:
    pex_v3/output/baselines/B4_compact_gam/seed{N}/
        eval_predictions_valid.csv     — per-net pred + golden on valid
        eval_predictions_test.csv      — per-net pred + golden on test (OOD)
        per_channel_summary.json       — gnd/cpl/total MAPE + runtime
        stratified_per_design.csv      — per-design × per-channel breakdown
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "pex_v3"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.baselines.features import NetFeatureVector  # noqa: E402
from src.baselines.compact_gam_v3 import (  # noqa: E402
    linear_compact_baseline,
    compact_plus_gbdt_residual,
    compact_plus_log_gbdt_residual,
    per_channel_mape,
    CompactGAMResult,
)


def parse_args():
    p = argparse.ArgumentParser(description="B4 compact + GAM eval")
    p.add_argument(
        "--features-csv", type=Path,
        default=Path("/data/PINNPEX/data/processed_v3/intel22/features/all_designs.csv"),
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=_PROJECT_ROOT / "pex_v3" / "output" / "baselines" / "B4_compact_gam",
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    return p.parse_args()


def _split(df: pd.DataFrame):
    train = df[df["split"] == "train"].reset_index(drop=True)
    valid = df[df["split"] == "valid"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    # Filter zero-cap rows (per B1 convention)
    def _filt(d):
        return d[(d["c_gnd_fF"] + d["c_cpl_total_fF"]) > 1e-4].reset_index(drop=True)
    return _filt(train), _filt(valid), _filt(test)


def _save_per_net(out_path: Path, df_split: pd.DataFrame, result: CompactGAMResult) -> None:
    """Save per-net predictions for downstream stratified analysis."""
    cols_keep = ["design_name", "net_name", "split"]
    cols_keep = [c for c in cols_keep if c in df_split.columns]
    pred_df = df_split[cols_keep].copy()
    pred_df["pred_gnd_fF"] = result.pred_gnd
    pred_df["pred_cpl_fF"] = result.pred_cpl
    pred_df["pred_total_fF"] = result.pred_total
    pred_df["golden_gnd_fF"] = result.golden_gnd
    pred_df["golden_cpl_fF"] = result.golden_cpl
    pred_df["golden_total_fF"] = result.golden_total
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_path, index=False)


def _stratified_per_design(eval_df: pd.DataFrame, result: CompactGAMResult, out_path: Path):
    """Per-design × per-channel breakdown."""
    rows = []
    pred_df = eval_df[["design_name"]].copy()
    pred_df["pred_gnd"] = result.pred_gnd
    pred_df["pred_cpl"] = result.pred_cpl
    pred_df["pred_total"] = result.pred_total
    pred_df["gold_gnd"] = result.golden_gnd
    pred_df["gold_cpl"] = result.golden_cpl
    pred_df["gold_total"] = result.golden_total
    for design, sub in pred_df.groupby("design_name"):
        for ch in ["gnd", "cpl", "total"]:
            pred = sub[f"pred_{ch}"].to_numpy()
            gold = sub[f"gold_{ch}"].to_numpy()
            gold_safe = np.clip(gold, 1e-3, None)
            rel = np.abs(pred - gold) / gold_safe
            rows.append({
                "design": design,
                "channel": ch,
                "n_nets": len(sub),
                "median_mape": float(np.median(rel)),
                "mean_mape": float(np.mean(rel)),
                "p95_mape": float(np.percentile(rel, 95)),
            })
    pd.DataFrame(rows).to_csv(out_path, index=False)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f">>> Loading features: {args.features_csv}")
    df = pd.read_csv(args.features_csv)
    train, valid, test = _split(df)
    print(f">>> splits: train={len(train):,}  valid={len(valid):,}  test={len(test):,}")

    feature_cols = NetFeatureVector.field_names()
    print(f">>> using {len(feature_cols)} feature columns")

    # Three variants — fit linear once (deterministic; doesn't use seed)
    print()
    print(">>> Variant 1: linear-only (sanity floor)")
    linear = linear_compact_baseline(train, valid)
    linear_test = linear_compact_baseline(train, test)
    print(f"  valid: {per_channel_mape(linear)}")
    print(f"  test:  {per_channel_mape(linear_test)}")

    # Variants 2/3 across seeds
    summaries: list[dict] = []
    for seed in args.seeds:
        print(f">>> Variant 2 (compact+gbdt resid) seed {seed}")
        v2_v = compact_plus_gbdt_residual(train, valid, feature_cols, seed=seed)
        v2_t = compact_plus_gbdt_residual(train, test, feature_cols, seed=seed)
        v2_v_summary = per_channel_mape(v2_v)
        v2_t_summary = per_channel_mape(v2_t)
        print(f"  valid total median: {v2_v_summary['total']['median']*100:.2f}%  "
              f"gnd: {v2_v_summary['gnd']['median']*100:.2f}%  "
              f"cpl: {v2_v_summary['cpl']['median']*100:.2f}%  "
              f"train_t={v2_v.train_seconds:.1f}s  "
              f"inf_t={v2_v.inference_seconds:.3f}s")
        print(f"  test  total median: {v2_t_summary['total']['median']*100:.2f}%  "
              f"gnd: {v2_t_summary['gnd']['median']*100:.2f}%  "
              f"cpl: {v2_t_summary['cpl']['median']*100:.2f}%")

        print(f">>> Variant 3 (compact+log-gbdt resid) seed {seed}")
        v3_v = compact_plus_log_gbdt_residual(train, valid, feature_cols, seed=seed)
        v3_t = compact_plus_log_gbdt_residual(train, test, feature_cols, seed=seed)
        v3_v_summary = per_channel_mape(v3_v)
        v3_t_summary = per_channel_mape(v3_t)
        print(f"  valid total median: {v3_v_summary['total']['median']*100:.2f}%  "
              f"gnd: {v3_v_summary['gnd']['median']*100:.2f}%  "
              f"cpl: {v3_v_summary['cpl']['median']*100:.2f}%")
        print(f"  test  total median: {v3_t_summary['total']['median']*100:.2f}%  "
              f"gnd: {v3_t_summary['gnd']['median']*100:.2f}%  "
              f"cpl: {v3_t_summary['cpl']['median']*100:.2f}%")

        # Save per-seed artifacts for variant 2 (additive — paper anchor)
        seed_dir = args.output_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        _save_per_net(seed_dir / "eval_predictions_valid.csv", valid, v2_v)
        _save_per_net(seed_dir / "eval_predictions_test.csv", test, v2_t)
        _stratified_per_design(valid, v2_v, seed_dir / "stratified_per_design_valid.csv")
        _stratified_per_design(test, v2_t, seed_dir / "stratified_per_design_test.csv")

        with open(seed_dir / "per_channel_summary.json", "w") as f:
            json.dump({
                "linear_valid": per_channel_mape(linear),
                "linear_test":  per_channel_mape(linear_test),
                "compact_gbdt_resid_valid": v2_v_summary,
                "compact_gbdt_resid_test":  v2_t_summary,
                "compact_log_gbdt_resid_valid": v3_v_summary,
                "compact_log_gbdt_resid_test":  v3_t_summary,
            }, f, indent=2)

        summaries.append({
            "seed": seed,
            "v2_valid": v2_v_summary, "v2_test": v2_t_summary,
            "v3_valid": v3_v_summary, "v3_test": v3_t_summary,
        })

    # Aggregate per-seed
    print()
    print(">>> 5-seed aggregate (Variant 2 — additive residual, paper anchor)")
    v2_valid_totals = [s["v2_valid"]["total"]["median"] for s in summaries]
    v2_test_totals = [s["v2_test"]["total"]["median"] for s in summaries]
    v2_valid_gnds = [s["v2_valid"]["gnd"]["median"] for s in summaries]
    v2_valid_cpls = [s["v2_valid"]["cpl"]["median"] for s in summaries]
    print(f"  valid total: median={np.median(v2_valid_totals)*100:.3f}%  "
          f"mean={np.mean(v2_valid_totals)*100:.3f}%  "
          f"stdev={np.std(v2_valid_totals)*100:.3f}pp")
    print(f"  valid gnd:   median={np.median(v2_valid_gnds)*100:.3f}%")
    print(f"  valid cpl:   median={np.median(v2_valid_cpls)*100:.3f}%")
    print(f"  test  total: median={np.median(v2_test_totals)*100:.3f}%  "
          f"mean={np.mean(v2_test_totals)*100:.3f}%  "
          f"stdev={np.std(v2_test_totals)*100:.3f}pp")

    with open(args.output_dir / "five_seed_summary.json", "w") as f:
        json.dump({
            "n_seeds": len(args.seeds),
            "linear_valid": per_channel_mape(linear),
            "linear_test":  per_channel_mape(linear_test),
            "v2_valid_total_median": float(np.median(v2_valid_totals)),
            "v2_valid_total_mean":   float(np.mean(v2_valid_totals)),
            "v2_valid_total_stdev":  float(np.std(v2_valid_totals)),
            "v2_test_total_median":  float(np.median(v2_test_totals)),
            "v2_test_total_mean":    float(np.mean(v2_test_totals)),
            "v2_test_total_stdev":   float(np.std(v2_test_totals)),
            "v2_valid_gnd_median":   float(np.median(v2_valid_gnds)),
            "v2_valid_cpl_median":   float(np.median(v2_valid_cpls)),
            "feature_cols": feature_cols,
        }, f, indent=2)
    print(f"✅ B4 complete. Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
