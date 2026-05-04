#!/usr/bin/env python3
"""
15_b1_test_aggregate.py — Aggregate B1 XGBoost test (OOD) 5-seed predictions.

After P2 (xgboost_baseline.py modified to write `eval_predictions_test.csv`
+ `per_channel_summary.json`), this script reads each per-seed summary and
emits a `test_5seed_summary.json` that mirrors Option F's structure:
    primary (median-of-per-seed-medians on TEST split):
      per-channel × per-split (valid + test)
    per_design_test:
      nova / tv80s aggregates
    secondary_pooled_predictions:
      pooled-prediction MAPE diagnostic on test

This is post-processing only — re-run anytime after B1 5-seed completes.
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


def parse_args():
    p = argparse.ArgumentParser(description="B1 XGBoost test 5-seed aggregator")
    p.add_argument(
        "--input-dir", type=Path,
        default=_PROJECT_ROOT / "pex_v3" / "output" / "baselines" / "B1_xgboost_real",
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    return p.parse_args()


_EPS_FF = 1e-3


def _mape(pred: np.ndarray, gold: np.ndarray) -> np.ndarray:
    return np.abs(pred - gold) / np.clip(np.abs(gold), _EPS_FF, None)


def _per_channel_summary(pred_df: pd.DataFrame) -> dict:
    out: dict = {}
    for ch in ["gnd", "cpl", "total"]:
        rel = _mape(
            pred_df[f"pred_{ch}_fF"].to_numpy(),
            pred_df[f"golden_{ch}_fF"].to_numpy(),
        )
        out[ch] = {
            "n_nets": int(len(rel)),
            "median": float(np.median(rel)),
            "mean": float(np.mean(rel)),
            "p95": float(np.percentile(rel, 95)),
        }
    return out


def _per_design_summary(pred_df: pd.DataFrame) -> dict:
    return {
        str(design): _per_channel_summary(sub)
        for design, sub in pred_df.groupby("design_name")
    }


def _aggregate_channel(per_seed: list[dict], split: str, ch: str,
                       stat: str = "median") -> dict:
    vals = np.array([s[split][ch][stat] for s in per_seed], dtype=np.float64)
    return {
        "median": float(np.median(vals)),
        "mean": float(np.mean(vals)),
        "stdev": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "values": [float(v) for v in vals],
    }


def main() -> None:
    args = parse_args()
    print(f">>> input dir: {args.input_dir}")
    print(f">>> seeds:     {args.seeds}")

    per_seed: list[dict] = []
    pool_valid_frames: list[pd.DataFrame] = []
    pool_test_frames: list[pd.DataFrame] = []

    for seed in args.seeds:
        sd = args.input_dir / f"seed{seed}"
        summ_path = sd / "per_channel_summary.json"
        if not summ_path.exists():
            raise SystemExit(
                f"Missing {summ_path}. Re-run B1 5-seed with the updated "
                "xgboost_baseline.py first."
            )
        with open(summ_path) as f:
            entry = json.load(f)
        per_seed.append({"seed": seed, **entry})

        v = pd.read_csv(sd / "eval_predictions_valid.csv")
        v["seed"] = seed
        pool_valid_frames.append(v)
        t = pd.read_csv(sd / "eval_predictions_test.csv")
        t["seed"] = seed
        pool_test_frames.append(t)

    summary: dict = {
        "n_seeds": len(args.seeds),
        "seeds": list(args.seeds),
        "primary_per_seed_median_aggregate": {
            "valid": {
                ch: _aggregate_channel(per_seed, "valid", ch)
                for ch in ["gnd", "cpl", "total"]
            },
            "test": {
                ch: _aggregate_channel(per_seed, "test", ch)
                for ch in ["gnd", "cpl", "total"]
            },
        },
        "per_design_test": {},
    }

    # Per-design aggregate on test
    test_designs = sorted(pool_test_frames[0]["design_name"].unique().tolist())
    for design in test_designs:
        per_seed_design = []
        for s in per_seed:
            d = s.get("test_per_design", {}).get(design)
            if d is None:
                continue
            per_seed_design.append(
                {"test": {ch: d[ch] for ch in ["gnd", "cpl", "total"]}}
            )
        if not per_seed_design:
            continue
        summary["per_design_test"][design] = {
            ch: _aggregate_channel(per_seed_design, "test", ch)
            for ch in ["gnd", "cpl", "total"]
        }

    # Secondary pooled diagnostic
    pool_valid = pd.concat(pool_valid_frames, ignore_index=True)
    pool_test = pd.concat(pool_test_frames, ignore_index=True)
    summary["secondary_pooled_predictions"] = {
        "note": "MAPE computed by pooling all 5-seed per-net predictions then taking median; tighter CI than per-seed-median aggregation, kept as diagnostic only.",
        "valid": _per_channel_summary(pool_valid),
        "test": _per_channel_summary(pool_test),
        "test_per_design": _per_design_summary(pool_test),
    }

    out_path = args.input_dir / "test_5seed_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    primary = summary["primary_per_seed_median_aggregate"]
    print()
    print("=" * 72)
    print("B1 XGBoost — 5-seed primary (per-seed-median aggregation)")
    print("=" * 72)
    for split in ["valid", "test"]:
        print(f"\n{split.upper()}:")
        for ch in ["total", "gnd", "cpl"]:
            agg = primary[split][ch]
            print(
                f"  {ch:5s}  median = {agg['median']*100:6.3f}%  "
                f"mean = {agg['mean']*100:6.3f}%  "
                f"stdev = {agg['stdev']*100:5.3f}pp  "
                f"min = {agg['min']*100:5.3f}%  "
                f"max = {agg['max']*100:5.3f}%"
            )
    print("\nPer-design TEST (OOD):")
    for design, ch_agg in summary["per_design_test"].items():
        agg = ch_agg["total"]
        print(
            f"  {design:25s} total median = {agg['median']*100:6.3f}% ± "
            f"{agg['stdev']*100:.3f}pp"
        )
    print(f"\n✅ test_5seed_summary written to {out_path}")


if __name__ == "__main__":
    main()
