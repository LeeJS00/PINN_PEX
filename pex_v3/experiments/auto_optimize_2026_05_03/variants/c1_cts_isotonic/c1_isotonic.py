#!/usr/bin/env python3
"""
c1_isotonic.py — CTS Mode B isotonic post-correction (variants C1a + C1b).

Per-seed fit on val, apply on test. Capacity-zero post-process; no model
changes. Anti-leak: isotonic fit NEVER sees test data.

Usage:
    python3 c1_isotonic.py --variant both
    python3 c1_isotonic.py --variant c1a_modeB --seeds 0,1,2,3,4
    python3 c1_isotonic.py --variant c1b_full --smoke    # seed 0 only

Outputs land at:
    pex_v3/experiments/auto_optimize_2026_05_03/outputs/c1_cts_isotonic/
        {c1a_modeB,c1b_full}/seed{0..4}/{eval_logger_test.parquet,
                                          eval_logger_valid.parquet,
                                          summary.json}
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

_PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_PROJECT_ROOT / "pex_v3"))

from src.utils.eval_logger import (  # noqa: E402
    read_eval_parquet, write_eval_parquet, add_error_columns,
)


BASELINE_DIR = _PROJECT_ROOT / "pex_v3" / "output" / "phase1_mesh_5seed"
OUTPUT_ROOT = (
    _PROJECT_ROOT / "pex_v3" / "experiments" / "auto_optimize_2026_05_03"
    / "outputs" / "c1_cts_isotonic"
)
SMOKE_PATH = (
    _PROJECT_ROOT / "pex_v3" / "experiments" / "auto_optimize_2026_05_03"
    / "variants" / "c1_cts_isotonic" / "smoke_seed0.json"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--variant",
        choices=["c1a_modeB", "c1b_full", "both"],
        default="both",
        help="Which isotonic recipe to run",
    )
    p.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated seed list (default: 0,1,2,3,4)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Single-seed (seed 0) smoke run; writes smoke_seed0.json",
    )
    p.add_argument(
        "--mode-b-quantile",
        type=float,
        default=0.99,
        help="Top-(1-q)% threshold for C1a Mode B selection (default 0.99 = top 1pct)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-seed fit + apply
# ---------------------------------------------------------------------------


def _fit_apply_c1a(
    df_val: pd.DataFrame, df_test: pd.DataFrame, q: float = 0.99
) -> tuple[np.ndarray, dict]:
    """Mode B-only isotonic. Threshold from val compact_gnd quantile q.

    Fit IR(gnd_pred -> gnd_gold) on val nets where compact_gnd >= thr.
    Apply to test nets where compact_gnd >= thr (same thr).
    """
    thr = float(df_val["compact_gnd_estimate_fF"].quantile(q))
    mb_v = df_val[df_val["compact_gnd_estimate_fF"] >= thr]
    if len(mb_v) < 10:
        # Not enough Mode B nets; identity correction
        return df_test["gnd_pred"].to_numpy(), {
            "variant": "c1a_modeB",
            "threshold_compact_gnd_fF": thr,
            "n_val_modeB": int(len(mb_v)),
            "n_test_modeB": int((df_test["compact_gnd_estimate_fF"] >= thr).sum()),
            "skipped": "n_val_modeB < 10",
        }

    ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-9)
    ir.fit(mb_v["gnd_pred"].to_numpy(), mb_v["gnd_gold"].to_numpy())

    test_mask = (df_test["compact_gnd_estimate_fF"] >= thr).to_numpy()
    new_pred = df_test["gnd_pred"].to_numpy().copy()
    new_pred[test_mask] = ir.predict(df_test.loc[test_mask, "gnd_pred"].to_numpy())

    info = {
        "variant": "c1a_modeB",
        "threshold_compact_gnd_fF": thr,
        "n_val_modeB": int(len(mb_v)),
        "n_test_modeB": int(test_mask.sum()),
        "ir_x_min": float(ir.X_min_),
        "ir_x_max": float(ir.X_max_),
    }
    return new_pred, info


def _fit_apply_c1b(
    df_val: pd.DataFrame, df_test: pd.DataFrame
) -> tuple[np.ndarray, dict]:
    """Full-distribution log-space isotonic.

    Fit IR(log gnd_pred -> log gnd_gold) on entire val. Apply to entire test.
    """
    eps = 1e-9
    val_mask = (df_val["gnd_pred"] > 0) & (df_val["gnd_gold"] > 0)
    x = np.log(df_val.loc[val_mask, "gnd_pred"].to_numpy() + eps)
    y = np.log(df_val.loc[val_mask, "gnd_gold"].to_numpy() + eps)

    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(x, y)

    test_pred = df_test["gnd_pred"].to_numpy()
    test_pred_safe = np.where(test_pred > 0, test_pred, eps)
    new_pred = np.exp(ir.predict(np.log(test_pred_safe)))
    # Preserve any zeros/negatives in original (shouldn't happen, but defensive)
    new_pred = np.where(test_pred > 0, new_pred, test_pred)

    info = {
        "variant": "c1b_full",
        "n_val_fit": int(val_mask.sum()),
        "n_test_apply": int(len(df_test)),
        "log_x_min": float(ir.X_min_),
        "log_x_max": float(ir.X_max_),
    }
    return new_pred, info


# ---------------------------------------------------------------------------
# Driver per seed + summary build
# ---------------------------------------------------------------------------


def _mape_summary(df: pd.DataFrame) -> dict:
    df = add_error_columns(df.copy())
    return {
        "gnd_mape_median": float(df["gnd_rel_err"].median()),
        "gnd_mape_mean": float(df["gnd_rel_err"].mean()),
        "cpl_mape_median": float(df["cpl_rel_err"].median()),
        "cpl_mape_mean": float(df["cpl_rel_err"].mean()),
        "total_mape_median": float(df["total_rel_err"].median()),
        "total_mape_mean": float(df["total_rel_err"].mean()),
        "n_nets": int(len(df)),
    }


def _top50_gnd(df: pd.DataFrame) -> dict:
    df = add_error_columns(df.copy())
    top = df.nlargest(50, "gnd_rel_err")
    return {
        "gnd_rel_err_median": float(top["gnd_rel_err"].median()),
        "gnd_rel_err_mean": float(top["gnd_rel_err"].mean()),
        "gnd_rel_err_max": float(top["gnd_rel_err"].max()),
        "compact_gnd_median": float(top["compact_gnd_estimate_fF"].median()),
        "fanout_median": float(top["fanout"].median()),
        "n": int(len(top)),
    }


def run_seed(seed: int, variant: str, q_modeB: float) -> dict:
    seed_dir = BASELINE_DIR / f"seed{seed}"
    df_v = read_eval_parquet(seed_dir / "eval_logger_valid.parquet")
    df_t = read_eval_parquet(seed_dir / "eval_logger_test.parquet")

    baseline_test = _mape_summary(df_t)
    baseline_top50 = _top50_gnd(df_t)
    baseline_valid = _mape_summary(df_v)

    if variant == "c1a_modeB":
        new_pred_t, info = _fit_apply_c1a(df_v, df_t, q=q_modeB)
        new_pred_v, _ = _fit_apply_c1a(df_v, df_v, q=q_modeB)  # apply to val for consistency
    elif variant == "c1b_full":
        new_pred_t, info = _fit_apply_c1b(df_v, df_t)
        new_pred_v, _ = _fit_apply_c1b(df_v, df_v)
    else:
        raise ValueError(f"unknown variant {variant}")

    df_t_corr = df_t.copy()
    df_t_corr["gnd_pred"] = new_pred_t
    df_t_corr["total_pred"] = df_t_corr["gnd_pred"] + df_t_corr["cpl_pred"]

    df_v_corr = df_v.copy()
    df_v_corr["gnd_pred"] = new_pred_v
    df_v_corr["total_pred"] = df_v_corr["gnd_pred"] + df_v_corr["cpl_pred"]

    out_dir = OUTPUT_ROOT / variant / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_eval_parquet(df_t_corr, out_dir / "eval_logger_test.parquet")
    write_eval_parquet(df_v_corr, out_dir / "eval_logger_valid.parquet")

    corrected_test = _mape_summary(df_t_corr)
    corrected_top50 = _top50_gnd(df_t_corr)
    corrected_valid = _mape_summary(df_v_corr)

    summary = {
        "seed": seed,
        "variant": variant,
        "isotonic_info": info,
        "baseline_valid": baseline_valid,
        "baseline_test": baseline_test,
        "baseline_top50_gnd": baseline_top50,
        "corrected_valid": corrected_valid,
        "corrected_test": corrected_test,
        "corrected_top50_gnd": corrected_top50,
        # baseline schema fields used by aggregate_ablation_summary.py
        "final_valid": corrected_valid,
        "final_test": corrected_test,
        "best_epoch": -1,  # post-process; no training
        "best_valid_total_mape": float(corrected_valid["total_mape_median"]),
        "best_valid_gnd_mape": float(corrected_valid["gnd_mape_median"]),
        "best_valid_cpl_mape": float(corrected_valid["cpl_mape_median"]),
        "elapsed_train_sec": 0.0,
        "calibration": "none_already_applied_in_baseline",
        "verdict": (
            f"C1 {variant}: gnd {baseline_test['gnd_mape_median']*100:.3f}%"
            f" -> {corrected_test['gnd_mape_median']*100:.3f}%, "
            f"top50 {baseline_top50['gnd_rel_err_median']*100:.1f}%"
            f" -> {corrected_top50['gnd_rel_err_median']*100:.1f}%"
        ),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    delta_gnd = (corrected_test["gnd_mape_median"] - baseline_test["gnd_mape_median"]) * 100
    delta_total = (corrected_test["total_mape_median"] - baseline_test["total_mape_median"]) * 100
    delta_top50 = (corrected_top50["gnd_rel_err_median"] - baseline_top50["gnd_rel_err_median"]) * 100
    print(
        f"  [seed {seed} | {variant}] "
        f"gnd {baseline_test['gnd_mape_median']*100:.3f}% -> "
        f"{corrected_test['gnd_mape_median']*100:.3f}% ({delta_gnd:+.3f}pp)  "
        f"total {baseline_test['total_mape_median']*100:.3f}% -> "
        f"{corrected_test['total_mape_median']*100:.3f}% ({delta_total:+.3f}pp)  "
        f"top50 {baseline_top50['gnd_rel_err_median']*100:.1f}% -> "
        f"{corrected_top50['gnd_rel_err_median']*100:.1f}% ({delta_top50:+.1f}pp)"
    )
    return summary


def main() -> None:
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.smoke:
        seeds = [seeds[0]]

    variants = ["c1a_modeB", "c1b_full"] if args.variant == "both" else [args.variant]

    all_summaries = {}
    for variant in variants:
        print(f">>> variant: {variant}")
        seed_summaries = []
        for s in seeds:
            seed_summaries.append(run_seed(s, variant, q_modeB=args.mode_b_quantile))
        all_summaries[variant] = seed_summaries

    if args.smoke:
        SMOKE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SMOKE_PATH, "w") as f:
            json.dump(all_summaries, f, indent=2)
        print(f">>> wrote smoke artifact: {SMOKE_PATH}")


if __name__ == "__main__":
    main()
