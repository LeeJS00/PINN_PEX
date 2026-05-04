#!/usr/bin/env python3
"""02_train_residual.py — fit LightGBM log-residual on TRAIN per-pair features."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/jslee/projects/PINNPEX")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pex_v3" / "joint_pareto" / "allocators" / "cpl"))
import per_pair_residual as ppr

OUT_DIR = Path("/home/jslee/projects/PINNPEX/pex_v3/joint_pareto/experiments/exp_013_per_pair/results")


def main():
    df = pd.read_parquet(OUT_DIR / "train_pairs.parquet")
    print(f">>> loaded {len(df):,} train rows")
    # Drop pairs with bad analytic (zero) — they break log
    df = df[(df["c_analytic_pair_fF"] > 1e-9) & (df["c_golden_pair_fF"] > 1e-9)].copy()
    print(f"    after analytic > 0 filter: {len(df):,}")

    # Hold out one design (random) for valid; use remaining for fit
    designs = sorted(df["design_name"].unique())
    print(f"    designs available: {designs}")
    candidates = ["intel22_wb_conmax_top_f3", "intel22_ibex_core_f3", "intel22_aes_cipher_top_f3"]
    holdout = next((d for d in candidates if d in designs), designs[-1])
    fit_df = df[df["design_name"] != holdout].copy()
    val_df = df[df["design_name"] == holdout].copy()
    print(f"    fit set: {len(fit_df):,}  valid (holdout {holdout}): {len(val_df):,}")

    print(">>> training (n_estimators=200, MSE on log-residual)")
    import time as _time
    _t0 = _time.time()
    booster = ppr.train_residual_model(fit_df, n_estimators=200, seed=0, n_jobs=8)
    print(f"    trained in {_time.time()-_t0:.1f}s")

    # Eval on holdout
    pred_val = ppr.predict_per_pair(val_df, booster)
    g_val = val_df["c_golden_pair_fF"].values
    a_val = val_df["c_analytic_pair_fF"].values
    ape_pred = 100 * np.abs(pred_val - g_val) / np.maximum(g_val, 1e-9)
    ape_an = 100 * np.abs(a_val - g_val) / np.maximum(g_val, 1e-9)

    print()
    print(f"=== holdout {holdout} ({len(val_df):,} pairs) ===")
    print(f"  ANALYTIC ALONE       mean={ape_an.mean():.2f}%  median={np.median(ape_an):.2f}%  p90={np.percentile(ape_an,90):.2f}%")
    print(f"  ANALYTIC × RESIDUAL  mean={ape_pred.mean():.2f}%  median={np.median(ape_pred):.2f}%  p90={np.percentile(ape_pred,90):.2f}%")

    # Save model
    out_model = OUT_DIR / "residual_model.lgb"
    booster.save_model(str(out_model))
    print(f"saved {out_model}")

    # Save fitted importances
    imp = pd.DataFrame({
        "feature": ppr.FEATURE_COLS,
        "gain": booster.feature_importance(importance_type="gain"),
        "split": booster.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    imp.to_csv(OUT_DIR / "feature_importance.csv", index=False)
    print(imp.to_string())


if __name__ == "__main__":
    main()
