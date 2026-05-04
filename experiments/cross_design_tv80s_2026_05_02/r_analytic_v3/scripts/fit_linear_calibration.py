"""Phase 2 — Fit physics-interpretable linear calibration on training designs.

Model:
    R_gold(net) = Σ_layer  c_metal[layer] × n_squares_M{layer}(net)
                + Σ_via    c_via[name]    × n_via_<NAME>(net)
                + (optional) c_rect[layer] × rect_squares_M{layer}(net)

Coefficients are non-negative (NNLS) so they remain physically interpretable
(sheet R per layer ≥ 0, R per via name ≥ 0).

Cost: minimize ||X·c - y||₂  s.t.  c ≥ 0   on training nets.

Compared baselines:
  - v2 raw analytic (no fit)
  - v2 + global-α
  - v3 NNLS
  - v3 NNLS + per-design intercept correction (optional)

Outputs:
  - r_analytic_v3/outputs/coefs_v3.json         : learned c
  - r_analytic_v3/outputs/feature_columns.json  : ordered column list
  - r_analytic_v3/reports/phase2_summary.txt    : train/test MAPE
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parent
_WS = _V3.parent
sys.path.insert(0, str(_HERE))

DESIGNS_TRAIN = [
    "intel22_aes_cipher_top_f3", "intel22_gcd_f3", "intel22_ibex_core_f3",
    "intel22_mc_top_f3", "intel22_spi_top_f3", "intel22_usbf_top_f3",
    "intel22_vga_enh_top_f3", "intel22_wb_conmax_top_f3", "intel22_nova_f3",
]
DESIGN_TEST = "intel22_tv80s_f3"


def _load(design):
    p = _V3 / "cache" / f"feat_{design}.parquet"
    df = pd.read_parquet(p)
    df = df.dropna(subset=["R_gold"])
    df = df[df["R_gold"] > 0.1].reset_index(drop=True).copy()
    return df


def _select_feature_cols(dfs):
    cols = set()
    for df in dfs:
        for c in df.columns:
            if c.startswith("nsq_M") or c.startswith("rsq_M") or c.startswith("nvian_"):
                cols.add(c)
    return sorted(cols)


def _design_matrix(df, fcols):
    X = np.zeros((len(df), len(fcols)), dtype=np.float64)
    for j, c in enumerate(fcols):
        if c in df.columns:
            X[:, j] = df[c].values.astype(np.float64)
    return X


def _stats(label, pred, y, rng=None, n_boot=2000):
    ape  = 100 * np.abs(pred - y) / y
    bias = 100 * (pred - y) / y
    if rng is None:
        rng = np.random.default_rng(0)
    boots = [ape[rng.integers(0, len(ape), len(ape))].mean() for _ in range(n_boot)]
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    print(f"  {label:>40s}: MAPE={ape.mean():7.4f}%  med={np.median(ape):7.4f}%  "
          f"P90={np.percentile(ape,90):7.3f}%  bias={bias.mean():+7.4f}%  CI=[{ci[0]:.3f}, {ci[1]:.3f}]")
    return ape.mean(), ci, bias.mean()


def main():
    print("Loading per-design feature parquets ...", flush=True)
    train_dfs = []
    for d in DESIGNS_TRAIN:
        try:
            df = _load(d)
        except FileNotFoundError:
            print(f"  [skip] {d}: cache missing")
            continue
        train_dfs.append(df)
    test_df = _load(DESIGN_TEST)
    print(f"  train designs: {len(train_dfs)}, total train nets: {sum(len(d) for d in train_dfs)}")
    print(f"  test design: {DESIGN_TEST}, n_nets: {len(test_df)}")

    fcols = _select_feature_cols(train_dfs + [test_df])
    print(f"  feature columns: {len(fcols)}")
    print(f"    metal: {[c for c in fcols if c.startswith('nsq_')]}")
    print(f"    rect:  {[c for c in fcols if c.startswith('rsq_')]}")
    print(f"    via:   {[c for c in fcols if c.startswith('nvian_')]}")

    X_train = np.vstack([_design_matrix(df, fcols) for df in train_dfs])
    y_train = np.concatenate([df["R_gold"].values for df in train_dfs])

    X_test  = _design_matrix(test_df, fcols)
    y_test  = test_df["R_gold"].values

    print(f"\nX_train shape: {X_train.shape}, y range [{y_train.min():.2f}, {y_train.max():.2f}]")

    # -----------------------------------------------------------------
    # NNLS with optional weighted MAPE objective.
    # NNLS minimizes ||Xc - y||² (L2 on absolute residuals).
    # For MAPE we want |Xc - y| / y → weight rows by 1/y.
    # -----------------------------------------------------------------
    print("\nFitting NNLS (L2 on absolute residual) ...", flush=True)
    c_l2, rnorm_l2 = nnls(X_train, y_train)
    print(f"  fit done.  rnorm = {rnorm_l2:.3f}")

    print("\nFitting NNLS (weighted, MAPE-like, weights = 1/y) ...", flush=True)
    w = 1.0 / np.maximum(y_train, 1e-3)
    Xw = X_train * w[:, None]
    yw = y_train * w
    c_mape, rnorm_mape = nnls(Xw, yw)
    print(f"  fit done.  rnorm_w = {rnorm_mape:.3f}")

    # -----------------------------------------------------------------
    # Pretty-print learned coefficients
    # -----------------------------------------------------------------
    print("\n=== Learned coefficients (NNLS-MAPE) ===")
    print("  metal sheet R (Ω/sq, == coefficient × wire width / 1):")
    for j, c in enumerate(fcols):
        if c.startswith("nsq_M"):
            print(f"    {c}: {c_mape[j]:.4f}  (L2 fit: {c_l2[j]:.4f})")
    print("  RECT sheet R (Ω/sq):")
    for j, c in enumerate(fcols):
        if c.startswith("rsq_M"):
            print(f"    {c}: {c_mape[j]:.4f}  (L2 fit: {c_l2[j]:.4f})")
    print("  per-VIA-name R (Ω):")
    for j, c in enumerate(fcols):
        if c.startswith("nvian_"):
            print(f"    {c}: {c_mape[j]:.4f}  (L2 fit: {c_l2[j]:.4f})")

    # -----------------------------------------------------------------
    # TRAIN/TEST evaluation
    # -----------------------------------------------------------------
    print(f"\n=== TRAIN evaluation (n={len(y_train)}) ===")
    pred_train_l2   = X_train @ c_l2
    pred_train_mape = X_train @ c_mape
    _stats("NNLS L2",       pred_train_l2,   y_train)
    _stats("NNLS MAPE-like", pred_train_mape, y_train)

    print(f"\n=== TEST evaluation: {DESIGN_TEST} (n={len(y_test)}) ===")
    pred_test_l2   = X_test @ c_l2
    pred_test_mape = X_test @ c_mape
    _stats("NNLS L2",       pred_test_l2,   y_test)
    mape_test, ci_test, bias_test = _stats("NNLS MAPE-like", pred_test_mape, y_test)

    # length-stratified
    qs = pd.qcut(y_test, 4, labels=["Q1_short", "Q2", "Q3", "Q4_long"])
    print(f"\nLength-stratified MAPE (NNLS-MAPE on test):")
    ape = 100 * np.abs(pred_test_mape - y_test) / y_test
    bias = 100 * (pred_test_mape - y_test) / y_test
    qs_arr = np.asarray(qs)
    for q in ["Q1_short", "Q2", "Q3", "Q4_long"]:
        m = (qs_arr == q)
        print(f"  {q:>9s}: n={m.sum():4d}  R_med={np.median(y_test[m]):.1f}Ω  "
              f"  MAPE={ape[m].mean():7.4f}%  bias={bias[m].mean():+7.4f}%")

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------
    out = {
        "feature_columns": fcols,
        "coefficients_l2":   {fcols[i]: float(c_l2[i])   for i in range(len(fcols))},
        "coefficients_mape": {fcols[i]: float(c_mape[i]) for i in range(len(fcols))},
        "train_mape_mape_obj":  float(np.mean(np.abs(pred_train_mape - y_train) / y_train) * 100),
        "test_mape_mape_obj":   mape_test,
        "test_mape_CI":         list(ci_test),
        "test_bias":            bias_test,
        "n_train_nets":         int(len(y_train)),
        "n_test_nets":          int(len(y_test)),
    }
    out_path = _V3 / "outputs" / "coefs_v3.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved coefficients: {out_path}")

    # also save per-net test predictions for downstream analysis
    test_df["R_pred_v3"] = pred_test_mape
    test_df["ape_v3"]    = ape
    test_df["bias_v3"]   = bias
    test_df.to_parquet(_V3 / "outputs" / "test_predictions_v3.parquet")
    print(f"Saved per-net test preds: {_V3 / 'outputs' / 'test_predictions_v3.parquet'}")


if __name__ == "__main__":
    main()
