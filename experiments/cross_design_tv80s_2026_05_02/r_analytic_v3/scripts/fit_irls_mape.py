"""Phase 3 — Push to <1% MAPE.

Strategies stacked:

  (a) Iteratively-reweighted NNLS for true L1/MAPE objective:
        loop:
          pred = X @ c
          w_i  = 1 / (y_i * (|resid_i|/y_i + eps))
          c    = NNLS(X * w[:,None], y * w)
      Converges to argmin Σ |X@c - y|/y subject to c ≥ 0 (in the limit).

  (b) Log-space NNLS via Gauss-Newton: minimize Σ (log(X@c) - log(y))²
      with positivity. Different bias profile than MAPE-IRLS — provides
      diversity for ensembling (c).

  (c) Per-design refinement: add per-design intercept α_d that absorbs
      design-level systematic offset. Closed-form per-design after main fit.

Selection: pick best on TRAIN MAPE, evaluate on TEST.
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

DESIGNS_TRAIN = [
    "intel22_aes_cipher_top_f3", "intel22_gcd_f3", "intel22_ibex_core_f3",
    "intel22_mc_top_f3", "intel22_spi_top_f3", "intel22_usbf_top_f3",
    "intel22_vga_enh_top_f3", "intel22_wb_conmax_top_f3", "intel22_nova_f3",
]
DESIGN_TEST = "intel22_tv80s_f3"


def _load(design):
    df = pd.read_parquet(_V3 / "cache" / f"feat_{design}.parquet")
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


def irls_nnls(X, y, n_iter=20, eps=1e-3, c_init=None, verbose=False):
    """Iteratively reweighted NNLS toward MAPE objective."""
    n = len(y)
    if c_init is None:
        # Bootstrap with weighted-MAPE-style NNLS (1/y weights)
        w = 1.0 / np.maximum(y, eps)
        c, _ = nnls(X * w[:, None], y * w)
    else:
        c = c_init.copy()
    last_mape = None
    for it in range(n_iter):
        pred = X @ c
        # MAPE-aligned IRLS weights: weight = 1 / (y * (|res|/y + eps))
        # The resulting weighted L2 minimization → ||X@c - y||_{w}^2 ≈ Σ |res|/y
        rel_err = np.abs(pred - y) / np.maximum(y, eps)
        w = 1.0 / (np.maximum(y, eps) * np.sqrt(rel_err + eps))
        c_new, _ = nnls(X * w[:, None], y * w)
        pred_new = X @ c_new
        mape = float(np.mean(np.abs(pred_new - y) / y) * 100)
        if verbose:
            print(f"  IRLS iter {it+1:2d}: MAPE={mape:.5f}%  ‖Δc‖={np.linalg.norm(c_new-c):.6f}")
        if last_mape is not None and abs(last_mape - mape) < 1e-5:
            c = c_new
            break
        last_mape = mape
        c = c_new
    return c


def log_space_nnls(X, y, n_iter=30, eps=1e-3, c_init=None, verbose=False):
    """Gauss-Newton on minimize Σ (log(X@c) - log(y))² with c ≥ 0.

    Linearize: log(pred) ≈ log(pred₀) + (pred - pred₀)/pred₀
    Substitute pred = X@c:  log(pred₀) + X@(c-c₀)/pred₀ ≈ log(y)
    => X / pred₀ @ (c - c₀) ≈ log(y) - log(pred₀)
    Solve for δc, update c with positivity projection.
    """
    if c_init is None:
        w = 1.0 / np.maximum(y, eps)
        c, _ = nnls(X * w[:, None], y * w)
    else:
        c = c_init.copy()
    for it in range(n_iter):
        pred = X @ c
        if (pred <= 0).any():
            pred = np.maximum(pred, eps)
        Xs = X / pred[:, None]
        rhs = np.log(np.maximum(y, eps)) - np.log(pred) + Xs @ c
        c_new, _ = nnls(Xs, rhs)
        # damping to ensure progress
        for damp in [1.0, 0.5, 0.25, 0.125]:
            c_try = c + damp * (c_new - c)
            c_try = np.maximum(c_try, 0.0)
            pred_try = X @ c_try
            if (pred_try > 0).all():
                resid = np.log(pred_try) - np.log(np.maximum(y, eps))
                cost = float(np.mean(resid**2))
                break
        c = c_try
        mape = float(np.mean(np.abs(pred_try - y) / y) * 100)
        if verbose:
            print(f"  LOG-NNLS iter {it+1:2d}: log-MSE={cost:.6f}  MAPE={mape:.5f}%")
    return c


def _stats(label, pred, y, n_boot=2000):
    ape  = 100 * np.abs(pred - y) / y
    bias = 100 * (pred - y) / y
    rng = np.random.default_rng(0)
    boots = [ape[rng.integers(0, len(ape), len(ape))].mean() for _ in range(n_boot)]
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    print(f"  {label:>40s}: MAPE={ape.mean():7.4f}%  med={np.median(ape):7.4f}%  "
          f"P90={np.percentile(ape,90):7.3f}%  bias={bias.mean():+7.4f}%  CI=[{ci[0]:.3f}, {ci[1]:.3f}]")
    return ape.mean(), ci, bias.mean(), ape, bias


def main():
    print("Loading caches ...", flush=True)
    train_dfs = [_load(d) for d in DESIGNS_TRAIN]
    test_df   = _load(DESIGN_TEST)
    fcols = _select_feature_cols(train_dfs + [test_df])

    Xt = np.vstack([_design_matrix(df, fcols) for df in train_dfs])
    yt = np.concatenate([df["R_gold"].values for df in train_dfs])
    Xs = _design_matrix(test_df, fcols); ys = test_df["R_gold"].values
    print(f"  train: {Xt.shape}, test: {Xs.shape}, fcols={len(fcols)}")

    # ---------------- (a) IRLS NNLS toward MAPE ----------------
    print("\nFitting IRLS-NNLS (MAPE objective) ...", flush=True)
    c_irls = irls_nnls(Xt, yt, n_iter=20, verbose=True)

    # ---------------- (b) log-space NNLS ----------------
    print("\nFitting LOG-NNLS (log-MSE objective) ...", flush=True)
    c_log  = log_space_nnls(Xt, yt, n_iter=30, verbose=True)

    print("\n=== TRAIN evaluation ===")
    _stats("IRLS-NNLS",        Xt @ c_irls, yt)
    _stats("LOG-NNLS",         Xt @ c_log,  yt)

    print(f"\n=== TEST evaluation: {DESIGN_TEST} (n={len(ys)}) ===")
    mape_irls, _, _, ape_irls, bias_irls = _stats("IRLS-NNLS",  Xs @ c_irls, ys)
    mape_log,  _, _, ape_log,  bias_log  = _stats("LOG-NNLS",   Xs @ c_log,  ys)

    # ---------------- Per-design intercept correction ----------------
    # For each train design, fit one residual scalar β_d minimizing MAPE on that design
    # (only useful if test sees a familiar design — but we use TEST-DEPENDENT correction
    # only for diagnosis, not as the production policy.)
    # For production, we report a single global α_post on top of IRLS pred.

    print("\n--- post-hoc global α (oracle) ---")
    pred = Xs @ c_irls
    alpha_oracle_irls = float(np.median(ys / np.maximum(pred, 1e-3)))
    _stats(f"IRLS × α={alpha_oracle_irls:.4f} (oracle)",  alpha_oracle_irls * pred, ys)
    pred_l = Xs @ c_log
    alpha_oracle_log = float(np.median(ys / np.maximum(pred_l, 1e-3)))
    _stats(f"LOG × α={alpha_oracle_log:.4f} (oracle)",   alpha_oracle_log * pred_l, ys)

    # train-fit α (median ratio on TRAIN nets)
    alpha_train_irls = float(np.median(yt / np.maximum(Xt @ c_irls, 1e-3)))
    alpha_train_log  = float(np.median(yt / np.maximum(Xt @ c_log,  1e-3)))
    _stats(f"IRLS × α_train={alpha_train_irls:.4f}",   alpha_train_irls * pred,  ys)
    _stats(f"LOG × α_train={alpha_train_log:.4f}",     alpha_train_log * pred_l, ys)

    # ---------------- Length stratified ----------------
    print(f"\nLength-stratified (TEST, IRLS-NNLS):")
    qs = np.asarray(pd.qcut(ys, 4, labels=["Q1_short", "Q2", "Q3", "Q4_long"]))
    for q in ["Q1_short", "Q2", "Q3", "Q4_long"]:
        m = (qs == q)
        print(f"  {q:>9s}: n={m.sum():4d}  R_med={np.median(ys[m]):.1f}Ω  "
              f"  MAPE={ape_irls[m].mean():7.4f}%  bias={bias_irls[m].mean():+7.4f}%")

    # ---------------- Save ----------------
    out = {
        "feature_columns": fcols,
        "coefficients_irls":   {fcols[i]: float(c_irls[i]) for i in range(len(fcols))},
        "coefficients_log":    {fcols[i]: float(c_log[i])  for i in range(len(fcols))},
        "test_MAPE_IRLS":      mape_irls,
        "test_MAPE_LOG":       mape_log,
        "alpha_train_irls":    alpha_train_irls,
        "alpha_train_log":     alpha_train_log,
    }
    out_path = _V3 / "outputs" / "coefs_irls_log.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved coefficients: {out_path}")


if __name__ == "__main__":
    main()
