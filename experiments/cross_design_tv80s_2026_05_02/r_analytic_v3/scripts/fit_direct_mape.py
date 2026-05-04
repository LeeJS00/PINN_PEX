"""Phase 5 — Direct MAPE optimization via L-BFGS-B with smoothed objective.

We minimize:
    L(c) = mean( huber(pred - y, δ=1e-3) / y )
    s.t. c ≥ 0

Huber smoothing makes the objective differentiable so L-BFGS-B works cleanly.
The IRLS-NNLS solution from Phase 3 is used as warm start.

Also tries:
  - "MAPE-symmetric" Huber on log-residual (different bias profile)
  - per-design intercept (training designs only) with shared shape coeffs

Goal: see how much room remains under the IRLS plateau (3.30%).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear, minimize

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parent

DESIGNS_TRAIN = [
    "intel22_aes_cipher_top_f3", "intel22_gcd_f3", "intel22_ibex_core_f3",
    "intel22_mc_top_f3", "intel22_spi_top_f3", "intel22_usbf_top_f3",
    "intel22_vga_enh_top_f3", "intel22_wb_conmax_top_f3", "intel22_nova_f3",
]
DESIGN_TEST = "intel22_tv80s_f3"


def _load(d):
    df = pd.read_parquet(_V3 / "cache" / f"feat_v2_{d}.parquet")
    pins = pd.read_parquet(_V3 / "cache" / f"pins_{d}.parquet")
    df = df.merge(pins, on="net_name", how="left").fillna(0.0)
    df = df.dropna(subset=["R_gold"])
    df = df[df["R_gold"] > 0.1].reset_index(drop=True).copy()
    return df


def _select(dfs, prefixes):
    cols = set()
    for df in dfs:
        for c in df.columns:
            for p in prefixes:
                if c == p or c.startswith(p):
                    cols.add(c); break
    return sorted(cols)


def _design_matrix(df, fcols):
    X = np.zeros((len(df), len(fcols)), dtype=np.float64)
    for j, c in enumerate(fcols):
        if c in df.columns:
            X[:, j] = df[c].values.astype(np.float64)
    return X


def _solve_bnd(A, b):
    res = lsq_linear(A, b, bounds=(0.0, np.inf),
                      method="bvls", max_iter=2000, lsmr_tol=1e-8, tol=1e-10)
    return res.x


def irls_nnls(X, y, n_iter=20, eps=1e-3):
    w = 1.0 / np.maximum(y, eps)
    c = _solve_bnd(X * w[:, None], y * w)
    last = None
    for it in range(n_iter):
        pred = X @ c
        rel = np.abs(pred - y) / np.maximum(y, eps)
        w = 1.0 / (np.maximum(y, eps) * np.sqrt(rel + eps))
        c_new = _solve_bnd(X * w[:, None], y * w)
        mape = float(np.mean(np.abs(X @ c_new - y) / y) * 100)
        if last is not None and abs(last - mape) < 1e-5:
            c = c_new; break
        last = mape; c = c_new
    return c


def fit_direct_mape(X, y, c0, smooth=1e-3, max_iter=300):
    """L-BFGS-B on MAPE-Huber smooth objective with c >= 0."""
    eps = max(smooth, 1e-6)
    inv_y = 1.0 / np.maximum(y, eps)
    n_feat = X.shape[1]
    bounds = [(0.0, None)] * n_feat

    def fg(c):
        pred = X @ c
        diff = pred - y
        rel = diff * inv_y                         # signed relative error
        # Huber-smooth |rel|: sqrt(rel^2 + eps^2) - eps
        s = np.sqrt(rel * rel + smooth * smooth)
        f = float(np.mean(s - smooth))
        # gradient: ∂f/∂c = X.T @ (rel / s * inv_y) / N
        g = (X.T @ (rel / s * inv_y)) / len(y)
        return f, g

    res = minimize(fg, c0.copy(), jac=True, method="L-BFGS-B", bounds=bounds,
                    options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-9})
    return res.x, res


def _stats(label, pred, y):
    ape = 100 * np.abs(pred - y) / y
    bias = 100 * (pred - y) / y
    rng = np.random.default_rng(0)
    boots = [ape[rng.integers(0, len(ape), len(ape))].mean() for _ in range(2000)]
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    print(f"  {label:>50s}: MAPE={ape.mean():7.4f}%  med={np.median(ape):7.4f}%  "
          f"P90={np.percentile(ape,90):7.3f}%  bias={bias.mean():+7.4f}%  CI=[{ci[0]:.3f}, {ci[1]:.3f}]")
    return ape.mean(), bias.mean(), ape, bias


def main():
    train_dfs = [_load(d) for d in DESIGNS_TRAIN]
    test_df   = _load(DESIGN_TEST)
    print(f"Train nets: {sum(len(d) for d in train_dfs):,}, test nets: {len(test_df)}")

    # Feature set: best from prev (intercept + n_pins split)
    prefixes = ["nsq_M", "rsq_M", "nvian_", "one", "n_pin_PIN", "n_pin_inst"]
    fcols = _select(train_dfs + [test_df], prefixes)
    print(f"Feature columns: {len(fcols)}")
    Xt = np.vstack([_design_matrix(d, fcols) for d in train_dfs])
    yt = np.concatenate([d["R_gold"].values for d in train_dfs])
    Xs = _design_matrix(test_df, fcols); ys = test_df["R_gold"].values

    # warm start
    print("\nWarm-start IRLS-NNLS ...", flush=True)
    c_irls = irls_nnls(Xt, yt)
    train_mape_irls = float(np.mean(np.abs(Xt @ c_irls - yt) / yt) * 100)
    test_mape_irls  = float(np.mean(np.abs(Xs @ c_irls - ys) / ys) * 100)
    print(f"  IRLS train MAPE = {train_mape_irls:.4f}%   test = {test_mape_irls:.4f}%")

    # direct MAPE via L-BFGS-B
    for smooth in [1e-3, 1e-4, 1e-5]:
        print(f"\nL-BFGS-B direct MAPE (smooth={smooth}) ...", flush=True)
        c_dm, res = fit_direct_mape(Xt, yt, c_irls, smooth=smooth, max_iter=500)
        train_mape = float(np.mean(np.abs(Xt @ c_dm - yt) / yt) * 100)
        test_mape  = float(np.mean(np.abs(Xs @ c_dm - ys) / ys) * 100)
        print(f"  converged: {res.success}, msg: {res.message}, n_iter: {res.nit}")
        print(f"  train MAPE = {train_mape:.4f}%   test = {test_mape:.4f}%")
        # use the smallest smoothing's result
        c_best = c_dm

    # ---------------- per-design intercept augmentation ----------------
    # Add D dummy columns (one per train design) to the design matrix during
    # training. At inference, use the MEAN of learned intercepts.
    n_designs = len(DESIGNS_TRAIN)
    print(f"\nL-BFGS-B with per-design intercept (D={n_designs}) ...", flush=True)
    Xt_aug = []
    for di, df in enumerate(train_dfs):
        block = np.zeros((len(df), n_designs))
        block[:, di] = 1.0
        Xt_aug.append(np.hstack([_design_matrix(df, fcols), block]))
    Xt_aug = np.vstack(Xt_aug)
    fcols_aug = fcols + [f"design_{i}_intercept" for i in range(n_designs)]
    c0_aug = np.concatenate([c_best, np.zeros(n_designs)])
    c_aug, res_aug = fit_direct_mape(Xt_aug, yt, c0_aug, smooth=1e-4, max_iter=500)
    print(f"  converged: {res_aug.success}, n_iter: {res_aug.nit}")
    train_mape_aug = float(np.mean(np.abs(Xt_aug @ c_aug - yt) / yt) * 100)
    print(f"  train MAPE w/ per-design intercept = {train_mape_aug:.4f}%")

    # split into shared coefs and per-design intercepts
    c_shared = c_aug[:len(fcols)]
    intercepts = c_aug[len(fcols):]
    avg_int = float(np.mean(intercepts))
    med_int = float(np.median(intercepts))
    print(f"  per-design intercepts (Ω, sorted): {sorted(intercepts.round(3))}")
    print(f"  mean = {avg_int:.3f}  median = {med_int:.3f}")

    # Apply to test with avg/med intercept
    pred_avg = Xs @ c_shared + avg_int
    pred_med = Xs @ c_shared + med_int
    print(f"\n=== TEST evaluation ===")
    _stats("IRLS-NNLS",                 Xt @ c_irls if False else (Xs @ c_irls), ys)
    _stats("L-BFGS-B direct MAPE",      Xs @ c_best, ys)
    _stats("+ per-design intercept (avg)", pred_avg, ys)
    _stats("+ per-design intercept (med)", pred_med, ys)

    # Length-stratified for direct-MAPE
    qs = np.asarray(pd.qcut(ys, 4, labels=["Q1_short", "Q2", "Q3", "Q4_long"]))
    for label, pred in [("L-BFGS-B direct", Xs @ c_best),
                         ("+ avg intercept", pred_avg)]:
        print(f"\nLength-stratified ({label}):")
        ape = 100 * np.abs(pred - ys) / ys
        bias = 100 * (pred - ys) / ys
        for q in ["Q1_short", "Q2", "Q3", "Q4_long"]:
            m = (qs == q)
            print(f"  {q:>9s}: n={m.sum():4d}  MAPE={ape[m].mean():7.4f}%  bias={bias[m].mean():+7.4f}%")

    # save
    out = {"feature_columns": fcols,
           "coefficients_lbfgsb": {fcols[i]: float(c_best[i]) for i in range(len(fcols))},
           "per_design_intercepts": {DESIGNS_TRAIN[i]: float(intercepts[i]) for i in range(n_designs)},
           "intercept_avg":   avg_int,
           "intercept_med":   med_int}
    with open(_V3 / "outputs" / "coefs_direct_mape.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {_V3 / 'outputs' / 'coefs_direct_mape.json'}")


if __name__ == "__main__":
    main()
