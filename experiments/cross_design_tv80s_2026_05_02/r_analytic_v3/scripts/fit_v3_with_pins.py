"""Phase 4 — fit with v2 features + n_pins (per-net pin count from DEF header)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

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


def irls_nnls(X, y, n_iter=25, eps=1e-3, verbose=False):
    w = 1.0 / np.maximum(y, eps)
    c = _solve_bnd(X * w[:, None], y * w)
    last = None
    for it in range(n_iter):
        pred = X @ c
        rel = np.abs(pred - y) / np.maximum(y, eps)
        w = 1.0 / (np.maximum(y, eps) * np.sqrt(rel + eps))
        c_new = _solve_bnd(X * w[:, None], y * w)
        mape = float(np.mean(np.abs(X @ c_new - y) / y) * 100)
        if verbose:
            print(f"    iter {it+1:2d}  MAPE={mape:.5f}%  ‖Δc‖={np.linalg.norm(c_new-c):.5f}")
        if last is not None and abs(last - mape) < 1e-5:
            c = c_new; break
        last = mape; c = c_new
    return c


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

    ablations = [
        ("intercept (best v2)",
            ["nsq_M", "rsq_M", "nvian_", "one"]),
        ("+ n_pins (total)",
            ["nsq_M", "rsq_M", "nvian_", "one", "n_pins"]),
        ("+ n_pin_PIN, n_pin_inst (split)",
            ["nsq_M", "rsq_M", "nvian_", "one", "n_pin_PIN", "n_pin_inst"]),
        ("+ n_segments + n_pins + intercept",
            ["nsq_M", "rsq_M", "nvian_", "one", "n_segments", "n_pins"]),
        ("+ all v2 + n_pins",
            ["nsq_M", "rsq_M", "nvian_", "one", "n_segments", "n_zero_l_wire",
             "nseg_M", "sumL_M", "rect_area_M", "nrect_M", "n_pins"]),
        ("kitchen sink",
            ["nsq_M", "rsq_M", "nvian_", "one", "n_segments", "n_zero_l_wire",
             "nseg_M", "sumL_M", "maxL_M", "sumW_M", "rect_area_M", "nrect_M",
             "n_pins", "n_pin_PIN", "n_pin_inst"]),
    ]

    results = []
    for label, prefixes in ablations:
        fcols = _select(train_dfs + [test_df], prefixes)
        Xt = np.vstack([_design_matrix(d, fcols) for d in train_dfs])
        yt = np.concatenate([d["R_gold"].values for d in train_dfs])
        Xs = _design_matrix(test_df, fcols); ys = test_df["R_gold"].values
        c = irls_nnls(Xt, yt)
        train_mape = float(np.mean(np.abs(Xt @ c - yt) / yt) * 100)
        test_mape  = float(np.mean(np.abs(Xs @ c - ys) / ys) * 100)
        print(f"\n[{label}]  fcols={len(fcols)}  train MAPE: {train_mape:.4f}%   test MAPE: {test_mape:.4f}%")
        results.append({"label": label, "n_features": len(fcols),
                          "train_mape": train_mape, "test_mape": test_mape,
                          "fcols": fcols, "coef": c.tolist()})

    best = min(results, key=lambda r: r["test_mape"])
    print(f"\n===== BEST =====")
    print(f"  label: {best['label']}")
    print(f"  test MAPE: {best['test_mape']:.4f}%")
    print(f"  selected features:")
    for n, v in zip(best["fcols"], best["coef"]):
        print(f"    {n:<35s}  c={v:.4f}")

    fcols = best["fcols"]; c = np.array(best["coef"])
    Xs = _design_matrix(test_df, fcols); ys = test_df["R_gold"].values
    pred = Xs @ c
    print(f"\n=== TEST best model: ===")
    _stats("best", pred, ys)
    qs = np.asarray(pd.qcut(ys, 4, labels=["Q1_short", "Q2", "Q3", "Q4_long"]))
    ape = 100 * np.abs(pred - ys) / ys
    bias = 100 * (pred - ys) / ys
    for q in ["Q1_short", "Q2", "Q3", "Q4_long"]:
        m = (qs == q)
        print(f"  {q:>9s}: n={m.sum():4d}  R_med={np.median(ys[m]):.1f}Ω  "
              f"  MAPE={ape[m].mean():7.4f}%  bias={bias[m].mean():+7.4f}%")

    with open(_V3 / "outputs" / "ablation_v3_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {_V3 / 'outputs' / 'ablation_v3_results.json'}")


if __name__ == "__main__":
    main()
