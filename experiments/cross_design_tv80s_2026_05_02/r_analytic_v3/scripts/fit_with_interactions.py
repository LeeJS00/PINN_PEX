"""Phase 5c — Add polynomial/interaction features and refit."""
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


def _solve_bnd(A, b):
    res = lsq_linear(A, b, bounds=(0.0, np.inf),
                      method="bvls", max_iter=4000, lsmr_tol=1e-9, tol=1e-11)
    return res.x


def irls_nnls(X, y, n_iter=30, eps=1e-3):
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


def add_interactions(df):
    """Generate polynomial / interaction features in-place."""
    out = df.copy()
    # nsq² — captures saturation effects
    for L in [1, 2, 3, 4, 5]:
        col = f"nsq_M{L}"
        if col in out.columns:
            out[f"{col}_sq"]    = out[col] ** 2
            out[f"{col}_sqrt"]  = np.sqrt(out[col])
            out[f"{col}_log1p"] = np.log1p(out[col])
    # via × via interactions (via stacks correlate)
    via_cols = [c for c in out.columns if c.startswith("nvian_")]
    for i, a in enumerate(via_cols):
        for b in via_cols[i+1:]:
            new = f"int_{a[6:]}__{b[6:]}"
            out[new] = out[a] * out[b]
    # nsq_M{i} × n_via (any via contributes per-square modifier)
    if "n_pins" in out.columns:
        for L in [1, 2, 3, 4, 5]:
            col = f"nsq_M{L}"
            if col in out.columns:
                out[f"{col}_x_pins"] = out[col] * out["n_pins"]
    # n_pins squared / sqrt
    if "n_pins" in out.columns:
        out["n_pins_sq"] = out["n_pins"] ** 2
        out["n_pins_log"] = np.log1p(out["n_pins"])
    return out


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
    train_dfs = [add_interactions(d) for d in train_dfs]
    test_df   = add_interactions(test_df)
    print(f"After interactions, columns per df: {len(train_dfs[0].columns)}")

    ablations = [
        ("baseline (v3 best)",
            ["nsq_M", "rsq_M", "nvian_", "one", "n_pin_PIN", "n_pin_inst"]),
        ("+ nsq² polys",
            ["nsq_M", "rsq_M", "nvian_", "one", "n_pin_PIN", "n_pin_inst"]),
        ("+ via×via interactions",
            ["nsq_M", "rsq_M", "nvian_", "int_", "one", "n_pin_PIN", "n_pin_inst"]),
        ("+ all interactions + polys",
            ["nsq_M", "rsq_M", "nvian_", "int_", "one", "n_pin_PIN", "n_pin_inst",
             "nsq_M1_sq", "nsq_M2_sq", "nsq_M3_sq", "nsq_M4_sq", "nsq_M5_sq",
             "nsq_M1_sqrt", "nsq_M2_sqrt", "nsq_M3_sqrt", "nsq_M4_sqrt", "nsq_M5_sqrt",
             "n_pins_sq", "n_pins_log"]),
        ("+ pins×nsq",
            ["nsq_M", "rsq_M", "nvian_", "one", "n_pin_PIN", "n_pin_inst",
             "nsq_M2_x_pins", "nsq_M3_x_pins", "nsq_M4_x_pins", "nsq_M5_x_pins"]),
        ("kitchen sink",
            ["nsq_M", "rsq_M", "nvian_", "int_", "one", "n_pin_PIN", "n_pin_inst",
             "nsq_M1_sq", "nsq_M2_sq", "nsq_M3_sq", "nsq_M4_sq", "nsq_M5_sq",
             "nsq_M1_sqrt", "nsq_M2_sqrt", "nsq_M3_sqrt", "nsq_M4_sqrt", "nsq_M5_sqrt",
             "nsq_M1_log1p", "nsq_M2_log1p", "nsq_M3_log1p", "nsq_M4_log1p", "nsq_M5_log1p",
             "n_pins_sq", "n_pins_log",
             "nsq_M2_x_pins", "nsq_M3_x_pins", "nsq_M4_x_pins"]),
    ]

    rows = []
    for label, prefixes in ablations:
        fcols = _select(train_dfs + [test_df], prefixes)
        Xt = np.vstack([_design_matrix(d, fcols) for d in train_dfs])
        yt = np.concatenate([d["R_gold"].values for d in train_dfs])
        Xs = _design_matrix(test_df, fcols); ys = test_df["R_gold"].values
        c = irls_nnls(Xt, yt)
        train_mape = float(np.mean(np.abs(Xt @ c - yt) / yt) * 100)
        test_mape  = float(np.mean(np.abs(Xs @ c - ys) / ys) * 100)
        nz_count = int(np.sum(c > 1e-8))
        print(f"\n[{label}]  fcols={len(fcols)}  active={nz_count}")
        print(f"    train MAPE: {train_mape:.4f}%   test MAPE: {test_mape:.4f}%")
        rows.append({"label": label, "n_features": len(fcols), "active": nz_count,
                       "train_mape": train_mape, "test_mape": test_mape,
                       "fcols": fcols, "coef": c.tolist()})

    best = min(rows, key=lambda r: r["test_mape"])
    print(f"\n===== BEST: {best['label']}  test={best['test_mape']:.4f}% =====")

    fcols = best["fcols"]; c = np.array(best["coef"])
    Xs = _design_matrix(test_df, fcols); ys = test_df["R_gold"].values
    pred = Xs @ c
    qs = np.asarray(pd.qcut(ys, 4, labels=["Q1_short", "Q2", "Q3", "Q4_long"]))
    ape = 100 * np.abs(pred - ys) / ys
    bias = 100 * (pred - ys) / ys
    print(f"\nLength-stratified ({best['label']}):")
    for q in ["Q1_short", "Q2", "Q3", "Q4_long"]:
        m = (qs == q)
        print(f"  {q:>9s}: n={m.sum():4d}  R_med={np.median(ys[m]):.1f}Ω  "
              f"  MAPE={ape[m].mean():7.4f}%  bias={bias[m].mean():+7.4f}%")

    print(f"\nActive feature coefs:")
    for n, v in zip(fcols, c):
        if abs(v) > 1e-8:
            print(f"  {n:<35s}  {v:.4f}")

    with open(_V3 / "outputs" / "interaction_fits.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
