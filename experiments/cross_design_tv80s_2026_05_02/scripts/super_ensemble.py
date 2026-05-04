"""Super-ensemble: average all 1D and 2D stratification predictions.

Honest fixed aggregation (no test tuning).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent
sys.path.insert(0, str(_WS))

from configs import cfg


def main():
    rd = cfg.REPORTS_DIR
    files = []
    for b in [4, 6, 8, 10, 12, 15, 20]:
        p = rd / f"stratum_mape_b{b}_test.csv"
        if p.exists(): files.append(("1d_b%d" % b, p))
    for c in [4, 5, 6, 7, 8, 10]:
        for a in [3, 4]:
            p = rd / f"stratum_2d_c{c}_a{a}_test.csv"
            if p.exists(): files.append(("2d_c%d_a%d" % (c, a), p))

    print(f"Loading {len(files)} stratifications:")
    dfs = []
    for label, p in files:
        df = pd.read_csv(p).set_index(["design_name", "net_name"])
        dfs.append((label, df))
        print(f"  {label}: {len(df)} rows")

    base = dfs[0][1]
    yt = base["y_true"].to_numpy()
    P = np.stack([df["y_pred"].to_numpy() for _, df in dfs], axis=1)

    rng = np.random.default_rng(0)
    def report(yhat, label):
        ape = 100 * np.abs(yhat - yt) / np.maximum(yt, 1e-3)
        boots = []
        for _ in range(2000):
            idx = rng.integers(0, len(ape), len(ape))
            boots.append(ape[idx].mean())
        lo = np.percentile(boots, 2.5); hi = np.percentile(boots, 97.5)
        print(f"  [{label}]  mean={ape.mean():.4f}%  median={np.median(ape):.3f}%  p90={np.percentile(ape, 90):.2f}%  CI=[{lo:.3f}, {hi:.3f}]")
        return ape.mean()

    print("\n=== Super-ensemble ===")
    report(P.mean(axis=1), "all_mean")
    report(np.median(P, axis=1), "all_median")
    report(np.exp(np.log(np.clip(P, 1e-6, None)).mean(axis=1)), "all_geomean")

    # Top-N by MAPE: identify on val NOT possible without val csvs; just use
    # the published test-MAPE order. Honest because we're picking BUCKETING SCHEMES,
    # not test labels — and the underlying weights were val-fit.
    test_mape = []
    for i, (label, df) in enumerate(dfs):
        ape = 100 * np.abs(P[:, i] - yt) / np.maximum(yt, 1e-3)
        test_mape.append((ape.mean(), label, i))
    test_mape.sort()

    print("\nIndividual stratifications sorted by test MAPE:")
    for tm, label, _ in test_mape:
        print(f"  {tm:.4f}%  {label}")

    # Top-K aggregations (these ARE test-tuned in selection of K, so report but flag)
    for K in [3, 5, 8]:
        idx = [t[2] for t in test_mape[:K]]
        Pk = P[:, idx]
        print(f"\n  Top-{K} (using TEST mape ranking — biased):")
        report(Pk.mean(axis=1), f"top{K}_mean (biased)")
        report(np.exp(np.log(np.clip(Pk, 1e-6, None)).mean(axis=1)), f"top{K}_geomean (biased)")

    # Save the honest all_mean as canonical
    yhat_best = P.mean(axis=1)
    out = pd.DataFrame({"design_name": base.reset_index()["design_name"],
                        "net_name": base.reset_index()["net_name"],
                        "y_true": yt, "y_pred": yhat_best})
    out.to_csv(cfg.REPORTS_DIR / "super_ensemble_test.csv", index=False)
    print(f"\nsaved super_ensemble_test.csv (uniform mean of {len(files)} stratifications)")


if __name__ == "__main__":
    main()
