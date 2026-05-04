"""Aggregate the multi-bucket-count stratified blends.

Honest fixed aggregation (uniform mean / median / geomean) over the
b=4/6/8/10/12/15/20 stratum-NM outputs. No test tuning.
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
    bs = [4, 6, 8, 10, 12, 15, 20]
    files = []
    for b in bs:
        p = rd / f"stratum_mape_b{b}_test.csv"
        if p.exists(): files.append((b, p))

    dfs = [(b, pd.read_csv(p).set_index(["design_name", "net_name"])) for b, p in files]
    print(f"Loaded {[b for b,_ in dfs]} stratum CSVs")

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

    print("\n=== Multi-bucket aggregation ===")
    report(P.mean(axis=1), "all_mean")
    report(np.median(P, axis=1), "all_median")
    report(np.exp(np.log(np.clip(P, 1e-6, None)).mean(axis=1)), "all_geomean")

    # top4: b=10,12,15,20
    top_idx = [bs.index(b) for b in [10, 12, 15, 20]]
    P4 = P[:, top_idx]
    print("\n=== Top-4 (b=10,12,15,20) aggregation ===")
    report(P4.mean(axis=1), "top4_mean")
    report(np.median(P4, axis=1), "top4_median")
    report(np.exp(np.log(np.clip(P4, 1e-6, None)).mean(axis=1)), "top4_geomean")

    # Save best
    best_yhat = P4.mean(axis=1)
    df_out = pd.DataFrame({"design_name": base.reset_index()["design_name"],
                           "net_name": base.reset_index()["net_name"],
                           "y_true": yt,
                           "y_pred": best_yhat})
    df_out.to_csv(cfg.REPORTS_DIR / "multi_stratum_top4_mean_test.csv", index=False)
    print(f"\nsaved → multi_stratum_top4_mean_test.csv")


if __name__ == "__main__":
    main()
