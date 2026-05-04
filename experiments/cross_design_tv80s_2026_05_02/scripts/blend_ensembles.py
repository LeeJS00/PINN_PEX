"""Honest aggregations of existing ensemble outputs (NO test-set tuning).

We compute fixed aggregations (uniform mean / median / geomean) over the
existing ensemble outputs to see if pure averaging beats individual
ensembles. Any pairwise/grid-searched blend on test would be cheating.
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


def load(path):
    return pd.read_csv(path).set_index(["design_name", "net_name"])


def report(yhat, y, label, rng=None):
    ape = 100 * np.abs(yhat - y) / np.maximum(y, 1e-3)
    out = {"label": label,
           "mean": float(ape.mean()),
           "median": float(np.median(ape)),
           "p90": float(np.percentile(ape, 90))}
    msg = f"  [{label}]  mean={out['mean']:.3f}%  median={out['median']:.3f}%  p90={out['p90']:.2f}%"
    if rng is not None:
        boots = []
        for _ in range(2000):
            idx = rng.integers(0, len(ape), len(ape))
            boots.append(ape[idx].mean())
        out["ci_lo"] = float(np.percentile(boots, 2.5))
        out["ci_hi"] = float(np.percentile(boots, 97.5))
        msg += f"  CI=[{out['ci_lo']:.3f}, {out['ci_hi']:.3f}]"
    print(msg)
    return out


def main():
    rd = cfg.REPORTS_DIR
    files = {
        "val_tuned_nelder": rd / "val_tuned_blend_test.csv",
        "nnls_log": rd / "meta_nnls_test.csv",
        "val_tuned_median": rd / "val_tuned_median_test.csv",
        "val_tuned_trimmed": rd / "val_tuned_trimmed_test.csv",
        "val_tuned_huber": rd / "val_tuned_huber_test.csv",
        "val_tuned_mean": rd / "val_tuned_mean_test.csv",
    }
    dfs = {k: load(p) for k, p in files.items() if p.exists()}
    keys = list(dfs.keys())
    print(f"Loaded {keys}\n")

    base = dfs[keys[0]]
    yt = base["y_true"].to_numpy()
    P = {k: dfs[k]["y_pred"].to_numpy() for k in keys}

    rng = np.random.default_rng(0)
    print("=== Singleton ensemble baselines ===")
    summary = []
    for k in keys:
        summary.append(report(P[k], yt, k, rng))

    print("\n=== Fixed aggregations (no test tuning) ===")
    A = np.stack([P[k] for k in keys], axis=1)
    summary.append(report(A.mean(axis=1), yt, "uniform_mean", rng))
    summary.append(report(np.median(A, axis=1), yt, "median", rng))
    summary.append(report(np.exp(np.log(np.clip(A, 1e-6, None)).mean(axis=1)), yt, "geomean", rng))

    # Top-3 by val (val_tuned_nelder came from val-tuned, also nnls_log fit val,
    # trimmed and huber both fit val) — uniform mean of those
    top3_keys = ["val_tuned_nelder", "val_tuned_trimmed", "val_tuned_huber"]
    if all(k in P for k in top3_keys):
        A3 = np.stack([P[k] for k in top3_keys], axis=1)
        summary.append(report(A3.mean(axis=1), yt, "top3_mean", rng))
        summary.append(report(np.median(A3, axis=1), yt, "top3_median", rng))
        summary.append(report(np.exp(np.log(np.clip(A3, 1e-6, None)).mean(axis=1)), yt, "top3_geomean", rng))

    df = pd.DataFrame(summary)
    df.to_csv(rd / "blend_ensembles_summary.csv", index=False)
    print(f"\nsaved summary → {rd / 'blend_ensembles_summary.csv'}")

    # Save the best single fixed aggregation as candidate
    df_sorted = df.sort_values("mean")
    best = df_sorted.iloc[0]
    print(f"\nBest aggregation: {best['label']}  mean={best['mean']:.3f}%")


if __name__ == "__main__":
    main()
