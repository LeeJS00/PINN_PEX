"""Advanced stacking: geometric mean, trimmed mean, weighted blend.

Reads test/val CSVs from given roots and tries multiple ensembling strategies.
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


def collect(roots, kind="test"):
    out = {}
    for root in roots:
        rp = Path(root)
        if not rp.exists():
            continue
        for csv in sorted(rp.rglob(f"*__{kind}.csv")):
            tag = f"{rp.name}::{csv.parent.name}::{csv.stem.replace(f'__{kind}','')}"
            try:
                df = pd.read_csv(csv)
                if {"y_true","y_pred"}.issubset(df.columns):
                    out[tag] = df.set_index(["design_name","net_name"]) if "design_name" in df.columns else df
            except Exception:
                continue
    return out


def mape(y_true, y_pred, label=""):
    ape = 100.0 * np.abs(y_pred - y_true) / np.maximum(y_true, 1e-3)
    finite = ape[np.isfinite(ape)]
    return dict(label=label,
                n=int(len(finite)),
                mape_mean=float(finite.mean()),
                mape_median=float(np.median(finite)),
                mape_p90=float(np.percentile(finite, 90)),
                mape_p99=float(np.percentile(finite, 99)))


def main():
    roots = [str(cfg.OUTPUT_DIR / "final_pipe"),
             str(cfg.OUTPUT_DIR / "final_pipe_nova"),
             str(cfg.OUTPUT_DIR / "final_pipe_v3"),
             str(cfg.OUTPUT_DIR / "resmlp_v2"),
             str(cfg.OUTPUT_DIR / "resmlp_v3"),
             str(cfg.OUTPUT_DIR / "mlp_hand_v2")]
    roots = [r for r in roots if Path(r).exists()]
    test_csvs = collect(roots, "test")
    print(f"Found {len(test_csvs)} test CSVs in {roots}")

    if not test_csvs:
        return
    keys = sorted(test_csvs.keys())
    base = test_csvs[keys[0]]
    yt = base["y_true"].to_numpy()
    P = np.stack([test_csvs[k]["y_pred"].to_numpy() for k in keys], axis=1)   # (N, M)
    P = np.where(np.isfinite(P), P, np.nan)
    print(f"P shape: {P.shape}")

    # Strategies
    strategies = {}
    strategies["mean"]   = np.nanmean(P, axis=1)
    strategies["median"] = np.nanmedian(P, axis=1)

    # Geometric mean (mean of logs)
    strategies["geomean"] = np.exp(np.nanmean(np.log(np.maximum(P, 1e-4)), axis=1))

    # Trimmed mean (drop top/bottom 10%)
    if P.shape[1] >= 5:
        trimmed_pct = 0.1
        sorted_p = np.sort(P, axis=1)
        n_keep_lo = int(P.shape[1] * trimmed_pct)
        n_keep_hi = P.shape[1] - n_keep_lo
        strategies["trim10_mean"] = np.nanmean(sorted_p[:, n_keep_lo:n_keep_hi], axis=1)

    # Per-model-bucket mean then trimmed
    by_bucket = {}
    for k in keys:
        bucket = k.split("::")[1]
        by_bucket.setdefault(bucket, []).append(test_csvs[k]["y_pred"].to_numpy())
    bucket_means = {}
    for b, preds in by_bucket.items():
        bucket_means[b] = np.mean(np.stack(preds, axis=0), axis=0)
    if bucket_means:
        Pb = np.stack(list(bucket_means.values()), axis=1)
        strategies["bucket_mean"]   = np.nanmean(Pb, axis=1)
        strategies["bucket_median"] = np.nanmedian(Pb, axis=1)

    # Print all
    print()
    rows = []
    for name, p in strategies.items():
        m = mape(yt, p, name)
        rows.append(m)
        print(f"  {name:22s} mean={m['mape_mean']:.3f}% median={m['mape_median']:.3f}% p90={m['mape_p90']:.2f}%")
    rows = sorted(rows, key=lambda x: x["mape_mean"])
    print(f"\nBest: {rows[0]['label']} → {rows[0]['mape_mean']:.3f}% mean MAPE")


if __name__ == "__main__":
    main()
