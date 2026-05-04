"""Stratified per-bucket blend for total_cap on e2e pipeline.

Reuses Pass 7 strategy: stratify nets by predicted-cap quantile, fit
positive Nelder-Mead weights per bucket on val.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent.parent
sys.path.insert(0, str(_WS))

from configs import cfg


def collect(root):
    out = {}
    for csv in sorted(Path(root).rglob("*__val.csv")):
        tag = csv.stem.replace("__val", "")
        df_v = pd.read_csv(csv).set_index(["design_name", "net_name"])
        ts = csv.parent / f"{tag}__test.csv"
        if not ts.exists():
            continue
        df_t = pd.read_csv(ts).set_index(["design_name", "net_name"])
        out[tag] = (df_v, df_t)
    return out


def fit_nelder_mape(Pv, yv, n_trials=3, n_iter=2000, max_n=4000):
    rng = np.random.default_rng(0)
    if len(yv) > max_n:
        idx = rng.choice(len(yv), size=max_n, replace=False)
        Pv = Pv[idx]; yv = yv[idx]
    if len(yv) < 30:
        return None, None

    def loss(w):
        w = np.clip(w, 0, None)
        if w.sum() == 0: return 1e6
        w = w / w.sum()
        yhat = Pv @ w
        ape = np.abs(yhat - yv) / np.maximum(yv, 1e-3)
        return float(np.mean(ape))

    best = float("inf"); best_w = None
    for trial in range(n_trials):
        np.random.seed(trial)
        w0 = np.random.rand(Pv.shape[1])
        res = minimize(loss, w0, method="Nelder-Mead",
                       options={"maxiter": n_iter, "xatol": 1e-5, "fatol": 1e-6, "adaptive": True})
        if res.fun < best:
            best = res.fun; best_w = res.x.copy()
    best_w = np.clip(best_w, 0, None)
    if best_w.sum() > 0: best_w = best_w / best_w.sum()
    return best_w, best


def main():
    nb = int(__import__("os").environ.get("N_BUCKETS", 12))
    print(f"n_buckets={nb}")

    root = cfg.OUTPUT_DIR / "spef_e2e" / "total_cap" / "preds_per_model"
    pool = collect(root)
    print(f"Loaded {len(pool)} models")

    keys = sorted(pool.keys())
    Pv = np.stack([pool[k][0]["y_pred"].to_numpy() for k in keys], axis=1)
    yv = pool[keys[0]][0]["y_true"].to_numpy()
    Pt = np.stack([pool[k][1]["y_pred"].to_numpy() for k in keys], axis=1)
    yt = pool[keys[0]][1]["y_true"].to_numpy()

    eps = 1e-4
    val_assigner = np.exp(np.log(np.clip(Pv, eps, None)).mean(axis=1))
    test_assigner = np.exp(np.log(np.clip(Pt, eps, None)).mean(axis=1))

    qs = np.linspace(0, 1, nb + 1)[1:-1]
    boundaries = np.quantile(val_assigner, qs)
    val_b = np.digitize(val_assigner, boundaries)
    test_b = np.digitize(test_assigner, boundaries)

    yhat_t = np.zeros_like(yt)
    for b in range(nb):
        mv = val_b == b
        if mv.sum() < 30:
            w, _ = fit_nelder_mape(Pv, yv)
        else:
            w, _ = fit_nelder_mape(Pv[mv], yv[mv])
        if w is None:
            yhat_t[test_b == b] = test_assigner[test_b == b]
            continue
        yhat_t[test_b == b] = Pt[test_b == b] @ w

    ape = 100 * np.abs(yhat_t - yt) / np.maximum(yt, 1e-3)
    rng = np.random.default_rng(0)
    boots = [ape[rng.integers(0, len(ape), len(ape))].mean() for _ in range(2000)]
    print(f"Stratum blend (n_buckets={nb}):")
    print(f"  mean MAPE = {ape.mean():.4f}%")
    print(f"  median = {np.median(ape):.3f}%")
    print(f"  CI = [{np.percentile(boots, 2.5):.3f}, {np.percentile(boots, 97.5):.3f}]")

    # Save
    out_path = cfg.REPORTS_DIR / f"stratum_total_cap_b{nb}_test.csv"
    pd.DataFrame({"design_name": pool[keys[0]][1].reset_index()["design_name"],
                  "net_name": pool[keys[0]][1].reset_index()["net_name"],
                  "y_true": yt, "y_pred": yhat_t}).to_csv(out_path, index=False)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
