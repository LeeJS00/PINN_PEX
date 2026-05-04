"""Generic stratum blend: takes a preds_per_model dir, fits per-bucket weights on val.

Usage:
  PREDS_DIR=output/spef_e2e/total_r/preds_per_model N_BUCKETS=12 python3 stratum_generic.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent.parent
sys.path.insert(0, str(_WS))

from configs import cfg


def fit_nelder_mape(Pv, yv, n_trials=3, n_iter=2500, max_n=4000):
    rng = np.random.default_rng(0)
    if len(yv) > max_n:
        idx = rng.choice(len(yv), size=max_n, replace=False)
        Pv = Pv[idx]; yv = yv[idx]
    if len(yv) < 30: return None

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
    return best_w


def main():
    preds_dir = Path(os.environ.get("PREDS_DIR"))
    nb = int(os.environ.get("N_BUCKETS", 20))
    save_to = os.environ.get("SAVE_WEIGHTS_TO", None)
    print(f"preds_dir={preds_dir}, n_buckets={nb}")

    keys = sorted(set(p.stem.replace("__val", "").replace("__test", "")
                       for p in preds_dir.glob("*.csv")))
    print(f"models: {len(keys)}")

    Pv = np.stack([pd.read_csv(preds_dir / f"{k}__val.csv")["y_pred"].to_numpy() for k in keys], axis=1)
    yv = pd.read_csv(preds_dir / f"{keys[0]}__val.csv")["y_true"].to_numpy()
    Pt = np.stack([pd.read_csv(preds_dir / f"{k}__test.csv")["y_pred"].to_numpy() for k in keys], axis=1)
    yt = pd.read_csv(preds_dir / f"{keys[0]}__test.csv")["y_true"].to_numpy()

    # Filter zero-target val rows
    mask_v = yv > 1e-6
    Pv = Pv[mask_v]; yv = yv[mask_v]

    eps = 1e-4
    val_assigner = np.exp(np.log(np.clip(Pv, eps, None)).mean(axis=1))
    test_assigner = np.exp(np.log(np.clip(Pt, eps, None)).mean(axis=1))
    qs = np.linspace(0, 1, nb + 1)[1:-1]
    boundaries = np.quantile(val_assigner, qs)
    val_b = np.digitize(val_assigner, boundaries)
    test_b = np.digitize(test_assigner, boundaries)

    bucket_weights = []
    for b in range(nb):
        mv = val_b == b
        if mv.sum() < 30:
            w = fit_nelder_mape(Pv, yv)
        else:
            w = fit_nelder_mape(Pv[mv], yv[mv])
        bucket_weights.append(w.tolist() if w is not None else None)

    yhat_t = np.zeros_like(yt)
    for b, w in enumerate(bucket_weights):
        m = test_b == b
        if w is None or sum(w) == 0:
            yhat_t[m] = test_assigner[m]
            continue
        wa = np.array(w)
        yhat_t[m] = Pt[m] @ wa

    nz = yt > 1e-6
    ape = 100 * np.abs(yhat_t[nz] - yt[nz]) / np.maximum(yt[nz], 1e-3)
    rng = np.random.default_rng(0)
    boots = [ape[rng.integers(0, len(ape), len(ape))].mean() for _ in range(2000)]
    print(f"Stratum (nb={nb}, n_models={len(keys)}): "
          f"MAPE={ape.mean():.4f}%  CI=[{np.percentile(boots, 2.5):.3f}, {np.percentile(boots, 97.5):.3f}]")

    if save_to:
        out = {"n_buckets": nb, "model_keys": keys,
               "boundaries": boundaries.tolist(), "bucket_weights": bucket_weights}
        with open(save_to, "w") as f:
            json.dump(out, f, indent=2)
        print(f"saved {save_to}")


if __name__ == "__main__":
    main()
