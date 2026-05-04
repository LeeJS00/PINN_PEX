"""Pre-fit per-bucket weights and save for inference-time use.

Reads val predictions from output/spef_e2e/total_cap/preds_per_model/,
fits Nelder-Mead positive weights per bucket, saves to JSON.
"""
from __future__ import annotations

import json
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
    return best_w.tolist()


def main():
    nb = int(__import__("os").environ.get("N_BUCKETS", 40))
    root = cfg.OUTPUT_DIR / "spef_e2e" / "total_cap" / "preds_per_model"
    keys = sorted(set(p.stem.replace("__val", "").replace("__test", "")
                       for p in root.glob("*.csv")))
    print(f"models ({len(keys)}): {keys}")

    Pv = np.stack([pd.read_csv(root / f"{k}__val.csv")["y_pred"].to_numpy() for k in keys], axis=1)
    yv = pd.read_csv(root / f"{keys[0]}__val.csv")["y_true"].to_numpy()

    eps = 1e-4
    val_assigner = np.exp(np.log(np.clip(Pv, eps, None)).mean(axis=1))
    qs = np.linspace(0, 1, nb + 1)[1:-1]
    boundaries = np.quantile(val_assigner, qs)
    val_b = np.digitize(val_assigner, boundaries)

    weights = []
    for b in range(nb):
        mv = val_b == b
        if mv.sum() < 30:
            w = fit_nelder_mape(Pv, yv)
        else:
            w = fit_nelder_mape(Pv[mv], yv[mv])
        weights.append(w)

    out = {
        "n_buckets": nb,
        "model_keys": keys,
        "boundaries": boundaries.tolist(),
        "bucket_weights": weights,
    }
    out_path = cfg.OUTPUT_DIR / "spef_e2e" / "total_cap" / "stratum_weights.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {out_path}")
    print(f"  n_buckets={nb}, n_models={len(keys)}")


if __name__ == "__main__":
    main()
