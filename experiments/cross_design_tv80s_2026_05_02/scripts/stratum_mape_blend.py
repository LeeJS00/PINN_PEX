"""Per-stratum MAPE-objective blend.

Fit per-bucket positive weights via Nelder-Mead minimizing val MAPE
within the bucket (subsampled). Apply per-bucket at test time using
predicted-quantile bucketing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent
sys.path.insert(0, str(_WS))

from configs import cfg


def collect(roots, kind):
    out = {}
    for root in roots:
        rp = Path(root)
        if not rp.exists():
            continue
        for csv in sorted(rp.rglob(f"*__{kind}.csv")):
            tag = f"{rp.name}::{csv.parent.name}::{csv.stem.replace(f'__{kind}', '')}"
            if "residual" in tag:
                continue
            try:
                df = pd.read_csv(csv).set_index(["design_name", "net_name"])
                out[tag] = df
            except Exception:
                continue
    return out


def report(yhat, y, label, rng=None):
    ape = 100 * np.abs(yhat - y) / np.maximum(y, 1e-3)
    msg = f"  [{label}]  mean={ape.mean():.3f}%  median={np.median(ape):.3f}%  p90={np.percentile(ape, 90):.2f}%"
    if rng is not None:
        boots = []
        for _ in range(2000):
            idx = rng.integers(0, len(ape), len(ape))
            boots.append(ape[idx].mean())
        lo = np.percentile(boots, 2.5); hi = np.percentile(boots, 97.5)
        msg += f"  CI=[{lo:.3f}, {hi:.3f}]"
    print(msg, flush=True)
    return ape.mean()


def fit_nelder_mape(Pv, yv, n_trials=3, n_iter=2000, max_n=5000):
    rng = np.random.default_rng(0)
    n = len(yv)
    if n > max_n:
        idx = rng.choice(n, size=max_n, replace=False)
        Pv = Pv[idx]; yv = yv[idx]

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
    if best_w.sum() > 0:
        best_w = best_w / best_w.sum()
    return best_w, best


def main():
    roots = [str(cfg.OUTPUT_DIR / "final_pipe"),
             str(cfg.OUTPUT_DIR / "final_pipe_nova"),
             str(cfg.OUTPUT_DIR / "final_pipe_v3"),
             str(cfg.OUTPUT_DIR / "final_pipe_v3_nova"),
             str(cfg.OUTPUT_DIR / "resmlp_v2"),
             str(cfg.OUTPUT_DIR / "resmlp_v3"),
             str(cfg.OUTPUT_DIR / "resmlp_v3_nova"),
             str(cfg.OUTPUT_DIR / "mlp_hand_v2"),
             str(cfg.OUTPUT_DIR / "deepset_v2")]
    val_csvs = collect(roots, "val")
    test_csvs = collect(roots, "test")

    val_sizes = {k: len(val_csvs[k]) for k in val_csvs if k in test_csvs}
    nova_size = max(val_sizes.values())
    pool = sorted([k for k in val_sizes if val_sizes[k] == nova_size])
    print(f"Pool size: {len(pool)}", flush=True)

    v0 = val_csvs[pool[0]]
    Pv = np.stack([val_csvs[k]["y_pred"].to_numpy() for k in pool], axis=1)
    yv = v0["y_true"].to_numpy()
    Pt = np.stack([test_csvs[k]["y_pred"].to_numpy() for k in pool], axis=1)
    t0 = test_csvs[pool[0]]
    yt = t0["y_true"].to_numpy()

    eps = 1e-4
    val_assigner = np.exp(np.log(np.clip(Pv, eps, None)).mean(axis=1))
    test_assigner = np.exp(np.log(np.clip(Pt, eps, None)).mean(axis=1))

    n_buckets = int(__import__("os").environ.get("N_BUCKETS", 6))
    qs = np.linspace(0, 1, n_buckets + 1)[1:-1]
    boundaries = np.quantile(val_assigner, qs)
    val_bucket = np.digitize(val_assigner, boundaries)
    test_bucket = np.digitize(test_assigner, boundaries)
    print(f"n_buckets={n_buckets}", flush=True)

    yhat_t = np.zeros_like(yt)
    for b in range(n_buckets):
        mv = val_bucket == b
        if mv.sum() < 100:
            w, val_loss = fit_nelder_mape(Pv, yv)
        else:
            w, val_loss = fit_nelder_mape(Pv[mv], yv[mv])
        mt = test_bucket == b
        yhat_t[mt] = Pt[mt] @ w
        order = np.argsort(-w)
        top = ", ".join(f"{w[i]:.2f}*{pool[i].split('::')[1][:6]}" for i in order[:3])
        print(f"  bucket {b}: val_n={mv.sum()}, test_n={mt.sum()}, val_mape={val_loss*100:.2f}%, top: {top}", flush=True)

    rng = np.random.default_rng(0)
    print("\n=== Stratified MAPE-objective blend ===")
    report(yhat_t, yt, "stratum_mape", rng)

    out = pd.DataFrame({"design_name": t0.reset_index()["design_name"],
                        "net_name": t0.reset_index()["net_name"],
                        "y_true": yt,
                        "y_pred": yhat_t,
                        "bucket": test_bucket})
    out_path = cfg.REPORTS_DIR / f"stratum_mape_b{n_buckets}_test.csv"
    out.to_csv(out_path, index=False)
    print(f"saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
