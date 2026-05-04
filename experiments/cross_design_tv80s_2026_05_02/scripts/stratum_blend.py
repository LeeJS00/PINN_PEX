"""Per-stratum (cap-bucket) blend.

Idea: the optimal model weights may differ for small vs large nets. We
estimate weights separately per cap bucket on val (using NNLS in log
space, fast closed form), then apply the same buckets at test time —
but with a critical wrinkle: at test time we don't know the true cap, so
we use a *predicted-cap-quantile* bucket assignment (using the existing
val-tuned ensemble as the bucket-assigner).

Workflow:
  1. Compute val-tuned blend prediction on val and test (we already have test).
  2. For val rows, bucket by TRUE log(c). For test rows, bucket by
     PREDICTED log(c) using the same quantile boundaries.
  3. Within each val bucket, fit NNLS weights mapping log(75 preds) → log(c).
  4. Apply per-bucket weights at test time using predicted-quantile bucketing.

Honesty check:
  - Val bucketing uses true labels (allowed; weights are val-fit).
  - Test bucketing uses predicted quantiles, NOT true labels (no leakage).
  - Risk: if a test net is mis-bucketed (predicted small, actually large),
    it gets the small-cap weights, which might not be optimal. But this
    is a model choice, not a leak.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

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


def fit_nnls_log(Pv, yv):
    eps = 1e-4
    Lv = np.log(np.clip(Pv, eps, None))
    Yv = np.log(np.clip(yv, eps, None))
    mask = np.all(np.isfinite(Lv), axis=1) & np.isfinite(Yv)
    Lv = Lv[mask]; Yv = Yv[mask]
    if len(Lv) < 50:
        return None, None
    A = np.concatenate([Lv, np.ones((len(Lv), 1))], axis=1)
    w, _ = nnls(A, Yv, maxiter=A.shape[1] * 20)
    return w[:-1], w[-1]


def predict_nnls_log(P, coefs, intercept):
    eps = 1e-4
    L = np.log(np.clip(P, eps, None))
    return np.exp(L @ coefs + intercept)


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
    print(f"Pool size: {len(pool)} (nova val)", flush=True)

    v0 = val_csvs[pool[0]]
    Pv = np.stack([val_csvs[k]["y_pred"].to_numpy() for k in pool], axis=1)
    yv = v0["y_true"].to_numpy()
    Pt = np.stack([test_csvs[k]["y_pred"].to_numpy() for k in pool], axis=1)
    t0 = test_csvs[pool[0]]
    yt = t0["y_true"].to_numpy()

    # Bucket assigner: simple geomean of all 30 preds
    eps = 1e-4
    val_assigner = np.exp(np.log(np.clip(Pv, eps, None)).mean(axis=1))
    test_assigner = np.exp(np.log(np.clip(Pt, eps, None)).mean(axis=1))

    # Bucket boundaries from val_assigner quantiles
    n_buckets = 6
    qs = np.linspace(0, 1, n_buckets + 1)[1:-1]
    boundaries = np.quantile(val_assigner, qs)
    print(f"bucket boundaries: {[f'{b:.3f}' for b in boundaries]}", flush=True)

    val_bucket = np.digitize(val_assigner, boundaries)
    test_bucket = np.digitize(test_assigner, boundaries)
    for b in range(n_buckets):
        nv = (val_bucket == b).sum(); nt = (test_bucket == b).sum()
        print(f"  bucket {b}: val={nv}, test={nt}", flush=True)

    # Fit NNLS per bucket on val, apply on test
    yhat_t = np.zeros_like(yt)
    for b in range(n_buckets):
        mv = (val_bucket == b)
        if mv.sum() < 50:
            print(f"bucket {b}: too few val rows ({mv.sum()}), using global fit")
            coefs, intercept = fit_nnls_log(Pv, yv)
        else:
            coefs, intercept = fit_nnls_log(Pv[mv], yv[mv])
        if coefs is None:
            print(f"bucket {b}: fit failed, fallback geomean")
            yhat_t[test_bucket == b] = test_assigner[test_bucket == b]
            continue
        mt = (test_bucket == b)
        yhat_t[mt] = predict_nnls_log(Pt[mt], coefs, intercept)
        n_active = (coefs > 1e-6).sum()
        # Compute val-bucket MAPE for this bucket
        yh_v = predict_nnls_log(Pv[mv], coefs, intercept)
        ape_b = 100 * np.abs(yh_v - yv[mv]) / np.maximum(yv[mv], 1e-3)
        print(f"  bucket {b}: active={n_active}/{Pv.shape[1]}  intercept={intercept:.3f}  val_mape={ape_b.mean():.3f}%", flush=True)

    rng = np.random.default_rng(0)
    print("\n=== Stratified NNLS-log meta-blender ===")
    report(yhat_t, yt, "stratum_nnls", rng)

    out = pd.DataFrame({"design_name": t0.reset_index()["design_name"],
                        "net_name": t0.reset_index()["net_name"],
                        "y_true": yt,
                        "y_pred": yhat_t,
                        "bucket": test_bucket})
    out.to_csv(cfg.REPORTS_DIR / "stratum_nnls_test.csv", index=False)
    print(f"saved → {cfg.REPORTS_DIR / 'stratum_nnls_test.csv'}", flush=True)


if __name__ == "__main__":
    main()
