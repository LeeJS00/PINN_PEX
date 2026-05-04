"""Meta-blender via NNLS in log space (closed-form fast).

Solve: min ||Lv @ w + b - Yv||^2 s.t. w >= 0
where Lv = log(val_predictions), Yv = log(val_target).

NNLS only handles non-negative coefficients with no intercept; we add a column
of 1s and let it learn an offset (forced non-negative — fine since log val
is centered around log(mean) ~ -2 and bias is small).
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
    print(f"Found {len(val_csvs)} val, {len(test_csvs)} test CSVs", flush=True)

    val_sizes = {k: len(val_csvs[k]) for k in val_csvs if k in test_csvs}
    nova_size = max(val_sizes.values())
    pool = sorted([k for k in val_sizes if val_sizes[k] == nova_size])
    print(f"Pool size (nova val): {len(pool)}", flush=True)

    v0 = val_csvs[pool[0]]
    Pv = np.stack([val_csvs[k]["y_pred"].to_numpy() for k in pool], axis=1)
    yv = v0["y_true"].to_numpy()
    Pt = np.stack([test_csvs[k]["y_pred"].to_numpy() for k in pool], axis=1)
    t0 = test_csvs[pool[0]]
    yt = t0["y_true"].to_numpy()

    eps = 1e-4
    Lv = np.log(np.clip(Pv, eps, None))
    Lt = np.log(np.clip(Pt, eps, None))
    Yv = np.log(np.clip(yv, eps, None))

    mask = np.all(np.isfinite(Lv), axis=1) & np.isfinite(Yv)
    Lv = Lv[mask]; Yv = Yv[mask]; yv_orig = yv[mask]

    # NNLS with intercept column
    A = np.concatenate([Lv, np.ones((len(Lv), 1))], axis=1)
    print(f"NNLS on {A.shape}…", flush=True)
    w, rss = nnls(A, Yv, maxiter=A.shape[1] * 20)
    coefs = w[:-1]; intercept = w[-1]
    n_active = (coefs > 1e-6).sum()
    print(f"intercept={intercept:.4f}  active={n_active}/{len(coefs)}", flush=True)

    yhat_v = np.exp(Lv @ coefs + intercept)
    ape_v = 100 * np.abs(yhat_v - yv_orig) / np.maximum(yv_orig, 1e-3)
    print(f"  val mean MAPE: {ape_v.mean():.3f}%  median: {np.median(ape_v):.3f}%", flush=True)

    yhat_t = np.exp(Lt @ coefs + intercept)
    ape_t = 100 * np.abs(yhat_t - yt) / np.maximum(yt, 1e-3)
    print(f"  Test MAPE: mean={ape_t.mean():.3f}%  median={np.median(ape_t):.3f}%  p90={np.percentile(ape_t, 90):.2f}%", flush=True)

    rng = np.random.default_rng(0)
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, len(ape_t), len(ape_t))
        boots.append(ape_t[idx].mean())
    lo = np.percentile(boots, 2.5); hi = np.percentile(boots, 97.5)
    print(f"  Bootstrap 95% CI: [{lo:.3f}, {hi:.3f}]", flush=True)

    # Top weights
    order = np.argsort(-coefs)
    print("Top weights:")
    for i in order[:10]:
        if coefs[i] > 1e-6:
            print(f"  {coefs[i]:.4f}  {pool[i]}", flush=True)

    out = pd.DataFrame({"design_name": t0.reset_index()["design_name"],
                        "net_name": t0.reset_index()["net_name"],
                        "y_true": yt,
                        "y_pred": yhat_t})
    out.to_csv(cfg.REPORTS_DIR / "meta_nnls_test.csv", index=False)
    print(f"saved → {cfg.REPORTS_DIR / 'meta_nnls_test.csv'}", flush=True)


if __name__ == "__main__":
    main()
