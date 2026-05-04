"""Meta-blender: positive Lasso on val log-predictions, trained on full val.

We log-transform predictions and the target so multiplicative model errors
become additive. Lasso on val auto-selects a sparse model subset. Predict
in log space then exp back.
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


def report(yhat, y, label):
    ape = 100 * np.abs(yhat - y) / np.maximum(y, 1e-3)
    print(f"  [{label}]  mean={ape.mean():.3f}%  median={np.median(ape):.3f}%  p90={np.percentile(ape, 90):.2f}%")


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
    print(f"Found {len(val_csvs)} val, {len(test_csvs)} test CSVs")

    val_sizes = {k: len(val_csvs[k]) for k in val_csvs if k in test_csvs}
    if not val_sizes:
        print("no overlap"); return

    nova_size = max(val_sizes.values())
    pool = sorted([k for k in val_sizes if val_sizes[k] == nova_size])
    print(f"Pool size (nova val): {len(pool)}")

    v0 = val_csvs[pool[0]]
    Pv = np.stack([val_csvs[k]["y_pred"].to_numpy() for k in pool], axis=1)
    yv = v0["y_true"].to_numpy()
    Pt = np.stack([test_csvs[k]["y_pred"].to_numpy() for k in pool], axis=1)
    t0 = test_csvs[pool[0]]
    yt = t0["y_true"].to_numpy()

    # Log-space + positive Lasso
    eps = 1e-4
    Lv = np.log(np.clip(Pv, eps, None))
    Lt = np.log(np.clip(Pt, eps, None))
    Yv = np.log(np.clip(yv, eps, None))
    mask = np.all(np.isfinite(Lv), axis=1) & np.isfinite(Yv)
    Lv = Lv[mask]; Yv = Yv[mask]

    from sklearn.linear_model import Lasso
    best = None
    for alpha in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]:
        try:
            mdl = Lasso(alpha=alpha, positive=True, max_iter=20000, fit_intercept=True)
            mdl.fit(Lv, Yv)
            yhat_v_log = mdl.predict(Lv)
            yhat_v = np.exp(yhat_v_log)
            yv_orig = np.exp(Yv)
            ape_v = 100 * np.abs(yhat_v - yv_orig) / np.maximum(yv_orig, 1e-3)
            n_active = (mdl.coef_ > 1e-6).sum()
            print(f"alpha={alpha:.0e}  active={n_active:>3}  val_mean_mape={ape_v.mean():.3f}%")
            if best is None or ape_v.mean() < best[0]:
                best = (ape_v.mean(), alpha, mdl)
        except Exception as e:
            print(f"alpha={alpha} failed: {e}")
    if best is None:
        print("no fit"); return
    val_mape, alpha, mdl = best
    print(f"\nBest alpha={alpha:.0e}  val_mape={val_mape:.3f}%")
    yhat_t = np.exp(mdl.predict(Lt))
    report(yhat_t, yt, f"meta_lasso test (alpha={alpha:.0e})")

    rng = np.random.default_rng(0)
    ape_t = 100 * np.abs(yhat_t - yt) / np.maximum(yt, 1e-3)
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, len(ape_t), len(ape_t))
        boots.append(ape_t[idx].mean())
    lo = np.percentile(boots, 2.5); hi = np.percentile(boots, 97.5)
    print(f"  Bootstrap 95% CI: [{lo:.3f}, {hi:.3f}]")

    out = pd.DataFrame({"design_name": t0.reset_index()["design_name"],
                        "net_name": t0.reset_index()["net_name"],
                        "y_true": yt,
                        "y_pred": yhat_t})
    out.to_csv(cfg.REPORTS_DIR / "meta_lasso_test.csv", index=False)
    print(f"saved → {cfg.REPORTS_DIR / 'meta_lasso_test.csv'}")


if __name__ == "__main__":
    main()
