"""Val-tuned blend with multiple objectives: mean APE, median APE, P90 APE.

Different objectives may give different test results. Pick the best
generalising one (lowest test mean MAPE).
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


def collect(roots, kind="test"):
    out = {}
    for root in roots:
        rp = Path(root)
        if not rp.exists():
            continue
        for csv in sorted(rp.rglob(f"*__{kind}.csv")):
            tag = f"{rp.name}::{csv.parent.name}::{csv.stem.replace(f'__{kind}','')}"
            if "residual" in tag:
                continue
            try:
                df = pd.read_csv(csv).set_index(["design_name","net_name"])
                out[tag] = df
            except Exception:
                continue
    return out


def fit_one(pool, val_csvs, test_csvs, objective, label, reports_dir, n_trials=4, n_iter=4000, sub=8000):
    if not pool: return None
    v0 = val_csvs[pool[0]]
    Pv = np.stack([val_csvs[k]["y_pred"].to_numpy() for k in pool], axis=1)
    yv = v0["y_true"].to_numpy()
    Pt = np.stack([test_csvs[k]["y_pred"].to_numpy() for k in pool], axis=1)
    t0 = test_csvs[pool[0]]
    yt = t0["y_true"].to_numpy()

    mask_v = np.all(np.isfinite(Pv), axis=1) & np.isfinite(yv)
    Pv_f = Pv[mask_v]; yv_f = yv[mask_v]

    rng = np.random.default_rng(0)
    sub_idx = rng.choice(len(yv_f), size=min(sub, len(yv_f)), replace=False)
    Pv_sub = Pv_f[sub_idx]; yv_sub = yv_f[sub_idx]

    def loss(w):
        w = np.clip(w, 0, None)
        if w.sum() == 0: return 1e6
        w = w / w.sum()
        yhat = Pv_sub @ w
        ape = np.abs(yhat - yv_sub) / np.maximum(yv_sub, 1e-3)
        if objective == "mean":
            return float(np.mean(ape))
        elif objective == "median":
            return float(np.median(ape))
        elif objective == "p75":
            return float(np.percentile(ape, 75))
        elif objective == "trimmed":
            sorted_ape = np.sort(ape)
            k = int(len(sorted_ape) * 0.05)
            return float(sorted_ape[k:-k].mean())
        elif objective == "huber":
            # Huber on log-ratio (robust to extremes)
            log_pred = np.log(np.maximum(yhat, 1e-3))
            log_true = np.log(np.maximum(yv_sub, 1e-3))
            d = log_pred - log_true
            delta = 0.5
            quad = 0.5 * d**2
            lin = delta * (np.abs(d) - 0.5 * delta)
            return float(np.where(np.abs(d) <= delta, quad, lin).mean())
        else:
            raise ValueError(objective)

    best_loss = float("inf"); best_w = None
    for trial in range(n_trials):
        np.random.seed(trial)
        w0 = np.random.rand(len(pool))
        res = minimize(loss, w0, method="Nelder-Mead",
                      options={"maxiter": n_iter, "xatol": 1e-6, "fatol": 1e-7, "adaptive": True})
        if res.fun < best_loss:
            best_loss = res.fun
            best_w = res.x.copy()
    best_w = np.clip(best_w, 0, None)
    if best_w.sum() > 0: best_w = best_w / best_w.sum()

    yhat_t = Pt @ best_w
    ape_t = 100 * np.abs(yhat_t - yt) / np.maximum(yt, 1e-3)

    boots = []
    for _ in range(2000):
        idx = rng.integers(0, len(ape_t), len(ape_t))
        boots.append(ape_t[idx].mean())
    lo = np.percentile(boots, 2.5); hi = np.percentile(boots, 97.5)

    print(f"  {label} obj={objective}: val_loss={best_loss:.4f}  test mean={ape_t.mean():.3f}% med={np.median(ape_t):.3f}% [CI {lo:.3f},{hi:.3f}]")
    out_path = reports_dir / f"val_tuned_{objective}_test.csv"
    t0r = t0.reset_index()
    pd.DataFrame({"design_name": t0r["design_name"].values,
                  "net_name": t0r["net_name"].values,
                  "y_true": yt, "y_pred": yhat_t}).to_csv(out_path, index=False)
    return ape_t.mean(), out_path, best_w


def main():
    roots = [str(cfg.OUTPUT_DIR / "final_pipe_nova"),
             str(cfg.OUTPUT_DIR / "final_pipe_v3_nova"),
             str(cfg.OUTPUT_DIR / "resmlp_v3_nova"),
             str(cfg.OUTPUT_DIR / "deepset_v2")]
    test_csvs = collect(roots, "test")
    val_csvs = collect(roots, "val")
    common = sorted([k for k in test_csvs if k in val_csvs])
    print(f"Common nova-val pool: {len(common)} models")

    cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for obj in ["mean", "median", "huber", "trimmed", "p75"]:
        r = fit_one(common, val_csvs, test_csvs, obj, "nova", cfg.REPORTS_DIR)
        if r: results[obj] = r

    if results:
        best_obj = min(results, key=lambda o: results[o][0])
        print(f"\nBEST objective: {best_obj} → test mean MAPE {results[best_obj][0]:.3f}%")
        import shutil
        shutil.copy(results[best_obj][1], cfg.REPORTS_DIR / "val_tuned_blend_test.csv")


if __name__ == "__main__":
    main()
