"""2D stratified blend: bucket by (predicted_cap, agg_total_count).

Adds a second stratification dimension orthogonal to predicted-cap. Per
2D-bucket Nelder-Mead positive-weight blend on val MAPE, applied to test
using the same boundary scheme.
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


def fit_nelder_mape(Pv, yv, n_trials=3, n_iter=2000, max_n=4000, seed_base=0):
    rng = np.random.default_rng(seed_base)
    n = len(yv)
    if n > max_n:
        idx = rng.choice(n, size=max_n, replace=False)
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
    if best_w.sum() > 0:
        best_w = best_w / best_w.sum()
    return best_w, best


def main():
    nb_cap = int(__import__("os").environ.get("NB_CAP", 6))
    nb_agg = int(__import__("os").environ.get("NB_AGG", 4))
    print(f"Bucketing: nb_cap={nb_cap}, nb_agg={nb_agg} → {nb_cap*nb_agg} 2D buckets", flush=True)

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

    # Load agg_total_count from features
    fcache = cfg.CACHE_DIR / "features_v3"
    val_design = "intel22_nova_f3"
    test_design = "intel22_tv80s_f3"
    feat_v = pd.read_parquet(fcache / f"{val_design}.parquet")[["net_name", "agg_total_count"]]
    feat_t = pd.read_parquet(fcache / f"{test_design}.parquet")[["net_name", "agg_total_count"]]

    val_idx = v0.reset_index()[["design_name", "net_name"]]
    test_idx = t0.reset_index()[["design_name", "net_name"]]

    val_merge = val_idx.merge(feat_v, on="net_name", how="left")
    test_merge = test_idx.merge(feat_t, on="net_name", how="left")
    agg_v = val_merge["agg_total_count"].fillna(0).to_numpy()
    agg_t = test_merge["agg_total_count"].fillna(0).to_numpy()
    print(f"agg val [min/median/max]: {agg_v.min()}/{np.median(agg_v):.0f}/{agg_v.max()}", flush=True)
    print(f"agg test [min/median/max]: {agg_t.min()}/{np.median(agg_t):.0f}/{agg_t.max()}", flush=True)

    eps = 1e-4
    val_assigner = np.exp(np.log(np.clip(Pv, eps, None)).mean(axis=1))
    test_assigner = np.exp(np.log(np.clip(Pt, eps, None)).mean(axis=1))

    qs_cap = np.linspace(0, 1, nb_cap + 1)[1:-1]
    qs_agg = np.linspace(0, 1, nb_agg + 1)[1:-1]
    bnd_cap = np.quantile(val_assigner, qs_cap)
    bnd_agg = np.quantile(agg_v, qs_agg)
    val_b_cap = np.digitize(val_assigner, bnd_cap)
    val_b_agg = np.digitize(agg_v, bnd_agg)
    test_b_cap = np.digitize(test_assigner, bnd_cap)
    test_b_agg = np.digitize(agg_t, bnd_agg)
    val_b = val_b_cap * nb_agg + val_b_agg
    test_b = test_b_cap * nb_agg + test_b_agg

    yhat_t = np.zeros_like(yt)
    for b in range(nb_cap * nb_agg):
        mv = val_b == b
        if mv.sum() < 30:
            # fallback to global
            w, vl = fit_nelder_mape(Pv, yv)
        else:
            w, vl = fit_nelder_mape(Pv[mv], yv[mv])
        if w is None:
            yhat_t[test_b == b] = test_assigner[test_b == b]
            continue
        mt = test_b == b
        yhat_t[mt] = Pt[mt] @ w
        if mv.sum() >= 30 and (b % max(1, (nb_cap*nb_agg)//8) == 0 or mt.sum() > 0):
            print(f"  bkt {b:3d} (cap={b//nb_agg},agg={b%nb_agg}): val_n={mv.sum():>5} test_n={mt.sum():>4} val_mape={vl*100:.2f}%", flush=True)

    rng = np.random.default_rng(0)
    ape = 100 * np.abs(yhat_t - yt) / np.maximum(yt, 1e-3)
    print(f"\n=== 2D stratified MAPE blend ({nb_cap}×{nb_agg}={nb_cap*nb_agg} buckets) ===")
    print(f"  mean={ape.mean():.4f}%  median={np.median(ape):.3f}%  p90={np.percentile(ape, 90):.2f}%")
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, len(ape), len(ape))
        boots.append(ape[idx].mean())
    print(f"  CI=[{np.percentile(boots, 2.5):.3f}, {np.percentile(boots, 97.5):.3f}]")

    out_path = cfg.REPORTS_DIR / f"stratum_2d_c{nb_cap}_a{nb_agg}_test.csv"
    out = pd.DataFrame({"design_name": t0.reset_index()["design_name"],
                        "net_name": t0.reset_index()["net_name"],
                        "y_true": yt, "y_pred": yhat_t,
                        "bucket_cap": test_b_cap, "bucket_agg": test_b_agg})
    out.to_csv(out_path, index=False)
    print(f"saved → {out_path}")


if __name__ == "__main__":
    main()
