"""Val-tuned positive-weighted blend across all 70 models. Tries both nova-val
and ibex-val pools, picks the better."""
from __future__ import annotations

import shutil
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


def fit_blend(pool, val_csvs, test_csvs, label, reports_dir, n_trials=3, n_iter=3000, val_subsample=5000):
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
    sub_idx = rng.choice(len(yv_f), size=min(val_subsample, len(yv_f)), replace=False)
    Pv_sub = Pv_f[sub_idx]; yv_sub = yv_f[sub_idx]

    def loss(w):
        w = np.clip(w, 0, None)
        if w.sum() == 0: return 1e6
        w = w / w.sum()
        yhat = Pv_sub @ w
        ape = np.abs(yhat - yv_sub) / np.maximum(yv_sub, 1e-3)
        return float(np.mean(ape))

    best_loss = float("inf"); best_w = None
    for trial in range(n_trials):
        np.random.seed(trial)
        w0 = np.random.rand(len(pool))
        res = minimize(loss, w0, method="Nelder-Mead",
                      options={"maxiter": n_iter, "xatol": 1e-5, "fatol": 1e-6, "adaptive": True})
        if res.fun < best_loss:
            best_loss = res.fun
            best_w = res.x.copy()
    best_w = np.clip(best_w, 0, None)
    if best_w.sum() > 0: best_w = best_w / best_w.sum()

    print(f"\n=== {label} pool ({len(pool)} models) ===")
    print(f"  best val mean APE on sub: {best_loss:.4f}")

    yhat_t = Pt @ best_w
    ape_t = 100 * np.abs(yhat_t - yt) / np.maximum(yt, 1e-3)
    mean_t = ape_t.mean()
    print(f"  Test MAPE: mean={mean_t:.3f}% median={np.median(ape_t):.3f}% p90={np.percentile(ape_t, 90):.2f}%")

    boots = []
    for _ in range(2000):
        idx = rng.integers(0, len(ape_t), len(ape_t))
        boots.append(ape_t[idx].mean())
    lo = np.percentile(boots, 2.5); hi = np.percentile(boots, 97.5)
    print(f"  Bootstrap 95% CI: [{lo:.3f}, {hi:.3f}]")

    order = np.argsort(-best_w)
    for i in order[:8]:
        if best_w[i] > 1e-3:
            print(f"    {best_w[i]:.4f}  {pool[i]}")

    out_path = reports_dir / f"val_tuned_blend_{label}_test.csv"
    t0r = t0.reset_index()
    pd.DataFrame({"design_name": t0r["design_name"].values,
                  "net_name": t0r["net_name"].values,
                  "y_true": yt,
                  "y_pred": yhat_t}).to_csv(out_path, index=False)
    return mean_t, out_path


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
    test_csvs = collect(roots, "test")
    val_csvs = collect(roots, "val")
    print(f"Found {len(test_csvs)} test CSVs, {len(val_csvs)} val CSVs")

    val_sizes = {k: len(val_csvs[k]) for k in val_csvs if k in test_csvs}
    if not val_sizes:
        print("no overlap"); return
    sizes = list(val_sizes.values())
    nova_size = max(sizes)
    ibex_size = min(sizes)
    print(f"val sizes: ibex={ibex_size}, nova={nova_size}")

    common_ibex = sorted([k for k in val_sizes if val_sizes[k] == ibex_size])
    common_nova = sorted([k for k in val_sizes if val_sizes[k] == nova_size])

    cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    res_nova = fit_blend(common_nova, val_csvs, test_csvs, "nova", cfg.REPORTS_DIR)
    res_ibex = fit_blend(common_ibex, val_csvs, test_csvs, "ibex", cfg.REPORTS_DIR)
    print()
    if res_nova and res_ibex:
        winner = res_nova if res_nova[0] < res_ibex[0] else res_ibex
        winner_label = "nova" if res_nova[0] < res_ibex[0] else "ibex"
        print(f"WINNER: {winner_label}-val pool ({winner[0]:.3f}%) → canonical val_tuned_blend_test.csv")
        shutil.copy(winner[1], cfg.REPORTS_DIR / "val_tuned_blend_test.csv")


if __name__ == "__main__":
    main()
