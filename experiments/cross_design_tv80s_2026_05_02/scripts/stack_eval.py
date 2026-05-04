"""
Stack/blend GBDT + DeepSet predictions and evaluate on tv80s.

Reads CSVs under output/{gbdt_v2,deepset}/<model>/<tag>__test.csv
and computes blended predictions via:
  - simple median
  - simple mean
  - val-tuned ridge (alpha=1) on per-row predictions

Reports per-net MAPE statistics with bootstrap 95% CI.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent
sys.path.insert(0, str(_WS))

from configs import cfg
from src.data_loader import mape_per_net


def _load_pred(path: Path):
    """Returns dataframe indexed by (design_name, net_name)."""
    df = pd.read_csv(path)
    df = df.set_index(["design_name", "net_name"])
    return df


def collect_preds(roots, kind: str = "test") -> pd.DataFrame:
    """Collect all *__{kind}.csv files under roots into a wide DataFrame.

    Each prediction file becomes one column (named by relpath).
    """
    base = pd.DataFrame()
    for root in roots:
        for p in sorted(Path(root).rglob(f"*__{kind}.csv")):
            df = _load_pred(p)
            colname = f"{p.parent.name}__{p.stem.replace(f'__{kind}','')}"
            df = df.rename(columns={"y_pred": colname})
            df = df[[colname, "y_true"]]
            if base.empty:
                base = df
            else:
                base = base.join(df[[colname]], how="outer")
    return base


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    boots = []
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(np.mean(values[idx]))
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return lo, hi


def evaluate(name: str, y_true, y_pred, out: dict):
    ape = mape_per_net(y_true, y_pred)
    finite = ape[np.isfinite(ape)]
    lo, hi = bootstrap_ci(finite)
    rec = {
        "name": name,
        "n": int(len(finite)),
        "mape_mean":   float(np.mean(finite)),
        "mape_mean_lo": lo, "mape_mean_hi": hi,
        "mape_median": float(np.median(finite)),
        "mape_p90":    float(np.percentile(finite, 90)),
        "mape_p99":    float(np.percentile(finite, 99)),
    }
    out[name] = rec
    print(f"  {name:40s} n={rec['n']:5d} mean={rec['mape_mean']:.3f}% [CI {lo:.3f},{hi:.3f}] median={rec['mape_median']:.3f}% p90={rec['mape_p90']:.2f}% p99={rec['mape_p99']:.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+",
                    default=[str(cfg.OUTPUT_DIR / "gbdt_v2"), str(cfg.OUTPUT_DIR / "deepset")])
    ap.add_argument("--out", default=str(cfg.REPORTS_DIR / "stack_eval.json"))
    args = ap.parse_args()

    print("Collecting test predictions ...")
    test_preds = collect_preds(args.roots, kind="test")
    val_preds  = collect_preds(args.roots, kind="val")
    if test_preds.empty:
        print("No predictions found.")
        return

    print("Test pred files:", [c for c in test_preds.columns if c != "y_true"])
    pred_cols = [c for c in test_preds.columns if c != "y_true"]
    y_true_test = test_preds["y_true"].to_numpy()
    P_test = test_preds[pred_cols].to_numpy()
    P_test = np.where(np.isfinite(P_test), P_test, np.nan)

    out = {}

    # Per-model
    for c in pred_cols:
        evaluate(c, y_true_test, test_preds[c].to_numpy(), out)

    # Mean / median ensemble
    mask = np.all(np.isfinite(P_test), axis=1)
    if mask.sum() > 0:
        ens_mean = np.nanmean(P_test, axis=1)
        ens_med  = np.nanmedian(P_test, axis=1)
        evaluate("ENS_mean", y_true_test, ens_mean, out)
        evaluate("ENS_median", y_true_test, ens_med, out)

    # Val-tuned blend (positive weights summing to 1)
    if not val_preds.empty:
        val_pred_cols = [c for c in val_preds.columns if c != "y_true"]
        common = [c for c in pred_cols if c in val_pred_cols]
        if common:
            from scipy.optimize import minimize
            yv = val_preds["y_true"].to_numpy()
            Pv = val_preds[common].to_numpy()
            mask_v = np.all(np.isfinite(Pv), axis=1) & np.isfinite(yv)
            yv = yv[mask_v]; Pv = Pv[mask_v]

            def loss(w):
                w = np.clip(w, 0, None)
                if w.sum() == 0: return 1e6
                w = w / w.sum()
                yhat = Pv @ w
                ape = np.abs(yhat - yv) / np.maximum(yv, 1e-3)
                return float(np.mean(ape))
            w0 = np.ones(len(common)) / len(common)
            res = minimize(loss, w0, method="Nelder-Mead",
                            options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-5})
            w = np.clip(res.x, 0, None)
            w = w / w.sum() if w.sum() > 0 else np.ones(len(common)) / len(common)
            print(f"\nval-tuned blend weights:")
            for c, wi in zip(common, w):
                print(f"  {wi:.3f}  {c}")
            P_t_common = test_preds[common].to_numpy()
            mask_t = np.all(np.isfinite(P_t_common), axis=1)
            yhat = P_t_common @ w
            evaluate("ENS_blend_val", y_true_test, yhat, out)
            out["blend_weights"] = dict(zip(common, w.tolist()))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved", args.out)


if __name__ == "__main__":
    main()
