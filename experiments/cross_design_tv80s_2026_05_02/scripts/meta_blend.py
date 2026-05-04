"""Meta-learner stacking on val predictions of all nova-val models.

Uses Ridge / Lasso regression on val (with log target) to learn weighted
combination, then applies to test. Compares with simple ensembles.
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


def main():
    roots = [str(cfg.OUTPUT_DIR / "final_pipe_nova"),
             str(cfg.OUTPUT_DIR / "final_pipe_v3_nova"),
             str(cfg.OUTPUT_DIR / "resmlp_v3_nova"),
             str(cfg.OUTPUT_DIR / "deepset_v2")]
    test_csvs = collect(roots, "test")
    val_csvs = collect(roots, "val")
    common = sorted(set(test_csvs) & set(val_csvs))
    print(f"Common models (nova-val pool): {len(common)}")
    if not common: return

    # Build matrices
    v0 = val_csvs[common[0]]
    yv = v0["y_true"].to_numpy()
    Pv = np.stack([np.log(np.maximum(val_csvs[k]["y_pred"].to_numpy(), 1e-4)) for k in common], axis=1)

    t0 = test_csvs[common[0]]
    yt = t0["y_true"].to_numpy()
    Pt = np.stack([np.log(np.maximum(test_csvs[k]["y_pred"].to_numpy(), 1e-4)) for k in common], axis=1)

    yv_log = np.log(np.maximum(yv, 1e-4))

    print(f"Pv: {Pv.shape}, Pt: {Pt.shape}")

    # Subsample val to 10000 for speed
    rng = np.random.default_rng(0)
    sub_idx = rng.choice(len(yv), size=min(10000, len(yv)), replace=False)
    Pv_sub = Pv[sub_idx]; yv_log_sub = np.log(np.maximum(yv[sub_idx], 1e-4))

    # Ridge on log preds → log target
    from sklearn.linear_model import Ridge
    for name, Cls, kwargs in [
        ("Ridge alpha=1", Ridge, dict(alpha=1.0)),
        ("Ridge alpha=10", Ridge, dict(alpha=10.0)),
        ("Ridge alpha=100", Ridge, dict(alpha=100.0)),
        ("Ridge alpha=1000", Ridge, dict(alpha=1000.0)),
    ]:
        m = Cls(**kwargs)
        m.fit(Pv_sub, yv_log_sub)
        pred_t = np.exp(m.predict(Pt))
        ape = 100*np.abs(pred_t - yt)/np.maximum(yt, 1e-3)
        # Bootstrap CI
        boots = []
        for _ in range(2000):
            idx = rng.integers(0, len(ape), len(ape))
            boots.append(ape[idx].mean())
        lo = np.percentile(boots, 2.5); hi = np.percentile(boots, 97.5)
        print(f"  {name}: test mean MAPE={ape.mean():.3f}% [CI {lo:.3f},{hi:.3f}], median={np.median(ape):.3f}%")

    # Also try: Ridge on log preds + log of compact_total feature
    parquet_path = cfg.CACHE_DIR / "features_v3" / "intel22_tv80s_f3.parquet"
    if parquet_path.exists():
        feat_df = pd.read_parquet(parquet_path).set_index(["design_name","net_name"])
        feat_df = feat_df.reindex(t0.index)
        compact = np.log1p(feat_df["compact_total_fF"].to_numpy(np.float64))

        # also val features
        val_pq = cfg.CACHE_DIR / "features_v3" / "intel22_nova_f3.parquet"
        val_feat = pd.read_parquet(val_pq).set_index(["design_name","net_name"])
        val_feat = val_feat.reindex(v0.index)
        compact_v = np.log1p(val_feat["compact_total_fF"].to_numpy(np.float64))

        Pv_aug = np.column_stack([Pv, compact_v])
        Pt_aug = np.column_stack([Pt, compact])
        Pv_aug_sub = Pv_aug[sub_idx]

        for name, Cls, kwargs in [
            ("Ridge+compact alpha=10", Ridge, dict(alpha=10.0)),
        ]:
            m = Cls(**kwargs)
            m.fit(Pv_aug_sub, yv_log_sub)
            pred_t = np.exp(m.predict(Pt_aug))
            ape = 100*np.abs(pred_t - yt)/np.maximum(yt, 1e-3)
            print(f"  {name}: test mean MAPE={ape.mean():.3f}%, median={np.median(ape):.3f}%")


if __name__ == "__main__":
    main()
