#!/usr/bin/env python3
"""diagnose.py — break down current (Path-2 v7 parallel) gnd matched MAPE.

Reads the seed0 5-seed comparison CSV (`compare_seed0_report.csv`) and the
seed0 XGB CSV (matched-net set), then loads the per-net topology pkl.gz to
compute per-net dominant metal layer and segment count. Reports:

  - per-layer M1..M8 gnd MAPE breakdown for matched and unmatched nets
  - per-quartile (g_tot size) gnd MAPE
  - matched vs unmatched summary

Output: `diagnose_summary.json` + console table.
"""
from __future__ import annotations
import gzip
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path("/home/jslee/projects/PINNPEX")
sys.path.insert(0, str(_ROOT))

DESIGN = "intel22_tv80s_f3"
COMPARE_CSV = _ROOT / "pex_v3/output/spef_e2e_fast_v3/tv80s_5seed/compare_seed0_report.csv"
XGB_CSV = _ROOT / "pex_v3/output/baselines/B1_xgboost_real/seed0/eval_predictions_test.csv"
TOPO_DIR = Path("/data/PINNPEX/data/processed_v3/intel22") / DESIGN / "topology"
OUT_DIR = _ROOT / "pex_v3/joint_pareto/experiments/exp_007_sakurai_gnd"


def per_net_dominant_layer(topo_dir: Path) -> dict[str, tuple[str, int, float]]:
    """Return {net_name: (dom_layer, n_segs, total_length_um)}.

    Dominant layer = layer with most total wire length within the net's
    `global_segments` filtered to `target_net=True` (i.e., the net itself,
    not aggressors). If the topology doesn't tag target_net, we fall back to
    matching by net_name in the segment dict.
    """
    out: dict[str, tuple[str, int, float]] = {}
    paths = list(topo_dir.rglob("*topo_*.pkl.gz"))
    print(f"  scanning {len(paths)} topology files...", flush=True)
    for i, path in enumerate(paths):
        if i % 500 == 0 and i > 0:
            print(f"  ... {i}/{len(paths)}", flush=True)
        net_stem = path.name.replace(".pkl.gz", "")
        if "topo_" not in net_stem:
            continue
        net_name = net_stem.split("topo_")[-1]
        try:
            with gzip.open(path, "rb") as f:
                d = pickle.load(f)
        except Exception:
            continue
        # try global_segments first
        layer_len: dict[str, float] = defaultdict(float)
        n_segs = 0
        total_len = 0.0
        # if any segment carries net_name use it; else assume all are target.
        segs = d.get("global_segments", [])
        if not segs:
            continue
        # Probe first segment for net_name
        has_net_name = any("net_name" in s for s in segs[:5])
        target_name = None
        if has_net_name:
            for s in segs:
                if "net_name" in s:
                    target_name = s["net_name"]
                    break
        for s in segs:
            if s.get("type") != "WIRE":
                continue
            if has_net_name and s.get("net_name") not in (None, target_name):
                # aggressor segment, skip
                continue
            start = np.asarray(s["start"], dtype=np.float64)
            end = np.asarray(s["end"], dtype=np.float64)
            length = float(np.linalg.norm(end - start))
            if length < 1e-9:
                continue
            layer = str(s.get("layer", "m1")).lower()
            layer_len[layer] += length
            n_segs += 1
            total_len += length
        if total_len <= 0 or not layer_len:
            continue
        dom_layer = max(layer_len.items(), key=lambda kv: kv[1])[0]
        out[net_name] = (dom_layer, n_segs, total_len)
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(">>> loading compare CSV + XGB CSV")
    df = pd.read_csv(COMPARE_CSV)
    xgb = pd.read_csv(XGB_CSV)
    matched = set(xgb[xgb["design_name"] == DESIGN]["net_name"].astype(str))
    df["net"] = df["net"].astype(str)
    df["matched"] = df["net"].isin(matched)
    df["mape_gnd"] = (df["p_gnd"] - df["g_gnd"]).abs() / df["g_gnd"].clip(lower=1e-9) * 100.0
    df["mape_cpl"] = (df["p_cpl"] - df["g_cpl"]).abs() / df["g_cpl"].clip(lower=1e-9) * 100.0
    df["mape_tot"] = (df["p_tot"] - df["g_tot"]).abs() / df["g_tot"].clip(lower=1e-9) * 100.0

    print(">>> probing dominant layer per net (slow first time)")
    layer_cache = OUT_DIR / "per_net_layer_cache.json"
    if layer_cache.exists():
        print(f"  using cache {layer_cache}")
        cache = json.loads(layer_cache.read_text())
        per_net = {k: tuple(v) for k, v in cache.items()}
    else:
        per_net = per_net_dominant_layer(TOPO_DIR)
        layer_cache.write_text(json.dumps({k: list(v) for k, v in per_net.items()}, indent=1))

    # Join
    rows = []
    for _, row in df.iterrows():
        info = per_net.get(row["net"])
        if info is None:
            rows.append((None, 0, 0.0))
        else:
            rows.append(info)
    df["dom_layer"] = [r[0] for r in rows]
    df["n_segs"] = [r[1] for r in rows]
    df["total_len_um"] = [r[2] for r in rows]
    df["dom_layer"] = df["dom_layer"].fillna("unknown")

    summary: dict = {
        "total_nets": int(len(df)),
        "n_with_layer": int((df["dom_layer"] != "unknown").sum()),
    }

    # Matched vs unmatched
    matched_df = df[df["matched"]]
    unmatched_df = df[~df["matched"]]
    summary["matched_n"] = int(len(matched_df))
    summary["unmatched_n"] = int(len(unmatched_df))
    summary["matched_gnd_mean"] = float(matched_df["mape_gnd"].mean())
    summary["matched_gnd_median"] = float(matched_df["mape_gnd"].median())
    summary["unmatched_gnd_mean"] = float(unmatched_df["mape_gnd"].mean())

    # Per-layer breakdown (matched + unmatched separately)
    per_layer: dict = {}
    print("\n>>> Per-layer gnd MAPE (matched / unmatched):")
    print(f"{'layer':<8} {'mch_n':>6} {'mch_mean':>10} {'mch_med':>10} {'umch_n':>6} {'umch_mean':>10}")
    for L in ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "unknown"]:
        m_sub = matched_df[matched_df["dom_layer"] == L]
        u_sub = unmatched_df[unmatched_df["dom_layer"] == L]
        per_layer[L] = {
            "matched_n": int(len(m_sub)),
            "matched_gnd_mean": float(m_sub["mape_gnd"].mean()) if len(m_sub) else None,
            "matched_gnd_median": float(m_sub["mape_gnd"].median()) if len(m_sub) else None,
            "unmatched_n": int(len(u_sub)),
            "unmatched_gnd_mean": float(u_sub["mape_gnd"].mean()) if len(u_sub) else None,
        }
        print(f"{L:<8} {len(m_sub):>6d} "
              f"{(m_sub['mape_gnd'].mean() if len(m_sub) else 0):>10.2f} "
              f"{(m_sub['mape_gnd'].median() if len(m_sub) else 0):>10.2f} "
              f"{len(u_sub):>6d} "
              f"{(u_sub['mape_gnd'].mean() if len(u_sub) else 0):>10.2f}")
    summary["per_layer"] = per_layer

    # Per-quartile breakdown by g_tot (matched only — XGB invariant per net)
    print("\n>>> Per-quartile gnd MAPE (by g_tot, matched nets):")
    print(f"{'quartile':<10} {'n':>6} {'g_tot_med':>10} {'gnd_mean':>10} {'gnd_med':>10} {'cpl_mean':>10}")
    quartiles = matched_df["g_tot"].quantile([0.25, 0.5, 0.75]).values
    per_q: dict = {}
    for q_idx, (lo, hi, label) in enumerate([
        (-np.inf, quartiles[0], "Q1"),
        (quartiles[0], quartiles[1], "Q2"),
        (quartiles[1], quartiles[2], "Q3"),
        (quartiles[2], np.inf, "Q4"),
    ]):
        sub = matched_df[(matched_df["g_tot"] > lo) & (matched_df["g_tot"] <= hi)]
        per_q[label] = {
            "n": int(len(sub)),
            "g_tot_lo": float(lo) if lo != -np.inf else None,
            "g_tot_hi": float(hi) if hi != np.inf else None,
            "g_tot_median": float(sub["g_tot"].median()) if len(sub) else None,
            "gnd_mean": float(sub["mape_gnd"].mean()) if len(sub) else None,
            "gnd_median": float(sub["mape_gnd"].median()) if len(sub) else None,
            "cpl_mean": float(sub["mape_cpl"].mean()) if len(sub) else None,
        }
        print(f"{label:<10} {len(sub):>6d} "
              f"{(sub['g_tot'].median() if len(sub) else 0):>10.4f} "
              f"{(sub['mape_gnd'].mean() if len(sub) else 0):>10.2f} "
              f"{(sub['mape_gnd'].median() if len(sub) else 0):>10.2f} "
              f"{(sub['mape_cpl'].mean() if len(sub) else 0):>10.2f}")
    summary["per_quartile"] = per_q

    # Per-segment-count breakdown (extra signal)
    print("\n>>> By n_segs bucket (matched):")
    print(f"{'bucket':<10} {'n':>6} {'gnd_mean':>10} {'gnd_med':>10}")
    bucket_edges = [(0, 1), (2, 5), (6, 20), (21, 100), (101, 1_000_000)]
    per_b: dict = {}
    for lo, hi in bucket_edges:
        label = f"{lo}-{hi}" if hi < 1_000_000 else f"{lo}+"
        sub = matched_df[(matched_df["n_segs"] >= lo) & (matched_df["n_segs"] <= hi)]
        per_b[label] = {
            "n": int(len(sub)),
            "gnd_mean": float(sub["mape_gnd"].mean()) if len(sub) else None,
            "gnd_median": float(sub["mape_gnd"].median()) if len(sub) else None,
        }
        print(f"{label:<10} {len(sub):>6d} "
              f"{(sub['mape_gnd'].mean() if len(sub) else 0):>10.2f} "
              f"{(sub['mape_gnd'].median() if len(sub) else 0):>10.2f}")
    summary["per_n_segs"] = per_b

    out_path = OUT_DIR / "diagnose_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n>>> wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
