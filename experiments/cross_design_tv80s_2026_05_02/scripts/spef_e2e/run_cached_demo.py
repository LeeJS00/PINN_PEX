"""End-to-end demo using cached super-ensemble predictions for tv80s.

Steps:
  1. Load super_ensemble_test.csv (3169 nets, total_cap_pred)
  2. Load tv80s features_v3.parquet (compact_gnd, compact_total etc.)
  3. Load tv80s pair_features parquet (built separately)
  4. Load tv80s cuboid_arr.npz (for resistance)
  5. Decompose: total_pred → c_gnd_pred, c_cpl_pred per net
  6. Distribute: c_cpl_pred → per-pair coupling
  7. Compute analytic R per net
  8. Write predicted SPEF
  9. Compare to golden SPEF using compare_spef.py — report MAPE on total_cap, c_gnd, c_cpl, R
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent.parent
sys.path.insert(0, str(_WS))
sys.path.insert(0, str(_WS.parent.parent))  # PINNPEX root for src.evaluation

from configs import cfg
from pex_pipeline.compute_resistance import total_resistance_for_design
from pex_pipeline.decompose_caps import (
    assemble_net_records,
    distribute_cpl_to_pairs,
    load_pair_features_design,
    split_total_to_gnd_cpl,
)
from pex_pipeline.write_spef import LumpedSPEFWriter


GOLDEN_SPEF = Path("/home/jslee/projects/PEX_SSL/golden_data/spef_data/intel22/intel22_tv80s_f3_starrc.spef")


def main():
    out_spef = cfg.OUTPUT_DIR / "spef_e2e" / "tv80s_predicted_v1.spef"
    out_spef.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    # 1. Load predictions
    pred_csv = cfg.REPORTS_DIR / "super_ensemble_test.csv"
    pred = pd.read_csv(pred_csv)
    print(f"[1/9] Loaded {len(pred)} cached predictions from {pred_csv.name}")

    # 2. Load features (for compact ratio)
    feat_path = cfg.CACHE_DIR / "features_v3" / "intel22_tv80s_f3.parquet"
    feat = pd.read_parquet(feat_path)
    feat_idx = feat.set_index("net_name")
    pred = pred.merge(feat[["net_name", "compact_gnd_fF", "compact_total_fF"]],
                      on="net_name", how="left")
    print(f"[2/9] Joined feature columns. valid_compact={pred['compact_total_fF'].notna().sum()}")

    # 3. Load pair features
    pair_pq = cfg.CACHE_DIR / "pair_features" / "intel22_tv80s_f3.parquet"
    if pair_pq.exists():
        pair_groups = load_pair_features_design(pair_pq)
        print(f"[3/9] Loaded pair_features: {len(pair_groups)} target nets, "
              f"{sum(len(v) for v in pair_groups.values())} total pairs")
    else:
        pair_groups = {}
        print(f"[3/9] No pair_features cached — pair distribution disabled")

    # 4. Load cuboid_arr for resistance
    cuboid_npz = cfg.CACHE_DIR / "cuboid_arr" / "intel22_tv80s_f3.npz"
    print(f"[4/9] Computing analytic R from {cuboid_npz.name}...")
    R_per_net = total_resistance_for_design(cuboid_npz)
    print(f"      n_R={len(R_per_net)} median={np.median(list(R_per_net.values())):.2f} ohm")

    # 5. Decompose total → c_gnd + c_cpl using LGBM-predicted ratio
    total_pred = pred["y_pred"].to_numpy(np.float64)
    ratio_csv = cfg.OUTPUT_DIR / "spef_e2e" / "gnd_ratio_preds.csv"
    if ratio_csv.exists():
        rat = pd.read_csv(ratio_csv)
        rat_map = dict(zip(rat["net_name"], rat["y_pred_ratio"]))
        ratio_arr = np.array([rat_map.get(n, 0.36) for n in pred["net_name"]])
        c_gnd_pred = total_pred * ratio_arr
        c_cpl_pred = total_pred * (1.0 - ratio_arr)
        print(f"[5/9] Split via LGBM-ratio model: ratio_mean={ratio_arr.mean():.3f} median={np.median(ratio_arr):.3f}")
    else:
        c_gnd_pred, c_cpl_pred = split_total_to_gnd_cpl(pred, total_pred)
        print(f"[5/9] Split via compact ratio (no LGBM): c_gnd_mean={c_gnd_pred.mean():.4f}, c_cpl_mean={c_cpl_pred.mean():.4f}")

    # 6. Per-pair distribution
    n_with_pairs = 0
    for n in pred["net_name"]:
        if n in pair_groups:
            n_with_pairs += 1
    print(f"[6/9] Per-pair: {n_with_pairs}/{len(pred)} nets have pair_features ({100*n_with_pairs/len(pred):.1f}%)")

    # 7. R per net — prefer LGBM-predicted R if available, fallback to analytic
    r_csv = cfg.OUTPUT_DIR / "spef_e2e" / "total_r_preds.csv"
    if r_csv.exists():
        rdf = pd.read_csv(r_csv)
        r_map = dict(zip(rdf["net_name"], rdf["y_pred_R"]))
        total_r = np.array([r_map.get(n, R_per_net.get(n, 0.0)) for n in pred["net_name"]], dtype=np.float64)
        n_lgbm = sum(1 for n in pred["net_name"] if n in r_map)
        print(f"[7/9] R distribution (LGBM): mean={total_r.mean():.2f} median={np.median(total_r):.2f} "
              f"(LGBM-coverage={n_lgbm}/{len(pred)})")
    else:
        total_r = np.array([R_per_net.get(n, 0.0) for n in pred["net_name"]], dtype=np.float64)
        print(f"[7/9] R distribution (analytic): mean={total_r.mean():.2f} median={np.median(total_r):.2f}")

    # 8. Build records and write SPEF
    records = assemble_net_records(
        features_df=pred[["net_name"]],
        total_pred=total_pred,
        pair_groups=pair_groups,
        c_gnd_pred=c_gnd_pred,
        c_cpl_pred=c_cpl_pred,
        total_r=total_r,
    )
    writer = LumpedSPEFWriter(design_name="tv80s",
                               vendor="PINNPEX",
                               program="PINNPEX-EDA-cached")
    writer.write(out_spef, records)
    n_pairs_total = sum(len(r["pairs"]) for r in records)
    sz_kb = out_spef.stat().st_size / 1024
    print(f"[8/9] Wrote {out_spef} — {len(records)} D_NETs, {n_pairs_total} coupling pairs, {sz_kb:.1f} KB")

    # 9. Compare to golden
    print(f"[9/9] Comparing to golden {GOLDEN_SPEF.name}...")
    from src.evaluation.compare_spef import parse_spef_with_coordinates
    g_nets = parse_spef_with_coordinates(GOLDEN_SPEF)
    p_nets = parse_spef_with_coordinates(out_spef)

    common = sorted(set(g_nets.keys()) & set(p_nets.keys()))
    n_common = len(common)
    print(f"    common nets: {n_common}")

    def compare_metric(getter, label):
        g_vals = np.array([getter(g_nets[n]) for n in common])
        p_vals = np.array([getter(p_nets[n]) for n in common])
        ape = 100 * np.abs(p_vals - g_vals) / np.maximum(np.abs(g_vals), 1e-6)
        # Filter zero-golden cases for the "interesting" metric
        nz = g_vals > 1e-6
        ape_nz = ape[nz]
        return {
            "label": label,
            "n": int(nz.sum()),
            "mape_mean": float(ape_nz.mean()),
            "mape_median": float(np.median(ape_nz)),
            "mape_p90": float(np.percentile(ape_nz, 90)),
            "g_mean": float(g_vals.mean()),
            "p_mean": float(p_vals.mean()),
            "bias_mean": float(((p_vals - g_vals) / np.maximum(np.abs(g_vals), 1e-6))[nz].mean() * 100),
        }

    metrics = [
        compare_metric(lambda x: x["total_cap"], "total_cap"),
        compare_metric(lambda x: x["sum_gnd_cap"], "c_gnd"),
        compare_metric(lambda x: x["sum_cpl_cap"], "c_cpl_total"),
        compare_metric(lambda x: x["total_res"], "total_res"),
    ]
    print()
    print(f"{'metric':<14}{'n':>6}{'mape_mean':>12}{'mape_median':>13}{'p90':>10}{'bias':>10}{'g_mean':>10}{'p_mean':>10}")
    for m in metrics:
        print(f"  {m['label']:<12}{m['n']:>6d}  {m['mape_mean']:>9.3f}%  "
              f"{m['mape_median']:>10.3f}%  {m['mape_p90']:>7.2f}%  "
              f"{m['bias_mean']:+8.2f}%  {m['g_mean']:>8.3f}  {m['p_mean']:>8.3f}")

    # Save metrics
    df_m = pd.DataFrame(metrics)
    metrics_path = cfg.REPORTS_DIR / "spef_e2e_cached_metrics.csv"
    df_m.to_csv(metrics_path, index=False)
    print(f"\nsaved {metrics_path}")
    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
