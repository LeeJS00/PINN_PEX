"""summarize_cold_results.py — aggregate all per-(design, model) cold-start
summary JSONs into a single Markdown table.

Full cold-start pipeline timing (per design) — these stages are shared across
all models because the feature parquet is produced once by pex_cold.py and
reused. Pulled from the actual run logs.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COLD_DIR = ROOT / "TreePEX" / "outputs" / "cold_reports"

MODELS = ["treepex", "b1_xgb", "catboost", "mesh_pinn"]
DESIGNS = ["intel22_tv80s_f3", "intel22_nova_f3"]

# Shared per-design feature pipeline stages (seconds), from the run that
# produced each design's cold_features.parquet:
#   tv80s: /tmp/cold_tv80s_save_feats.log   (XGB-Tweedie proxy + parquet save)
#   nova:  /tmp/cold_nova_rerun.log         (XGB-Tweedie proxy + parquet save)
SHARED_STAGES = {
    "intel22_tv80s_f3": {
        "t_pdk_parse_s":  0.767,
        "t_def_parse_s":  3.965,
        "t_v3_features_s": 69.789,
        "t_v4_h3_features_s": 87.647,
    },
    "intel22_nova_f3": {
        "t_pdk_parse_s":  0.391,
        "t_def_parse_s":  93.673,
        "t_v3_features_s": 5607.129,
        "t_v4_h3_features_s": 2348.967,
    },
}


def load(d, m):
    p = COLD_DIR / f"{d}_{m}_summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def fmt_row(d, m, s):
    if s is None:
        return f"| {m} | {d} | — | — | — | — | — | — | — | — | — | — | — | — |"
    sh = SHARED_STAGES.get(d, {})
    pdk = sh.get("t_pdk_parse_s", 0)
    defp = sh.get("t_def_parse_s", 0)
    v3 = sh.get("t_v3_features_s", 0)
    v4 = sh.get("t_v4_h3_features_s", 0)
    pre = s.get("t_preprocess_s", 0)
    inf = s.get("t_inference_s", 0)
    spef = s.get("t_spef_write_s", 0)
    total = pdk + defp + v3 + v4 + pre + inf + spef
    return (
        f"| {m} | {d} | {s['n_nets_compared']:,} | "
        f"{s['MAPE_tot_med']:.3f}% | {s['MAPE_gnd_med']:.2f}% | {s['MAPE_cpl_med']:.2f}% | "
        f"{s['R2_tot']:.4f} | "
        f"{s['pred_chip_total_fF']:,.0f} (gold {s['gold_chip_total_fF']:,.0f}) | "
        f"{pdk:.2f} | {defp:.2f} | {v3:.2f} | {v4:.2f} | "
        f"{inf:.2f}{'' if pre <= 0.01 else ' + ' + f'{pre:.2f} prep'} | "
        f"{spef:.2f} | "
        f"**{total:.2f}** |"
    )


def main():
    print("# Cold-start results — full pipeline timing (seconds)\n")
    print("Shared stages (PDK / DEF / V3 / V4) are per design — produced once "
          "by `pex_cold.py` and re-used across all models via the cold-features "
          "parquet cache. Inference + SPEF + total are per (design, model).\n")
    print("| Model | Design | n | MAPE_tot | gnd | cpl | R²_tot | chip_total fF (vs gold) | "
          "pdk | def | v3 feat | v4 H3 feat | infer | spef write | **TOTAL** |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in MODELS:
        for d in DESIGNS:
            s = load(d, m)
            print(fmt_row(d, m, s))
    print()
    print("**Warm-start reference** (per memory / archived results, not measured here):")
    print()
    print("| Model | tv80s tot | nova tot |")
    print("|---|---:|---:|")
    print("| TreePEX 5-seed Tweedie XGBoost | 4.98 % | 5.28 % |")
    print("| pex_v3 B1 XGBoost (5-seed) | 5.31 % | 5.86 % |")
    print("| CatBoost-Tweedie (5-seed, this work) | n/a (newly trained) | n/a |")
    print("| pex_v3 mesh-curriculum PINN (5-seed best-step) | 6.26 % | n/a |")


if __name__ == "__main__":
    main()
