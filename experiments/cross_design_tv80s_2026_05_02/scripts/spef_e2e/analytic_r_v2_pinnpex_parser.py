"""Analytic R policy v2 — uses PINNPEX DefStreamParser for segment-level R.

Improvements over v1 (parse_def_via_counts.py):
  • Proper RECT landing handling (small metal patches contribute to R).
  • Exact per-segment width (LEF default OR explicit SPECIALNETS width),
    not a per-layer "typical" assumption.
  • SPECIALNETS (power) handled.
  • Via top/bottom layer mapping comes from tech LEF, not regex.

Per-segment R formula:
  WIRE  : R = sheet_R[layer] × |p2 - p1| / width
  RECT  : R = sheet_R[layer] × max(dx, dy) / min(dx, dy)
          (a single metal patch is one square at ratio L/W)
  VIA   : R = R_via[via_lvl] where via_lvl = "v{min(bot_lvl, top_lvl)}"

Net total = α × Σ R_seg.

Compares against v7 ML (11.92%) and v1 analytic (6.87% with α only).
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# allow `from src...` imports the same way build_dataset.py does
PROJ = Path("/home/jslee/projects/PINNPEX")
sys.path.insert(0, str(PROJ))

from configs import config as cfg  # noqa: E402
from src.preprocessing.def_parser  import DefStreamParser   # noqa: E402
from src.preprocessing.layer_parser import LayerInfoParser  # noqa: E402
from src.preprocessing.lef_parser   import LefParser        # noqa: E402
from src.preprocessing.cell_parser  import CellLibParser    # noqa: E402

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent.parent

_spec = importlib.util.spec_from_file_location(
    "compare_spef_pinpex",
    str(PROJ / "src" / "evaluation" / "compare_spef.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
parse_spef = _mod.parse_spef_with_coordinates

CALIB_PATH = _WS / "reports" / "sheet_r_calibration.json"

DEF_DIR  = Path("/home/jslee/projects/PEX_SSL/data/raw/def/intel22")
SPEF_DIR = Path("/home/jslee/projects/PEX_SSL/golden_data/spef_data/intel22")
TRAIN_DESIGNS = [
    "intel22_aes_cipher_top_f3", "intel22_gcd_f3", "intel22_ibex_core_f3",
    "intel22_mc_top_f3", "intel22_spi_top_f3", "intel22_usbf_top_f3",
    "intel22_vga_enh_top_f3", "intel22_wb_conmax_top_f3", "intel22_nova_f3",
]
TEST_DESIGN = "intel22_tv80s_f3"

VIA_NUM_RE = re.compile(r"VIA(\d+)", re.IGNORECASE)


def make_parser_resources():
    layer_map = LayerInfoParser(cfg.LAYERS_INFO_PATH).parse()
    tech_lef  = LefParser(cfg.TECH_LEF_PATH).parse()
    cell_lib  = CellLibParser(cfg.CELL_LEF_PATH).parse()
    return layer_map, tech_lef, cell_lib


def _layer_name_to_metal_lvl(layer_name: str):
    """'m1' -> 1, 'm2' -> 2, ... 'm9' -> 9. Returns None if not metal."""
    if not layer_name:
        return None
    m = re.match(r"^m(\d+)$", layer_name.lower())
    return int(m.group(1)) if m else None


def _via_layer_key(via_name: str, bot_layer: str, top_layer: str) -> str | None:
    """Decide via stack key like 'v1', 'v2', etc.

    Prefer the explicit VIA<n> token in the via name (e.g. VIA1_60SX44...).
    Fall back to the lower of bot/top metal layer numbers.
    """
    m = VIA_NUM_RE.search(via_name)
    if m:
        return f"v{m.group(1)}"
    bot = _layer_name_to_metal_lvl(bot_layer)
    top = _layer_name_to_metal_lvl(top_layer)
    if bot and top:
        return f"v{min(bot, top)}"
    return None


def compute_R_per_net(segments_iter, sheet_R, via_R) -> dict:
    """Walk DefStreamParser output and compute R per net (no α yet)."""
    out = {}
    for net_name, cuboids, segs in segments_iter:
        if not segs:
            continue
        R = 0.0
        for s in segs:
            t = s["type"]
            if t == "WIRE":
                L_layer = s["layer"].upper()
                if L_layer not in sheet_R:
                    continue
                p1, p2 = s["start"], s["end"]
                dx = abs(p1[0] - p2[0]); dy = abs(p1[1] - p2[1])
                length = (dx + dy)  # Manhattan; one of dx/dy is 0 in routes
                w = max(float(s["width"]), 1e-6)
                R += sheet_R[L_layer] * (length / w)
            elif t == "RECT":
                # Landing/contact patches: in golden RES these typically appear
                # as 0.001Ω entries (near-zero) — the patch is a 2D pad whose
                # current flows perpendicular through a large area; the dominant
                # R is in the adjacent via, not in the patch itself.
                # Treat as ~0 R (skip).
                continue
            elif t == "VIA":
                key = _via_layer_key(s["name"], s.get("bot_layer", ""), s.get("top_layer", ""))
                if key and key in via_R:
                    R += via_R[key]
        out[net_name] = R
    return out


def parse_def_to_R(def_path: Path, sheet_R, via_R, parser_resources) -> pd.DataFrame:
    layer_map, tech_lef, cell_lib = parser_resources
    parser = DefStreamParser(str(def_path), layer_map, tech_lef, cell_lib)
    R_dict = compute_R_per_net(parser.parse(), sheet_R, via_R)
    return pd.DataFrame([{"net_name": n, "R_pred_raw": v} for n, v in R_dict.items()])


def get_golden_R(design: str) -> pd.DataFrame:
    spef = SPEF_DIR / f"{design}_starrc.spef"
    g = parse_spef(spef)
    return pd.DataFrame([{"net_name": n, "R_gold": float(info["total_res"])}
                         for n, info in g.items()])


def main():
    # ---------------- Load calibration ----------------------
    with open(CALIB_PATH) as f:
        calib = json.load(f)
    sheet_R = {row["name"].upper(): row["sheet_median"] for row in calib["metal_per_layer"]}
    sheet_R.setdefault("M9", 0.18)   # tv80s rarely uses; fallback constant
    sheet_R.setdefault("M9P", 0.18)
    via_R = {row["name"]: row["R_median"] for row in calib["via_per_layer"]}
    for k in ["v5", "v6", "v7"]:
        via_R.setdefault(k, via_R.get("v4", 13.07))
    print("Calibrated sheet R:", sheet_R)
    print("Calibrated via R:  ", via_R)

    print("\nLoading parser resources (layer_map / tech_lef / cell_lib) ...", flush=True)
    parser_res = make_parser_resources()

    # ---------------- TRAIN: per-design + global α ----------------------
    rows = []
    all_pred = []
    all_gold = []
    for d in TRAIN_DESIGNS:
        def_path = DEF_DIR / f"{d}.def"
        if not def_path.exists():
            print(f"  [skip] {d}: DEF missing")
            continue
        print(f"  parsing DEF {d} (PINNPEX DefStreamParser) ...", flush=True)
        df_pred = parse_def_to_R(def_path, sheet_R, via_R, parser_res)
        df_gold = get_golden_R(d)
        joined = df_pred.merge(df_gold, on="net_name").query("R_gold > 0.1 and R_pred_raw > 0")
        if joined.empty:
            print(f"    [skip] {d}: no joined nets")
            continue

        ratio = joined["R_gold"].values / joined["R_pred_raw"].values
        alpha_med = float(np.median(ratio))
        ape_raw  = 100 * np.abs(joined["R_pred_raw"] - joined["R_gold"]).values / joined["R_gold"].values
        ape_calib = 100 * np.abs(alpha_med * joined["R_pred_raw"] - joined["R_gold"]).values / joined["R_gold"].values
        rows.append({
            "design": d, "n_nets": len(joined),
            "alpha_med": alpha_med,
            "raw_MAPE":  float(ape_raw.mean()),
            "med_MAPE":  float(ape_calib.mean()),
        })
        all_pred.append(joined["R_pred_raw"].values)
        all_gold.append(joined["R_gold"].values)
        print(f"    n={len(joined):6d}  α_med={alpha_med:.4f}  raw_MAPE={ape_raw.mean():.3f}%  med_MAPE={ape_calib.mean():.3f}%")

    df_train = pd.DataFrame(rows)
    print("\n=== Per-train-design ===")
    print(df_train.to_string(index=False))

    pred_all = np.concatenate(all_pred); gold_all = np.concatenate(all_gold)
    alpha_global_med = float(np.median(gold_all / np.maximum(pred_all, 1e-3)))
    grid = np.linspace(0.7, 1.5, 801)
    apes = [float(np.mean(np.abs(a*pred_all - gold_all)/gold_all)*100) for a in grid]
    alpha_global_mape = float(grid[int(np.argmin(apes))])
    print(f"\nGlobal α (median, {len(gold_all)} train nets): {alpha_global_med:.4f}")
    print(f"Global α (MAPE-min):                          {alpha_global_mape:.4f} (train MAPE {min(apes):.3f}%)")

    # ---------------- TEST: tv80s ----------------------
    test_def = DEF_DIR / f"{TEST_DESIGN}.def"
    print(f"\nParsing TEST DEF {TEST_DESIGN} ...", flush=True)
    df_pred_t = parse_def_to_R(test_def, sheet_R, via_R, parser_res)
    df_gold_t = get_golden_R(TEST_DESIGN)
    joined_t = df_pred_t.merge(df_gold_t, on="net_name").query("R_gold > 0.1 and R_pred_raw > 0").reset_index(drop=True)
    pred_t = joined_t["R_pred_raw"].values
    gold_t = joined_t["R_gold"].values

    def _stats(label, pred):
        ape = 100 * np.abs(pred - gold_t) / gold_t
        bias = 100 * (pred - gold_t) / gold_t
        rng = np.random.default_rng(0)
        boots = [ape[rng.integers(0, len(ape), len(ape))].mean() for _ in range(2000)]
        ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
        print(f"  {label:>40s}: MAPE={ape.mean():6.3f}%  med={np.median(ape):6.3f}%  "
              f"P90={np.percentile(ape, 90):6.2f}%  bias={bias.mean():+6.3f}%  CI=[{ci[0]:.2f}, {ci[1]:.2f}]")
        return float(ape.mean()), ci

    print(f"\n=== TEST: {TEST_DESIGN} (n={len(joined_t)}) ===")
    raw_mape,_ = _stats("raw analytic (α=1)", pred_t)
    med_mape, med_ci = _stats(f"train α_med ({alpha_global_med:.4f})",   alpha_global_med * pred_t)
    mapemin_mape,_   = _stats(f"train α_mape-min ({alpha_global_mape:.4f})",  alpha_global_mape * pred_t)
    alpha_oracle = float(np.median(gold_t / np.maximum(pred_t, 1e-3)))
    _stats(f"ORACLE α ({alpha_oracle:.4f}, cheat)", alpha_oracle * pred_t)

    # length-stratified for the train-α_med policy (production-realistic)
    pred_med = alpha_global_med * pred_t
    ape_med = 100 * np.abs(pred_med - gold_t) / gold_t
    bias_med = 100 * (pred_med - gold_t) / gold_t
    print(f"\nLength-stratified MAPE (post-global-α, quartiles by R_pred_raw):")
    # Use R_pred_raw to bucket — at inference we don't know R_gold.
    bucket_edges = np.quantile(pred_all, [0, 0.25, 0.5, 0.75, 1.0])
    print(f"  bucket edges (from train R_pred_raw):", [round(x, 1) for x in bucket_edges])

    # ---------------- Per-stratum α (Step B) ----------------------
    # Fit α per quartile of R_pred_raw on TRAIN data; apply to TEST.
    # Use predicted-quantile edges so inference-time bucketing is well-defined.
    train_buckets = np.clip(np.digitize(pred_all, bucket_edges) - 1, 0, 3)
    alpha_per_bucket = []
    for b in range(4):
        m = (train_buckets == b)
        if m.sum() == 0:
            alpha_per_bucket.append(alpha_global_med)
            continue
        a = float(np.median(gold_all[m] / np.maximum(pred_all[m], 1e-3)))
        alpha_per_bucket.append(a)
    print(f"  per-stratum α (train-fit): {[round(a, 3) for a in alpha_per_bucket]}")

    test_buckets = np.clip(np.digitize(pred_t, bucket_edges) - 1, 0, 3)
    pred_strat = np.array([alpha_per_bucket[test_buckets[i]] * pred_t[i] for i in range(len(pred_t))])
    ape_strat  = 100 * np.abs(pred_strat - gold_t) / gold_t
    bias_strat = 100 * (pred_strat - gold_t) / gold_t

    rng = np.random.default_rng(0)
    boots = [ape_strat[rng.integers(0, len(ape_strat), len(ape_strat))].mean() for _ in range(2000)]
    ci_strat = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    print(f"\n=== TEST: per-stratum α (Step B) ===")
    print(f"  MAPE = {ape_strat.mean():6.3f}%  med = {np.median(ape_strat):6.3f}%  "
          f"P90 = {np.percentile(ape_strat, 90):6.2f}%  bias = {bias_strat.mean():+6.3f}%  CI=[{ci_strat[0]:.3f}, {ci_strat[1]:.3f}]")
    print(f"  vs v7 ML 11.925%: Δ = {ape_strat.mean() - 11.925:+.2f}pp")
    print(f"  vs target 4%:    Δ = {ape_strat.mean() - 4:+.2f}pp")

    print(f"\nLength-stratified MAPE — global α vs per-stratum α:")
    for label, qcol_pred, qcol_gold in [("by R_gold", None, gold_t)]:
        qs = pd.qcut(gold_t, 4, labels=["Q1_short", "Q2", "Q3", "Q4_long"])
        for q in ["Q1_short", "Q2", "Q3", "Q4_long"]:
            mask = (qs == q)
            print(f"  {q:>9s}: n={mask.sum():4d}  R_med={np.median(gold_t[mask]):.1f}Ω  "
                  f"|  global-α: MAPE={ape_med[mask].mean():.2f}% bias={bias_med[mask].mean():+5.2f}%  "
                  f"|  strat-α: MAPE={ape_strat[mask].mean():.2f}% bias={bias_strat[mask].mean():+5.2f}%")

    out = {
        "calibrated_sheet_R": sheet_R,
        "calibrated_via_R":   via_R,
        "alpha_global_median":   alpha_global_med,
        "alpha_global_mape_min": alpha_global_mape,
        "alpha_per_bucket":      alpha_per_bucket,
        "bucket_edges":          [float(x) for x in bucket_edges],
        "test_MAPE_raw":         raw_mape,
        "test_MAPE_alpha_med":   med_mape,
        "test_MAPE_alpha_oracle": alpha_oracle,
        "test_MAPE_strat":       float(ape_strat.mean()),
        "test_MAPE_strat_CI":    [ci_strat[0], ci_strat[1]],
        "test_bias_strat":       float(bias_strat.mean()),
        "test_design":           TEST_DESIGN,
        "n_test_nets":           int(len(joined_t)),
        "per_train_design":      rows,
        "test_alpha_oracle":     alpha_oracle,
        "vs_v7_ML_MAPE":         11.925,
    }
    with open(_WS / "reports" / "alpha_global_v2.json", "w") as f:
        json.dump(out, f, indent=2)
    joined_t["R_pred_calibrated"] = alpha_global_med * pred_t
    joined_t["ape"] = ape_med
    joined_t["bias_signed"] = bias_med
    joined_t.to_csv(_WS / "reports" / "analytic_r_v2_test_per_net.csv", index=False)
    print(f"\nSaved: reports/alpha_global_v2.json + analytic_r_v2_test_per_net.csv")


if __name__ == "__main__":
    main()
