#!/usr/bin/env python3
"""
aggregate_ablation_summary.py — anti-overclaim 5-seed aggregator.

For an ablation `--variant`, walks all `seed*/summary.json` files plus
all `seed*/eval_logger.parquet` files and produces:

    pex_v3/output/ablation/<variant>/aggregate.json

containing:
    - per-seed best-step + last-step metrics (valid total, test total,
      valid gnd, valid cpl, test gnd, test cpl)
    - across-seed median, stdev, range (min, max)  for each
    - Cohen's d vs `--baseline` (Hybrid mesh-curriculum 5-seed by default)
    - Paired Mann-Whitney U on per-net |relative error| (valid + test)
      vs the baseline (matches on net_id; rejects nets without baseline match)
    - Bootstrap 95% CI on test total median (1000 resamples; BCa where SciPy
      is available, percentile fallback otherwise)

Per project rule #2: a 5-seed run + MWU + Cohen's d is the ONLY way to
make an "improvement claim". Single-seed BEST is suspicion, not signal.

Usage:
    python3 pex_v3/scripts/aggregate_ablation_summary.py \\
        --variant HybridPexV3MeshAdditive \\
        --baseline HybridPexV3Mesh

Self-test (no real ablation yet):
    python3 pex_v3/scripts/aggregate_ablation_summary.py \\
        --variant-dir pex_v3/output/phase1_mesh_5seed \\
        --baseline-dir pex_v3/output/phase1_mesh_5seed
    → expects near-zero Cohen's d, MWU p≈1
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "pex_v3"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluation.seed_aggregator import (  # noqa: E402
    cohens_d, cohens_d_label, bootstrap_median_ci, mann_whitney_u_two_sided,
)
from src.utils.eval_logger import read_eval_parquet, EVAL_LOGGER_SCHEMA, add_error_columns  # noqa: E402


ABLATION_ROOT = _PROJECT_ROOT / "pex_v3" / "output" / "ablation"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="5-seed ablation summary + MWU/Cohen's d vs baseline")
    p.add_argument("--variant", type=str, default=None,
                   help="Variant name → resolves to pex_v3/output/ablation/<variant>/")
    p.add_argument("--baseline", type=str, default="HybridPexV3Mesh",
                   help="Baseline variant name → pex_v3/output/ablation/<baseline>/ "
                        "(falls back to phase1_mesh_5seed if not present in ablation/)")
    p.add_argument("--variant-dir", type=Path, default=None,
                   help="Override the variant directory (skips name resolution)")
    p.add_argument("--baseline-dir", type=Path, default=None,
                   help="Override the baseline directory (for self-test)")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--metric", choices=["valid_total", "test_total"], default="test_total",
                   help="Primary metric for Cohen's d / MWU on across-seed values")
    return p.parse_args()


def _resolve_dir(name: Optional[str], override: Optional[Path], fallback_phase1: bool = False) -> Path:
    if override is not None:
        return override
    if name is None:
        raise SystemExit("Must pass --variant or --variant-dir (and --baseline or --baseline-dir)")
    cand = ABLATION_ROOT / name
    if cand.exists():
        return cand
    # Fallback: legacy locked baseline path
    if fallback_phase1:
        legacy = _PROJECT_ROOT / "pex_v3" / "output" / "phase1_mesh_5seed"
        if legacy.exists():
            return legacy
    raise SystemExit(f"Could not resolve directory for {name!r} (looked at {cand})")


# ---------------------------------------------------------------------------
# Per-seed extraction
# ---------------------------------------------------------------------------


def _read_summary_metrics(seed_dir: Path) -> Optional[dict]:
    """Pull best/last metrics out of summary.json."""
    sf = seed_dir / "summary.json"
    if not sf.exists():
        return None
    with open(sf) as f:
        s = json.load(f)
    fv = s.get("final_valid", {})
    ft = s.get("final_test", {})
    return {
        "seed": s.get("seed", int(seed_dir.name.replace("seed", ""))),
        # last-step (preferred for paper claims per A1 audit)
        "last_valid_total": float(fv.get("total_mape_median", float("nan"))),
        "last_valid_gnd":   float(fv.get("gnd_mape_median", float("nan"))),
        "last_valid_cpl":   float(fv.get("cpl_mape_median", float("nan"))),
        "last_test_total":  float(ft.get("total_mape_median", float("nan"))),
        "last_test_gnd":    float(ft.get("gnd_mape_median", float("nan"))),
        "last_test_cpl":    float(ft.get("cpl_mape_median", float("nan"))),
        # best-step (diagnostic)
        "best_epoch":            int(s.get("best_epoch", -1)),
        "best_valid_total":      float(s.get("best_valid_total_mape", float("nan"))),
        "best_valid_gnd":        float(s.get("best_valid_gnd_mape", float("nan"))),
        "best_valid_cpl":        float(s.get("best_valid_cpl_mape", float("nan"))),
        "elapsed_train_sec":     float(s.get("elapsed_train_sec", float("nan"))),
    }


def collect_per_seed(variant_dir: Path) -> pd.DataFrame:
    rows = []
    for sd in sorted(p for p in variant_dir.glob("seed*") if p.is_dir()):
        m = _read_summary_metrics(sd)
        if m is not None:
            rows.append(m)
    if not rows:
        raise SystemExit(f"No seed*/summary.json found under {variant_dir}")
    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)


def across_seed_stats(per_seed: pd.DataFrame, metric_cols: list[str]) -> dict:
    out = {}
    for c in metric_cols:
        v = per_seed[c].to_numpy(dtype=float)
        v = v[~np.isnan(v)]
        if len(v) == 0:
            out[c] = {"median": float("nan"), "mean": float("nan"),
                      "stdev": float("nan"), "min": float("nan"), "max": float("nan"),
                      "n": 0}
            continue
        out[c] = {
            "median": float(np.median(v)),
            "mean": float(np.mean(v)),
            "stdev": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            "min": float(np.min(v)),
            "max": float(np.max(v)),
            "n": int(len(v)),
        }
    return out


# ---------------------------------------------------------------------------
# Per-net paired MWU (loads eval_logger.parquet from each seed)
# ---------------------------------------------------------------------------


def _load_seed_eval(seed_dir: Path, split: str = "test") -> Optional[pd.DataFrame]:
    """Try eval_logger_{split}.parquet first, then eval_logger.parquet, then csv."""
    for cand_name in (f"eval_logger_{split}.parquet", "eval_logger.parquet"):
        cand = seed_dir / cand_name
        try:
            df = read_eval_parquet(cand)
            df = add_error_columns(df)
            return df
        except (FileNotFoundError, ValueError):
            continue
    return None


def average_predictions_across_seeds(variant_dir: Path, split: str = "test") -> Optional[pd.DataFrame]:
    """Build an across-seed mean-prediction DataFrame for paired MWU.

    Returns None if no eval_logger files exist (legacy 5-seed dirs).
    """
    seed_dfs = []
    for sd in sorted(p for p in variant_dir.glob("seed*") if p.is_dir()):
        df = _load_seed_eval(sd, split=split)
        if df is None:
            continue
        seed_dfs.append(df)
    if not seed_dfs:
        return None
    cat = pd.concat(seed_dfs, ignore_index=True)
    grouped = cat.groupby("net_id", as_index=False).agg({
        "design": "first",
        "net_name": "first",
        "fanout": "first",
        "gnd_pred": "mean",
        "cpl_pred": "mean",
        "gnd_gold": "first",
        "cpl_gold": "first",
        "dominant_layer": "first",
    })
    grouped["total_pred"] = grouped["gnd_pred"] + grouped["cpl_pred"]
    grouped["total_gold"] = grouped["gnd_gold"] + grouped["cpl_gold"]
    return add_error_columns(grouped)


def paired_mwu_vs_baseline(variant_df: pd.DataFrame, baseline_df: pd.DataFrame,
                           col: str = "total_rel_err") -> dict:
    """Wilcoxon signed-rank-style paired MWU on common net_ids.

    Strictly: uses scipy.stats.wilcoxon (paired) when available; else falls
    back to mannwhitneyu (unpaired) on aligned values.
    """
    merged = variant_df[["net_id", col]].merge(
        baseline_df[["net_id", col]], on="net_id", suffixes=("_v", "_b"),
    )
    n = len(merged)
    if n == 0:
        return {"n_paired": 0, "p_value": float("nan"), "test": "none"}
    a = merged[f"{col}_v"].to_numpy(dtype=float)
    b = merged[f"{col}_b"].to_numpy(dtype=float)

    try:
        from scipy.stats import wilcoxon
        # zero_method='wilcox' drops zero-diffs; same as default
        res = wilcoxon(a, b, alternative="two-sided")
        return {
            "n_paired": int(n),
            "test": "wilcoxon_signed_rank",
            "statistic": float(res.statistic),
            "p_value": float(res.pvalue),
            "median_diff_v_minus_b": float(np.median(a - b)),
            "mean_diff_v_minus_b": float(np.mean(a - b)),
            "median_a": float(np.median(a)),
            "median_b": float(np.median(b)),
        }
    except Exception:
        mwu = mann_whitney_u_two_sided(a, b)
        return {
            "n_paired": int(n),
            "test": "mannwhitneyu_unpaired_fallback",
            "statistic": mwu["U"],
            "p_value": mwu["p_value"],
            "median_diff_v_minus_b": float(np.median(a) - np.median(b)),
        }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    variant_dir = _resolve_dir(args.variant, args.variant_dir, fallback_phase1=False)
    baseline_dir = _resolve_dir(args.baseline, args.baseline_dir, fallback_phase1=True)
    out_dir = variant_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f">>> variant : {variant_dir}")
    print(f">>> baseline: {baseline_dir}")

    metric_cols = [
        "last_valid_total", "last_valid_gnd", "last_valid_cpl",
        "last_test_total", "last_test_gnd", "last_test_cpl",
        "best_valid_total", "best_valid_gnd", "best_valid_cpl",
        "best_epoch", "elapsed_train_sec",
    ]

    var_per_seed = collect_per_seed(variant_dir)
    base_per_seed = collect_per_seed(baseline_dir)
    print(f">>> variant  seeds: {sorted(var_per_seed['seed'].tolist())}")
    print(f">>> baseline seeds: {sorted(base_per_seed['seed'].tolist())}")

    var_stats = across_seed_stats(var_per_seed, metric_cols)
    base_stats = across_seed_stats(base_per_seed, metric_cols)

    # Across-seed Cohen's d on the chosen metric
    metric_name = "last_test_total" if args.metric == "test_total" else "last_valid_total"
    a = var_per_seed[metric_name].to_numpy(dtype=float)
    b = base_per_seed[metric_name].to_numpy(dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    d = cohens_d(a, b) if (len(a) > 0 and len(b) > 0) else float("nan")

    # Across-seed MWU (n=5 vs n=5)
    across_mwu = mann_whitney_u_two_sided(a, b) if (len(a) and len(b)) else None

    # Bootstrap CI on variant test_total across seeds
    boot_med, boot_lo, boot_hi = bootstrap_median_ci(
        a, n_resamples=args.n_bootstrap, confidence=0.95, seed=0, method="bca",
    )

    # Per-net paired MWU (test split) — needs eval_logger.parquet from BOTH dirs
    var_per_net = average_predictions_across_seeds(variant_dir, split="test")
    base_per_net = average_predictions_across_seeds(baseline_dir, split="test")
    if var_per_net is not None and base_per_net is not None:
        per_net_test = paired_mwu_vs_baseline(var_per_net, base_per_net, col="total_rel_err")
        per_net_test_gnd = paired_mwu_vs_baseline(var_per_net, base_per_net, col="gnd_rel_err")
        per_net_test_cpl = paired_mwu_vs_baseline(var_per_net, base_per_net, col="cpl_rel_err")
    else:
        msg = ("eval_logger parquet not present in one of the dirs; "
               "per-net paired MWU skipped (across-seed MWU still reported)")
        per_net_test = {"skipped": True, "reason": msg}
        per_net_test_gnd = {"skipped": True}
        per_net_test_cpl = {"skipped": True}
        print(f"  [warn] {msg}")

    var_per_net_valid = average_predictions_across_seeds(variant_dir, split="valid")
    base_per_net_valid = average_predictions_across_seeds(baseline_dir, split="valid")
    if var_per_net_valid is not None and base_per_net_valid is not None:
        per_net_valid = paired_mwu_vs_baseline(var_per_net_valid, base_per_net_valid, col="total_rel_err")
    else:
        per_net_valid = {"skipped": True}

    aggregate = {
        "variant": args.variant or variant_dir.name,
        "baseline": args.baseline or baseline_dir.name,
        "variant_dir": str(variant_dir),
        "baseline_dir": str(baseline_dir),
        "primary_metric": metric_name,
        "variant_per_seed": var_per_seed.to_dict(orient="records"),
        "baseline_per_seed": base_per_seed.to_dict(orient="records"),
        "variant_across_seed": var_stats,
        "baseline_across_seed": base_stats,
        "across_seed": {
            "metric": metric_name,
            "variant_values": a.tolist(),
            "baseline_values": b.tolist(),
            "cohens_d": float(d),
            "cohens_d_label": cohens_d_label(d),
            "mwu": across_mwu,
            "bootstrap_95ci_variant": {
                "median": boot_med, "ci95_low": boot_lo, "ci95_high": boot_hi,
                "n_resamples": int(args.n_bootstrap),
            },
        },
        "per_net": {
            "test_total": per_net_test,
            "test_gnd": per_net_test_gnd,
            "test_cpl": per_net_test_cpl,
            "valid_total": per_net_valid,
        },
    }

    out_path = out_dir / "aggregate.json"
    with open(out_path, "w") as f:
        json.dump(aggregate, f, indent=2, default=str)
    print()
    print(f">>> wrote {out_path}")

    # Console summary
    print()
    print("=" * 60)
    print(f"VARIANT  ({args.variant or variant_dir.name})")
    for c in ("last_test_total", "last_test_gnd", "last_test_cpl",
              "last_valid_total", "best_valid_total"):
        s = var_stats[c]
        print(f"  {c:<22} median={s['median']*100:.3f}%  stdev={s['stdev']*100:.3f}pp  "
              f"min={s['min']*100:.3f}%  max={s['max']*100:.3f}%  n={s['n']}")
    print()
    print(f"BASELINE ({args.baseline or baseline_dir.name})")
    for c in ("last_test_total", "last_test_gnd", "last_test_cpl",
              "last_valid_total", "best_valid_total"):
        s = base_stats[c]
        print(f"  {c:<22} median={s['median']*100:.3f}%  stdev={s['stdev']*100:.3f}pp  "
              f"min={s['min']*100:.3f}%  max={s['max']*100:.3f}%  n={s['n']}")
    print()
    print(f"ACROSS-SEED ({metric_name}): "
          f"Cohen's d = {d:.3f} ({cohens_d_label(d)}); "
          f"MWU p = {across_mwu['p_value']:.4f}" if across_mwu else "(MWU skipped)")
    print(f"Bootstrap 95% CI on variant median: "
          f"{boot_med*100:.3f}% [{boot_lo*100:.3f}%, {boot_hi*100:.3f}%]")
    if not per_net_test.get("skipped"):
        print(f"PAIRED PER-NET (test, total): "
              f"{per_net_test['test']}  n={per_net_test['n_paired']}  "
              f"p={per_net_test['p_value']:.3e}  "
              f"median Δ(v-b)={per_net_test['median_diff_v_minus_b']*100:.3f}pp")


if __name__ == "__main__":
    main()
