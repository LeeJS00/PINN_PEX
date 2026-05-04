#!/usr/bin/env python3
"""
stratify_eval.py — produce stratified MAPE tables for an ablation variant.

Reads `pex_v3/output/ablation/<variant>/seed*/eval_logger.parquet` (or
falls back to `eval_logger.csv`), aggregates per-net errors across seeds
by averaging predictions (ensemble), then emits four stratification tables:

    1. Per-design (each train design + nova + tv80s)
    2. Per-quartile of compact_gnd_estimate_fF (Q1..Q4)
    3. Per-fanout bucket (1, 2-5, 6-20, >20)
    4. Per-dominant-layer (M1..top)

Outputs land at:
    pex_v3/output/ablation/<variant>/stratified/{
        per_design.csv,
        per_quartile.csv,
        per_fanout.csv,
        per_layer.csv,
        top50_outliers.csv,
        summary.md,
    }

Usage:
    python3 pex_v3/scripts/stratify_eval.py --variant HybridPexV3Mesh
    python3 pex_v3/scripts/stratify_eval.py \\
        --eval-dir pex_v3/output/phase1_mesh_5seed_ensemble \\
        --pred-csv pex_v3/output/phase1_mesh_5seed_ensemble/ensemble_predictions_test.csv \\
        --out-dir  pex_v3/output/phase1_mesh_5seed_ensemble/stratified

The second form is used in the self-test against the locked baseline.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "pex_v3"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils.eval_logger import (  # noqa: E402
    read_eval_parquet,
    load_ensemble_with_features,
    add_error_columns,
    stratify_per_design,
    stratify_per_quartile,
    stratify_per_fanout,
    stratify_per_layer,
    top_outliers,
    write_eval_parquet,
)


DEFAULT_FEATURES_CSV = Path("/data/PINNPEX/data/processed_v3/intel22/features/all_designs.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stratified MAPE for an ablation variant")
    p.add_argument("--variant", type=str, default=None,
                   help="Variant name; resolves to pex_v3/output/ablation/<variant>/")
    p.add_argument("--eval-dir", type=Path, default=None,
                   help="Override: directory containing seed*/eval_logger.parquet")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Override: where to write stratified tables")
    p.add_argument("--pred-csv", type=Path, default=None,
                   help="Optional: build eval parquet on-the-fly from an ensemble CSV "
                        "(for legacy 5-seed dirs that pre-date eval_logger).")
    p.add_argument("--features-csv", type=Path, default=DEFAULT_FEATURES_CSV,
                   help="v3 features CSV for joining fanout/bbox/layer info")
    p.add_argument("--split", choices=["valid", "test"], default="test",
                   help="Which split to stratify (when reading per-seed parquet, "
                        "looks for eval_logger_{split}.parquet first, then eval_logger.parquet)")
    p.add_argument("--top-n-outliers", type=int, default=50)
    return p.parse_args()


def _resolve_eval_dir(args: argparse.Namespace) -> Path:
    if args.eval_dir is not None:
        return args.eval_dir
    if args.variant is None:
        raise SystemExit("--variant or --eval-dir required")
    return _PROJECT_ROOT / "pex_v3" / "output" / "ablation" / args.variant


def _resolve_out_dir(args: argparse.Namespace, eval_dir: Path) -> Path:
    if args.out_dir is not None:
        return args.out_dir
    return eval_dir / "stratified"


def _load_predictions(args: argparse.Namespace, eval_dir: Path) -> pd.DataFrame:
    """Resolve eval data either from per-seed parquet (and ensemble) or from a pred CSV."""
    # Path A: explicit ensemble CSV → build on the fly
    if args.pred_csv is not None:
        print(f">>> loading ensemble predictions from {args.pred_csv}")
        designs = None
        if args.split == "test":
            designs = ["intel22_nova_f3", "intel22_tv80s_f3"]
        df = load_ensemble_with_features(
            args.pred_csv, args.features_csv, designs_filter=designs,
        )
        return df

    # Path B: per-seed parquet → average across seeds
    suffixes = [f"eval_logger_{args.split}.parquet", "eval_logger.parquet"]
    seed_dirs = sorted([p for p in eval_dir.glob("seed*") if p.is_dir()])
    if not seed_dirs:
        raise SystemExit(f"No seed dirs under {eval_dir}")

    seed_dfs = []
    for sd in seed_dirs:
        for suf in suffixes:
            cand = sd / suf
            if cand.exists() or cand.with_suffix(".csv").exists():
                df = read_eval_parquet(cand)
                df["__seed"] = sd.name
                seed_dfs.append(df)
                break
    if not seed_dfs:
        raise SystemExit(
            f"No eval_logger parquet/csv found under {eval_dir}/seed*/. "
            f"If this is the locked phase1_mesh_5seed dir, pass --pred-csv "
            f"<ensemble_predictions_test.csv> instead."
        )
    print(f">>> aggregating {len(seed_dfs)} seed eval parquets ({args.split} split)")

    # Average predictions per net
    cat = pd.concat(seed_dfs, ignore_index=True)
    grouped = cat.groupby(["net_id"], as_index=False).agg({
        "design": "first",
        "net_name": "first",
        "fanout": "first",
        "bbox_xy_um2": "first",
        "compact_gnd_estimate_fF": "first",
        "dominant_layer": "first",
        "gnd_pred": "mean",
        "cpl_pred": "mean",
        "gnd_gold": "first",
        "cpl_gold": "first",
    })
    grouped["total_pred"] = grouped["gnd_pred"] + grouped["cpl_pred"]
    grouped["total_gold"] = grouped["gnd_gold"] + grouped["cpl_gold"]
    return grouped


def _df_to_md_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Tabulate-free pipe-table renderer (avoids optional `tabulate` dep)."""
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(format(v, floatfmt))
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


def _summary_md(df: pd.DataFrame, per_design, per_q, per_q_err, per_fan, per_layer, args) -> str:
    n = len(df)
    g_med = df["gnd_rel_err"].median()
    c_med = df["cpl_rel_err"].median()
    t_med = df["total_rel_err"].median()
    md = []
    md.append(f"# Stratified eval — variant `{args.variant or args.eval_dir.name}` "
              f"(split={args.split})\n\n")
    md.append(f"Baseline: n={n:,} nets, "
              f"gnd median {g_med*100:.3f}%, cpl median {c_med*100:.3f}%, "
              f"total median {t_med*100:.3f}%.\n\n")

    md.append("## Per-design\n\n")
    md.append(_df_to_md_table(per_design) + "\n\n")
    md.append("## Per-quartile (axis=compact_gnd_estimate_fF)\n\n")
    md.append(_df_to_md_table(per_q) + "\n\n")
    md.append("## Per-quartile (axis=gnd_rel_err — Mode B giant-CTS surface)\n\n")
    md.append(_df_to_md_table(per_q_err) + "\n\n")
    md.append("## Per-fanout bucket\n\n")
    md.append(_df_to_md_table(per_fan) + "\n\n")
    md.append("## Per-dominant-layer\n\n")
    md.append(_df_to_md_table(per_layer) + "\n\n")
    return "".join(md)


def main() -> None:
    args = parse_args()
    eval_dir = _resolve_eval_dir(args)
    out_dir = _resolve_out_dir(args, eval_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f">>> eval_dir : {eval_dir}")
    print(f">>> out_dir  : {out_dir}")
    print(f">>> split    : {args.split}")

    df = _load_predictions(args, eval_dir)
    df = add_error_columns(df)
    print(f">>> n_nets   : {len(df):,}")

    # Persist the joined/aggregated parquet for downstream re-use
    write_eval_parquet(
        df[[c for c in df.columns if not c.startswith("__")]],
        out_dir / "eval_aggregated.parquet",
    )

    per_design = stratify_per_design(df)
    per_q = stratify_per_quartile(df, axis="compact_gnd_estimate_fF")
    per_q_err = stratify_per_quartile(df, axis="gnd_rel_err")
    per_fan = stratify_per_fanout(df)
    per_layer = stratify_per_layer(df)
    top50 = top_outliers(df, n=args.top_n_outliers, by="gnd_rel_err")

    per_design.to_csv(out_dir / "per_design.csv", index=False)
    per_q.to_csv(out_dir / "per_quartile.csv", index=False)
    per_q_err.to_csv(out_dir / "per_quartile_by_gnd_err.csv", index=False)
    per_fan.to_csv(out_dir / "per_fanout.csv", index=False)
    per_layer.to_csv(out_dir / "per_layer.csv", index=False)
    top50.to_csv(out_dir / f"top{args.top_n_outliers}_outliers.csv", index=False)

    md = _summary_md(df, per_design, per_q, per_q_err, per_fan, per_layer, args)
    (out_dir / "summary.md").write_text(md)

    # Echo the most important sanity numbers
    print()
    print("--- per-design ---")
    print(per_design.to_string(index=False))
    print()
    print("--- per-quartile (compact_gnd_estimate_fF) ---")
    print(per_q.to_string(index=False))
    print()
    print("--- per-quartile (gnd_rel_err — Mode B surface) ---")
    print(per_q_err.to_string(index=False))
    print()
    print("--- per-fanout ---")
    print(per_fan.to_string(index=False))
    print()
    print("--- per-layer ---")
    print(per_layer.to_string(index=False))
    print()
    print(f">>> wrote: {out_dir}")


if __name__ == "__main__":
    main()
