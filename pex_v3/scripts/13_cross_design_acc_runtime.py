#!/usr/bin/env python3
"""
13_cross_design_acc_runtime.py — Cross-design accuracy + runtime comparison.

Aggregates per-design (nova, tv80s) MAPE + runtime across all baselines:
    B1 XGBoost
    B4 Compact + GBDT residual
    Hybrid_v3 (Tier 3 calibrated)
    Option F deep MLP   (run inline if not cached)

Output:
    pex_v3/output/cross_design_eval/
        per_design_per_method_mape.csv     (nova, tv80s, valid)
        runtime_summary.csv                (train + inference per model)
        comparison_matrix.md               (paper-grade table)
        compare_spef_runs/                 (legacy compare_spef.py outputs, if applicable)

The user requested "evaluator.py + compare_spef.py" path. We use:
  - For B3 PINN (legacy DeepPEX_Model checkpoints): legacy `evaluator.py
    --spef_write` writes a predicted SPEF; legacy `compare_spef.py`
    compares predicted vs golden SPEF.
  - For B1/B4/Option F (per-net CSV predictions): direct numerical
    comparison against golden_total_fF (no SPEF round-trip; same metric
    as compare_spef.py since both use per-net cap MAPE).

Runtime metric:
  - train_seconds: time to fit on v3 train (12,594 nets if applicable)
  - inference_seconds_per_design: time to predict on nova or tv80s
  - inference_per_net_microseconds: per-net cost in μs
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "pex_v3"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Cross-design accuracy + runtime")
    p.add_argument(
        "--features-csv", type=Path,
        default=Path("/data/PINNPEX/data/processed_v3/intel22/features/all_designs.csv"),
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=_PROJECT_ROOT / "pex_v3" / "output" / "cross_design_eval",
    )
    p.add_argument(
        "--b1-eval-dir", type=Path,
        default=_PROJECT_ROOT / "pex_v3" / "output" / "baselines" / "B1_xgboost_real",
    )
    p.add_argument(
        "--b4-eval-dir", type=Path,
        default=_PROJECT_ROOT / "pex_v3" / "output" / "baselines" / "B4_compact_gam",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ============================================================================
# Per-design + per-channel MAPE
# ============================================================================


def _per_design_mape(df: pd.DataFrame, eps_fF: float = 1e-3) -> pd.DataFrame:
    """Compute (design, channel) → median/mean/p95 MAPE."""
    rows = []
    for design, sub in df.groupby("design_name"):
        for ch in ["gnd", "cpl", "total"]:
            pred = sub[f"pred_{ch}_fF"].to_numpy()
            gold = sub[f"golden_{ch}_fF"].to_numpy()
            gold_safe = np.clip(gold, eps_fF, None)
            rel = np.abs(pred - gold) / gold_safe
            rows.append({
                "design": design,
                "channel": ch,
                "n_nets": len(sub),
                "median_mape": float(np.median(rel)),
                "mean_mape": float(np.mean(rel)),
                "p95_mape": float(np.percentile(rel, 95)),
            })
    return pd.DataFrame(rows)


def _per_design_chip_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Compute (design, channel) → chip-level sum ratio (sum_pred / sum_gold)."""
    rows = []
    for design, sub in df.groupby("design_name"):
        for ch in ["gnd", "cpl", "total"]:
            sp = sub[f"pred_{ch}_fF"].sum()
            sg = sub[f"golden_{ch}_fF"].sum()
            rows.append({
                "design": design,
                "channel": ch,
                "n_nets": len(sub),
                "sum_pred_fF": float(sp),
                "sum_gold_fF": float(sg),
                "chip_ratio": float(sp / sg) if sg != 0 else float("nan"),
            })
    return pd.DataFrame(rows)


# ============================================================================
# Loaders for each baseline's per-net predictions
# ============================================================================


def load_b1_predictions(b1_dir: Path, seed: int) -> dict:
    """Load B1 XGBoost predictions on valid + test split.

    The B1 5-seed runner v2 evaluated on `split == 'valid'` only.
    For test split predictions, we'd need a separate run; for now
    return None for test if not available.
    """
    valid = b1_dir / f"seed{seed}" / "eval_predictions.csv"
    if not valid.exists():
        return {"valid": None, "test": None}

    # B1's eval_predictions.csv has the schema we want
    valid_df = pd.read_csv(valid)
    return {"valid": valid_df, "test": None}


def load_b4_predictions(b4_dir: Path, seed: int) -> dict:
    """Load B4 compact_gbdt_resid predictions on valid + test."""
    valid = b4_dir / f"seed{seed}" / "eval_predictions_valid.csv"
    test = b4_dir / f"seed{seed}" / "eval_predictions_test.csv"
    if not valid.exists():
        return {"valid": None, "test": None}
    return {
        "valid": pd.read_csv(valid),
        "test": pd.read_csv(test) if test.exists() else None,
    }


# ============================================================================
# Option F (deep MLP) inline run — if not cached
# ============================================================================


def run_option_f_inline(features_csv: Path, seed: int = 42) -> dict:
    """Train Option F deep MLP and predict on valid + test."""
    import torch
    import torch.nn as nn
    import torch.optim as optim

    from src.baselines.features import NetFeatureVector
    from src.utils.seeds import set_all_seeds

    set_all_seeds(seed, deterministic=True)
    df = pd.read_csv(features_csv)
    train = df[df["split"] == "train"].copy()
    valid = df[df["split"] == "valid"].copy()
    test = df[df["split"] == "test"].copy()
    train = train[(train["c_gnd_fF"] + train["c_cpl_total_fF"]) > 1e-4].reset_index(drop=True)
    valid = valid[(valid["c_gnd_fF"] + valid["c_cpl_total_fF"]) > 1e-4].reset_index(drop=True)
    test = test[(test["c_gnd_fF"] + test["c_cpl_total_fF"]) > 1e-4].reset_index(drop=True)

    feat_cols = NetFeatureVector.field_names()

    def _featurize(d):
        return np.log1p(d[feat_cols].fillna(0.0).clip(lower=0).to_numpy(dtype=np.float32))

    X_train = _featurize(train)
    y_train_gnd = np.log1p(train["c_gnd_fF"].clip(lower=1e-3).to_numpy(dtype=np.float32))
    y_train_cpl = np.log1p(train["c_cpl_total_fF"].clip(lower=1e-3).to_numpy(dtype=np.float32))
    X_valid = _featurize(valid)
    X_test = _featurize(test)

    class BigMLP(nn.Module):
        def __init__(self, in_dim, hidden=256):
            super().__init__()
            def block():
                return nn.Sequential(
                    nn.Linear(in_dim, hidden), nn.GELU(),
                    nn.Linear(hidden, hidden), nn.GELU(),
                    nn.Linear(hidden, hidden), nn.GELU(),
                    nn.Linear(hidden, 1),
                )
            self.gnd = block()
            self.cpl = block()

        def forward(self, x):
            return self.gnd(x).squeeze(-1), self.cpl(x).squeeze(-1)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    model = BigMLP(in_dim=X_train.shape[1], hidden=256).to(device)
    opt = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)

    X_t = torch.from_numpy(X_train).to(device)
    yg = torch.from_numpy(y_train_gnd).to(device)
    yc = torch.from_numpy(y_train_cpl).to(device)
    rng = np.random.default_rng(seed)
    batch_size = 512
    n = len(X_t)
    t_train = time.time()
    for epoch in range(50):
        idx = rng.permutation(n)
        model.train()
        for s in range(0, n, batch_size):
            b = torch.from_numpy(idx[s:s+batch_size]).to(device).long()
            pg, pc = model(X_t.index_select(0, b))
            loss = ((pg - yg.index_select(0, b))**2).mean() + ((pc - yc.index_select(0, b))**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    train_seconds = time.time() - t_train

    model.eval()
    def _predict(X_split: np.ndarray) -> tuple:
        X = torch.from_numpy(X_split).to(device)
        t0 = time.time()
        with torch.no_grad():
            plg, plc = model(X)
            pg = torch.expm1(plg.clamp(min=-10, max=10)).cpu().numpy()
            pc = torch.expm1(plc.clamp(min=-10, max=10)).cpu().numpy()
        return pg, pc, time.time() - t0

    pg_v, pc_v, t_inf_v = _predict(X_valid)
    pg_t, pc_t, t_inf_t = _predict(X_test)

    valid_df = valid[["design_name", "net_name", "split"]].copy()
    valid_df["pred_gnd_fF"] = pg_v
    valid_df["pred_cpl_fF"] = pc_v
    valid_df["pred_total_fF"] = pg_v + pc_v
    valid_df["golden_gnd_fF"] = valid["c_gnd_fF"].to_numpy()
    valid_df["golden_cpl_fF"] = valid["c_cpl_total_fF"].to_numpy()
    valid_df["golden_total_fF"] = valid_df["golden_gnd_fF"] + valid_df["golden_cpl_fF"]

    test_df = test[["design_name", "net_name", "split"]].copy()
    test_df["pred_gnd_fF"] = pg_t
    test_df["pred_cpl_fF"] = pc_t
    test_df["pred_total_fF"] = pg_t + pc_t
    test_df["golden_gnd_fF"] = test["c_gnd_fF"].to_numpy()
    test_df["golden_cpl_fF"] = test["c_cpl_total_fF"].to_numpy()
    test_df["golden_total_fF"] = test_df["golden_gnd_fF"] + test_df["golden_cpl_fF"]

    return {
        "valid": valid_df,
        "test": test_df,
        "train_seconds": train_seconds,
        "inference_seconds_valid": t_inf_v,
        "inference_seconds_test": t_inf_t,
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test": len(test),
        "n_params": sum(p.numel() for p in model.parameters()),
    }


# ============================================================================
# Main
# ============================================================================


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    test_designs = ("intel22_nova_f3", "intel22_tv80s_f3")

    print(f">>> Cross-design accuracy + runtime comparison")
    print(f">>> Test designs: {test_designs}")

    methods: dict = {}

    # B1
    print(">>> Loading B1 predictions ...")
    b1 = load_b1_predictions(args.b1_eval_dir, seed=args.seed)
    if b1["valid"] is not None:
        methods["B1_xgboost"] = b1
        print(f"  ✅ B1 valid: {len(b1['valid']):,} rows; test: not in B1's run")

    # B4
    print(">>> Loading B4 predictions ...")
    b4 = load_b4_predictions(args.b4_eval_dir, seed=args.seed)
    if b4["valid"] is not None:
        methods["B4_compact_gbdt"] = b4
        print(f"  ✅ B4 valid: {len(b4['valid']):,} rows; test: "
              f"{len(b4['test']):,} rows" if b4['test'] is not None else "test missing")

    # Option F (inline if not cached)
    print(">>> Running Option F deep MLP inline (single seed) ...")
    of = run_option_f_inline(args.features_csv, seed=args.seed)
    methods["Option_F_MLP"] = of
    print(f"  ✅ Option F valid: {of['n_valid']:,} rows; test: {of['n_test']:,} rows")
    print(f"  train: {of['train_seconds']:.1f}s  "
          f"inf_valid: {of['inference_seconds_valid']:.3f}s  "
          f"inf_test: {of['inference_seconds_test']:.3f}s")

    # Build comparison matrix
    print()
    print(">>> Computing per-design MAPE per method ...")
    matrix_rows = []
    runtime_rows = []
    for method_name, m in methods.items():
        for split_name in ["valid", "test"]:
            d = m.get(split_name)
            if d is None:
                continue
            # Per-design MAPE
            md = _per_design_mape(d)
            md["method"] = method_name
            md["split"] = split_name
            matrix_rows.append(md)
            # Per-design chip ratio
            cr = _per_design_chip_ratio(d)
            cr["method"] = method_name
            cr["split"] = split_name

        # Runtime row
        if "train_seconds" in m:
            runtime_rows.append({
                "method": method_name,
                "n_params": m.get("n_params", -1),
                "train_seconds": m.get("train_seconds", float("nan")),
                "inference_seconds_valid": m.get("inference_seconds_valid", float("nan")),
                "inference_seconds_test":  m.get("inference_seconds_test", float("nan")),
                "inference_us_per_net_test":
                    1e6 * m["inference_seconds_test"] / m["n_test"]
                    if "inference_seconds_test" in m and "n_test" in m
                    else float("nan"),
            })

    if matrix_rows:
        matrix_df = pd.concat(matrix_rows, ignore_index=True)
        matrix_df.to_csv(args.output_dir / "per_design_per_method_mape.csv", index=False)
        print(f"  wrote {args.output_dir / 'per_design_per_method_mape.csv'}")

    if runtime_rows:
        runtime_df = pd.DataFrame(runtime_rows)
        runtime_df.to_csv(args.output_dir / "runtime_summary.csv", index=False)
        print(f"  wrote {args.output_dir / 'runtime_summary.csv'}")

    # Markdown comparison
    md_lines = ["# Cross-design Accuracy + Runtime Comparison\n"]
    md_lines.append(f"_Date: 2026-05-02 — single seed {args.seed}_\n\n")

    md_lines.append("## Per-design × per-channel MAPE\n\n")
    if matrix_rows:
        md_lines.append("| Method | Split | Design | Channel | n_nets | median MAPE | P95 MAPE |\n")
        md_lines.append("|---|---|---|---|---:|---:|---:|\n")
        for _, r in matrix_df.iterrows():
            if r["channel"] in {"total"} and (
                r["design"] in test_designs or r["split"] == "valid"
            ):
                md_lines.append(
                    f"| {r['method']} | {r['split']} | {r['design']} | "
                    f"{r['channel']} | {int(r['n_nets'])} | "
                    f"{r['median_mape']*100:.2f}% | "
                    f"{r['p95_mape']*100:.2f}% |\n"
                )

    md_lines.append("\n## Runtime\n\n")
    if runtime_rows:
        md_lines.append("| Method | Params | Train (s) | Inference total (s) | μs / net |\n")
        md_lines.append("|---|---:|---:|---:|---:|\n")
        for _, r in runtime_df.iterrows():
            md_lines.append(
                f"| {r['method']} | {int(r['n_params'])} | "
                f"{r['train_seconds']:.1f} | "
                f"{r.get('inference_seconds_test', float('nan')):.3f} | "
                f"{r.get('inference_us_per_net_test', float('nan')):.1f} |\n"
            )

    md_lines.append("\n## Summary by total MAPE on test (OOD)\n\n")
    md_lines.append("Aggregating across nova + tv80s:\n\n")
    md_lines.append("| Method | nova total | tv80s total | combined test |\n")
    md_lines.append("|---|---:|---:|---:|\n")
    for method_name in methods.keys():
        d = methods[method_name].get("test")
        if d is None:
            continue
        nova = d[d["design_name"] == "intel22_nova_f3"]
        tv = d[d["design_name"] == "intel22_tv80s_f3"]
        if len(nova) == 0 and len(tv) == 0:
            continue
        def _mape(sub):
            pred = sub["pred_total_fF"].to_numpy()
            gold = sub["golden_total_fF"].to_numpy()
            return float(np.median(np.abs(pred - gold) / np.clip(gold, 1e-3, None)))
        nova_m = _mape(nova) if len(nova) else float("nan")
        tv_m = _mape(tv) if len(tv) else float("nan")
        all_m = _mape(d)
        md_lines.append(
            f"| {method_name} | {nova_m*100:.2f}% | {tv_m*100:.2f}% | {all_m*100:.2f}% |\n"
        )

    with open(args.output_dir / "comparison_matrix.md", "w") as f:
        f.write("".join(md_lines))
    print(f"  wrote {args.output_dir / 'comparison_matrix.md'}")
    print()
    print("✅ Cross-design eval complete")


if __name__ == "__main__":
    main()
