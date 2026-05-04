#!/usr/bin/env python3
"""
14_option_f_5seed.py — Phase 0.5 Option F (deep MLP) 5-seed run.

Locks the variance estimate for the "Option F deep MLP" hand-feature ceiling
baseline. Single-seed run (script 13) gave 4.66% valid / 5.67% test on
v3 (intel22). This script reruns 5 seeds [0..4] to put error bars on
the ceiling number for paper #1A.

Architecture (mirrors `13_cross_design_acc_runtime.run_option_f_inline`):
    BigMLP — two heads (gnd, cpl), each Linear-GELU x4, hidden=256
    AdamW lr=2e-3 wd=1e-4, CosineAnnealingLR T_max=50, batch=512
    Loss: MSE on log1p(target)
    Pred: expm1(model(log1p(x)).clamp(-10, 10))

Per Codex review (2026-05-03):
  - Variance comes from torch.manual_seed (init) + np permutation (batch order);
    no dropout added — paper number must be the natural baseline.
  - Last-epoch eval (no best-val tracking) per A1 anti-overclaim.
  - Aggregation: median-of-per-seed-medians (primary, matches B4 convention)
    AND pooled-prediction MAPE (secondary diagnostic field).

Output:
    pex_v3/output/baselines/Option_F_MLP/seed{N}/
        eval_predictions_valid.csv     — per-net pred + golden on valid
        eval_predictions_test.csv      — per-net pred + golden on test (OOD)
        per_channel_summary.json       — per-channel × per-split + per-design
        metrics_row.csv                — single-row MetricsRow (paper schema)
    pex_v3/output/baselines/Option_F_MLP/
        five_seed_summary.json         — aggregate stats (incl. pooled secondary)
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

from src.baselines.features import NetFeatureVector  # noqa: E402
from src.evaluation.metrics import build_metrics_row  # noqa: E402
from src.utils.seeds import set_all_seeds  # noqa: E402


# ============================================================================
# Args
# ============================================================================


def parse_args():
    p = argparse.ArgumentParser(description="Option F deep MLP 5-seed")
    p.add_argument(
        "--features-csv", type=Path,
        default=Path("/data/PINNPEX/data/processed_v3/intel22/features/all_designs.csv"),
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=_PROJECT_ROOT / "pex_v3" / "output" / "baselines" / "Option_F_MLP",
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    return p.parse_args()


# ============================================================================
# Per-seed train + eval (verbatim from script 13's run_option_f_inline,
# parameterized for hyperparams, returning the same dict shape)
# ============================================================================


def run_one_seed(
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
    feat_cols: list[str],
    seed: int,
    device: str,
    hidden: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> dict:
    """Train + predict on valid+test for a single seed. Returns prediction
    DataFrames + runtime stats."""
    import torch
    import torch.nn as nn
    import torch.optim as optim

    set_all_seeds(seed, deterministic=True)
    torch.manual_seed(seed)

    def _featurize(d: pd.DataFrame) -> np.ndarray:
        return np.log1p(
            d[feat_cols].fillna(0.0).clip(lower=0).to_numpy(dtype=np.float32)
        )

    X_train = _featurize(df_train)
    y_train_gnd = np.log1p(
        df_train["c_gnd_fF"].clip(lower=1e-3).to_numpy(dtype=np.float32)
    )
    y_train_cpl = np.log1p(
        df_train["c_cpl_total_fF"].clip(lower=1e-3).to_numpy(dtype=np.float32)
    )
    X_valid = _featurize(df_valid)
    X_test = _featurize(df_test)

    class BigMLP(nn.Module):
        def __init__(self, in_dim: int, hidden: int):
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

    model = BigMLP(in_dim=X_train.shape[1], hidden=hidden).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    X_t = torch.from_numpy(X_train).to(device)
    yg = torch.from_numpy(y_train_gnd).to(device)
    yc = torch.from_numpy(y_train_cpl).to(device)
    rng = np.random.default_rng(seed)
    n = len(X_t)
    t_train = time.time()
    for _epoch in range(epochs):
        idx = rng.permutation(n)
        model.train()
        for s in range(0, n, batch_size):
            b = torch.from_numpy(idx[s:s + batch_size]).to(device).long()
            pg, pc = model(X_t.index_select(0, b))
            loss = (
                ((pg - yg.index_select(0, b)) ** 2).mean()
                + ((pc - yc.index_select(0, b)) ** 2).mean()
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    train_seconds = time.time() - t_train

    model.eval()

    def _predict(X_split: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        X = torch.from_numpy(X_split).to(device)
        t0 = time.time()
        with torch.no_grad():
            plg, plc = model(X)
            pg = torch.expm1(plg.clamp(min=-10, max=10)).cpu().numpy()
            pc = torch.expm1(plc.clamp(min=-10, max=10)).cpu().numpy()
        return pg, pc, time.time() - t0

    pg_v, pc_v, t_inf_v = _predict(X_valid)
    pg_t, pc_t, t_inf_t = _predict(X_test)

    valid_pred = df_valid[["design_name", "net_name", "split"]].copy()
    valid_pred["pred_gnd_fF"] = pg_v
    valid_pred["pred_cpl_fF"] = pc_v
    valid_pred["pred_total_fF"] = pg_v + pc_v
    valid_pred["golden_gnd_fF"] = df_valid["c_gnd_fF"].to_numpy()
    valid_pred["golden_cpl_fF"] = df_valid["c_cpl_total_fF"].to_numpy()
    valid_pred["golden_total_fF"] = (
        valid_pred["golden_gnd_fF"] + valid_pred["golden_cpl_fF"]
    )

    test_pred = df_test[["design_name", "net_name", "split"]].copy()
    test_pred["pred_gnd_fF"] = pg_t
    test_pred["pred_cpl_fF"] = pc_t
    test_pred["pred_total_fF"] = pg_t + pc_t
    test_pred["golden_gnd_fF"] = df_test["c_gnd_fF"].to_numpy()
    test_pred["golden_cpl_fF"] = df_test["c_cpl_total_fF"].to_numpy()
    test_pred["golden_total_fF"] = (
        test_pred["golden_gnd_fF"] + test_pred["golden_cpl_fF"]
    )

    return {
        "valid": valid_pred,
        "test": test_pred,
        "train_seconds": float(train_seconds),
        "inference_seconds_valid": float(t_inf_v),
        "inference_seconds_test": float(t_inf_t),
        "n_params": int(sum(p.numel() for p in model.parameters())),
    }


# ============================================================================
# Per-channel + per-design MAPE
# ============================================================================

_EPS_FF = 1e-3


def _mape(pred: np.ndarray, gold: np.ndarray) -> np.ndarray:
    return np.abs(pred - gold) / np.clip(np.abs(gold), _EPS_FF, None)


def _per_channel_summary(pred_df: pd.DataFrame) -> dict:
    out: dict = {}
    for ch in ["gnd", "cpl", "total"]:
        rel = _mape(
            pred_df[f"pred_{ch}_fF"].to_numpy(),
            pred_df[f"golden_{ch}_fF"].to_numpy(),
        )
        out[ch] = {
            "n_nets": int(len(rel)),
            "median": float(np.median(rel)),
            "mean": float(np.mean(rel)),
            "p95": float(np.percentile(rel, 95)),
        }
    return out


def _per_design_summary(pred_df: pd.DataFrame) -> dict:
    out: dict = {}
    for design, sub in pred_df.groupby("design_name"):
        out[str(design)] = _per_channel_summary(sub)
    return out


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    if torch.cuda.is_available():
        device = f"cuda:{args.gpu}"
    else:
        device = "cpu"
    print(f">>> device: {device}")
    print(f">>> features: {args.features_csv}")
    print(f">>> output:   {args.output_dir}")
    print(f">>> seeds:    {args.seeds}")

    # Load + split features once
    df = pd.read_csv(args.features_csv)
    train = df[df["split"] == "train"].copy()
    valid = df[df["split"] == "valid"].copy()
    test = df[df["split"] == "test"].copy()

    def _filt(d):
        return d[(d["c_gnd_fF"] + d["c_cpl_total_fF"]) > 1e-4].reset_index(drop=True)

    train, valid, test = _filt(train), _filt(valid), _filt(test)
    print(f">>> splits: train={len(train):,}  valid={len(valid):,}  test={len(test):,}")

    feat_cols = NetFeatureVector.field_names()
    print(f">>> features: {len(feat_cols)} columns")

    # Per-seed loop
    per_seed_summaries: list[dict] = []
    per_seed_runtimes: list[dict] = []
    pool_valid_frames: list[pd.DataFrame] = []
    pool_test_frames: list[pd.DataFrame] = []

    for seed in args.seeds:
        seed_dir = args.output_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        print()
        print(f">>> seed {seed} — training")
        t0 = time.time()
        result = run_one_seed(
            df_train=train,
            df_valid=valid,
            df_test=test,
            feat_cols=feat_cols,
            seed=seed,
            device=device,
            hidden=args.hidden,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        elapsed = time.time() - t0

        valid_pred = result["valid"]
        test_pred = result["test"]
        valid_pred.to_csv(seed_dir / "eval_predictions_valid.csv", index=False)
        test_pred.to_csv(seed_dir / "eval_predictions_test.csv", index=False)

        valid_summary = _per_channel_summary(valid_pred)
        test_summary = _per_channel_summary(test_pred)
        valid_per_design = _per_design_summary(valid_pred)
        test_per_design = _per_design_summary(test_pred)
        with open(seed_dir / "per_channel_summary.json", "w") as f:
            json.dump(
                {
                    "valid": valid_summary,
                    "test": test_summary,
                    "valid_per_design": valid_per_design,
                    "test_per_design": test_per_design,
                    "n_params": result["n_params"],
                    "train_seconds": result["train_seconds"],
                    "inference_seconds_valid": result["inference_seconds_valid"],
                    "inference_seconds_test": result["inference_seconds_test"],
                    "elapsed_seconds": float(elapsed),
                },
                f,
                indent=2,
            )

        # MetricsRow on TEST (OOD) for paper-grade aggregation parity with B1/B4
        row = build_metrics_row(
            method="Option_F_MLP",
            seed=seed,
            pred_fF=test_pred["pred_total_fF"].to_numpy(),
            golden_fF=test_pred["golden_total_fF"].to_numpy(),
        )
        pd.DataFrame([vars(row)]).to_csv(seed_dir / "metrics_row.csv", index=False)

        print(
            f"    seed {seed}: valid total median={valid_summary['total']['median']*100:.3f}%  "
            f"test total median={test_summary['total']['median']*100:.3f}%  "
            f"({elapsed:.1f}s)"
        )

        per_seed_summaries.append(
            {
                "seed": seed,
                "valid": valid_summary,
                "test": test_summary,
                "valid_per_design": valid_per_design,
                "test_per_design": test_per_design,
            }
        )
        per_seed_runtimes.append(
            {
                "seed": seed,
                "n_params": result["n_params"],
                "train_seconds": result["train_seconds"],
                "inference_seconds_valid": result["inference_seconds_valid"],
                "inference_seconds_test": result["inference_seconds_test"],
                "elapsed_seconds": float(elapsed),
            }
        )

        # Track pooled predictions for diagnostic secondary metric
        v_pool = valid_pred[
            ["design_name", "net_name", "pred_gnd_fF", "pred_cpl_fF",
             "pred_total_fF", "golden_gnd_fF", "golden_cpl_fF", "golden_total_fF"]
        ].copy()
        v_pool["seed"] = seed
        pool_valid_frames.append(v_pool)

        t_pool = test_pred[
            ["design_name", "net_name", "pred_gnd_fF", "pred_cpl_fF",
             "pred_total_fF", "golden_gnd_fF", "golden_cpl_fF", "golden_total_fF"]
        ].copy()
        t_pool["seed"] = seed
        pool_test_frames.append(t_pool)

    # ---- Aggregate (primary: median-of-per-seed-medians; B4 parity) -------
    def _aggregate_channel(
        per_seed: list[dict], split: str, ch: str, stat: str = "median"
    ) -> dict:
        vals = np.array([s[split][ch][stat] for s in per_seed], dtype=np.float64)
        return {
            "median": float(np.median(vals)),
            "mean": float(np.mean(vals)),
            "stdev": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "values": [float(v) for v in vals],
        }

    summary: dict = {
        "n_seeds": len(args.seeds),
        "seeds": list(args.seeds),
        "n_train": int(len(train)),
        "n_valid": int(len(valid)),
        "n_test": int(len(test)),
        "primary_per_seed_median_aggregate": {
            "valid": {
                ch: _aggregate_channel(per_seed_summaries, "valid", ch)
                for ch in ["gnd", "cpl", "total"]
            },
            "test": {
                ch: _aggregate_channel(per_seed_summaries, "test", ch)
                for ch in ["gnd", "cpl", "total"]
            },
        },
        "per_design_test": {},  # filled below
        "runtime": {
            "n_params": int(per_seed_runtimes[0]["n_params"]),
            "train_seconds_mean": float(
                np.mean([r["train_seconds"] for r in per_seed_runtimes])
            ),
            "train_seconds_stdev": float(
                np.std([r["train_seconds"] for r in per_seed_runtimes])
            ),
            "inference_seconds_valid_mean": float(
                np.mean([r["inference_seconds_valid"] for r in per_seed_runtimes])
            ),
            "inference_seconds_test_mean": float(
                np.mean([r["inference_seconds_test"] for r in per_seed_runtimes])
            ),
            "inference_us_per_net_test_mean": float(
                np.mean(
                    [
                        1e6 * r["inference_seconds_test"] / max(len(test), 1)
                        for r in per_seed_runtimes
                    ]
                )
            ),
        },
        "per_seed": per_seed_summaries,
        "per_seed_runtimes": per_seed_runtimes,
    }

    # Per-design aggregate on test (nova / tv80s separately + combined)
    test_designs = sorted(test["design_name"].unique().tolist())
    for design in test_designs:
        per_seed_design = []
        for s in per_seed_summaries:
            if design in s["test_per_design"]:
                per_seed_design.append(
                    {"test": {ch: s["test_per_design"][design][ch]
                               for ch in ["gnd", "cpl", "total"]}}
                )
        if not per_seed_design:
            continue
        summary["per_design_test"][design] = {
            ch: _aggregate_channel(per_seed_design, "test", ch)
            for ch in ["gnd", "cpl", "total"]
        }

    # ---- Secondary diagnostic: pooled-prediction MAPE (across-seed pool) ----
    pool_valid = pd.concat(pool_valid_frames, ignore_index=True)
    pool_test = pd.concat(pool_test_frames, ignore_index=True)
    summary["secondary_pooled_predictions"] = {
        "note": "MAPE computed by pooling all 5-seed per-net predictions then taking median; tighter CI than per-seed-median aggregation, kept as diagnostic only.",
        "valid": _per_channel_summary(pool_valid),
        "test": _per_channel_summary(pool_test),
        "test_per_design": _per_design_summary(pool_test),
    }

    out_path = args.output_dir / "five_seed_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Console report ---------------------------------------------------
    primary = summary["primary_per_seed_median_aggregate"]
    print()
    print("=" * 72)
    print("5-seed Option F deep MLP — primary (per-seed-median aggregation)")
    print("=" * 72)
    for split in ["valid", "test"]:
        print(f"\n{split.upper()}:")
        for ch in ["total", "gnd", "cpl"]:
            agg = primary[split][ch]
            print(
                f"  {ch:5s}  median = {agg['median']*100:6.3f}%  "
                f"mean = {agg['mean']*100:6.3f}%  "
                f"stdev = {agg['stdev']*100:5.3f}pp  "
                f"min = {agg['min']*100:5.3f}%  "
                f"max = {agg['max']*100:5.3f}%"
            )
    print("\nPer-design TEST (OOD):")
    for design, ch_agg in summary["per_design_test"].items():
        agg = ch_agg["total"]
        print(
            f"  {design:25s} total median = {agg['median']*100:6.3f}% ± "
            f"{agg['stdev']*100:.3f}pp"
        )
    rt = summary["runtime"]
    print(
        f"\nRuntime: n_params={rt['n_params']:,}  "
        f"train_mean={rt['train_seconds_mean']:.1f}s  "
        f"inf_test_mean={rt['inference_seconds_test_mean']:.3f}s  "
        f"({rt['inference_us_per_net_test_mean']:.1f} μs/net)"
    )
    print(f"\n✅ five_seed_summary written to {out_path}")


if __name__ == "__main__":
    main()
