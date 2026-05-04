#!/usr/bin/env python3
"""
17_finetune_hybrid_calibrated_smoke.py — Phase 1 Week 1: NNLS-calibrated PINN.

Same as scripts/10_finetune_hybrid_smoke.py BUT applies per-layer NNLS
calibration to compact_gnd_estimate_fF / compact_cpl_estimate_total_fF
BEFORE fine-tuning. Calibration fit on TRAIN only (no leakage), applied
to train+valid+test.

Hypothesis (Codex Round 3, 2026-05-03):
    The bounded multiplicative residual (clamp=log(1.5)=±50%) cannot fight
    the analytic prior's median 0.35 ratio (3× under-estimate for ground).
    NNLS recalibration brings median ratio → 1.0 → bounded paradigm becomes
    viable → stable training → expected gain: total -1.5~2.5pp, gnd -5~8pp.

Go/No-Go (Codex):
    3-seed valid median <= 8.5% total AND gnd < 18% → Week 2 (mesh_v3)
    Otherwise → bottleneck is representation, skip to mesh_v3 directly.

This is the SINGLE-SEED smoke. After pass, 5-seed runner goes next.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "pex_v3"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.models.hybrid_v3 import HybridPexV3  # noqa: E402
from src.trainers.finetune_hybrid_v3 import (  # noqa: E402
    df_to_tensors,
    split_by_manifest_column,
    finetune_hybrid,
    evaluate_per_channel,
    evaluate_beta_gate,
    FinetuneConfig,
)
from src.baselines.calibration_v3 import (  # noqa: E402
    fit_per_layer_calibration,
    apply_per_layer_calibration,
    validate_calibration,
)
from src.utils.seeds import set_all_seeds  # noqa: E402
from src.utils.manifest_hash import write_provenance  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Phase 1 Week 1 — NNLS-calibrated smoke")
    p.add_argument(
        "--features-csv", type=Path,
        default=Path("/data/PINNPEX/data/processed_v3/intel22/features/all_designs.csv"),
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=_PROJECT_ROOT / "pex_v3" / "output" / "phase1_finetune_calibrated_smoke",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--calibration", choices=["scalar", "per_layer", "none"],
        default="per_layer",
        help="Type of analytic prior calibration to apply.",
    )
    p.add_argument("--hidden-dim", type=int, default=64,
                   help="Residual MLP hidden width (default 64 = ~11K params).")
    p.add_argument("--n-hidden", type=int, default=2,
                   help="Residual MLP depth (default 2).")
    p.add_argument("--clamp-init", type=float, default=None,
                   help="Override initial clamp_bound (None = use HybridPexV3 default log(1.5)).")
    p.add_argument("--device", type=str,
                   default="cuda:0" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_all_seeds(args.seed, deterministic=True)

    print(f">>> Phase 1 Week 1 — NNLS-calibrated smoke — seed {args.seed}")
    print(f">>> features:    {args.features_csv}")
    print(f">>> output:      {args.output_dir}")
    print(f">>> device:      {args.device}")
    print(f">>> calibration: {args.calibration}")

    # Load + split
    df = pd.read_csv(args.features_csv)
    print(f">>> loaded {len(df):,} rows × {len(df.columns)} cols")
    train_df, valid_df, test_df = split_by_manifest_column(df)
    train_df = train_df[(train_df["c_gnd_fF"] + train_df["c_cpl_total_fF"]) > 1e-4].reset_index(drop=True)
    valid_df = valid_df[(valid_df["c_gnd_fF"] + valid_df["c_cpl_total_fF"]) > 1e-4].reset_index(drop=True)
    test_df  = test_df[ (test_df["c_gnd_fF"]  + test_df["c_cpl_total_fF"])  > 1e-4].reset_index(drop=True)
    print(f">>> splits: train={len(train_df):,}  valid={len(valid_df):,}  test={len(test_df):,}")

    # ---- WEEK 1 ADDITION: fit NNLS calibration on TRAIN, apply to all splits ----
    cal_summary: dict = {"calibration_type": args.calibration}
    if args.calibration != "none":
        print()
        print(f">>> Fitting NNLS calibration ({args.calibration}) on TRAIN only ...")

        # Validate before
        before_valid = validate_calibration(valid_df)
        print(f"  BEFORE (valid): gnd median={before_valid['median_ratio_gnd']:.3f}  "
              f"cpl median={before_valid['median_ratio_cpl']:.3f}")

        if args.calibration == "scalar":
            from src.baselines.calibration_v3 import (
                fit_scalar_calibration, apply_scalar_calibration,
            )
            calib = fit_scalar_calibration(train_df)
            print(f"  scalar calibration: s_gnd={calib.s_gnd:.4f}  s_cpl={calib.s_cpl:.4f}")
            train_df = apply_scalar_calibration(train_df, calib)
            valid_df = apply_scalar_calibration(valid_df, calib)
            test_df  = apply_scalar_calibration(test_df,  calib)
            cal_summary["calibration"] = {
                "s_gnd": calib.s_gnd, "s_cpl": calib.s_cpl,
                "median_ratio_gnd_before": calib.median_ratio_gnd_before,
                "median_ratio_cpl_before": calib.median_ratio_cpl_before,
                "n_train_nets": calib.n_train_nets,
            }
        else:  # per_layer
            calib = fit_per_layer_calibration(train_df)
            print(f"  per-layer calibration: "
                  f"{len(calib.s_gnd_per_layer)} gnd-layers fit, "
                  f"default s_gnd={calib.s_gnd_default:.4f} "
                  f"s_cpl_default={calib.s_cpl_default:.4f}")
            for L, s in sorted(calib.s_gnd_per_layer.items()):
                print(f"    layer {L}: s_gnd={s:.4f}  "
                      f"s_cpl={calib.s_cpl_per_layer.get(L, calib.s_cpl_default):.4f}")
            train_df = apply_per_layer_calibration(train_df, calib)
            valid_df = apply_per_layer_calibration(valid_df, calib)
            test_df  = apply_per_layer_calibration(test_df,  calib)
            cal_summary["calibration"] = {
                "s_gnd_per_layer": calib.s_gnd_per_layer,
                "s_cpl_per_layer": calib.s_cpl_per_layer,
                "s_gnd_default": calib.s_gnd_default,
                "s_cpl_default": calib.s_cpl_default,
                "n_train_nets": calib.n_train_nets,
            }

        # Validate after
        after_valid = validate_calibration(valid_df)
        after_test  = validate_calibration(test_df)
        print(f"  AFTER  (valid): gnd median={after_valid['median_ratio_gnd']:.3f}  "
              f"cpl median={after_valid['median_ratio_cpl']:.3f}")
        print(f"  AFTER  (test):  gnd median={after_test['median_ratio_gnd']:.3f}   "
              f"cpl median={after_test['median_ratio_cpl']:.3f}")
        cal_summary["validation_before_valid"] = before_valid
        cal_summary["validation_after_valid"] = after_valid
        cal_summary["validation_after_test"] = after_test

    # Provenance
    from configs import config_v3 as cfg
    snap = cfg.v3_snapshot()
    snap["task"] = "phase1_finetune_calibrated_smoke"
    snap["calibration"] = args.calibration
    snap["n_epochs"] = args.n_epochs
    write_provenance(args.output_dir, args.features_csv, snap, args.seed)

    # Model + config
    torch.manual_seed(args.seed)
    import math as _math
    hybrid_kwargs = {
        "hidden_dim": args.hidden_dim,
        "n_hidden": args.n_hidden,
    }
    if args.clamp_init is not None:
        hybrid_kwargs["clamp_bound"] = args.clamp_init
    model = HybridPexV3(**hybrid_kwargs)
    pc = model.parameter_count()
    print(f">>> model: hidden_dim={args.hidden_dim} n_hidden={args.n_hidden}  "
          f"params={pc}")

    config = FinetuneConfig(
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        log_every_n_steps=200,
        eval_every_n_epochs=1,
        curriculum_enabled=True,
        early_stop_patience=8,
    )

    # Day-1 eval (zero-init multiplier; should track CALIBRATED analytic baseline)
    print()
    print(f">>> Day-1 evaluation (zero-init residual; output = calibrated analytic) ...")
    valid_tensors = df_to_tensors(valid_df)
    day1 = evaluate_per_channel(model, valid_tensors, args.device)
    print(f"  day-1 valid: gnd={day1['gnd_mape_median']*100:.2f}%  "
          f"cpl={day1['cpl_mape_median']*100:.2f}%  "
          f"total={day1['total_mape_median']*100:.2f}%")

    # Train
    t0 = time.time()
    print()
    print(f">>> training {args.n_epochs} epochs ...")
    history = finetune_hybrid(model, train_df, valid_df, config, args.device)
    elapsed = time.time() - t0
    print(f">>> training done in {elapsed:.1f}s")

    # Final eval
    final_valid = evaluate_per_channel(model, valid_tensors, args.device)
    print(f">>> final valid: gnd={final_valid['gnd_mape_median']*100:.2f}%  "
          f"cpl={final_valid['cpl_mape_median']*100:.2f}%  "
          f"total={final_valid['total_mape_median']*100:.2f}%")

    test_tensors = df_to_tensors(test_df)
    final_test = evaluate_per_channel(model, test_tensors, args.device)
    print(f">>> final test (OOD nova+tv80s): gnd={final_test['gnd_mape_median']*100:.2f}%  "
          f"cpl={final_test['cpl_mape_median']*100:.2f}%  "
          f"total={final_test['total_mape_median']*100:.2f}%")

    # β gate
    beta = evaluate_beta_gate(model, valid_df, config, args.device)
    print(f">>> β-strategy gate: {beta['verdict']}")
    print(f"  gnd<8%? {beta['gate_gnd']}  cpl<8%? {beta['gate_cpl']}  total<4%? {beta['gate_total']}")

    # Week 1 Go/No-Go (Codex spec: total <= 8.5% AND gnd < 18%)
    week1_pass = (
        final_valid["total_mape_median"] <= 0.085
        and final_valid["gnd_mape_median"] < 0.18
    )
    print()
    print(f"=" * 60)
    print(f">>> WEEK 1 GO/NO-GO (Codex: total≤8.5% AND gnd<18%):")
    print(f"    total: {final_valid['total_mape_median']*100:.3f}% "
          f"(≤ 8.5%? {final_valid['total_mape_median'] <= 0.085})")
    print(f"    gnd:   {final_valid['gnd_mape_median']*100:.3f}% "
          f"(< 18%? {final_valid['gnd_mape_median'] < 0.18})")
    print(f"    → {'✅ PASS — proceed to 3-seed' if week1_pass else '❌ FAIL — go directly to mesh_v3 (Week 2)'}")
    print(f"=" * 60)

    # Save
    torch.save(model.state_dict(), args.output_dir / "model.pt")
    summary = {
        "seed": args.seed,
        "n_epochs": args.n_epochs,
        "elapsed_sec": elapsed,
        **cal_summary,
        "day1_valid": day1,
        "final_valid": final_valid,
        "final_test": final_test,
        "beta_gate": {
            "gate_gnd": beta["gate_gnd"],
            "gate_cpl": beta["gate_cpl"],
            "gate_total": beta["gate_total"],
            "beta_passed": beta["beta_passed"],
            "verdict": beta["verdict"],
        },
        "week1_pass": week1_pass,
        "best_epoch": history.best_epoch,
        "best_valid_total_mape": history.best_valid_total_mape,
        "best_valid_gnd_mape": history.best_valid_gnd_mape,
        "best_valid_cpl_mape": history.best_valid_cpl_mape,
    }
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    with open(args.output_dir / "history.json", "w") as f:
        json.dump({
            "step": history.step,
            "train_loss": history.train_loss,
            "valid_total_mape": history.valid_total_mape,
            "valid_gnd_mape": history.valid_gnd_mape,
            "valid_cpl_mape": history.valid_cpl_mape,
            "epoch_complete": history.epoch_complete,
        }, f, indent=2)

    print(f"✅ smoke complete. Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
