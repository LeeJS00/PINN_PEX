#!/usr/bin/env python3
"""
10_finetune_hybrid_smoke.py — Phase 1 Tier 2 single-seed smoke on real v3.

Runs HybridPexV3 fine-tune on real v3 features for ONE seed to validate
the end-to-end pipeline + measure time/seed before committing to 5-seed.

After smoke pass, the 5-seed multi-GPU launcher (`11_finetune_hybrid_5seed.py`)
takes over.
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
from src.utils.seeds import set_all_seeds  # noqa: E402
from src.utils.manifest_hash import write_provenance  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Phase 1 Tier 2 smoke")
    p.add_argument(
        "--features-csv", type=Path,
        default=Path("/data/PINNPEX/data/processed_v3/intel22/features/all_designs.csv"),
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=_PROJECT_ROOT / "pex_v3" / "output" / "phase1_finetune_smoke",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_all_seeds(args.seed, deterministic=True)

    print(f">>> Phase 1 Tier 2 SMOKE — seed {args.seed}")
    print(f">>> features: {args.features_csv}")
    print(f">>> output:   {args.output_dir}")
    print(f">>> device:   {args.device}")

    # Load v3 features
    df = pd.read_csv(args.features_csv)
    print(f">>> loaded {len(df):,} rows × {len(df.columns)} cols")
    train_df, valid_df, test_df = split_by_manifest_column(df)
    # Filter zero-cap rows
    train_df = train_df[(train_df["c_gnd_fF"] + train_df["c_cpl_total_fF"]) > 1e-4].reset_index(drop=True)
    valid_df = valid_df[(valid_df["c_gnd_fF"] + valid_df["c_cpl_total_fF"]) > 1e-4].reset_index(drop=True)
    test_df  = test_df[ (test_df["c_gnd_fF"]  + test_df["c_cpl_total_fF"])  > 1e-4].reset_index(drop=True)
    print(f">>> splits: train={len(train_df):,}  valid={len(valid_df):,}  test={len(test_df):,}")

    # Provenance
    from configs import config_v3 as cfg
    snap = cfg.v3_snapshot()
    snap["task"] = "phase1_finetune_smoke"
    snap["n_epochs"] = args.n_epochs
    write_provenance(args.output_dir, args.features_csv, snap, args.seed)

    # Model + config
    torch.manual_seed(args.seed)
    model = HybridPexV3()
    pc = model.parameter_count()
    print(f">>> model params: {pc}")

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

    print(f">>> Day-1 evaluation (zero-init, expected: hybrid output = analytic baseline) ...")
    valid_tensors = df_to_tensors(valid_df)
    day1 = evaluate_per_channel(model, valid_tensors, args.device)
    print(f"  day-1 valid: gnd={day1['gnd_mape_median']*100:.2f}%  "
          f"cpl={day1['cpl_mape_median']*100:.2f}%  "
          f"total={day1['total_mape_median']*100:.2f}%  "
          f"(should ≈ Sakurai-Tamaru baseline since multiplier=1)")

    # Train
    t0 = time.time()
    print(f">>> training {args.n_epochs} epochs ...")
    history = finetune_hybrid(model, train_df, valid_df, config, args.device)
    elapsed = time.time() - t0
    print(f">>> training done in {elapsed:.1f}s")

    # Final eval
    final_valid = evaluate_per_channel(model, valid_tensors, args.device)
    print(f">>> final valid: gnd={final_valid['gnd_mape_median']*100:.2f}%  "
          f"cpl={final_valid['cpl_mape_median']*100:.2f}%  "
          f"total={final_valid['total_mape_median']*100:.2f}%")

    # Test split eval (OOD)
    test_tensors = df_to_tensors(test_df)
    final_test = evaluate_per_channel(model, test_tensors, args.device)
    print(f">>> final test (OOD nova+tv80s): gnd={final_test['gnd_mape_median']*100:.2f}%  "
          f"cpl={final_test['cpl_mape_median']*100:.2f}%  "
          f"total={final_test['total_mape_median']*100:.2f}%")

    # β gate
    beta = evaluate_beta_gate(model, valid_df, config, args.device)
    print(f">>> β-strategy gate: {beta['verdict']}")
    print(f"  gnd<8%? {beta['gate_gnd']}  cpl<8%? {beta['gate_cpl']}  total<4%? {beta['gate_total']}")

    # Save artifacts
    torch.save(model.state_dict(), args.output_dir / "model.pt")
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump({
            "seed": args.seed,
            "n_epochs": args.n_epochs,
            "elapsed_sec": elapsed,
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
            "best_epoch": history.best_epoch,
            "best_valid_total_mape": history.best_valid_total_mape,
            "best_valid_gnd_mape": history.best_valid_gnd_mape,
            "best_valid_cpl_mape": history.best_valid_cpl_mape,
        }, f, indent=2, default=str)

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
