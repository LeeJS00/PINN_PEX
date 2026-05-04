#!/usr/bin/env python3
"""
09_pretrain_and_canary.py — Phase 1 K3 hard kill criterion check.

End-to-end: pretrain hybrid_v3 on synthetic Stage 1 + Stage 2 Mode A,
then run K3 transfer canary on real v3 features.

Verdict (per Codex round 2 mandate):
    PASS — pretrained init drops loss ≥50% faster than control over
           1k fine-tune steps. Synthetic strategy is a useful prior;
           proceed to Stage 3+ Q3D oracle (when needed).
    FAIL — K3 fires; abort synthetic strategy. Pretrain is not earning
           its compute. Fall back to direct real-BEOL fine-tune.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "pex_v3"))

import torch  # noqa: E402

from src.models.hybrid_v3 import HybridPexV3  # noqa: E402
from src.trainers.pretrain_synthetic_v3 import (  # noqa: E402
    PretrainConfig,
    pretrain_hybrid,
    check_pretrain_converged,
)
from src.trainers.transfer_canary_v3 import (  # noqa: E402
    CanaryConfig,
    run_transfer_canary,
)
from src.utils.seeds import set_all_seeds  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Phase 1 pretrain + K3 canary")
    p.add_argument("--features-csv", type=Path,
                   default=Path("/data/PINNPEX/data/processed_v3/intel22/features/all_designs.csv"))
    p.add_argument("--output-dir", type=Path,
                   default=_PROJECT_ROOT / "pex_v3" / "output" / "phase1_canary")
    p.add_argument("--pretrain-samples", type=int, default=5_000)
    p.add_argument("--pretrain-epochs", type=int, default=3)
    p.add_argument("--canary-nets", type=int, default=200)
    p.add_argument("--canary-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_all_seeds(args.seed, deterministic=True)

    print(f">>> Phase 1 pretrain + K3 canary, output: {args.output_dir}")
    t_start = time.time()

    # --- Pretrain ---
    print(f">>> Step 1/2: synthetic pretrain ({args.pretrain_samples} samples, "
          f"{args.pretrain_epochs} epochs)")
    model = HybridPexV3()
    pretrain_cfg = PretrainConfig(
        n_samples=args.pretrain_samples,
        n_epochs=args.pretrain_epochs,
        batch_size=128,
        lr=1e-3,
        seed=args.seed,
        stage_2_fraction=0.5,
    )
    history = pretrain_hybrid(model, pretrain_cfg, device="cpu")
    verdict_pretrain = check_pretrain_converged(history)
    print(f"  pretrain converged: {verdict_pretrain}")

    # Save pretrained ckpt + history
    ckpt_path = args.output_dir / "pretrained_ckpt.pt"
    torch.save(model.state_dict(), ckpt_path)
    with open(args.output_dir / "pretrain_history.json", "w") as f:
        json.dump({
            "step": history.step,
            "loss": history.loss,
            "multiplier_mean": history.multiplier_mean,
            "multiplier_max_dev": history.multiplier_max_dev,
        }, f, indent=2)
    with open(args.output_dir / "pretrain_verdict.json", "w") as f:
        json.dump(verdict_pretrain, f, indent=2)

    if not verdict_pretrain["converged"]:
        print(f"⚠️  pretrain did NOT converge — aborting before canary")
        sys.exit(1)

    # --- K3 Canary ---
    print(f">>> Step 2/2: K3 transfer canary ({args.canary_nets} nets, "
          f"{args.canary_steps} fine-tune steps)")
    canary_cfg = CanaryConfig(
        n_nets=args.canary_nets,
        n_finetune_steps=args.canary_steps,
        batch_size=64,
        lr=1e-3,
        seed=args.seed + 1,
        speedup_threshold=0.50,
    )
    canary = run_transfer_canary(
        pretrained_state_dict=torch.load(ckpt_path, map_location="cpu", weights_only=True),
        features_csv=args.features_csv,
        config=canary_cfg,
        device="cpu",
    )
    print(f"  K3 verdict: {canary['verdict']}  speedup={canary['speedup']*100:+.1f}%  "
          f"control={canary['control_final_loss']:.4f}  "
          f"pretrained={canary['pretrained_final_loss']:.4f}")

    with open(args.output_dir / "canary_verdict.json", "w") as f:
        json.dump(canary, f, indent=2)

    elapsed = time.time() - t_start
    print(f">>> Total elapsed: {elapsed:.1f}s")

    if canary["verdict"] == "PASS":
        print("✅ K3 PASSED — synthetic pretrain is a useful prior")
    else:
        print("❌ K3 FAILED — abort synthetic strategy or revisit feature/pretrain design")

    return canary["verdict"] == "PASS"


if __name__ == "__main__":
    main()
