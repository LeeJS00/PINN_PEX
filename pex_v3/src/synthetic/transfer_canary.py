"""
transfer_canary.py — Hard kill K3 gate for synthetic pretraining.

Codex round 2 mandate: do NOT commit Q3D compute (Stage 3+, ~3000 GPU-h)
unless Stage 1-2 pretraining transfers to real BEOL data.

Procedure:
    1. Take Stage 1+2 pretrained checkpoint
    2. Finetune on a 500-1000 net subset of real intel22 data (v3 manifest)
    3. Compare initial loss + 1000-step loss to a no-pretrain control
    4. Decide:
       - if 1000-step loss is ≥ 50% lower than control → CONTINUE to Stage 3
       - else → STOP synthetic strategy entirely (K3 fired)

This is THE go/no-go gate for the synthetic investment.
"""
from __future__ import annotations
from pathlib import Path


def run_transfer_canary(
    pretrained_ckpt_path: Path,
    canary_subset_size: int,
    n_finetune_steps: int,
    output_dir: Path,
    seed: int,
) -> dict:
    """Run the canary protocol. Returns dict with verdict.

    Returns:
        {
            "verdict": "PASS" | "FAIL",
            "control_loss_at_1000_steps": float,
            "pretrained_loss_at_1000_steps": float,
            "improvement_pct": float,    # negative = pretrained worse
            "k3_fired": bool,
            "rationale": str,
        }
    """
    raise NotImplementedError(
        "Phase 1 scaffold. Implementation:\n"
        "  1. Load pretrained ckpt; clone weights\n"
        "  2. Build 2 trainers (control = fresh init, pretrained = ckpt init)\n"
        "  3. Run both for n_finetune_steps with same seed, same data\n"
        "  4. Compare losses; emit verdict\n"
    )
