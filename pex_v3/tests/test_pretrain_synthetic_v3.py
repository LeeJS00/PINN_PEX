"""
test_pretrain_synthetic_v3.py — Phase 1 pretrain harness invariants.

Validates:
  1. Dataset generates correct analytic == golden ground truth
  2. Pretrain converges multiplier → 1.0 (sanity gate)
  3. Stage 1-only and Stage 1+2 mix both work
  4. Day-1 (no training) loss = 0 because hybrid output = analytic
  5. Convergence threshold enforced
"""
from __future__ import annotations
import math

import numpy as np
import pytest
import torch

from src.models.hybrid_v3 import HybridPexV3
from src.synthetic.ground_truth import (
    parallel_plate_capacitance_fF,
    stacked_dielectric_capacitance_fF,
)
from src.trainers.pretrain_synthetic_v3 import (
    SyntheticPretrainDataset,
    PretrainConfig,
    pretrain_hybrid,
    check_pretrain_converged,
)


# ============================================================================
# Dataset correctness
# ============================================================================


def test_dataset_stage_1_analytic_matches_closed_form():
    """Each Stage-1 sample's analytic_C must match parallel_plate_capacitance_fF."""
    ds = SyntheticPretrainDataset(n_samples=50, seed=1, stage_2_fraction=0.0)
    for s in ds._cache:
        geo = s.geometry
        c_gt = parallel_plate_capacitance_fF(
            w_um=geo["w_um"], h_um=geo["h_um"],
            d_um=geo["d_um"], eps_r=geo["eps_r"],
        )
        rel = abs(s.analytic_C_gnd_fF - c_gt) / c_gt
        assert rel < 1e-9, f"stage 1 mismatch: {s.analytic_C_gnd_fF} vs {c_gt}"


def test_dataset_stage_2_analytic_matches_closed_form():
    ds = SyntheticPretrainDataset(n_samples=50, seed=2, stage_2_fraction=1.0)
    for s in ds._cache:
        geo = s.geometry
        c_gt = stacked_dielectric_capacitance_fF(
            w_um=geo["w_um"], h_um=geo["h_um"],
            layer_thicknesses_um=geo["thicknesses"],
            layer_eps_r=geo["eps_layers"],
        )
        rel = abs(s.analytic_C_gnd_fF - c_gt) / c_gt
        assert rel < 1e-9, f"stage 2 mismatch: {s.analytic_C_gnd_fF} vs {c_gt}"


def test_dataset_mix_fraction():
    """stage_2_fraction = 0.5 → roughly half stage-1 half stage-2."""
    ds = SyntheticPretrainDataset(n_samples=2000, seed=3, stage_2_fraction=0.5)
    n_stage_2 = sum(1 for s in ds._cache if s.geometry["stage"] == 2)
    # Tolerance ±5%
    frac = n_stage_2 / len(ds)
    assert 0.45 <= frac <= 0.55


def test_dataset_features_correct_dim():
    ds = SyntheticPretrainDataset(n_samples=10, seed=4, stage_2_fraction=0.5)
    for s in ds._cache:
        assert s.self_features.shape == (16,)
        assert s.pair_features.shape == (24,)


def test_dataset_dataloader_batches():
    """Smoke test: DataLoader iterates."""
    from torch.utils.data import DataLoader
    from src.trainers.pretrain_synthetic_v3 import _collate_pretrain
    ds = SyntheticPretrainDataset(n_samples=64, seed=5, stage_2_fraction=0.5)
    dl = DataLoader(ds, batch_size=16, collate_fn=_collate_pretrain)
    batches = list(dl)
    assert len(batches) == 4
    for b in batches:
        assert b["analytic_gnd"].shape == (16,)
        assert b["self_features"].shape == (16, 16)


# ============================================================================
# Day-1 invariant
# ============================================================================


def test_day1_pretrain_loss_is_zero():
    """Before any training, hybrid output = analytic ⇒ MAPE loss = 0."""
    torch.manual_seed(0)
    model = HybridPexV3()
    model.eval()
    ds = SyntheticPretrainDataset(n_samples=100, seed=6, stage_2_fraction=0.5)
    losses = []
    for i in range(len(ds)):
        sample = ds[i]
        analytic = sample["analytic_gnd"].unsqueeze(0)
        feats = sample["self_features"].unsqueeze(0)
        golden = sample["golden_gnd"].unsqueeze(0)
        pred = model.predict_gnd(analytic, feats)
        rel_err = (pred - golden).abs() / golden.clamp(min=1e-3)
        losses.append(float(rel_err.item()))
    max_err = max(losses)
    assert max_err < 1e-6, f"day-1 should be 0, got max {max_err}"


# ============================================================================
# Convergence (sanity gate)
# ============================================================================


@pytest.mark.parametrize("stage_2_fraction", [0.0, 0.5, 1.0])
def test_pretrain_converges_to_multiplier_1(stage_2_fraction):
    """Sanity gate: residual must learn multiplier ≈ 1.0 on synthetic."""
    torch.manual_seed(0)
    model = HybridPexV3()
    config = PretrainConfig(
        n_samples=2_000,
        n_epochs=3,
        batch_size=128,
        lr=1e-3,
        seed=7,
        stage_2_fraction=stage_2_fraction,
        log_every_n_steps=10,
    )
    history = pretrain_hybrid(model, config, device="cpu")
    verdict = check_pretrain_converged(history, threshold=0.05)
    assert verdict["converged"], f"FAILED: {verdict}"
    # Loss should be < 1% (since analytic == truth, residual just stays at 1)
    assert verdict["final_loss"] < 0.01


def test_pretrain_history_recorded():
    torch.manual_seed(0)
    model = HybridPexV3()
    config = PretrainConfig(
        n_samples=200, n_epochs=1, batch_size=64, log_every_n_steps=1
    )
    history = pretrain_hybrid(model, config, device="cpu")
    # 200/64 = ~4 steps per epoch
    assert len(history.step) >= 3
    assert len(history.loss) == len(history.step)


def test_check_pretrain_converged_thresholds():
    """The convergence checker enforces the multiplier-≈-1 invariant."""
    from src.trainers.pretrain_synthetic_v3 import PretrainHistory
    # Synthetic history that should pass
    h_ok = PretrainHistory(
        step=[100], loss=[0.001],
        multiplier_mean=[1.001], multiplier_max_dev=[0.02],
    )
    v = check_pretrain_converged(h_ok, threshold=0.05)
    assert v["converged"]

    # History that should fail (max_dev too high)
    h_bad = PretrainHistory(
        step=[100], loss=[0.5],
        multiplier_mean=[1.5], multiplier_max_dev=[2.0],
    )
    v = check_pretrain_converged(h_bad, threshold=0.05)
    assert not v["converged"]
