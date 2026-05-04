"""
test_finetune_hybrid_v3.py — Phase 1 Tier 2 fine-tune harness invariants.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.hybrid_v3 import HybridPexV3
from src.trainers.finetune_hybrid_v3 import (
    df_to_tensors,
    split_by_manifest_column,
    evaluate_per_channel,
    evaluate_beta_gate,
    FinetuneConfig,
    finetune_hybrid,
)


def _make_real_features_fixture(n_per_split: int = 100, seed: int = 0) -> pd.DataFrame:
    """Mimic the v3 features CSV schema."""
    rng = np.random.default_rng(seed)
    rows = []
    for split in ["train", "valid", "test"]:
        for i in range(n_per_split):
            d = {
                "design_name": f"intel22_{split}",
                "net_name": f"{split}_net_{i}",
                "split": split,
                "c_gnd_fF": float(np.abs(rng.normal(1.0, 2.0)) + 0.1),
                "c_cpl_total_fF": float(np.abs(rng.normal(0.5, 1.0)) + 0.05),
                "compact_gnd_estimate_fF": float(np.abs(rng.normal(1.0, 2.0)) + 0.1),
                "compact_cpl_estimate_total_fF": float(np.abs(rng.normal(0.5, 1.0)) + 0.05),
                "total_wire_length_um": float(rng.uniform(1, 100)),
                "total_metal_area_um2": float(rng.uniform(0.1, 10)),
                "n_cuboids": float(rng.integers(1, 100)),
                "bbox_xy_um2": float(rng.uniform(0.1, 100)),
                "bbox_z_um": float(rng.uniform(0.05, 5)),
                "n_aggressor_nets": float(rng.integers(0, 256)),
                "n_layers_present": float(rng.integers(1, 9)),
                "eps_mean": float(rng.uniform(2, 6)),
                "vss_shield_M1_M3": float(rng.uniform(0, 10)),
                "vss_shield_M4_M5": float(rng.uniform(0, 10)),
                "vss_shield_M6_plus": float(rng.uniform(0, 5)),
                "density_M1_M3": float(rng.uniform(0, 0.5)),
                "density_M4_M5": float(rng.uniform(0, 0.5)),
                "density_M6_plus": float(rng.uniform(0, 0.2)),
                "broadside_overlap_total_um2": float(rng.uniform(0, 5)),
                "broadside_overlap_p95_um2": float(rng.uniform(0, 1)),
                "lateral_overlap_total_um2": float(rng.uniform(0, 5)),
                "lateral_overlap_p95_um2": float(rng.uniform(0, 1)),
                "spacing_min_um": float(rng.uniform(0.1, 4)),
                "spacing_p25_um": float(rng.uniform(0.5, 4)),
                "spacing_p50_um": float(rng.uniform(0.5, 4)),
                "spacing_p95_um": float(rng.uniform(2, 4)),
                "n_edges_lt_1um": float(rng.integers(0, 50)),
                "n_edges_1_to_3um": float(rng.integers(0, 100)),
                "n_edges_3_to_4um": float(rng.integers(0, 100)),
                "fanout": float(rng.integers(1, 20)),
                "aspect_ratio": float(rng.uniform(0.1, 10)),
            }
            rows.append(d)
    return pd.DataFrame(rows)


# ============================================================================
# Data loading
# ============================================================================


def test_df_to_tensors_shapes():
    df = _make_real_features_fixture(n_per_split=10)
    tensors = df_to_tensors(df.head(10))
    assert tensors["analytic_gnd"].shape == (10,)
    assert tensors["analytic_cpl"].shape == (10,)
    assert tensors["self_features"].shape == (10, 16)
    assert tensors["pair_features"].shape == (10, 24)
    assert tensors["golden_gnd"].shape == (10,)
    assert tensors["golden_cpl"].shape == (10,)


def test_split_by_manifest_column():
    df = _make_real_features_fixture(n_per_split=20)
    train, valid, test = split_by_manifest_column(df)
    assert len(train) == 20
    assert len(valid) == 20
    assert len(test) == 20


def test_df_to_tensors_no_nan():
    df = _make_real_features_fixture(n_per_split=50)
    tensors = df_to_tensors(df)
    assert torch.isfinite(tensors["self_features"]).all()
    assert torch.isfinite(tensors["pair_features"]).all()
    assert torch.isfinite(tensors["analytic_gnd"]).all()


# ============================================================================
# Day-1 evaluation = analytic
# ============================================================================


def test_day1_evaluation_loss_is_zero_when_analytic_equals_golden():
    """If we synthesize df where compact_gnd == c_gnd, day-1 model loss = 0."""
    df = _make_real_features_fixture(n_per_split=50)
    df["compact_gnd_estimate_fF"] = df["c_gnd_fF"]
    df["compact_cpl_estimate_total_fF"] = df["c_cpl_total_fF"]

    torch.manual_seed(0)
    model = HybridPexV3()
    model.eval()

    tensors = df_to_tensors(df.head(50))
    metrics = evaluate_per_channel(model, tensors, device="cpu")
    # All MAPEs should be 0 (or close)
    assert metrics["gnd_mape_median"] < 1e-4
    assert metrics["cpl_mape_median"] < 1e-4
    assert metrics["total_mape_median"] < 1e-4


# ============================================================================
# Training loop
# ============================================================================


def test_finetune_runs_end_to_end():
    df = _make_real_features_fixture(n_per_split=80)
    train, valid, _ = split_by_manifest_column(df)
    torch.manual_seed(0)
    model = HybridPexV3()
    config = FinetuneConfig(
        n_epochs=2,
        batch_size=32,
        lr=1e-3,
        seed=42,
        log_every_n_steps=1,
        eval_every_n_epochs=1,
        curriculum_enabled=True,
    )
    history = finetune_hybrid(model, train, valid, config, device="cpu")
    assert len(history.train_loss) > 0
    assert len(history.valid_total_mape) > 0
    assert history.best_epoch >= 0


def test_finetune_loss_decreases():
    """On synthetic data where analytic ≠ golden, loss should decrease."""
    df = _make_real_features_fixture(n_per_split=100, seed=1)
    # Make analytic systematically biased so model has signal to learn
    df["compact_gnd_estimate_fF"] = df["c_gnd_fF"] * 0.5
    df["compact_cpl_estimate_total_fF"] = df["c_cpl_total_fF"] * 1.5
    train, valid, _ = split_by_manifest_column(df)
    torch.manual_seed(0)
    model = HybridPexV3()
    config = FinetuneConfig(
        n_epochs=15,
        batch_size=32,
        lr=5e-3,
        seed=42,
        log_every_n_steps=1,
        eval_every_n_epochs=1,
        early_stop_patience=20,    # disable early stopping for this test
        curriculum_enabled=False,  # constant clamp for cleaner signal
    )
    history = finetune_hybrid(model, train, valid, config, device="cpu")
    # Train loss should decrease
    n = len(history.train_loss)
    early = sum(history.train_loss[: n // 4]) / max(1, n // 4)
    late = sum(history.train_loss[-n // 4 :]) / max(1, n // 4)
    assert late < early, f"loss did not decrease: early={early:.4f} late={late:.4f}"


# ============================================================================
# Beta gate
# ============================================================================


def test_beta_gate_returns_verdict():
    df = _make_real_features_fixture(n_per_split=50)
    _, valid, _ = split_by_manifest_column(df)
    torch.manual_seed(0)
    model = HybridPexV3()
    config = FinetuneConfig()
    out = evaluate_beta_gate(model, valid, config)
    assert "verdict" in out
    assert "beta_passed" in out
    assert "gate_gnd" in out
    assert "gate_cpl" in out
    assert "gate_total" in out


def test_beta_gate_pass_when_analytic_matches():
    """If analytic == golden, day-1 model passes β gate trivially."""
    df = _make_real_features_fixture(n_per_split=50)
    df["compact_gnd_estimate_fF"] = df["c_gnd_fF"]
    df["compact_cpl_estimate_total_fF"] = df["c_cpl_total_fF"]
    _, valid, _ = split_by_manifest_column(df)
    torch.manual_seed(0)
    model = HybridPexV3()
    config = FinetuneConfig()
    out = evaluate_beta_gate(model, valid, config)
    assert out["beta_passed"]
    assert out["gate_gnd"]
    assert out["gate_cpl"]
    assert out["gate_total"]


def test_beta_gate_fail_when_analytic_misses():
    df = _make_real_features_fixture(n_per_split=50)
    # Analytic systematically wrong
    df["compact_gnd_estimate_fF"] = df["c_gnd_fF"] * 0.1   # analytic =  golden / 10
    df["compact_cpl_estimate_total_fF"] = df["c_cpl_total_fF"] * 5.0
    _, valid, _ = split_by_manifest_column(df)
    torch.manual_seed(0)
    model = HybridPexV3()  # day-1: pred = analytic
    config = FinetuneConfig()
    out = evaluate_beta_gate(model, valid, config)
    assert not out["beta_passed"]
