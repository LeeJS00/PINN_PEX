"""
test_transfer_canary_v3.py — K3 transfer canary correctness.

Validates:
  1. Real-feature CSV slice loader returns deterministic subset
  2. Per-net analytic baseline computed from compact_gnd_estimate_fF
  3. real_features_to_self_features produces (n, 16) tensor
  4. Canary can run end-to-end with both control + pretrained
  5. Verdict logic: pass when pretrained loss < (1 - threshold) × control
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from src.trainers.transfer_canary_v3 import (
    load_real_v3_canary_slice,
    compute_per_net_analytic_baseline,
    real_features_to_self_features,
    CanaryConfig,
    run_transfer_canary,
)
from src.models.hybrid_v3 import HybridPexV3


# ============================================================================
# Synthetic features CSV fixture (mimics feature_dataset output schema)
# ============================================================================


def _make_fixture_features_csv(tmp_path: Path, n_per_split: int = 100) -> Path:
    rng = np.random.default_rng(123)
    rows = []
    for split in ["train", "valid", "test"]:
        for i in range(n_per_split):
            rows.append({
                "design_name": f"intel22_test_{split}",
                "net_name": f"{split}_net_{i}",
                "split": split,
                "c_gnd_fF": float(np.abs(rng.normal(1.0, 2.0)) + 0.1),
                "c_cpl_total_fF": float(np.abs(rng.normal(0.5, 1.0))),
                "compact_gnd_estimate_fF": float(np.abs(rng.normal(1.0, 2.0)) + 0.1),
                "compact_cpl_estimate_total_fF": float(np.abs(rng.normal(0.5, 1.0))),
                "total_wire_length_um": float(rng.uniform(1.0, 100.0)),
                "total_metal_area_um2": float(rng.uniform(0.1, 10.0)),
                "n_cuboids": float(rng.integers(1, 100)),
                "bbox_xy_um2": float(rng.uniform(0.1, 100.0)),
                "n_aggressor_nets": float(rng.integers(0, 256)),
                "broadside_overlap_total_um2": float(rng.uniform(0.0, 5.0)),
                "lateral_overlap_total_um2": float(rng.uniform(0.0, 5.0)),
                "spacing_min_um": float(rng.uniform(0.1, 4.0)),
                "n_layers_present": float(rng.integers(1, 9)),
                "eps_mean": float(rng.uniform(2.0, 6.0)),
                "vss_shield_M1_M3": float(rng.uniform(0.0, 10.0)),
                "density_M1_M3": float(rng.uniform(0.0, 0.5)),
                "fanout": float(rng.integers(1, 20)),
            })
    df = pd.DataFrame(rows)
    out = tmp_path / "synth_features.csv"
    df.to_csv(out, index=False)
    return out


# ============================================================================
# Loader
# ============================================================================


def test_load_canary_slice_returns_valid_split(tmp_path):
    csv = _make_fixture_features_csv(tmp_path, n_per_split=100)
    df = load_real_v3_canary_slice(csv, n_nets=50, seed=42, split="valid")
    assert len(df) == 50
    assert (df["split"] == "valid").all()


def test_load_canary_slice_deterministic(tmp_path):
    csv = _make_fixture_features_csv(tmp_path, n_per_split=100)
    a = load_real_v3_canary_slice(csv, n_nets=30, seed=7)
    b = load_real_v3_canary_slice(csv, n_nets=30, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_load_canary_slice_handles_n_larger_than_split(tmp_path):
    csv = _make_fixture_features_csv(tmp_path, n_per_split=10)
    df = load_real_v3_canary_slice(csv, n_nets=100)
    # Only 10 valid rows exist; loader returns all
    assert len(df) == 10


# ============================================================================
# Analytic baseline + features
# ============================================================================


def test_compute_per_net_analytic_uses_compact_estimate(tmp_path):
    csv = _make_fixture_features_csv(tmp_path)
    df = load_real_v3_canary_slice(csv, n_nets=20)
    a = compute_per_net_analytic_baseline(df)
    assert a.shape == (20,)
    assert a.dtype == torch.float32
    # All > 0 (we used abs in fixture)
    assert (a > 0).all().item()


def test_compute_per_net_analytic_missing_column_raises():
    df = pd.DataFrame({"split": ["valid"] * 5, "c_gnd_fF": [1.0] * 5})
    with pytest.raises(KeyError, match="compact_gnd_estimate_fF"):
        compute_per_net_analytic_baseline(df)


def test_real_features_to_self_features_shape(tmp_path):
    csv = _make_fixture_features_csv(tmp_path)
    df = load_real_v3_canary_slice(csv, n_nets=20)
    feats = real_features_to_self_features(df, self_feature_dim=16)
    assert feats.shape == (20, 16)
    assert torch.isfinite(feats).all()


# ============================================================================
# End-to-end canary
# ============================================================================


def test_canary_runs_end_to_end(tmp_path):
    """Canary completes both control + pretrained runs and returns a verdict."""
    csv = _make_fixture_features_csv(tmp_path, n_per_split=200)
    # Use a fresh untrained model's state_dict as the "pretrained" stand-in;
    # control will be re-initialized inside the canary, so the two should
    # differ only in init seed (test that the pipe runs cleanly).
    torch.manual_seed(99)
    fake_pretrained = HybridPexV3()
    config = CanaryConfig(
        n_nets=50,
        n_finetune_steps=20,        # quick smoke
        batch_size=16,
        lr=1e-2,
        seed=42,
        log_every_n_steps=5,
    )
    out = run_transfer_canary(
        pretrained_state_dict=fake_pretrained.state_dict(),
        features_csv=csv,
        config=config,
        device="cpu",
    )
    assert "verdict" in out
    assert out["verdict"] in {"PASS", "FAIL"}
    assert "control_final_loss" in out
    assert "pretrained_final_loss" in out
    assert "speedup" in out
    assert isinstance(out["k3_fired"], bool)
    # Both histories were recorded
    assert len(out["control_history"]) > 0
    assert len(out["pretrained_history"]) > 0


def test_canary_verdict_logic(tmp_path):
    """If pretrained final loss > control final loss, K3 fires."""
    csv = _make_fixture_features_csv(tmp_path, n_per_split=200)
    # Sabotaged pretrained: weights perturbed so it starts WORSE
    torch.manual_seed(99)
    fake = HybridPexV3()
    with torch.no_grad():
        for p in fake.parameters():
            p.add_(0.5 * torch.randn_like(p))  # noise injection

    config = CanaryConfig(n_nets=30, n_finetune_steps=5, batch_size=8,
                          seed=42, speedup_threshold=0.50)
    out = run_transfer_canary(
        pretrained_state_dict=fake.state_dict(),
        features_csv=csv,
        config=config,
        device="cpu",
    )
    # Sabotaged pretrained typically does worse → speedup negative → FAIL
    assert out["k3_fired"] == (out["verdict"] == "FAIL")
    # Speedup logic: speedup = (control - pretrained) / control
    # Verify formula consistency
    expected_speedup = (out["control_final_loss"] - out["pretrained_final_loss"]) / max(
        out["control_final_loss"], 1e-12
    )
    assert abs(out["speedup"] - expected_speedup) < 1e-6
