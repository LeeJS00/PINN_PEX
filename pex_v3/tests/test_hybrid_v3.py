"""
test_hybrid_v3.py — Phase 1 hybrid model invariants.

Validates:
  1. Day-1 (zero-init) hybrid predictions equal analytic baseline
  2. Gradients flow through both gnd_residual and cpl_residual
  3. Per-channel loss does NOT collapse to single total
  4. Bounded multiplier respected
  5. Per-pair-type variant works
  6. Param budget within A5 spec envelope (~30K)
"""
from __future__ import annotations
import math

import pytest
import torch

from src.models.hybrid_v3 import (
    HybridPexV3,
    per_channel_mape_loss,
    DEFAULT_SELF_FEATURE_DIM,
    DEFAULT_PAIR_FEATURE_DIM,
)
from src.models.analytic_base_v3 import analytic_parallel_plate


# ============================================================================
# 1. Day-1 invariant — hybrid output = analytic
# ============================================================================


def test_hybrid_day1_gnd_equals_analytic():
    """Zero-init residual → multiplier=1.0 → gnd_pred = analytic."""
    torch.manual_seed(0)
    model = HybridPexV3()
    model.eval()

    B = 50
    analytic = torch.randn(B).abs() + 0.1
    self_features = torch.randn(B, DEFAULT_SELF_FEATURE_DIM)

    pred = model.predict_gnd(analytic, self_features)
    assert torch.allclose(pred, analytic, rtol=1e-12), (
        f"day-1 gnd should equal analytic, got max diff {(pred - analytic).abs().max()}"
    )


def test_hybrid_day1_cpl_equals_analytic():
    torch.manual_seed(0)
    model = HybridPexV3()
    model.eval()

    E = 100
    analytic = torch.randn(E).abs() + 0.1
    pair_features = torch.randn(E, DEFAULT_PAIR_FEATURE_DIM)

    pred = model.predict_cpl(analytic, pair_features)
    assert torch.allclose(pred, analytic, rtol=1e-12)


# ============================================================================
# 2. Gradient flow
# ============================================================================


def test_hybrid_gradients_flow_through_both_heads():
    """Loss gradient reaches both gnd_residual and cpl_residual params."""
    torch.manual_seed(0)
    model = HybridPexV3()

    # Perturb model so gradients aren't trivially zero
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01 * torch.randn_like(p))

    B, E = 20, 30
    gnd_analytic = torch.randn(B).abs() + 0.5
    self_feats = torch.randn(B, DEFAULT_SELF_FEATURE_DIM)
    cpl_analytic = torch.randn(E).abs() + 0.5
    pair_feats = torch.randn(E, DEFAULT_PAIR_FEATURE_DIM)
    gold_gnd = gnd_analytic + 0.1 * torch.randn(B)
    gold_cpl = cpl_analytic + 0.1 * torch.randn(E)

    pred_gnd = model.predict_gnd(gnd_analytic, self_feats)
    pred_cpl = model.predict_cpl(cpl_analytic, pair_feats)
    losses = per_channel_mape_loss(pred_gnd, gold_gnd, pred_cpl, gold_cpl)
    losses["total_loss"].backward()

    # Both heads have non-zero gradient
    gnd_grads = [p.grad for p in model.gnd_residual.parameters() if p.requires_grad]
    cpl_grads = [p.grad for p in model.cpl_residual.parameters() if p.requires_grad]
    assert all(g is not None and g.abs().max() > 0 for g in gnd_grads)
    assert all(g is not None and g.abs().max() > 0 for g in cpl_grads)


# ============================================================================
# 3. Per-channel loss separation
# ============================================================================


def test_per_channel_loss_returns_separate_metrics():
    """Loss dict exposes gnd_mape and cpl_mape separately."""
    # gnd: rel errs (0.1/1.0, 0.1/2.0, 0.1/3.0) = (0.10, 0.05, 0.0333)
    pred_gnd = torch.tensor([1.1, 2.1, 3.1])
    gold_gnd = torch.tensor([1.0, 2.0, 3.0])
    # cpl: rel errs (0.05/0.5, 0.05/0.5) = (0.10, 0.10)
    pred_cpl = torch.tensor([0.45, 0.55])
    gold_cpl = torch.tensor([0.5, 0.5])
    out = per_channel_mape_loss(pred_gnd, gold_gnd, pred_cpl, gold_cpl)
    assert "gnd_mape" in out
    assert "cpl_mape" in out
    assert "total_loss" in out
    # gnd mean: (0.10 + 0.05 + 0.0333...) / 3 ≈ 0.0611
    assert abs(out["gnd_mape"].item() - 0.0611111) < 1e-4
    # cpl mean: 0.10
    assert abs(out["cpl_mape"].item() - 0.10) < 1e-4


def test_per_channel_loss_does_not_collapse_to_total():
    """A2 mandate: a model that overestimates gnd and underestimates cpl by
    equal absolute amount should NOT have zero per-channel loss."""
    pred_gnd = torch.tensor([2.0])  # overestimate by 1.0
    gold_gnd = torch.tensor([1.0])
    pred_cpl = torch.tensor([0.0])  # underestimate by 1.0
    gold_cpl = torch.tensor([1.0])
    out = per_channel_mape_loss(pred_gnd, gold_gnd, pred_cpl, gold_cpl)
    # total_loss = w_gnd * 1.0 + w_cpl * 1.0 = 2.0 (NOT zero)
    assert out["total_loss"].item() == pytest.approx(2.0, rel=1e-5)
    assert out["gnd_mape"].item() == pytest.approx(1.0, rel=1e-5)
    assert out["cpl_mape"].item() == pytest.approx(1.0, rel=1e-5)


def test_per_channel_loss_zero_target_handled():
    """When golden is zero, eps_fF clamping prevents divergence."""
    pred = torch.tensor([0.001, 0.0])
    gold = torch.tensor([0.0, 0.0])
    out = per_channel_mape_loss(pred, gold, pred, gold, eps_fF=1e-3)
    assert torch.isfinite(out["total_loss"]).all()


# ============================================================================
# 4. Bounded multiplier
# ============================================================================


def test_bounded_multiplier_after_perturbation():
    """Even after extreme MLP perturbation, output multiplier stays bounded."""
    torch.manual_seed(0)
    model = HybridPexV3(clamp_bound=math.log(2.0))
    # Force MLPs to large outputs
    with torch.no_grad():
        for layer in model.gnd_residual.mlp:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.fill_(50.0)
                layer.bias.fill_(50.0)

    analytic = torch.tensor([1.0])
    feats = torch.randn(1, DEFAULT_SELF_FEATURE_DIM) * 10
    pred = model.predict_gnd(analytic, feats)
    # Multiplier ∈ [0.5, 2.0] → pred ∈ [0.5, 2.0]
    assert 0.5 - 1e-3 <= pred.item() <= 2.0 + 1e-3


# ============================================================================
# 5. Curriculum hook
# ============================================================================


def test_set_clamp_bounds_updates_both_heads():
    model = HybridPexV3(clamp_bound=math.log(1.5))
    assert abs(model.gnd_residual.get_clamp_bound() - math.log(1.5)) < 1e-6
    assert abs(model.cpl_residual.get_clamp_bound() - math.log(1.5)) < 1e-6
    model.set_clamp_bounds(math.log(4.0))
    assert abs(model.gnd_residual.get_clamp_bound() - math.log(4.0)) < 1e-6
    assert abs(model.cpl_residual.get_clamp_bound() - math.log(4.0)) < 1e-6


# ============================================================================
# 6. Per-pair-type variant
# ============================================================================


def test_per_pair_type_variant_works():
    """When per_pair_clamp=True, predict_cpl requires pair_type_idx."""
    torch.manual_seed(0)
    model = HybridPexV3(per_pair_clamp=True)
    E = 30
    analytic = torch.randn(E).abs() + 0.5
    pair_feats = torch.randn(E, DEFAULT_PAIR_FEATURE_DIM)
    pair_type = torch.randint(0, 2, (E,))
    pred = model.predict_cpl(analytic, pair_feats, pair_type)
    assert pred.shape == (E,)


def test_per_pair_type_variant_requires_idx():
    model = HybridPexV3(per_pair_clamp=True)
    analytic = torch.tensor([1.0, 2.0])
    feats = torch.zeros(2, DEFAULT_PAIR_FEATURE_DIM)
    with pytest.raises(ValueError, match="pair_type_idx"):
        model.predict_cpl(analytic, feats, pair_type_idx=None)


# ============================================================================
# 7. Param budget (A5 spec §3)
# ============================================================================


def test_param_count_within_envelope():
    """A5 spec: ~5.7K + ~5.2K = ~11K minimal model. With our 24/16 dims
    should be within ~30K total."""
    model = HybridPexV3()
    pc = model.parameter_count()
    assert pc["total"] < 50_000, f"too many params: {pc}"
    assert pc["total"] > 5_000, f"suspiciously few params: {pc}"


# ============================================================================
# 8. End-to-end with real analytic_base
# ============================================================================


def test_hybrid_e2e_with_analytic_parallel_plate():
    """Full path: parallel-plate analytic → hybrid → loss → backward."""
    torch.manual_seed(0)
    model = HybridPexV3()
    # Perturb so we have non-zero gradient
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.005 * torch.randn_like(p))

    B = 20
    w = torch.tensor([2.0] * B, requires_grad=False)
    h = torch.tensor([3.0] * B, requires_grad=False)
    d = torch.tensor([0.3] * B, requires_grad=False)
    eps = torch.tensor([4.0] * B, requires_grad=False)
    analytic = analytic_parallel_plate(w, h, d, eps)
    feats = torch.randn(B, DEFAULT_SELF_FEATURE_DIM)
    pred_gnd = model.predict_gnd(analytic, feats)

    # Same for cpl
    E = 30
    cpl_analytic = analytic_parallel_plate(
        torch.tensor([1.0] * E),
        torch.tensor([1.0] * E),
        torch.tensor([0.5] * E),
        torch.tensor([4.0] * E),
    )
    pair_feats = torch.randn(E, DEFAULT_PAIR_FEATURE_DIM)
    pred_cpl = model.predict_cpl(cpl_analytic, pair_feats)

    gold_gnd = analytic + 0.1 * torch.randn(B)
    gold_cpl = cpl_analytic + 0.05 * torch.randn(E)
    losses = per_channel_mape_loss(pred_gnd, gold_gnd, pred_cpl, gold_cpl)
    losses["total_loss"].backward()
    assert torch.isfinite(losses["total_loss"]).item()
