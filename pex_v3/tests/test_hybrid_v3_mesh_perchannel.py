"""
test_hybrid_v3_mesh_perchannel.py — Phase 1 Vector A1 invariants.

Validates:
  1. Param count within budget (≤ 100K, ideally < 90K).
  2. Day-1 invariant: zero-init residual heads → forward output = analytic.
  3. Gradient isolation: ∂loss_gnd/∂cpl_encoder == 0 AND vice-versa.
  4. Drop-in replacement for HybridPexV3Mesh.
"""
from __future__ import annotations
import math

import pytest
import torch

from src.models.hybrid_v3_mesh_perchannel import (
    HybridPexV3MeshPerChannel,
    DEFAULT_SELF_FEATURE_DIM,
    DEFAULT_PAIR_FEATURE_DIM,
)
from src.models.hybrid_v3_mesh import HybridPexV3Mesh


# Fixtures ---------------------------------------------------------------


def _make_batch(B: int = 8, N_max: int = 32, cuboid_in_dim: int = 10):
    torch.manual_seed(0)
    cuboids = torch.randn(B, N_max, cuboid_in_dim)
    # Random padding pattern
    n_valid = torch.randint(low=4, high=N_max + 1, size=(B,))
    padding_mask = torch.zeros(B, N_max)
    for b, nv in enumerate(n_valid):
        padding_mask[b, : int(nv)] = 1.0
    analytic_gnd = torch.rand(B).abs() + 0.1
    analytic_cpl = torch.rand(B).abs() + 0.1
    self_features = torch.randn(B, DEFAULT_SELF_FEATURE_DIM)
    pair_features = torch.randn(B, DEFAULT_PAIR_FEATURE_DIM)
    return {
        "cuboids": cuboids,
        "padding_mask": padding_mask,
        "analytic_gnd": analytic_gnd,
        "analytic_cpl": analytic_cpl,
        "self_features": self_features,
        "pair_features": pair_features,
    }


# 1. Param count ---------------------------------------------------------


def test_param_count():
    """Total params ≤ 100K (target ~ 53.8K)."""
    model = HybridPexV3MeshPerChannel()
    pc = model.parameter_count()
    assert pc["total"] <= 100_000, (
        f"param budget exceeded: {pc['total']} > 100K  (breakdown: {pc})"
    )
    # Sanity: encoders should have equal params.
    assert pc["gnd_encoder"] == pc["cpl_encoder"], (
        f"encoders should have identical param count, got {pc}"
    )
    # Sanity: at least one encoder + one residual head non-empty.
    assert pc["gnd_encoder"] > 0
    assert pc["gnd_residual"] > 0
    # Echo for visibility.
    print("PARAM COUNT:", pc)


# 2. Day-1 invariant -----------------------------------------------------


def test_day1_analytic_gnd():
    """Zero-init residual head → multiplier = 1.0 → pred_gnd = analytic_gnd."""
    torch.manual_seed(0)
    model = HybridPexV3MeshPerChannel()
    model.eval()
    batch = _make_batch()

    pred = model.predict_gnd(
        batch["analytic_gnd"],
        batch["self_features"],
        batch["cuboids"],
        batch["padding_mask"],
    )
    assert torch.allclose(pred, batch["analytic_gnd"], atol=1e-5), (
        f"day-1 gnd should equal analytic, got max abs diff "
        f"{(pred - batch['analytic_gnd']).abs().max().item():.3e}"
    )


def test_day1_analytic_cpl():
    """Zero-init cpl residual head → pred_cpl = analytic_cpl."""
    torch.manual_seed(0)
    model = HybridPexV3MeshPerChannel()
    model.eval()
    batch = _make_batch()

    pred = model.predict_cpl(
        batch["analytic_cpl"],
        batch["pair_features"],
        batch["cuboids"],
        batch["padding_mask"],
    )
    assert torch.allclose(pred, batch["analytic_cpl"], atol=1e-5), (
        f"day-1 cpl should equal analytic, got max abs diff "
        f"{(pred - batch['analytic_cpl']).abs().max().item():.3e}"
    )


# 3. Gradient isolation --------------------------------------------------


def test_gradient_isolation():
    """∂loss_gnd / ∂cpl_encoder == 0  AND  ∂loss_cpl / ∂gnd_encoder == 0.

    The whole point of A1: gnd training must NOT update cpl encoder, and
    cpl training must NOT update gnd encoder. With zero-init residual
    heads the residuals collapse to multiplier=1.0, but we need a real
    gradient signal — so we perturb the residual MLPs (NOT the encoders)
    so that downstream gradients are non-trivial.
    """
    torch.manual_seed(0)
    model = HybridPexV3MeshPerChannel()
    # Perturb ONLY the residual heads so encoders remain zero-init.
    with torch.no_grad():
        for p in model.gnd_residual.parameters():
            p.add_(0.01 * torch.randn_like(p))
        for p in model.cpl_residual.parameters():
            p.add_(0.01 * torch.randn_like(p))

    batch = _make_batch()
    # Make the cuboids require_grad off (they're inputs, not parameters).

    # --- gnd loss only ---
    model.zero_grad()
    pred_gnd = model.predict_gnd(
        batch["analytic_gnd"],
        batch["self_features"],
        batch["cuboids"],
        batch["padding_mask"],
    )
    loss_gnd = (pred_gnd - batch["analytic_gnd"] * 1.5).pow(2).mean()
    loss_gnd.backward()

    # gnd_encoder MUST have non-zero gradient (verify path is alive).
    gnd_enc_grad = sum(
        (p.grad.abs().sum() if p.grad is not None else torch.tensor(0.0))
        for p in model.gnd_encoder.parameters()
    )
    assert gnd_enc_grad > 0, (
        "gnd_encoder grad should be non-zero after gnd-only loss; "
        "path may be broken"
    )
    # cpl_encoder MUST have zero gradient.
    for name, p in model.cpl_encoder.named_parameters():
        if p.grad is None:
            continue  # None == zero, also fine
        assert p.grad.abs().max().item() == 0.0, (
            f"cpl_encoder.{name} grad leaked from gnd loss "
            f"(max |grad| = {p.grad.abs().max().item():.3e})"
        )

    # --- cpl loss only ---
    model.zero_grad()
    pred_cpl = model.predict_cpl(
        batch["analytic_cpl"],
        batch["pair_features"],
        batch["cuboids"],
        batch["padding_mask"],
    )
    loss_cpl = (pred_cpl - batch["analytic_cpl"] * 1.5).pow(2).mean()
    loss_cpl.backward()

    cpl_enc_grad = sum(
        (p.grad.abs().sum() if p.grad is not None else torch.tensor(0.0))
        for p in model.cpl_encoder.parameters()
    )
    assert cpl_enc_grad > 0, (
        "cpl_encoder grad should be non-zero after cpl-only loss; "
        "path may be broken"
    )
    for name, p in model.gnd_encoder.named_parameters():
        if p.grad is None:
            continue
        assert p.grad.abs().max().item() == 0.0, (
            f"gnd_encoder.{name} grad leaked from cpl loss "
            f"(max |grad| = {p.grad.abs().max().item():.3e})"
        )


# 4. Drop-in replacement for HybridPexV3Mesh -----------------------------


def test_drop_in_replacement():
    """API parity: same forward signature & set_clamp_bounds / parameter_count.

    A trainer currently using HybridPexV3Mesh should be able to swap the
    class name in (no other change) and still get a valid forward.
    """
    torch.manual_seed(0)
    baseline = HybridPexV3Mesh()
    a1 = HybridPexV3MeshPerChannel()

    batch = _make_batch()

    # Both expose predict_gnd / predict_cpl with the same arg names.
    out_b_gnd = baseline.predict_gnd(
        batch["analytic_gnd"], batch["self_features"],
        batch["cuboids"], batch["padding_mask"],
    )
    out_a_gnd = a1.predict_gnd(
        batch["analytic_gnd"], batch["self_features"],
        batch["cuboids"], batch["padding_mask"],
    )
    out_b_cpl = baseline.predict_cpl(
        batch["analytic_cpl"], batch["pair_features"],
        batch["cuboids"], batch["padding_mask"],
    )
    out_a_cpl = a1.predict_cpl(
        batch["analytic_cpl"], batch["pair_features"],
        batch["cuboids"], batch["padding_mask"],
    )
    # Output shapes must match.
    assert out_b_gnd.shape == out_a_gnd.shape
    assert out_b_cpl.shape == out_a_cpl.shape

    # set_clamp_bounds works without raising.
    a1.set_clamp_bounds(math.log(2.5))
    assert a1.gnd_residual.get_clamp_bound() == pytest.approx(math.log(2.5))
    assert a1.cpl_residual.get_clamp_bound() == pytest.approx(math.log(2.5))

    # parameter_count returns dict with 'total' key (trainer relies on it).
    pc = a1.parameter_count()
    assert "total" in pc
    assert isinstance(pc["total"], int)
