"""
test_hybrid_v3_mesh_clamp_norm.py — ClampNorm variant invariants.

Validates:
  1. Param count == HybridPexV3Mesh baseline (44,738).
  2. Day-1 invariant: zero-init residuals → forward output = analytic.
  3. Below-threshold identity: ||δ|| < C → δ_eff == δ exactly.
  4. At/above-threshold projection: ||δ|| > C → ||δ_eff|| == C exactly.
  5. Drop-in replacement for HybridPexV3Mesh API.
"""
from __future__ import annotations
import math

import pytest
import torch

from src.models.hybrid_v3_mesh_clamp_norm import (
    HybridPexV3MeshClampNorm,
    DEFAULT_SELF_FEATURE_DIM,
    DEFAULT_PAIR_FEATURE_DIM,
)
from src.models.hybrid_v3_mesh import HybridPexV3Mesh


# Fixtures ---------------------------------------------------------------


def _make_batch(B: int = 8, N_max: int = 32, cuboid_in_dim: int = 10):
    torch.manual_seed(0)
    cuboids = torch.randn(B, N_max, cuboid_in_dim)
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
    """Total params must equal HybridPexV3Mesh baseline (44,738) exactly."""
    torch.manual_seed(0)
    baseline = HybridPexV3Mesh()
    cn = HybridPexV3MeshClampNorm()

    pc_b = baseline.parameter_count()
    pc_c = cn.parameter_count()

    assert pc_c["total"] == pc_b["total"], (
        f"ClampNorm total ({pc_c['total']}) must equal baseline "
        f"({pc_b['total']}); breakdown CN={pc_c}, baseline={pc_b}"
    )
    assert pc_c["total"] == 44_738, (
        f"Expected 44,738 params, got {pc_c['total']}"
    )
    # Per-submodule parity (encoder + each head).
    assert pc_c["cuboid_encoder"] == pc_b["cuboid_encoder"]
    assert pc_c["gnd_residual"] == pc_b["gnd_residual"]
    assert pc_c["cpl_residual"] == pc_b["cpl_residual"]
    print("PARAM COUNT (ClampNorm):", pc_c)


# 2. Day-1 invariant -----------------------------------------------------


def test_day1_analytic():
    """Zero-init residuals → multiplier = 1.0 → pred == analytic for both heads.

    Tests `_predict_joint` (the canonical training entry point) to ensure
    the joint-norm clamp does not perturb the day-1 baseline.
    """
    torch.manual_seed(0)
    model = HybridPexV3MeshClampNorm()
    model.eval()
    batch = _make_batch()

    pred_gnd, pred_cpl = model._predict_joint(
        batch["analytic_gnd"],
        batch["analytic_cpl"],
        batch["self_features"],
        batch["pair_features"],
        batch["cuboids"],
        batch["padding_mask"],
    )
    assert torch.allclose(pred_gnd, batch["analytic_gnd"], atol=1e-5), (
        f"day-1 gnd should equal analytic; max abs diff = "
        f"{(pred_gnd - batch['analytic_gnd']).abs().max().item():.3e}"
    )
    assert torch.allclose(pred_cpl, batch["analytic_cpl"], atol=1e-5), (
        f"day-1 cpl should equal analytic; max abs diff = "
        f"{(pred_cpl - batch['analytic_cpl']).abs().max().item():.3e}"
    )

    # Also probe the standalone-API fallbacks.
    pg = model.predict_gnd(
        batch["analytic_gnd"], batch["self_features"],
        batch["cuboids"], batch["padding_mask"],
    )
    pc = model.predict_cpl(
        batch["analytic_cpl"], batch["pair_features"],
        batch["cuboids"], batch["padding_mask"],
    )
    assert torch.allclose(pg, batch["analytic_gnd"], atol=1e-5)
    assert torch.allclose(pc, batch["analytic_cpl"], atol=1e-5)


# 3. Below-threshold identity --------------------------------------------


def test_clamp_below_threshold():
    """When ||δ|| < C, the projection is the identity: δ_eff == δ."""
    torch.manual_seed(0)
    model = HybridPexV3MeshClampNorm()
    cap = 1.386  # Phase 2 cap value
    model.set_clamp_bounds(cap)

    # Construct logits well within the cap.
    B = 16
    logit_gnd = torch.randn(B) * 0.1   # ||·|| ≈ O(0.1)
    logit_cpl = torch.randn(B) * 0.1
    n = torch.sqrt(logit_gnd ** 2 + logit_cpl ** 2)
    assert (n < cap).all(), (
        f"setup error: some n exceed cap ({n.max().item():.3f} > {cap})"
    )

    eff_gnd, eff_cpl = model._norm_project(logit_gnd, logit_cpl)
    assert torch.allclose(eff_gnd, logit_gnd, atol=1e-7), (
        f"below-threshold identity failed for gnd; max diff = "
        f"{(eff_gnd - logit_gnd).abs().max().item():.3e}"
    )
    assert torch.allclose(eff_cpl, logit_cpl, atol=1e-7), (
        f"below-threshold identity failed for cpl; max diff = "
        f"{(eff_cpl - logit_cpl).abs().max().item():.3e}"
    )


# 4. At/above-threshold projection ---------------------------------------


def test_clamp_at_threshold():
    """When ||δ|| > C, ||δ_eff|| == C exactly (projection onto sphere)."""
    torch.manual_seed(0)
    model = HybridPexV3MeshClampNorm()
    cap = 0.5
    model.set_clamp_bounds(cap)

    # Construct logits with ||·|| significantly > cap.
    B = 16
    raw = torch.randn(B, 2)
    raw = raw / raw.norm(dim=1, keepdim=True) * 2.0   # ||·|| = 2.0 each
    logit_gnd = raw[:, 0]
    logit_cpl = raw[:, 1]
    n = torch.sqrt(logit_gnd ** 2 + logit_cpl ** 2)
    assert torch.allclose(n, torch.full_like(n, 2.0), atol=1e-5), (
        f"setup error: norms not 2.0, got {n.tolist()}"
    )

    eff_gnd, eff_cpl = model._norm_project(logit_gnd, logit_cpl)
    n_eff = torch.sqrt(eff_gnd ** 2 + eff_cpl ** 2)
    assert torch.allclose(n_eff, torch.full_like(n_eff, cap), atol=1e-5), (
        f"above-threshold projection failed; expected ||δ_eff||={cap}, "
        f"got {n_eff.tolist()}"
    )

    # Direction preserved: δ_eff parallel to δ → cosine = 1.
    cos = (logit_gnd * eff_gnd + logit_cpl * eff_cpl) / (
        n * n_eff
    )
    assert torch.allclose(cos, torch.ones_like(cos), atol=1e-5), (
        f"projection should preserve direction; got cosines {cos.tolist()}"
    )


# 4b. Gradient stability at day-1 (NaN regression guard) ----------------


def test_gradient_finite_at_day1():
    """Backward pass at day-1 (δ=0) must not produce NaN/Inf gradients.

    The naive `sqrt(δ_gnd² + δ_cpl²)` has infinite derivative at δ=0,
    so without the soft-sqrt mitigation the very first training step
    explodes. This test guards against that regression.
    """
    torch.manual_seed(0)
    model = HybridPexV3MeshClampNorm()
    model.train()
    batch = _make_batch()

    pg, pc = model._predict_joint(
        batch["analytic_gnd"], batch["analytic_cpl"],
        batch["self_features"], batch["pair_features"],
        batch["cuboids"], batch["padding_mask"],
    )
    # Trivial loss; values don't matter as long as gradient is finite.
    loss = (pg - 1.0).pow(2).mean() + (pc - 1.0).pow(2).mean()
    loss.backward()

    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        finite = torch.isfinite(p.grad).all().item()
        assert finite, (
            f"non-finite gradient at day-1 for parameter {name!r}: "
            f"any-NaN={torch.isnan(p.grad).any().item()}, "
            f"any-Inf={torch.isinf(p.grad).any().item()}"
        )


# 5. Drop-in replacement -------------------------------------------------


def test_drop_in_replacement():
    """API parity with HybridPexV3Mesh: same predict_* signatures, set_clamp,
    parameter_count returns dict with 'total'.
    """
    torch.manual_seed(0)
    baseline = HybridPexV3Mesh()
    cn = HybridPexV3MeshClampNorm()
    batch = _make_batch()

    # Same forward signature.
    out_b_gnd = baseline.predict_gnd(
        batch["analytic_gnd"], batch["self_features"],
        batch["cuboids"], batch["padding_mask"],
    )
    out_c_gnd = cn.predict_gnd(
        batch["analytic_gnd"], batch["self_features"],
        batch["cuboids"], batch["padding_mask"],
    )
    out_b_cpl = baseline.predict_cpl(
        batch["analytic_cpl"], batch["pair_features"],
        batch["cuboids"], batch["padding_mask"],
    )
    out_c_cpl = cn.predict_cpl(
        batch["analytic_cpl"], batch["pair_features"],
        batch["cuboids"], batch["padding_mask"],
    )
    assert out_b_gnd.shape == out_c_gnd.shape
    assert out_b_cpl.shape == out_c_cpl.shape

    # set_clamp_bounds works.
    cn.set_clamp_bounds(math.log(2.5))
    assert cn.gnd_residual.get_clamp_bound() == pytest.approx(math.log(2.5))
    assert cn.cpl_residual.get_clamp_bound() == pytest.approx(math.log(2.5))

    # parameter_count exposes 'total'.
    pc = cn.parameter_count()
    assert "total" in pc
    assert isinstance(pc["total"], int)
