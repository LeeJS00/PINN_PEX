"""
test_hybrid_v3_mesh_input_subset_clamp_norm.py — combined-stack invariants.

Validates the combined `HybridPexV3MeshInputSubsetClampNorm` against:
  1. Param count == 44,738 (exact baseline parity).
  2. Day-1 invariant: zero-init residuals + joint-norm clamp + masked input
     → multiplier = 1.0 → pred = analytic.
  3. InputSubset mask correctness on the gnd encoder input (channels 6, 7, 9
     zeroed) and identity on the cpl encoder input.
  4. Shared encoder identity: exactly ONE CuboidSetEncoder instance, used
     by both gnd and cpl paths.
  5. ClampNorm below-threshold identity: ||δ|| < C → δ_eff == δ.
  6. ClampNorm at/above-threshold projection: ||δ|| > C → ||δ_eff|| == C.
  7. Gradient finite at day-1 (no NaN/Inf from softened sqrt).
  8. Drop-in replacement for HybridPexV3Mesh (API parity).
"""
from __future__ import annotations
import math

import pytest
import torch

from src.models.hybrid_v3_mesh_input_subset_clamp_norm import (
    HybridPexV3MeshInputSubsetClampNorm,
    DEFAULT_SELF_FEATURE_DIM,
    DEFAULT_PAIR_FEATURE_DIM,
)
from src.models.hybrid_v3_mesh_input_subset import DEFAULT_INTERACTION_CHANNELS
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
    """Total params must equal HybridPexV3Mesh baseline (44,738) exactly.

    Both InputSubset and ClampNorm add 0 trainable parameters; the only
    extras are non-trainable buffers (channel masks, clamp_bound).
    """
    torch.manual_seed(0)
    baseline = HybridPexV3Mesh()
    combined = HybridPexV3MeshInputSubsetClampNorm()

    pc_b = baseline.parameter_count()
    pc_c = combined.parameter_count()

    assert pc_c["total"] == pc_b["total"], (
        f"Combined total ({pc_c['total']}) must equal baseline "
        f"({pc_b['total']}); breakdown CC={pc_c}, baseline={pc_b}"
    )
    assert pc_c["total"] == 44_738, (
        f"Expected 44,738 params, got {pc_c['total']}"
    )
    assert pc_c["cuboid_encoder"] == pc_b["cuboid_encoder"]
    assert pc_c["gnd_residual"] == pc_b["gnd_residual"]
    assert pc_c["cpl_residual"] == pc_b["cpl_residual"]
    print("PARAM COUNT (Combined):", pc_c)


# 2. Day-1 invariant -----------------------------------------------------


def test_day1_analytic():
    """Zero-init residuals → multiplier = 1.0 → pred == analytic.

    Tests `_predict_joint` (the canonical training entry point) and the
    standalone API fallbacks. The composition (input mask + softened-sqrt
    norm clamp) must NOT perturb the day-1 baseline.
    """
    torch.manual_seed(0)
    model = HybridPexV3MeshInputSubsetClampNorm()
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

    pg = model.predict_gnd(
        batch["analytic_gnd"], batch["self_features"],
        batch["cuboids"], batch["padding_mask"],
    )
    pc_ = model.predict_cpl(
        batch["analytic_cpl"], batch["pair_features"],
        batch["cuboids"], batch["padding_mask"],
    )
    assert torch.allclose(pg, batch["analytic_gnd"], atol=1e-5)
    assert torch.allclose(pc_, batch["analytic_cpl"], atol=1e-5)


# 3. InputSubset mask correctness ----------------------------------------


def test_input_mask_correctness():
    """gnd input zeros interaction columns (6, 7, 9); cpl input keeps them."""
    torch.manual_seed(0)
    model = HybridPexV3MeshInputSubsetClampNorm()
    batch = _make_batch()

    gnd_input = model._gnd_input(batch["cuboids"])
    cpl_input = model._cpl_input(batch["cuboids"])

    # GEO_CORE (0..5) and MATERIAL (8) untouched in both paths.
    keep_channels = [0, 1, 2, 3, 4, 5, 8]
    for c in keep_channels:
        assert torch.allclose(gnd_input[:, :, c], batch["cuboids"][:, :, c]), (
            f"gnd input mutated channel {c} (must be untouched)"
        )
        assert torch.allclose(cpl_input[:, :, c], batch["cuboids"][:, :, c]), (
            f"cpl input mutated channel {c} (must be untouched)"
        )

    # INTERACTION channels: zeroed in gnd input, present in cpl input.
    for c in DEFAULT_INTERACTION_CHANNELS:
        assert torch.allclose(
            gnd_input[:, :, c], torch.zeros_like(gnd_input[:, :, c])
        ), (
            f"gnd input channel {c} should be zeroed, got max |val| "
            f"{gnd_input[:, :, c].abs().max().item():.3e}"
        )
        assert torch.allclose(cpl_input[:, :, c], batch["cuboids"][:, :, c]), (
            f"cpl input channel {c} should keep raw values"
        )

    # Buffer shapes / values.
    assert model.gnd_channel_mask.shape == (1, 1, 10)
    assert model.cpl_channel_mask.shape == (1, 1, 10)
    for c in DEFAULT_INTERACTION_CHANNELS:
        assert model.gnd_channel_mask[0, 0, c].item() == 0.0
        assert model.cpl_channel_mask[0, 0, c].item() == 1.0


# 4. Shared encoder identity ---------------------------------------------


def test_shared_encoder():
    """Exactly ONE CuboidSetEncoder instance; both paths read it.

    Static guards:
      - Only one encoder attribute exists.
      - `predict_gnd` and `predict_cpl` and `_predict_joint` source code
        all reference `self.cuboid_encoder` (no per-channel encoder).
      - Encoder param count == baseline (no duplication).
    """
    model = HybridPexV3MeshInputSubsetClampNorm()

    encoder_attrs = [
        name for name, mod in model.named_modules()
        if mod.__class__.__name__ == "CuboidSetEncoder"
    ]
    assert encoder_attrs == ["cuboid_encoder"], (
        f"Combined model must have exactly ONE CuboidSetEncoder attribute, "
        f"got {encoder_attrs}"
    )

    import inspect
    gnd_src = inspect.getsource(model.predict_gnd)
    cpl_src = inspect.getsource(model.predict_cpl)
    joint_src = inspect.getsource(model._predict_joint)
    assert "self.cuboid_encoder" in gnd_src, (
        "predict_gnd must call self.cuboid_encoder, not a per-channel encoder"
    )
    assert "self.cuboid_encoder" in cpl_src, (
        "predict_cpl must call self.cuboid_encoder, not a per-channel encoder"
    )
    # _predict_joint must call self.cuboid_encoder TWICE (once per masked input)
    # but on the SAME shared module — count occurrences as a proxy.
    assert joint_src.count("self.cuboid_encoder") >= 2, (
        "_predict_joint must call self.cuboid_encoder for both gnd and cpl paths"
    )

    baseline_enc_params = sum(
        p.numel() for p in HybridPexV3Mesh().cuboid_encoder.parameters()
    )
    cc_enc_params = sum(p.numel() for p in model.cuboid_encoder.parameters())
    assert cc_enc_params == baseline_enc_params, (
        f"encoder param count diverged from baseline: "
        f"{cc_enc_params} vs {baseline_enc_params}"
    )


# 5. ClampNorm below-threshold identity ----------------------------------


def test_clamp_below_threshold():
    """When ||δ|| < C, the projection is the identity: δ_eff == δ."""
    torch.manual_seed(0)
    model = HybridPexV3MeshInputSubsetClampNorm()
    cap = 1.386  # Phase 2 cap value
    model.set_clamp_bounds(cap)

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


# 6. ClampNorm at/above-threshold projection -----------------------------


def test_clamp_at_threshold():
    """When ||δ|| > C, ||δ_eff|| == C exactly (projection onto sphere)."""
    torch.manual_seed(0)
    model = HybridPexV3MeshInputSubsetClampNorm()
    cap = 0.5
    model.set_clamp_bounds(cap)

    B = 16
    raw = torch.randn(B, 2)
    raw = raw / raw.norm(dim=1, keepdim=True) * 2.0   # ||·|| = 2.0
    logit_gnd = raw[:, 0]
    logit_cpl = raw[:, 1]
    n = torch.sqrt(logit_gnd ** 2 + logit_cpl ** 2)
    assert torch.allclose(n, torch.full_like(n, 2.0), atol=1e-5)

    eff_gnd, eff_cpl = model._norm_project(logit_gnd, logit_cpl)
    n_eff = torch.sqrt(eff_gnd ** 2 + eff_cpl ** 2)
    assert torch.allclose(n_eff, torch.full_like(n_eff, cap), atol=1e-5), (
        f"above-threshold projection failed; expected ||δ_eff||={cap}, "
        f"got {n_eff.tolist()}"
    )

    # Direction preserved.
    cos = (logit_gnd * eff_gnd + logit_cpl * eff_cpl) / (n * n_eff)
    assert torch.allclose(cos, torch.ones_like(cos), atol=1e-5)


# 7. Gradient finite at day-1 --------------------------------------------


def test_gradient_finite_at_day1():
    """Backward at day-1 (δ=0) must produce finite gradients.

    Naive sqrt(δ²) has infinite derivative at zero; the softened sqrt
    `sqrt(sum_sq + eps²)` keeps the gradient finite. This test is the
    same regression guard ClampNorm-alone enforces, applied to the
    combined model where the masked-input path also runs.
    """
    torch.manual_seed(0)
    model = HybridPexV3MeshInputSubsetClampNorm()
    model.train()
    batch = _make_batch()

    pg, pc_ = model._predict_joint(
        batch["analytic_gnd"], batch["analytic_cpl"],
        batch["self_features"], batch["pair_features"],
        batch["cuboids"], batch["padding_mask"],
    )
    loss = (pg - 1.0).pow(2).mean() + (pc_ - 1.0).pow(2).mean()
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


# 8. Drop-in replacement -------------------------------------------------


def test_drop_in_replacement():
    """API parity with HybridPexV3Mesh: predict_* shapes, set_clamp,
    parameter_count.dict.
    """
    torch.manual_seed(0)
    baseline = HybridPexV3Mesh()
    combined = HybridPexV3MeshInputSubsetClampNorm()
    batch = _make_batch()

    out_b_gnd = baseline.predict_gnd(
        batch["analytic_gnd"], batch["self_features"],
        batch["cuboids"], batch["padding_mask"],
    )
    out_c_gnd = combined.predict_gnd(
        batch["analytic_gnd"], batch["self_features"],
        batch["cuboids"], batch["padding_mask"],
    )
    out_b_cpl = baseline.predict_cpl(
        batch["analytic_cpl"], batch["pair_features"],
        batch["cuboids"], batch["padding_mask"],
    )
    out_c_cpl = combined.predict_cpl(
        batch["analytic_cpl"], batch["pair_features"],
        batch["cuboids"], batch["padding_mask"],
    )
    assert out_b_gnd.shape == out_c_gnd.shape
    assert out_b_cpl.shape == out_c_cpl.shape

    combined.set_clamp_bounds(math.log(2.5))
    assert combined.gnd_residual.get_clamp_bound() == pytest.approx(math.log(2.5))
    assert combined.cpl_residual.get_clamp_bound() == pytest.approx(math.log(2.5))

    pc = combined.parameter_count()
    assert "total" in pc
    assert isinstance(pc["total"], int)
