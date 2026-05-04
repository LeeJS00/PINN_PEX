"""
test_hybrid_v3_mesh_input_subset.py — Phase 1 InputSubset invariants.

Validates:
  1. Param count ≤ 50K  (NO encoder duplication; should be ~ baseline 44.7K).
  2. Day-1 invariant: zero-init residual heads → forward output = analytic.
  3. Input mask correctness: gnd input has interaction columns zeroed,
     cpl input keeps them intact.
  4. Shared encoder identity: gnd & cpl forward paths use the SAME
     `CuboidSetEncoder` instance (id check).
  5. Drop-in replacement for `HybridPexV3Mesh` (API parity).
"""
from __future__ import annotations
import math

import pytest
import torch

from src.models.hybrid_v3_mesh_input_subset import (
    HybridPexV3MeshInputSubset,
    DEFAULT_SELF_FEATURE_DIM,
    DEFAULT_PAIR_FEATURE_DIM,
    DEFAULT_INTERACTION_CHANNELS,
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
    """Total params ≤ 50K  (target ~ baseline 44.7K + zero new trainables)."""
    model = HybridPexV3MeshInputSubset()
    pc = model.parameter_count()
    assert pc["total"] <= 50_000, (
        f"param budget exceeded: {pc['total']} > 50K  (breakdown: {pc})"
    )
    # Sanity: parity with baseline (no encoder duplication).
    baseline_pc = HybridPexV3Mesh().parameter_count()
    assert pc["total"] == baseline_pc["total"], (
        f"InputSubset must NOT add trainable params over baseline. "
        f"got {pc['total']} vs baseline {baseline_pc['total']}"
    )
    print("PARAM COUNT:", pc, "(baseline:", baseline_pc, ")")


# 2. Day-1 invariant -----------------------------------------------------


def test_day1_analytic_gnd():
    """Zero-init residual head → multiplier = 1.0 → pred_gnd = analytic_gnd.

    Even with the gnd input mask zeroing interaction channels, the residual
    head's last linear is zero-init → exp(0) = 1 → forward output unchanged.
    """
    torch.manual_seed(0)
    model = HybridPexV3MeshInputSubset()
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
    model = HybridPexV3MeshInputSubset()
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


# 3. Input mask correctness ----------------------------------------------


def test_input_mask_correctness():
    """gnd input zeros interaction columns (6, 7, 9); cpl input keeps them."""
    torch.manual_seed(0)
    model = HybridPexV3MeshInputSubset()
    batch = _make_batch()

    gnd_input = model._gnd_input(batch["cuboids"])
    cpl_input = model._cpl_input(batch["cuboids"])

    # GEO_CORE channels (0..5) and MATERIAL channel (8) must be untouched
    # in both gnd and cpl inputs.
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
        assert torch.allclose(gnd_input[:, :, c], torch.zeros_like(gnd_input[:, :, c])), (
            f"gnd input channel {c} should be zeroed, got max |val| "
            f"{gnd_input[:, :, c].abs().max().item():.3e}"
        )
        assert torch.allclose(cpl_input[:, :, c], batch["cuboids"][:, :, c]), (
            f"cpl input channel {c} should keep raw values"
        )

    # Sanity on the registered buffers.
    assert model.gnd_channel_mask.shape == (1, 1, 10)
    assert model.cpl_channel_mask.shape == (1, 1, 10)
    for c in DEFAULT_INTERACTION_CHANNELS:
        assert model.gnd_channel_mask[0, 0, c].item() == 0.0
        assert model.cpl_channel_mask[0, 0, c].item() == 1.0


# 4. Shared encoder identity ---------------------------------------------


def test_shared_encoder():
    """gnd_emb and cpl_emb must come from THE SAME `cuboid_encoder` instance.

    Pure id() check on the module attribute, plus a parameter-count check
    confirming no shadow duplicate encoder exists.
    """
    model = HybridPexV3MeshInputSubset()

    # Only one encoder attribute exists.
    encoder_attrs = [
        name for name, mod in model.named_modules()
        if mod.__class__.__name__ == "CuboidSetEncoder"
    ]
    assert encoder_attrs == ["cuboid_encoder"], (
        f"InputSubset must have exactly ONE CuboidSetEncoder attribute, got {encoder_attrs}"
    )

    # gnd path and cpl path resolve to the same encoder object.
    enc_id = id(model.cuboid_encoder)
    # Both predict_gnd and predict_cpl read self.cuboid_encoder; verify
    # by tracing the source attribute name.
    import inspect
    gnd_src = inspect.getsource(model.predict_gnd)
    cpl_src = inspect.getsource(model.predict_cpl)
    assert "self.cuboid_encoder" in gnd_src, (
        "predict_gnd must call self.cuboid_encoder, not a per-channel encoder"
    )
    assert "self.cuboid_encoder" in cpl_src, (
        "predict_cpl must call self.cuboid_encoder, not a per-channel encoder"
    )

    # Encoder param count should equal baseline (no duplication factor).
    baseline_enc_params = sum(
        p.numel() for p in HybridPexV3Mesh().cuboid_encoder.parameters()
    )
    is_enc_params = sum(p.numel() for p in model.cuboid_encoder.parameters())
    assert is_enc_params == baseline_enc_params, (
        f"encoder param count diverged from baseline: "
        f"{is_enc_params} vs {baseline_enc_params}"
    )


def test_shared_encoder_gradient_coupling():
    """A consequence of sharing: cpl loss MUST update the encoder, and
    gnd loss MUST also update the encoder (no gradient isolation).

    Contrast with A1 (HybridPexV3MeshPerChannel) which DEMANDS isolation.
    InputSubset DEMANDS coupling — same weights, different input slices.
    """
    torch.manual_seed(0)
    model = HybridPexV3MeshInputSubset()
    # Perturb residual heads so gradients are non-trivial.
    with torch.no_grad():
        for p in model.gnd_residual.parameters():
            p.add_(0.01 * torch.randn_like(p))
        for p in model.cpl_residual.parameters():
            p.add_(0.01 * torch.randn_like(p))

    batch = _make_batch()

    # gnd-only loss → encoder grads non-zero.
    model.zero_grad()
    pred_gnd = model.predict_gnd(
        batch["analytic_gnd"], batch["self_features"],
        batch["cuboids"], batch["padding_mask"],
    )
    loss_gnd = (pred_gnd - batch["analytic_gnd"] * 1.5).pow(2).mean()
    loss_gnd.backward()
    enc_grad_after_gnd = sum(
        (p.grad.abs().sum() if p.grad is not None else torch.tensor(0.0))
        for p in model.cuboid_encoder.parameters()
    )
    assert enc_grad_after_gnd > 0, (
        "shared encoder should receive gradient from gnd loss"
    )

    # cpl-only loss → encoder grads non-zero (same encoder).
    model.zero_grad()
    pred_cpl = model.predict_cpl(
        batch["analytic_cpl"], batch["pair_features"],
        batch["cuboids"], batch["padding_mask"],
    )
    loss_cpl = (pred_cpl - batch["analytic_cpl"] * 1.5).pow(2).mean()
    loss_cpl.backward()
    enc_grad_after_cpl = sum(
        (p.grad.abs().sum() if p.grad is not None else torch.tensor(0.0))
        for p in model.cuboid_encoder.parameters()
    )
    assert enc_grad_after_cpl > 0, (
        "shared encoder should receive gradient from cpl loss"
    )


# 5. Drop-in replacement for HybridPexV3Mesh -----------------------------


def test_drop_in_replacement():
    """API parity: same forward signature & set_clamp_bounds / parameter_count."""
    torch.manual_seed(0)
    baseline = HybridPexV3Mesh()
    is_model = HybridPexV3MeshInputSubset()

    batch = _make_batch()

    out_b_gnd = baseline.predict_gnd(
        batch["analytic_gnd"], batch["self_features"],
        batch["cuboids"], batch["padding_mask"],
    )
    out_a_gnd = is_model.predict_gnd(
        batch["analytic_gnd"], batch["self_features"],
        batch["cuboids"], batch["padding_mask"],
    )
    out_b_cpl = baseline.predict_cpl(
        batch["analytic_cpl"], batch["pair_features"],
        batch["cuboids"], batch["padding_mask"],
    )
    out_a_cpl = is_model.predict_cpl(
        batch["analytic_cpl"], batch["pair_features"],
        batch["cuboids"], batch["padding_mask"],
    )
    assert out_b_gnd.shape == out_a_gnd.shape
    assert out_b_cpl.shape == out_a_cpl.shape

    is_model.set_clamp_bounds(math.log(2.5))
    assert is_model.gnd_residual.get_clamp_bound() == pytest.approx(math.log(2.5))
    assert is_model.cpl_residual.get_clamp_bound() == pytest.approx(math.log(2.5))

    pc = is_model.parameter_count()
    assert "total" in pc
    assert isinstance(pc["total"], int)
