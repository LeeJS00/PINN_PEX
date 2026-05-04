"""
test_residual_head_v3.py — Phase 1 bounded residual head invariants.

Per A5 (neural-operator-architect) Tier 0 mandate.
"""
from __future__ import annotations
import math

import torch

from src.models.residual_head_v3 import (
    BoundedResidualHead,
    PerPairTypeBoundedResidualHead,
    res_clamp_for_epoch,
)


# ============================================================================
# RES_CLAMP curriculum
# ============================================================================


def test_curriculum_phase_0():
    """Epoch 0-49: clamp = log(1.5)."""
    assert res_clamp_for_epoch(0) == math.log(1.5)
    assert res_clamp_for_epoch(25) == math.log(1.5)
    assert res_clamp_for_epoch(49) == math.log(1.5)


def test_curriculum_phase_1():
    """Epoch 50-149: clamp = log(2.5)."""
    assert res_clamp_for_epoch(50) == math.log(2.5)
    assert res_clamp_for_epoch(100) == math.log(2.5)
    assert res_clamp_for_epoch(149) == math.log(2.5)


def test_curriculum_phase_2():
    """Epoch 150+: clamp = log(4.0)."""
    assert res_clamp_for_epoch(150) == math.log(4.0)
    assert res_clamp_for_epoch(500) == math.log(4.0)


# ============================================================================
# Day-1 zero output (analytic-equivalent)
# ============================================================================


def test_day_1_multiplier_is_one():
    """A5 mandate: zero-init last layer → day-1 output = 0 → multiplier = 1.0.

    Without this, the analytic baseline contribution attribution is broken.
    """
    torch.manual_seed(0)
    head = BoundedResidualHead(in_dim=24, hidden_dim=64)
    head.eval()
    # ANY input — the multiplier should be exactly 1.0 at init
    x = torch.randn(50, 24)
    mul = head(x)
    assert torch.allclose(mul, torch.ones(50)), (
        f"day-1 multiplier should be 1.0, got max diff {(mul - 1.0).abs().max()}"
    )


def test_day_1_self_capacitance_is_one_too():
    """Same property for arbitrary in_dim (e.g., self-cap MLP_self with 16 features)."""
    torch.manual_seed(0)
    head = BoundedResidualHead(in_dim=16, hidden_dim=32, n_hidden=2)
    head.eval()
    x = torch.randn(20, 16)
    mul = head(x)
    assert torch.allclose(mul, torch.ones(20))


# ============================================================================
# Bounded output
# ============================================================================


def test_clamp_enforces_bound():
    """Even with extreme inputs, multiplier stays in [exp(-clamp), exp(+clamp)]."""
    torch.manual_seed(0)
    head = BoundedResidualHead(in_dim=4, hidden_dim=8, clamp_bound=math.log(2.0))
    # Manually push the MLP to extreme values
    with torch.no_grad():
        for layer in head.mlp:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.fill_(100.0)
                layer.bias.fill_(100.0)
    x = torch.randn(50, 4) * 10
    mul = head(x)
    # All outputs in [0.5, 2.0]
    assert (mul >= 0.5 - 1e-6).all()
    assert (mul <= 2.0 + 1e-6).all()


def test_set_clamp_bound_updates():
    """`set_clamp_bound` mutates the bound used at forward time."""
    head = BoundedResidualHead(in_dim=4, hidden_dim=8, clamp_bound=math.log(1.5))
    assert abs(head.get_clamp_bound() - math.log(1.5)) < 1e-6
    head.set_clamp_bound(math.log(4.0))
    assert abs(head.get_clamp_bound() - math.log(4.0)) < 1e-6


# ============================================================================
# Gradient flows
# ============================================================================


def test_gradients_flow_through_residual():
    """Backward pass should produce non-zero gradients on MLP weights when
    fed non-zero gradient signal (after non-trivial training)."""
    torch.manual_seed(0)
    head = BoundedResidualHead(in_dim=4, hidden_dim=8)
    # Tweak the last layer slightly so the day-1 output isn't 0 forever
    with torch.no_grad():
        for p in head.parameters():
            p.add_(0.01 * torch.randn_like(p))
    x = torch.randn(20, 4, requires_grad=False)
    out = head(x)
    loss = (out - 1.5).pow(2).mean()
    loss.backward()
    # All trainable params have gradient
    grads = [p.grad for p in head.parameters() if p.requires_grad]
    assert all(g is not None and g.abs().max() > 0 for g in grads)


# ============================================================================
# Per-pair-type variant
# ============================================================================


def test_per_pair_type_zero_init_at_day_1():
    """Per-type variant also obeys day-1 multiplier=1."""
    torch.manual_seed(0)
    head = PerPairTypeBoundedResidualHead(
        in_dim=24, hidden_dim=32,
        clamp_bounds_by_type=(math.log(1.5), math.log(4.0)),
    )
    head.eval()
    x = torch.randn(20, 24)
    pair_type = torch.randint(0, 2, (20,))
    mul = head(x, pair_type)
    assert torch.allclose(mul, torch.ones(20))


def test_per_pair_type_different_bounds():
    """Type-0 should be tighter clamp than type-1."""
    torch.manual_seed(0)
    head = PerPairTypeBoundedResidualHead(
        in_dim=4, hidden_dim=8,
        clamp_bounds_by_type=(math.log(1.5), math.log(4.0)),
    )
    # Force MLP to large positive output
    with torch.no_grad():
        for layer in head.mlp:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.fill_(50.0)
                layer.bias.fill_(50.0)
    x = torch.randn(50, 4)
    pair_type_0 = torch.zeros(50, dtype=torch.long)
    pair_type_1 = torch.ones(50, dtype=torch.long)
    mul_0 = head(x, pair_type_0)
    mul_1 = head(x, pair_type_1)
    # Type-0 should saturate at exp(log(1.5)) = 1.5
    # Type-1 at exp(log(4.0)) = 4.0
    assert mul_0.max().item() <= 1.5 + 1e-4
    assert mul_1.max().item() <= 4.0 + 1e-4
    assert mul_0.max().item() < mul_1.max().item()


# ============================================================================
# Param budget (A5 spec §3)
# ============================================================================


def test_param_count_matches_spec_envelope():
    """A5 spec §3: 24-dim input MLP 64×64 → ~5.7K params."""
    head = BoundedResidualHead(in_dim=24, hidden_dim=64, n_hidden=2)
    n_params = sum(p.numel() for p in head.parameters())
    # 24*64+64 + 64*64+64 + 64*1+1 = 1600+4160+65 ≈ 5825
    assert 5_000 < n_params < 7_000, f"unexpected param count: {n_params}"
