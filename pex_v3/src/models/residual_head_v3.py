"""
residual_head_v3.py — Phase 1 bounded multiplicative residual.

A5 (neural-operator-architect) Tier 0 ship: residual head with bounded
multiplier `exp(clamp(logit, -RES_CLAMP, +RES_CLAMP))` so day-1 output
≈ analytic baseline (zero-init last layer) and the model cannot deviate
more than the clamp permits.

Curriculum (per A5):
    Phase 0 (epochs 0-50):   RES_CLAMP = log(1.5) ≈ 0.405  → mul ∈ [0.67, 1.50]
    Phase 1 (epochs 50-150): RES_CLAMP = log(2.5) ≈ 0.916  → mul ∈ [0.40, 2.50]
    Phase 2 (epochs 150+):   RES_CLAMP = log(4.0) ≈ 1.386  → mul ∈ [0.25, 4.00]

Per-pair-type bounding (A5 recommendation):
    parallel-plate-dominated pairs (broadside_overlap > 0.5*area): tighter clamp
    cross-layer / lateral pairs:                                     looser clamp
"""
from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn


# ============================================================================
# RES_CLAMP curriculum schedule
# ============================================================================


_DEFAULT_CURRICULUM = (
    (0,   math.log(1.5)),     # Phase 0
    (50,  math.log(2.5)),     # Phase 1
    (150, math.log(4.0)),     # Phase 2
)


def res_clamp_for_epoch(
    epoch: int,
    schedule: tuple = _DEFAULT_CURRICULUM,
) -> float:
    """Return the RES_CLAMP value for the given epoch per the curriculum."""
    last_value = schedule[0][1]
    for boundary, value in schedule:
        if epoch >= boundary:
            last_value = value
        else:
            break
    return float(last_value)


# ============================================================================
# Bounded residual head
# ============================================================================


class BoundedResidualHead(nn.Module):
    """MLP residual head with `exp(clamp(...))` bounded multiplier.

    Pattern (A5 spec §4.1):
        residual_logit = MLP(pair_features)
        residual_logit = clamp(residual_logit, -RES_CLAMP, +RES_CLAMP)
        multiplier      = exp(residual_logit)
        C_pred          = C_analytic * multiplier

    Initialization (A5 mandate):
        Last linear layer's weight + bias zero-initialized → day-1 output is
        exactly zero → multiplier = 1.0 → C_pred = C_analytic.

    This is the **cleanest attribution mechanism**: any deviation from
    multiplier=1.0 is the residual head's earned contribution, measurable
    against the day-0 baseline.

    Args:
        in_dim:        feature vector dimension (24 per A5 spec §4.1)
        hidden_dim:    MLP hidden dimension (default 64; A5 budget ~5.7K params)
        n_hidden:      number of hidden layers (default 2)
        clamp_bound:   initial RES_CLAMP value; can be updated via
                       `set_clamp_bound(c)` per epoch curriculum.
    """

    def __init__(
        self,
        in_dim: int = 24,
        hidden_dim: int = 64,
        n_hidden: int = 2,
        clamp_bound: float = math.log(1.5),
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.GELU())
            prev = hidden_dim
        # Final scalar output layer — zero-init so day-1 output = 0
        final = nn.Linear(prev, 1)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        layers.append(final)

        self.mlp = nn.Sequential(*layers)
        self.register_buffer("_clamp_bound", torch.tensor(float(clamp_bound)))

    def set_clamp_bound(self, value: float) -> None:
        """Update RES_CLAMP value for the curriculum (call per epoch)."""
        with torch.no_grad():
            self._clamp_bound.fill_(float(value))

    def get_clamp_bound(self) -> float:
        return float(self._clamp_bound)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute bounded multiplier from features.

        Args:
            features: shape (..., in_dim)

        Returns:
            multiplier: shape (...) — bounded in [exp(-clamp), exp(+clamp)]
        """
        logit = self.mlp(features).squeeze(-1)
        clamped = torch.clamp(logit, -self._clamp_bound, +self._clamp_bound)
        return torch.exp(clamped)


# ============================================================================
# Per-pair-type clamping (A5 recommendation)
# ============================================================================


class PerPairTypeBoundedResidualHead(nn.Module):
    """Per-pair-type clamp variant.

    Tighter for parallel-plate-dominated pairs (analytic is accurate),
    looser for cross-layer / lateral pairs (analytic less accurate).
    A5 recommendation, §2.

    The pair-type tag is provided per pair as a categorical index
    (0=parallel-plate, 1=cross-layer, ...) and selects from per-type
    clamp values.

    Args:
        in_dim, hidden_dim, n_hidden: same as BoundedResidualHead
        clamp_bounds_by_type: tuple of clamp values per pair type
    """

    def __init__(
        self,
        in_dim: int = 24,
        hidden_dim: int = 64,
        n_hidden: int = 2,
        clamp_bounds_by_type: tuple = (math.log(1.5), math.log(4.0)),
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.GELU())
            prev = hidden_dim
        final = nn.Linear(prev, 1)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.mlp = nn.Sequential(*layers)

        clamp_t = torch.tensor(list(clamp_bounds_by_type), dtype=torch.float32)
        self.register_buffer("_clamps", clamp_t)

    def forward(
        self,
        features: torch.Tensor,
        pair_type_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-pair multiplier with type-conditional clamp.

        Args:
            features:       (..., in_dim)
            pair_type_idx:  (...) integer tensor in [0, len(clamp_bounds_by_type))

        Returns:
            multiplier:     (...) bounded by per-type clamp.
        """
        logit = self.mlp(features).squeeze(-1)
        clamps = self._clamps[pair_type_idx]
        clamped = torch.clamp(logit, -clamps, +clamps)
        return torch.exp(clamped)
