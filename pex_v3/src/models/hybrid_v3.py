"""
hybrid_v3.py — Phase 1 hybrid analytic + neural residual model.

Combines `analytic_base_v3.AnalyticBase` (closed-form Mode A + Mode B)
with `residual_head_v3.BoundedResidualHead` (zero-init bounded multiplier).

Per A5 (neural-operator-architect) Tier 0 spec + A2 audit β-strategy:

    Per-channel separation:
        - gnd self head: predicts C_self for target net (M_s scalar features)
        - cpl pair head: predicts C_cpl[i,j] for each (target, aggressor) pair (M_p scalar features)

    NOT a single total head — total fitting causes gnd/cpl cancellation
    learning, exactly the artifact that inflates B1's headline 4.66%.

Day-1 invariant (zero-init residual heads): C_pred = C_analytic exactly.
Convergence target (post-pretrain on Stage 1+Mode-A): residual stays
near multiplier=1.0, deviation only when the geometry deviates from
parallel-plate / stacked-series.

Real-BEOL fine-tune target (after K3 canary): per-channel learned
correction. β-strategy gate:
    gnd MAPE < 8% AND cpl MAPE < 8% on v3 valid 5-seed
"""
from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn

from .analytic_base_v3 import (
    EPS0_FF_UM,
    analytic_parallel_plate,
    analytic_stacked_dielectric,
)
from .residual_head_v3 import (
    BoundedResidualHead,
    PerPairTypeBoundedResidualHead,
)


# Default scalar feature dimensions per A5 spec §4.1 / §4.4
DEFAULT_SELF_FEATURE_DIM = 16   # self-cap per-conductor features
DEFAULT_PAIR_FEATURE_DIM = 24   # pair-coupling features


class HybridPexV3(nn.Module):
    """Hybrid analytic + bounded neural residual, per-channel.

    Inference contract:
        gnd_pred[t] = gnd_analytic(t) * gnd_residual(self_features[t])     [fF]
        cpl_pred[i,j] = cpl_analytic(i,j) * cpl_residual(pair_features[i,j])  [fF]

    Day-1 (zero-init residual): outputs equal the analytic baselines.

    Args:
        self_feature_dim:  dim of per-conductor feature vector
        pair_feature_dim:  dim of per-pair feature vector
        hidden_dim:        residual MLP hidden width
        n_hidden:          residual MLP depth
        clamp_bound:       initial RES_CLAMP (curriculum sets per epoch)
        per_pair_clamp:    if True, use PerPairTypeBoundedResidualHead with
                           tighter clamp for parallel-plate-dominated pairs
        per_pair_clamp_bounds: tuple of clamp values when per_pair_clamp=True
                               (default (log(1.5), log(4.0)) per A5 recommendation)
    """

    def __init__(
        self,
        self_feature_dim: int = DEFAULT_SELF_FEATURE_DIM,
        pair_feature_dim: int = DEFAULT_PAIR_FEATURE_DIM,
        hidden_dim: int = 64,
        n_hidden: int = 2,
        clamp_bound: float = math.log(1.5),
        per_pair_clamp: bool = False,
        per_pair_clamp_bounds: tuple = (math.log(1.5), math.log(4.0)),
    ):
        super().__init__()
        self.gnd_residual = BoundedResidualHead(
            in_dim=self_feature_dim,
            hidden_dim=hidden_dim,
            n_hidden=n_hidden,
            clamp_bound=clamp_bound,
        )
        if per_pair_clamp:
            self.cpl_residual: nn.Module = PerPairTypeBoundedResidualHead(
                in_dim=pair_feature_dim,
                hidden_dim=hidden_dim,
                n_hidden=n_hidden,
                clamp_bounds_by_type=per_pair_clamp_bounds,
            )
        else:
            self.cpl_residual = BoundedResidualHead(
                in_dim=pair_feature_dim,
                hidden_dim=hidden_dim,
                n_hidden=n_hidden,
                clamp_bound=clamp_bound,
            )
        self.per_pair_clamp = per_pair_clamp

    # ------------------------------------------------------------------
    # GND channel — self-cap per target conductor
    # ------------------------------------------------------------------

    def predict_gnd(
        self,
        analytic_C_fF: torch.Tensor,
        self_features: torch.Tensor,
    ) -> torch.Tensor:
        """Per-conductor C_gnd prediction.

        Args:
            analytic_C_fF: shape (B,) — analytic baseline (e.g., parallel
                          plate to ground from `analytic_base_v3`).
            self_features: shape (B, self_feature_dim) — scalar features
                          for the residual head.

        Returns:
            C_gnd_pred:   shape (B,)
        """
        multiplier = self.gnd_residual(self_features)
        return analytic_C_fF * multiplier

    # ------------------------------------------------------------------
    # CPL channel — pair-coupling per (target, aggressor)
    # ------------------------------------------------------------------

    def predict_cpl(
        self,
        analytic_C_fF: torch.Tensor,
        pair_features: torch.Tensor,
        pair_type_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-pair C_cpl prediction.

        Args:
            analytic_C_fF: shape (E,) — analytic baseline for E pairs.
            pair_features: shape (E, pair_feature_dim).
            pair_type_idx: shape (E,) integer in [0, n_types). Required
                           if `per_pair_clamp=True`; ignored otherwise.

        Returns:
            C_cpl_pred:   shape (E,)
        """
        if self.per_pair_clamp:
            if pair_type_idx is None:
                raise ValueError("per_pair_clamp=True requires pair_type_idx")
            multiplier = self.cpl_residual(pair_features, pair_type_idx)
        else:
            multiplier = self.cpl_residual(pair_features)
        return analytic_C_fF * multiplier

    # ------------------------------------------------------------------
    # Curriculum hooks
    # ------------------------------------------------------------------

    def set_clamp_bounds(self, clamp_bound: float) -> None:
        """Update RES_CLAMP for both gnd and cpl heads (uniform variant only).

        For per-pair-type variant, modify `_clamps` buffer directly externally.
        """
        if hasattr(self.gnd_residual, "set_clamp_bound"):
            self.gnd_residual.set_clamp_bound(clamp_bound)
        if hasattr(self.cpl_residual, "set_clamp_bound") and not self.per_pair_clamp:
            self.cpl_residual.set_clamp_bound(clamp_bound)

    def parameter_count(self) -> dict:
        """Return total trainable params for budget reporting."""
        return {
            "gnd_residual": sum(p.numel() for p in self.gnd_residual.parameters()),
            "cpl_residual": sum(p.numel() for p in self.cpl_residual.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }


# ============================================================================
# Per-channel loss (β-strategy mandate)
# ============================================================================


def per_channel_mape_loss(
    pred_gnd: torch.Tensor,
    golden_gnd: torch.Tensor,
    pred_cpl: torch.Tensor,
    golden_cpl: torch.Tensor,
    eps_fF: float = 1e-3,
    w_gnd: float = 1.0,
    w_cpl: float = 1.0,
) -> dict:
    """Loss with explicit per-channel separation — never sums to total before MAPE.

    A2 audit's β-strategy gate enforcement: training on `total = gnd + cpl`
    causes gnd/cpl cancellation learning. Compute separate per-channel
    relative errors, then weighted sum.

    Args:
        pred_gnd, golden_gnd: shape (B,) — per-net ground cap predictions
        pred_cpl, golden_cpl: shape (E,) — per-pair coupling predictions
        eps_fF:               clamp for division (zero-target-safe)
        w_gnd, w_cpl:         channel weights

    Returns:
        dict with `total_loss`, `gnd_mape`, `cpl_mape` for logging.
    """
    gnd_safe = golden_gnd.clamp(min=eps_fF)
    gnd_rel = (pred_gnd - golden_gnd).abs() / gnd_safe
    gnd_mape = gnd_rel.mean()

    cpl_safe = golden_cpl.clamp(min=eps_fF)
    cpl_rel = (pred_cpl - golden_cpl).abs() / cpl_safe
    cpl_mape = cpl_rel.mean()

    total_loss = w_gnd * gnd_mape + w_cpl * cpl_mape
    return {
        "total_loss": total_loss,
        "gnd_mape": gnd_mape.detach(),
        "cpl_mape": cpl_mape.detach(),
    }
