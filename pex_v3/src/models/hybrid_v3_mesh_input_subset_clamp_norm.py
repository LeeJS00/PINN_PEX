"""
hybrid_v3_mesh_input_subset_clamp_norm.py — Phase 1 combined-stack variant.

Composes two architecturally orthogonal levers on top of `HybridPexV3Mesh`:

    INPUT lever  — InputSubset:  per-channel input zero-masking with ONE
                                shared `CuboidSetEncoder`. The gnd path
                                receives `cuboids * gnd_channel_mask`
                                (zeros at interaction columns 6, 7, 9);
                                the cpl path receives the unmasked
                                tensor.
    OUTPUT lever — ClampNorm:    joint per-net (gnd, cpl) L2 norm-projection
                                clamp on the residual logits, replacing
                                the element-wise `clamp(δ, ±C)`.

Both levers preserve:
  * Param count           — 44,738 (identical to HybridPexV3Mesh baseline).
  * Day-1 invariant       — zero-init residual heads → mul = 1 → pred = analytic.
  * Curriculum schedule   — 0.405 → 0.916 → 1.386 via `set_clamp_bounds`.

Composition contract (see DESIGN.md):
  * One `CuboidSetEncoder` instance — shared between gnd and cpl paths.
    Two encoder forwards per `_predict_joint` call (one per masked input)
    but the SAME weights — InputSubset's coupling guarantee.
  * The joint-norm projection is computed on the per-net residual VECTOR
    `(δ_gnd, δ_cpl) ∈ R²`. Below-cap → identity; above-cap → smooth
    rank-1 Jacobian.

Drop-in API parity with `HybridPexV3Mesh`:
  predict_gnd / predict_cpl (standalone, with logit_other = 0 fallback),
  set_clamp_bounds, parameter_count.

For training and joint evaluation, callers SHOULD use `_predict_joint`
to obtain both predictions under the correct joint-norm clamp.
"""
from __future__ import annotations
import math
from typing import Sequence

import torch
import torch.nn as nn

from .residual_head_v3 import BoundedResidualHead
from .cuboid_set_encoder import CuboidSetEncoder
from .hybrid_v3_mesh_input_subset import (
    DEFAULT_INTERACTION_CHANNELS,
    _build_channel_mask,
)


DEFAULT_SELF_FEATURE_DIM = 16
DEFAULT_PAIR_FEATURE_DIM = 24
DEFAULT_CUBOID_IN_DIM = 10

# Numerical safety for n = ||δ||; chosen so that C / eps >> 1 in float32
# at day-1 (n = 0.0 exact), guaranteeing s = 1 and δ_eff = 0.
_NORM_EPS = 1.0e-12


class HybridPexV3MeshInputSubsetClampNorm(nn.Module):
    """Hybrid analytic + shared cuboid encoder + masked input + joint-norm clamp.

    Architectural identity to `HybridPexV3Mesh`:
        - Same `CuboidSetEncoder` (single instance, shared weights).
        - Same `BoundedResidualHead` MLPs for gnd and cpl.
        - Same `clamp_bound` buffer + `set_clamp_bounds` curriculum hook.

    Differs from baseline in TWO orthogonal points:
        1. Encoder INPUT (InputSubset): the gnd-path forward zeros
           interaction channels {6, 7, 9}; the cpl-path forward keeps
           the full tensor.
        2. Residual OUTPUT clamp (ClampNorm):
              baseline:  δ_eff = clamp(δ, -C, +C)        (per channel)
              combined:  δ_eff = δ × min(1, C / ||(δ_gnd, δ_cpl)||₂)
                                    (joint per-net 2-vector projection)

    Args:
        self_feature_dim:           scalar self features (16)
        pair_feature_dim:           scalar pair features (24)
        cuboid_in_dim:              per-cuboid feature dim (10)
        cuboid_hidden:              per-cuboid MLP hidden (64)
        cuboid_embed_dim:           per-pool head embedding (64) → out 192
        cuboid_n_layers:            per-cuboid MLP depth (2)
        residual_hidden:            bounded residual MLP hidden (64)
        residual_n_hidden:          bounded residual MLP depth (2)
        clamp_bound:                initial clamp_bound (default log(1.5))
        gnd_interaction_channels:   channels zeroed in the gnd encoder input
                                    (default (6, 7, 9): semantic_type,
                                    is_target, net_type).
    """

    def __init__(
        self,
        self_feature_dim: int = DEFAULT_SELF_FEATURE_DIM,
        pair_feature_dim: int = DEFAULT_PAIR_FEATURE_DIM,
        cuboid_in_dim: int = DEFAULT_CUBOID_IN_DIM,
        cuboid_hidden: int = 64,
        cuboid_embed_dim: int = 64,
        cuboid_n_layers: int = 2,
        residual_hidden: int = 64,
        residual_n_hidden: int = 2,
        clamp_bound: float = math.log(1.5),
        gnd_interaction_channels: Sequence[int] = DEFAULT_INTERACTION_CHANNELS,
    ):
        super().__init__()

        # ONE shared encoder — same architecture as baseline HybridPexV3Mesh.
        self.cuboid_encoder = CuboidSetEncoder(
            in_dim=cuboid_in_dim,
            hidden=cuboid_hidden,
            embed_dim=cuboid_embed_dim,
            n_layers=cuboid_n_layers,
        )
        emb_dim = self.cuboid_encoder.out_dim  # 3 * cuboid_embed_dim

        # Residual heads kept identical to baseline so parameter count is
        # exactly 44,738. We bypass `head.forward` (which does element-wise
        # clamp) and access `head.mlp` directly to apply the joint-norm
        # clamp ourselves. The `_clamp_bound` buffer remains the curriculum
        # source-of-truth via `set_clamp_bounds`.
        self.gnd_residual = BoundedResidualHead(
            in_dim=self_feature_dim + emb_dim,
            hidden_dim=residual_hidden,
            n_hidden=residual_n_hidden,
            clamp_bound=clamp_bound,
        )
        self.cpl_residual = BoundedResidualHead(
            in_dim=pair_feature_dim + emb_dim,
            hidden_dim=residual_hidden,
            n_hidden=residual_n_hidden,
            clamp_bound=clamp_bound,
        )

        self.self_feature_dim = self_feature_dim
        self.pair_feature_dim = pair_feature_dim
        self.cuboid_in_dim = cuboid_in_dim
        self.emb_dim = emb_dim
        self.gnd_interaction_channels = tuple(int(c) for c in gnd_interaction_channels)

        # InputSubset masks: non-trainable buffers shaped (1, 1, in_dim) so
        # they broadcast over (B, N, in_dim). The cpl mask is registered for
        # symmetric inspection (always ones).
        gnd_mask = _build_channel_mask(cuboid_in_dim, self.gnd_interaction_channels)
        self.register_buffer("gnd_channel_mask", gnd_mask)
        self.register_buffer("cpl_channel_mask", torch.ones_like(gnd_mask))

    # ------------------------------------------------------------------
    # InputSubset hooks (identical to HybridPexV3MeshInputSubset)
    # ------------------------------------------------------------------

    def _gnd_input(self, cuboids: torch.Tensor) -> torch.Tensor:
        """Zero out interaction channels in the gnd encoder input."""
        return cuboids * self.gnd_channel_mask

    def _cpl_input(self, cuboids: torch.Tensor) -> torch.Tensor:
        """Identity (cpl encoder sees the full tensor)."""
        return cuboids

    # ------------------------------------------------------------------
    # ClampNorm hook (identical math to HybridPexV3MeshClampNorm._norm_project)
    # ------------------------------------------------------------------

    def _norm_project(
        self,
        logit_gnd: torch.Tensor,    # (B,)
        logit_cpl: torch.Tensor,    # (B,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply joint norm-projection clamp to (logit_gnd, logit_cpl).

        Reads the curriculum cap C from `self.gnd_residual._clamp_bound`
        (both heads are kept in lock-step via `set_clamp_bounds`).
        Returns (logit_gnd_eff, logit_cpl_eff) with the same per-net
        scalar `s[b] = min(1, C / max(||δ[b]||₂, eps))` applied to both.

        Numerical care: softened sqrt `sqrt(sum_sq + eps²)` so the
        gradient is finite even at the day-1 zero-logit point.
        """
        cap = self.gnd_residual._clamp_bound  # 0-d tensor

        sum_sq = logit_gnd * logit_gnd + logit_cpl * logit_cpl
        n = torch.sqrt(sum_sq + (_NORM_EPS * _NORM_EPS))
        s = torch.clamp(cap / n, max=1.0)
        return s * logit_gnd, s * logit_cpl

    # ------------------------------------------------------------------
    # Joint forward (canonical training entry point)
    # ------------------------------------------------------------------

    def _predict_joint(
        self,
        analytic_gnd: torch.Tensor,     # (B,)
        analytic_cpl: torch.Tensor,     # (B,)
        self_features: torch.Tensor,    # (B, self_feature_dim)
        pair_features: torch.Tensor,    # (B, pair_feature_dim)
        cuboids: torch.Tensor,          # (B, N_max, in_dim)
        padding_mask: torch.Tensor,     # (B, N_max)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute (pred_gnd, pred_cpl) jointly under InputSubset + ClampNorm.

        Two encoder forwards (gnd masked input, cpl full input) using
        the SAME shared encoder weights. Joint-norm clamp on the
        per-net residual vector before exponentiation.
        """
        gnd_input = self._gnd_input(cuboids)
        cpl_input = self._cpl_input(cuboids)

        gnd_emb = self.cuboid_encoder(gnd_input, padding_mask)
        cpl_emb = self.cuboid_encoder(cpl_input, padding_mask)

        feats_gnd = torch.cat([self_features, gnd_emb], dim=-1)
        feats_cpl = torch.cat([pair_features, cpl_emb], dim=-1)

        # Bypass head.forward (element-wise clamp); use raw .mlp logit.
        logit_gnd = self.gnd_residual.mlp(feats_gnd).squeeze(-1)
        logit_cpl = self.cpl_residual.mlp(feats_cpl).squeeze(-1)

        logit_gnd_eff, logit_cpl_eff = self._norm_project(logit_gnd, logit_cpl)

        mul_gnd = torch.exp(logit_gnd_eff)
        mul_cpl = torch.exp(logit_cpl_eff)
        return analytic_gnd * mul_gnd, analytic_cpl * mul_cpl

    # ------------------------------------------------------------------
    # Public API (identical signature to HybridPexV3Mesh)
    # ------------------------------------------------------------------

    def predict_gnd(
        self,
        analytic_C_fF: torch.Tensor,    # (B,) — analytic gnd
        self_features: torch.Tensor,    # (B, self_feature_dim)
        cuboids: torch.Tensor,          # (B, N_max, in_dim)
        padding_mask: torch.Tensor,     # (B, N_max)
    ) -> torch.Tensor:
        """Compute pred_gnd under InputSubset + ClampNorm (cpl logit = 0 fallback).

        The standalone API does not receive `pair_features` or
        `analytic_cpl`, so we fall back to `logit_cpl = 0`. With the
        joint-norm clamp this degenerates to per-channel clamp on
        `δ_gnd` alone (n = |δ_gnd|, s = min(1, C/|δ_gnd|),
        δ_eff_gnd = sign(δ_gnd) · min(|δ_gnd|, C)). This matches the
        baseline element-wise clamp on a single channel exactly.

        For training and standard joint eval, the trainer uses
        `_predict_joint` to get correct joint-clamp behaviour. The smoke
        runner script for this variant is wired to `_predict_joint`.
        """
        gnd_input = self._gnd_input(cuboids)
        emb = self.cuboid_encoder(gnd_input, padding_mask)
        feats_gnd = torch.cat([self_features, emb], dim=-1)
        logit_gnd = self.gnd_residual.mlp(feats_gnd).squeeze(-1)
        logit_cpl_zero = torch.zeros_like(logit_gnd)
        logit_gnd_eff, _ = self._norm_project(logit_gnd, logit_cpl_zero)
        return analytic_C_fF * torch.exp(logit_gnd_eff)

    def predict_cpl(
        self,
        analytic_C_fF: torch.Tensor,
        pair_features: torch.Tensor,
        cuboids: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute pred_cpl under InputSubset + ClampNorm (gnd logit = 0 fallback).

        Symmetric to `predict_gnd`. The cpl encoder input is the full
        tensor (no masking). For correct joint-clamp behaviour at
        evaluation time, prefer `_predict_joint`.
        """
        cpl_input = self._cpl_input(cuboids)
        emb = self.cuboid_encoder(cpl_input, padding_mask)
        feats_cpl = torch.cat([pair_features, emb], dim=-1)
        logit_cpl = self.cpl_residual.mlp(feats_cpl).squeeze(-1)
        logit_gnd_zero = torch.zeros_like(logit_cpl)
        _, logit_cpl_eff = self._norm_project(logit_gnd_zero, logit_cpl)
        return analytic_C_fF * torch.exp(logit_cpl_eff)

    # ------------------------------------------------------------------
    # Curriculum + introspection (identical to baseline)
    # ------------------------------------------------------------------

    def set_clamp_bounds(self, clamp_bound: float) -> None:
        """Update curriculum cap on BOTH residual heads.

        The two heads share an identical `_clamp_bound` value because the
        joint-norm clamp uses ONE scalar cap per forward.
        """
        if hasattr(self.gnd_residual, "set_clamp_bound"):
            self.gnd_residual.set_clamp_bound(clamp_bound)
        if hasattr(self.cpl_residual, "set_clamp_bound"):
            self.cpl_residual.set_clamp_bound(clamp_bound)

    def parameter_count(self) -> dict:
        return {
            "cuboid_encoder": sum(p.numel() for p in self.cuboid_encoder.parameters()),
            "gnd_residual": sum(p.numel() for p in self.gnd_residual.parameters()),
            "cpl_residual": sum(p.numel() for p in self.cpl_residual.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }
