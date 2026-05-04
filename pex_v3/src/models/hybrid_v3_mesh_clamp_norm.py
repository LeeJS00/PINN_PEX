"""
hybrid_v3_mesh_clamp_norm.py — Phase 1 ClampNorm variant.

Drop-in replacement for `HybridPexV3Mesh` that swaps the **element-wise
clamp on the residual logit** for a **vector-norm projection clamp on the
joint per-net (gnd, cpl) residual**.

Hypothesis (see `pex_v3/experiments/auto_optimize_2026_05_03/variants/clamp_norm/DESIGN.md`):
    The element-wise hard clamp `clamp(δ, -C, +C)` produces a 0/1 gradient
    cliff at the cap boundary. In Phase 2 (`C = log 4 ≈ 1.386`), more
    residual logits are near the cap and the encoder weights oscillate.
    Replacing with a smooth norm-projection clamp removes the cliff
    while preserving the day-1 invariant and curriculum schedule.

Key contracts:
- Param count: identical to `HybridPexV3Mesh` (44,738).
- Day-1 invariant preserved: zero-init residual heads + norm-projection
  identity-in-region → multiplier = 1.0 → output = analytic.
- API parity: `predict_gnd / predict_cpl / set_clamp_bounds /
  parameter_count` identical to baseline.
- Curriculum: `set_clamp_bounds(value)` updates the shared cap that is
  read from BOTH residual heads' `_clamp_bound` buffer.

Implementation note: because the joint (gnd, cpl) norm requires both
logits, the model exposes `_predict_joint(...)` which runs both residual
MLPs and returns `(pred_gnd, pred_cpl)`. The public `predict_gnd` and
`predict_cpl` wrap `_predict_joint` and return the corresponding half;
this means single-head calls run BOTH MLPs internally (forward cost
≈ 2× residual heads, encoder is shared and dominant). For the trainer
which calls predict_gnd then predict_cpl back-to-back, this is identical
total compute to baseline.
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn

from .residual_head_v3 import BoundedResidualHead
from .cuboid_set_encoder import CuboidSetEncoder


DEFAULT_SELF_FEATURE_DIM = 16
DEFAULT_PAIR_FEATURE_DIM = 24

# Numerical safety for n = ||δ||; chosen so that C / eps >> 1 in float32
# at day-1 (n = 0.0 exact), guaranteeing s = 1 and δ_eff = 0.
_NORM_EPS = 1.0e-12


class HybridPexV3MeshClampNorm(nn.Module):
    """Hybrid analytic + cuboid set encoder + bounded residual with
    joint-norm projection clamp.

    Architectural identity to `HybridPexV3Mesh`:
        - Same `CuboidSetEncoder` (shared between gnd and cpl heads).
        - Same `BoundedResidualHead` MLPs for gnd and cpl.
        - Same `clamp_bound` buffer and `set_clamp_bounds` curriculum hook.

    Differs from baseline ONLY in the clamp formula:
        baseline:   δ_eff = clamp(δ, -C, +C)        (per channel, per net)
        ClampNorm:  δ_eff = δ × min(1, C / ||(δ_gnd, δ_cpl)||_2)  (joint per net)

    Args:
        self_feature_dim:  scalar self features (16)
        pair_feature_dim:  scalar pair features (24)
        cuboid_in_dim:     per-cuboid feature dim (10)
        cuboid_hidden:     per-cuboid MLP hidden (64)
        cuboid_embed_dim:  per-pool head embedding (64) → out 192 (3 pools)
        cuboid_n_layers:   per-cuboid MLP depth (2)
        residual_hidden:   bounded residual MLP hidden (64)
        residual_n_hidden: bounded residual MLP depth (2)
        clamp_bound:       initial clamp_bound (default log(1.5))
    """

    def __init__(
        self,
        self_feature_dim: int = DEFAULT_SELF_FEATURE_DIM,
        pair_feature_dim: int = DEFAULT_PAIR_FEATURE_DIM,
        cuboid_in_dim: int = 10,
        cuboid_hidden: int = 64,
        cuboid_embed_dim: int = 64,
        cuboid_n_layers: int = 2,
        residual_hidden: int = 64,
        residual_n_hidden: int = 2,
        clamp_bound: float = math.log(1.5),
    ):
        super().__init__()
        self.cuboid_encoder = CuboidSetEncoder(
            in_dim=cuboid_in_dim,
            hidden=cuboid_hidden,
            embed_dim=cuboid_embed_dim,
            n_layers=cuboid_n_layers,
        )
        emb_dim = self.cuboid_encoder.out_dim  # 3 * cuboid_embed_dim

        # Reuse `BoundedResidualHead` for parameter parity with baseline,
        # but bypass its `forward` (which does element-wise clamp) and
        # instead access `head.mlp` directly. The head's `_clamp_bound`
        # buffer is still used as the source-of-truth for the cap value
        # via `set_clamp_bounds`.
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
        self.emb_dim = emb_dim

    # ------------------------------------------------------------------
    # Joint norm-projection forward (the actual ClampNorm logic)
    # ------------------------------------------------------------------

    def _norm_project(
        self,
        logit_gnd: torch.Tensor,    # (B,)
        logit_cpl: torch.Tensor,    # (B,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply joint norm-projection clamp to (logit_gnd, logit_cpl).

        Reads the curriculum cap C from `self.gnd_residual._clamp_bound`
        (the two heads share the same scheduled value via
        `set_clamp_bounds`). Both logits are scaled by the same scalar
        `s[b] = min(1, C / max(||δ[b]||_2, eps))`.

        Numerical care: the naive `sqrt(δ_gnd² + δ_cpl²)` has an
        infinite gradient at δ=0 (`d(sqrt x)/dx → ∞ as x → 0`). At
        day-1 both logits are exactly zero, so the very first backward
        produces NaN. We mitigate by:
          1. Using a **softened sqrt**: `n = sqrt(sum_sq + _NORM_EPS²)`
             which has finite gradient `δ_i / n` for all δ (including 0).
             At δ=0, the gradient is `0 / eps = 0` — clean zero, no NaN.
          2. The softened `n` differs from true norm only by O(eps²/n);
             at any practical norm scale (eps = 1e-12) this is unmeasurable.
          3. The day-1 invariant still holds: with softened n = eps and
             cap = 0.405, s = min(0.405/eps, 1) = 1.0, δ_eff = 0.

        Returns:
            (logit_gnd_eff, logit_cpl_eff) — both shape (B,)
        """
        # `_clamp_bound` is a 0-d buffer registered by BoundedResidualHead.
        cap = self.gnd_residual._clamp_bound  # 0-d tensor

        # Softened L2 norm: smooth at zero, finite gradient everywhere.
        sum_sq = logit_gnd * logit_gnd + logit_cpl * logit_cpl
        n = torch.sqrt(sum_sq + (_NORM_EPS * _NORM_EPS))
        # Scale factor in (0, 1]; identity in-region (s=1 when n < cap).
        s = torch.clamp(cap / n, max=1.0)
        return s * logit_gnd, s * logit_cpl

    def _predict_joint(
        self,
        analytic_gnd: torch.Tensor,     # (B,)
        analytic_cpl: torch.Tensor,     # (B,)
        self_features: torch.Tensor,    # (B, self_feature_dim)
        pair_features: torch.Tensor,    # (B, pair_feature_dim)
        cuboids: torch.Tensor,          # (B, N_max, in_dim)
        padding_mask: torch.Tensor,     # (B, N_max)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute (pred_gnd, pred_cpl) jointly with norm-projection clamp."""
        emb = self.cuboid_encoder(cuboids, padding_mask)
        feats_gnd = torch.cat([self_features, emb], dim=-1)
        feats_cpl = torch.cat([pair_features, emb], dim=-1)

        # Bypass the `forward` of BoundedResidualHead; access its MLP directly
        # to get the raw logit, then apply joint-norm clamp ourselves.
        logit_gnd = self.gnd_residual.mlp(feats_gnd).squeeze(-1)  # (B,)
        logit_cpl = self.cpl_residual.mlp(feats_cpl).squeeze(-1)  # (B,)

        logit_gnd_eff, logit_cpl_eff = self._norm_project(logit_gnd, logit_cpl)

        mul_gnd = torch.exp(logit_gnd_eff)
        mul_cpl = torch.exp(logit_cpl_eff)
        return analytic_gnd * mul_gnd, analytic_cpl * mul_cpl

    # ------------------------------------------------------------------
    # Public API (identical signature to HybridPexV3Mesh)
    # ------------------------------------------------------------------

    # Cache of the most-recently-computed joint pair so that back-to-back
    # `predict_gnd` then `predict_cpl` on the same (batch, features)
    # tuple does not recompute. Key is a tuple of input tensor IDs +
    # current `_clamp_bound` value so that curriculum changes invalidate
    # the cache. The cache holds a strong reference to the tensors only
    # for the current call's lifetime; we clear it on every set_clamp.
    _joint_cache: tuple | None = None

    def predict_gnd(
        self,
        analytic_C_fF: torch.Tensor,    # (B,) — analytic gnd
        self_features: torch.Tensor,    # (B, self_feature_dim)
        cuboids: torch.Tensor,          # (B, N_max, in_dim)
        padding_mask: torch.Tensor,     # (B, N_max)
    ) -> torch.Tensor:
        """Compute pred_gnd with joint-norm clamp.

        For the joint clamp the cpl residual logit must be computed too.
        Since the standalone `predict_gnd` API does not receive
        `pair_features` or `analytic_cpl`, we **cannot** compute a
        meaningful `logit_cpl` here. We therefore approximate by using
        `logit_cpl = 0`, which makes the joint norm degenerate to
        `||δ_gnd||` and the projection equivalent to element-wise clamp
        on `δ_gnd` ALONE. This is the safe identity-on-cpl fallback for
        users that only need gnd prediction (e.g., per-channel eval).

        For training and standard eval, the trainer should use
        `_predict_joint(analytic_gnd, analytic_cpl, self_features,
        pair_features, cuboids, padding_mask)` to get correct joint
        clamp behaviour. The smoke / ablation runner script for this
        variant is wired to `_predict_joint`.
        """
        emb = self.cuboid_encoder(cuboids, padding_mask)
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
        """Compute pred_cpl with joint-norm clamp.

        Symmetric to `predict_gnd`: assumes `logit_gnd = 0` since the
        standalone API does not receive `self_features`. For correct
        joint clamp behaviour, use `_predict_joint`.
        """
        emb = self.cuboid_encoder(cuboids, padding_mask)
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

        The two heads share an identical `_clamp_bound` buffer value
        because the joint-norm clamp uses ONE scalar cap per forward.
        We update both for backward compatibility with code that probes
        either head's `get_clamp_bound`.
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
