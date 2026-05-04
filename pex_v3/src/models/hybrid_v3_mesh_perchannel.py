"""
hybrid_v3_mesh_perchannel.py — Phase 1 Vector A1: per-channel encoders.

Variant of `HybridPexV3Mesh` with TWO independent `CuboidSetEncoder`
instances — one feeding the gnd residual head, one feeding the cpl
residual head. Disjoint gradient paths let each channel specialize.

Motivation (see `pex_v3/docs/A1_PERCHANNEL_ENCODER_DESIGN.md`):
    Strike #8 diagnostic identified the shared cuboid encoder as the
    only untested architectural lever. gnd and cpl errors correlate
    only weakly (ρ=0.33), suggesting they need different per-cuboid
    aggregations.

Day-1 invariant preserved: both residual heads zero-init last linear →
multiplier = 1.0 → output = analytic baseline. The encoders are NOT in
the day-1 critical path.

Drop-in API parity with `HybridPexV3Mesh`:
    predict_gnd(analytic_C_fF, self_features, cuboids, padding_mask)
    predict_cpl(analytic_C_fF, pair_features, cuboids, padding_mask)
    set_clamp_bounds(value)
    parameter_count() -> dict

Param budget: ~53.8K (vs 44.7K shared baseline) — 1.20× factor, well
under the 100K cap.
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn

from .residual_head_v3 import BoundedResidualHead
from .cuboid_set_encoder import CuboidSetEncoder


DEFAULT_SELF_FEATURE_DIM = 16
DEFAULT_PAIR_FEATURE_DIM = 24


class HybridPexV3MeshPerChannel(nn.Module):
    """Hybrid analytic + per-channel cuboid encoders + bounded residuals.

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

        # Two independent encoders — separate parameters, identical architecture.
        self.gnd_encoder = CuboidSetEncoder(
            in_dim=cuboid_in_dim,
            hidden=cuboid_hidden,
            embed_dim=cuboid_embed_dim,
            n_layers=cuboid_n_layers,
        )
        self.cpl_encoder = CuboidSetEncoder(
            in_dim=cuboid_in_dim,
            hidden=cuboid_hidden,
            embed_dim=cuboid_embed_dim,
            n_layers=cuboid_n_layers,
        )
        emb_dim = self.gnd_encoder.out_dim  # 3 * cuboid_embed_dim, equal for both

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

    def predict_gnd(
        self,
        analytic_C_fF: torch.Tensor,    # (B,)
        self_features: torch.Tensor,    # (B, self_feature_dim)
        cuboids: torch.Tensor,          # (B, N_max, in_dim)
        padding_mask: torch.Tensor,     # (B, N_max), 1=valid
    ) -> torch.Tensor:
        emb = self.gnd_encoder(cuboids, padding_mask)
        feats = torch.cat([self_features, emb], dim=-1)
        multiplier = self.gnd_residual(feats)
        return analytic_C_fF * multiplier

    def predict_cpl(
        self,
        analytic_C_fF: torch.Tensor,    # (B,)
        pair_features: torch.Tensor,    # (B, pair_feature_dim)
        cuboids: torch.Tensor,          # (B, N_max, in_dim)
        padding_mask: torch.Tensor,     # (B, N_max)
    ) -> torch.Tensor:
        emb = self.cpl_encoder(cuboids, padding_mask)
        feats = torch.cat([pair_features, emb], dim=-1)
        multiplier = self.cpl_residual(feats)
        return analytic_C_fF * multiplier

    # ------------------------------------------------------------------

    def set_clamp_bounds(self, clamp_bound: float) -> None:
        if hasattr(self.gnd_residual, "set_clamp_bound"):
            self.gnd_residual.set_clamp_bound(clamp_bound)
        if hasattr(self.cpl_residual, "set_clamp_bound"):
            self.cpl_residual.set_clamp_bound(clamp_bound)

    def parameter_count(self) -> dict:
        return {
            "gnd_encoder": sum(p.numel() for p in self.gnd_encoder.parameters()),
            "cpl_encoder": sum(p.numel() for p in self.cpl_encoder.parameters()),
            "gnd_residual": sum(p.numel() for p in self.gnd_residual.parameters()),
            "cpl_residual": sum(p.numel() for p in self.cpl_residual.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }
