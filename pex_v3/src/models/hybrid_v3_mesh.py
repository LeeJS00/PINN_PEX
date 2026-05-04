"""
hybrid_v3_mesh.py — Phase 1 Path 2: Hybrid_v3 + Cuboid Set Encoder.

Same as `HybridPexV3` but the gnd/cpl residual heads receive an additional
per-net cuboid set embedding (DeepSet-style) on top of the existing
hand-engineered scalar features.

Day-1 invariant preserved: zero-init last layers → multiplier = 1.0
→ output = analytic baseline (calibrated).

Inference contract:
    embed = cuboid_encoder(cuboids[net], mask[net])              # (B, 3*embed_dim)
    gnd_pred = analytic_gnd × bounded_multiplier(self_features ⊕ embed)
    cpl_pred = analytic_cpl × bounded_multiplier(pair_features ⊕ embed)

Per Codex Round 4 (2026-05-03): if this MVP achieves ≤ 8% per-net total
MAPE, conclude spatial info sufficient (no full mesh_v3 BEM patches needed).
"""
from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn

from .residual_head_v3 import (
    BoundedResidualHead,
    PerPairTypeBoundedResidualHead,
)
from .cuboid_set_encoder import CuboidSetEncoder


DEFAULT_SELF_FEATURE_DIM = 16
DEFAULT_PAIR_FEATURE_DIM = 24


class HybridPexV3Mesh(nn.Module):
    """Hybrid analytic + cuboid set encoder + bounded residual.

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
        per_pair_clamp:    not used by mesh variant yet
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
        emb = self.cuboid_encoder(cuboids, padding_mask)
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
        emb = self.cuboid_encoder(cuboids, padding_mask)
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
            "cuboid_encoder": sum(p.numel() for p in self.cuboid_encoder.parameters()),
            "gnd_residual": sum(p.numel() for p in self.gnd_residual.parameters()),
            "cpl_residual": sum(p.numel() for p in self.cpl_residual.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }
