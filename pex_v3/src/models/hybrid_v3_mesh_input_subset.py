"""
hybrid_v3_mesh_input_subset.py — Phase 1 InputSubset variant.

Same shared `CuboidSetEncoder` as the locked baseline (`HybridPexV3Mesh`).
The gnd-head call passes a column-masked cuboid tensor (interaction
columns ch6/7/9 zeroed); the cpl-head call passes the full tensor.

Design rationale: see
`pex_v3/experiments/auto_optimize_2026_05_03/variants/input_subset/DESIGN.md`.

Codex Round 2 constraint (zero-masking ONLY, shared weights):
    - One `CuboidSetEncoder` instance — no encoder duplication.
    - The mask is a fixed `(1, 1, 10)` non-trainable buffer.
    - Multiplication is element-wise on the input tensor.
    - Separate input projections would be A1-in-disguise → FORBIDDEN.

Day-1 invariant preserved: zero-init residual head last linear →
multiplier = 1.0 → output = analytic baseline.

Drop-in API parity with `HybridPexV3Mesh`.
"""
from __future__ import annotations
import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn

from .residual_head_v3 import BoundedResidualHead
from .cuboid_set_encoder import CuboidSetEncoder


DEFAULT_SELF_FEATURE_DIM = 16
DEFAULT_PAIR_FEATURE_DIM = 24
DEFAULT_CUBOID_IN_DIM = 10

# Channel partition (verified from per_net_cuboids/intel22_*.npz):
#   0,1,2,3,4,5  : x_rel, y_rel, z_abs, w, h, d   (GEO_CORE)
#   6            : semantic_type      (INTERACTION)
#   7            : is_target          (INTERACTION)
#   8            : eps                (MATERIAL)
#   9            : net_type           (INTERACTION)
DEFAULT_INTERACTION_CHANNELS: tuple[int, ...] = (6, 7, 9)


def _build_channel_mask(
    in_dim: int,
    interaction_channels: Sequence[int],
) -> torch.Tensor:
    """Return a (1, 1, in_dim) float tensor with 0 at interaction channels, 1 elsewhere."""
    mask = torch.ones(in_dim, dtype=torch.float32)
    for c in interaction_channels:
        if c < 0 or c >= in_dim:
            raise ValueError(
                f"interaction channel {c} out of range for in_dim={in_dim}"
            )
        mask[c] = 0.0
    return mask.view(1, 1, in_dim)


class HybridPexV3MeshInputSubset(nn.Module):
    """Hybrid analytic + shared cuboid encoder + per-channel zero-masked input.

    Args:
        self_feature_dim:       scalar self features (16)
        pair_feature_dim:       scalar pair features (24)
        cuboid_in_dim:          per-cuboid feature dim (10)
        cuboid_hidden:          per-cuboid MLP hidden (64)
        cuboid_embed_dim:       per-pool head embedding (64) → out_dim 192 (3 pools)
        cuboid_n_layers:        per-cuboid MLP depth (2)
        residual_hidden:        bounded residual MLP hidden (64)
        residual_n_hidden:      bounded residual MLP depth (2)
        clamp_bound:            initial clamp_bound (default log(1.5))
        gnd_interaction_channels:
            channel indices to zero in the gnd encoder input
            (default (6, 7, 9): semantic_type, is_target, net_type).
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

        # Non-trainable channel mask buffer: (1, 1, in_dim).
        gnd_mask = _build_channel_mask(cuboid_in_dim, self.gnd_interaction_channels)
        self.register_buffer("gnd_channel_mask", gnd_mask)
        # cpl mask is identity but we register a buffer for symmetric inspection.
        self.register_buffer("cpl_channel_mask", torch.ones_like(gnd_mask))

    # ------------------------------------------------------------------

    def _gnd_input(self, cuboids: torch.Tensor) -> torch.Tensor:
        """Apply the gnd channel mask to the cuboid tensor (out-of-place)."""
        # Broadcast: (B, N, 10) * (1, 1, 10) → (B, N, 10)
        return cuboids * self.gnd_channel_mask

    def _cpl_input(self, cuboids: torch.Tensor) -> torch.Tensor:
        """Identity (cpl sees the full tensor)."""
        return cuboids

    # ------------------------------------------------------------------

    def predict_gnd(
        self,
        analytic_C_fF: torch.Tensor,    # (B,)
        self_features: torch.Tensor,    # (B, self_feature_dim)
        cuboids: torch.Tensor,          # (B, N_max, in_dim)
        padding_mask: torch.Tensor,     # (B, N_max), 1=valid
    ) -> torch.Tensor:
        masked = self._gnd_input(cuboids)
        emb = self.cuboid_encoder(masked, padding_mask)
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
        full = self._cpl_input(cuboids)
        emb = self.cuboid_encoder(full, padding_mask)
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
