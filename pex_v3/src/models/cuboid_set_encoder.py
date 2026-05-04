"""
cuboid_set_encoder.py — Phase 1 Path 2 MVP.

DeepSet/PointNet-style permutation-invariant encoder for variable-length
per-net cuboid sequences.

Input:  cuboids (B, N_max, in_dim), padding_mask (B, N_max) where 1=valid
Output: per-net embedding (B, 3 * embed_dim)  — concatenation of mean,
        masked-max, and sum pooling. Multi-pool gives the residual head
        access to both average geometry AND extreme values (e.g., longest
        wire) without needing to learn pooling.

Used by `HybridPexV3Mesh` to inject per-cuboid spatial information into
the gnd/cpl residual heads.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CuboidSetEncoder(nn.Module):
    """Permutation-invariant per-net cuboid encoder.

    Per-cuboid MLP → pool (mean + max + sum) → concat.

    Args:
        in_dim:     per-cuboid feature dim (10 for v3 build:
                    x,y,z,w,h,d,semantic_type,logic_flag,eps,net_type)
        hidden:     hidden width of per-cuboid MLP
        embed_dim:  embedding dim per pool head
        n_layers:   per-cuboid MLP depth (default 2)
        dropout:    dropout in per-cuboid MLP (default 0; turn on for regularization)
    """

    def __init__(
        self,
        in_dim: int = 10,
        hidden: int = 64,
        embed_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim

        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden
        layers.append(nn.Linear(prev, embed_dim))
        self.cuboid_mlp = nn.Sequential(*layers)

    @property
    def out_dim(self) -> int:
        return 3 * self.embed_dim  # mean + max + sum

    def forward(
        self,
        cuboids: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            cuboids: (B, N_max, in_dim) float
            padding_mask: (B, N_max) float, 1=valid, 0=pad
        Returns:
            embedding: (B, 3 * embed_dim) float
        """
        # Per-cuboid embedding
        h = self.cuboid_mlp(cuboids)  # (B, N_max, embed_dim)

        # Mask out padded positions
        mask = padding_mask.unsqueeze(-1)  # (B, N_max, 1)
        h_masked = h * mask

        # Mean pool
        n_valid = padding_mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)
        mean_pool = h_masked.sum(dim=1) / n_valid  # (B, embed_dim)

        # Max pool (masked)
        h_for_max = h.masked_fill(~mask.bool(), float("-inf"))
        max_pool, _ = h_for_max.max(dim=1)  # (B, embed_dim)
        # Replace -inf with 0 for nets with no valid cuboids (shouldn't happen but safe)
        max_pool = torch.where(
            torch.isinf(max_pool),
            torch.zeros_like(max_pool),
            max_pool,
        )

        # Sum pool (proxy for total length / count)
        sum_pool = h_masked.sum(dim=1)  # (B, embed_dim)

        return torch.cat([mean_pool, max_pool, sum_pool], dim=-1)
