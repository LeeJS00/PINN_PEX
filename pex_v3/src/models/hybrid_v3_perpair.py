"""
hybrid_v3_perpair.py — Strike #2: Per-pair coupling head.

Extends HybridPexV3Mesh with explicit per-pair (target, aggressor) cpl head.

Architecture:
    target_emb = cuboid_encoder(target_cuboids, target_mask)            # (B, 3D)
    aggr_emb   = cuboid_encoder(aggr_cuboids[b,j], aggr_mask[b,j])      # (B, K, 3D)

    # GND head: same as Mesh
    gnd_pred[b] = analytic_gnd[b] × bounded_residual(target_self ⊕ target_emb)

    # Per-pair CPL head
    pair_input[b,j] = [analytic_pair, target_self, aggr_self,
                      target_emb, aggr_emb]                             # (1+S+S+3D+3D)
    c_pair_pred[b,j] = analytic_pair_baseline × bounded_residual(pair_input[b,j])

    # Aggregate to net total (for joint loss)
    cpl_total_pred[b] = K^-1 · n_aggr_total[b] · sum_j c_pair_pred[b,j]
                       (unbiased estimator: averaged over K, scaled by total aggressor count)

Loss:
    L = w_gnd · MAPE(gnd_pred, gnd_gold) +
        w_pair · MAPE(c_pair_pred[mask], c_pair_gold[mask]) +
        w_total · MAPE(cpl_total_pred, cpl_total_gold)
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

from src.models.cuboid_set_encoder import CuboidSetEncoder
from src.models.residual_head_v3 import BoundedResidualHead


DEFAULT_SELF_FEATURE_DIM = 16
DEFAULT_PAIR_FEATURE_DIM = 24


class HybridPexV3PerPair(nn.Module):
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
            in_dim=cuboid_in_dim, hidden=cuboid_hidden,
            embed_dim=cuboid_embed_dim, n_layers=cuboid_n_layers,
        )
        emb_dim = self.cuboid_encoder.out_dim  # 3 × embed_dim

        # GND head: same as Mesh
        self.gnd_residual = BoundedResidualHead(
            in_dim=self_feature_dim + emb_dim,
            hidden_dim=residual_hidden,
            n_hidden=residual_n_hidden,
            clamp_bound=clamp_bound,
        )

        # Per-pair CPL head
        # input dim = analytic_pair (1) + target_self + aggr_self + target_emb + aggr_emb
        pair_in_dim = 1 + self_feature_dim + self_feature_dim + emb_dim + emb_dim
        self.pair_residual = BoundedResidualHead(
            in_dim=pair_in_dim,
            hidden_dim=residual_hidden,
            n_hidden=residual_n_hidden,
            clamp_bound=clamp_bound,
        )
        self.emb_dim = emb_dim
        self.self_dim = self_feature_dim
        self.pair_dim = pair_feature_dim

    def encode_target(self, target_cuboids, target_mask) -> torch.Tensor:
        return self.cuboid_encoder(target_cuboids, target_mask)

    def encode_aggressors(self, aggr_cuboids, aggr_mask) -> torch.Tensor:
        """aggr_cuboids: (B, K, Na, in_dim), aggr_mask: (B, K, Na)
        Returns aggr_emb: (B, K, 3D)."""
        B, K, Na, D_in = aggr_cuboids.shape
        # Flatten (B,K) → batch dim
        flat_cb = aggr_cuboids.view(B * K, Na, D_in)
        flat_mk = aggr_mask.view(B * K, Na)
        flat_emb = self.cuboid_encoder(flat_cb, flat_mk)  # (B*K, 3D)
        return flat_emb.view(B, K, -1)

    def predict_gnd(
        self,
        analytic_gnd: torch.Tensor,         # (B,)
        target_self: torch.Tensor,          # (B, S)
        target_emb: torch.Tensor,           # (B, 3D)
    ) -> torch.Tensor:
        feats = torch.cat([target_self, target_emb], dim=-1)
        return analytic_gnd * self.gnd_residual(feats)

    def predict_pair_cpl(
        self,
        analytic_pair: torch.Tensor,        # (B,)        baseline = analytic_cpl_total / n_aggr
        target_self: torch.Tensor,          # (B, S)
        aggr_self: torch.Tensor,            # (B, K, S)
        target_emb: torch.Tensor,           # (B, 3D)
        aggr_emb: torch.Tensor,             # (B, K, 3D)
    ) -> torch.Tensor:
        """Returns c_pair_pred: (B, K)."""
        B, K = aggr_self.shape[:2]
        # Broadcast target -> per-pair
        analytic_per_pair = analytic_pair.unsqueeze(1).expand(B, K)            # (B, K)
        target_self_pp = target_self.unsqueeze(1).expand(B, K, -1)              # (B, K, S)
        target_emb_pp = target_emb.unsqueeze(1).expand(B, K, -1)                # (B, K, 3D)

        feats = torch.cat([
            analytic_per_pair.unsqueeze(-1),    # (B, K, 1)
            target_self_pp,                     # (B, K, S)
            aggr_self,                          # (B, K, S)
            target_emb_pp,                      # (B, K, 3D)
            aggr_emb,                           # (B, K, 3D)
        ], dim=-1)                              # (B, K, pair_in_dim)

        # Flatten to apply per-pair MLP
        flat_feats = feats.view(B * K, -1)
        flat_mult = self.pair_residual(flat_feats)                              # (B*K,)
        mult = flat_mult.view(B, K)                                             # (B, K)
        return analytic_per_pair * mult

    def aggregate_cpl_total(
        self,
        c_pair_pred: torch.Tensor,          # (B, K)
        sampled_mask: torch.Tensor,         # (B, K) — which K positions are valid
        n_aggr_total: torch.Tensor,         # (B,) total aggressors per target
    ) -> torch.Tensor:
        """Estimate per-target cpl_total = mean(c_pair_pred over sampled) × n_aggr_total.

        This is the unbiased estimator under uniform sampling without replacement.
        """
        n_sampled = sampled_mask.sum(dim=1).clamp(min=1.0)                      # (B,)
        sum_pred = (c_pair_pred * sampled_mask).sum(dim=1)                      # (B,)
        mean_pred = sum_pred / n_sampled
        return mean_pred * n_aggr_total

    def set_clamp_bounds(self, clamp_bound: float) -> None:
        self.gnd_residual.set_clamp_bound(clamp_bound)
        self.pair_residual.set_clamp_bound(clamp_bound)

    def parameter_count(self) -> dict:
        return {
            "cuboid_encoder": sum(p.numel() for p in self.cuboid_encoder.parameters()),
            "gnd_residual": sum(p.numel() for p in self.gnd_residual.parameters()),
            "pair_residual": sum(p.numel() for p in self.pair_residual.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }
