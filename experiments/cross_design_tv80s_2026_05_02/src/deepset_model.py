"""
DeepSet/MLP hybrid model for per-net total_cap regression.

Architecture:
    1. Per-cuboid MLP encoder (shared) over (10) features.
    2. Symmetric pooling per kind: mean, max, sum, std → 4× embedding.
    3. Hand-feature MLP branch on the cached parquet features (~57 dims).
    4. Concat all branches → trunk MLP → head.

Output: log(total_cap_fF + eps).

Loss: weighted MAPE (the target metric) + 0.1 × MSE on log scale.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pad_mask(n: torch.Tensor, max_n: int) -> torch.Tensor:
    """n: (B,)  → mask of shape (B, max_n) where True = valid."""
    rng = torch.arange(max_n, device=n.device).unsqueeze(0)   # (1, max_n)
    return rng < n.unsqueeze(1)


def masked_pool(emb: torch.Tensor, mask: torch.Tensor, kind: str = "mean") -> torch.Tensor:
    """emb: (B, K, D), mask: (B, K) bool. Returns (B, D)."""
    m = mask.unsqueeze(-1).to(emb.dtype)
    if kind == "sum":
        return (emb * m).sum(dim=1)
    if kind == "mean":
        denom = m.sum(dim=1).clamp(min=1.0)
        return (emb * m).sum(dim=1) / denom
    if kind == "max":
        very_neg = torch.finfo(emb.dtype).min
        emb_masked = emb.masked_fill(~mask.unsqueeze(-1), very_neg)
        return emb_masked.max(dim=1).values
    if kind == "std":
        denom = m.sum(dim=1).clamp(min=1.0)
        mu = (emb * m).sum(dim=1, keepdim=True) / denom.unsqueeze(-1)
        var = ((emb - mu) ** 2 * m).sum(dim=1) / denom
        return torch.sqrt(var + 1e-8)
    raise ValueError(kind)


class CuboidEncoder(nn.Module):
    def __init__(self, in_dim: int = 10, hidden: int = 128, out: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out),
        )

    def _scale_input(self, x: torch.Tensor) -> torch.Tensor:
        # Conservative scaling so the encoder sees normalized inputs.
        # x: (B, N, 10): [x_rel, y_rel, z_abs, w, h, d, semantic, logic, eps, net_type]
        x = x.clone()
        SCALE = 4.0   # window half-extent in xy
        x[..., 0] = x[..., 0] / SCALE
        x[..., 1] = x[..., 1] / SCALE
        x[..., 2] = (x[..., 2] - 1.5) / 1.0          # z range ~0.5..3.0
        x[..., 3] = torch.log1p(x[..., 3])           # w
        x[..., 4] = torch.log1p(x[..., 4])           # h
        x[..., 5] = torch.log1p(x[..., 5])           # d
        x[..., 8] = (x[..., 8] - 3.5) / 1.0          # eps ~ 2.8..4.0
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self._scale_input(x))


class DeepSetCapModel(nn.Module):
    """3-stream DeepSet on (target, aggressor, power) cuboids + hand-feature branch."""

    def __init__(
        self,
        cuboid_dim: int = 10,
        hand_dim: int = 57,
        cuboid_hidden: int = 128,
        cuboid_out: int = 128,
        hand_hidden: int = 128,
        trunk_hidden: int = 256,
    ):
        super().__init__()
        self.enc_t = CuboidEncoder(cuboid_dim, cuboid_hidden, cuboid_out)
        self.enc_a = CuboidEncoder(cuboid_dim, cuboid_hidden, cuboid_out)
        self.enc_p = CuboidEncoder(cuboid_dim, cuboid_hidden, cuboid_out)

        # Per-stream pooled embedding has 4 pool stats × cuboid_out
        pool_dim = 4 * cuboid_out
        self.set_proj_t = nn.Sequential(nn.Linear(pool_dim, trunk_hidden), nn.GELU())
        self.set_proj_a = nn.Sequential(nn.Linear(pool_dim, trunk_hidden), nn.GELU())
        self.set_proj_p = nn.Sequential(nn.Linear(pool_dim, trunk_hidden), nn.GELU())

        self.hand_branch = nn.Sequential(
            nn.Linear(hand_dim, hand_hidden), nn.GELU(),
            nn.Linear(hand_hidden, hand_hidden), nn.GELU(),
        )

        in_trunk = 3 * trunk_hidden + hand_hidden
        self.trunk = nn.Sequential(
            nn.Linear(in_trunk, trunk_hidden), nn.GELU(),
            nn.Linear(trunk_hidden, trunk_hidden), nn.GELU(),
            nn.Linear(trunk_hidden, 1),
        )

        # Bias init from compact_gnd magnitude (rough median 0.5 fF → log -0.7).
        self.log_bias = nn.Parameter(torch.tensor(0.0))

    def _stream(self, enc, x, n, max_n, proj):
        emb = enc(x)                              # (B, K, D)
        mask = _pad_mask(n, max_n)                # (B, K)
        pools = [masked_pool(emb, mask, kind=k) for k in ("mean", "max", "sum", "std")]
        return proj(torch.cat(pools, dim=-1))

    def forward(self, batch):
        T = batch["target"]
        A = batch["aggressor"]
        P = batch["power"]
        nT = batch["n_target"]; nA = batch["n_agg"]; nP = batch["n_pwr"]
        hand = batch["hand"]

        zt = self._stream(self.enc_t, T, nT, T.shape[1], self.set_proj_t)
        za = self._stream(self.enc_a, A, nA, A.shape[1], self.set_proj_a)
        zp = self._stream(self.enc_p, P, nP, P.shape[1], self.set_proj_p)
        zh = self.hand_branch(hand)
        z = torch.cat([zt, za, zp, zh], dim=-1)
        return self.trunk(z).squeeze(-1) + self.log_bias


# ---------------------------------------------------------------------------
# Loss: log-MAPE-style (smoothed) + small MSE regulariser
# ---------------------------------------------------------------------------


def hybrid_loss(pred_log: torch.Tensor, y_lin: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    pred = torch.exp(pred_log)
    y = y_lin.clamp(min=eps)
    ape = torch.abs(pred - y) / y
    return ape.mean() + 0.05 * F.mse_loss(pred_log, torch.log(y))
