"""
NetGammaHead — per-net multiplicative scaling head for heteroscedastic
calibration correction.

Design rationale (after Codex review of /tmp/gamma_head_design.md):
- Apply γ_gnd ONLY (cpl_modifier already provides per-edge variance; adding
  γ_cpl would duplicate that role).
- Drop pred_*_pre_gamma from features — including them lets γ become a
  learned remapper rather than a geometry-aware corrector.
- Drop dominant_layer_idx — area_layer_dist already encodes layer info.
- Init last layer to zero → γ output ≈ 1.0 at start (identity).
- Warmup schedule: gamma_mix = min(1, step/2000); clamp range tightens
  over time. See _gamma_mix_and_clamp() in finetuner integration.
- Optimizer group: 0.1× base LR + identity-regularization penalty.

Features (14 dims):
  [0]    log1p(n_target_cuboids)
  [1]    log1p(total_gnd_area_um2)
  [2]    log1p(total_w_cpl_base_fF)
  [3]    n_layers_present (count of distinct z anchors)
  [4:14] area_layer_dist (10-vector, mirroring calibration_solver buckets)

Output: per-net γ_gnd ∈ exp(clamp([-2, 2])) = [0.135, 7.39].
"""
from __future__ import annotations
import torch
import torch.nn as nn


class NetGammaHead(nn.Module):
    """Per-net multiplicative scaling head.

    Input: (N_nets, 14) per-net feature tensor.
    Output: (N_nets,) γ_gnd scalar per net.
    """
    INPUT_DIM = 14
    HIDDEN_DIM = 32

    def __init__(self, in_dim: int = INPUT_DIM, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Zero-init last layer → output 0 → γ = exp(0) = 1.0 (identity start).
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feats: torch.Tensor, clamp_lo: float = -2.0,
                clamp_hi: float = 2.0, mix: float = 1.0) -> torch.Tensor:
        """Apply γ scaling.

        Args:
            feats:    (N_nets, INPUT_DIM) per-net feature tensor.
            clamp_lo, clamp_hi: clamp range for log γ before exp. Tighter
                                early in training, relaxed over time.
            mix:      [0, 1] mixing factor — γ = exp(mix × clamp(logits)).
                      mix=0 → γ=1 always (off). mix=1 → full γ.

        Returns: (N_nets,) γ_gnd ≥ 0.
        """
        logits = self.net(feats).squeeze(-1)        # (N_nets,)
        clipped = torch.clamp(logits, clamp_lo, clamp_hi)
        gamma = torch.exp(mix * clipped)
        return gamma

    def identity_regularizer(self, feats: torch.Tensor,
                             clamp_lo: float = -2.0,
                             clamp_hi: float = 2.0) -> torch.Tensor:
        """L2 penalty on log γ to keep γ near 1.0 (identity) by default.

        Returns scalar λ × mean(log_γ²) — the caller multiplies by its λ
        weight in the loss assembly.
        """
        logits = self.net(feats).squeeze(-1)
        clipped = torch.clamp(logits, clamp_lo, clamp_hi)
        return torch.mean(clipped ** 2)


def gamma_clamp_schedule(step: int) -> tuple[float, float, float]:
    """Codex-recommended schedule for γ activation.

    Returns (clamp_lo, clamp_hi, mix).
    - step ∈ [0, 2000)   : mix=step/2000, clamp [-0.5, 0.5]    — slow ramp
    - step ∈ [2000, 4000): mix=1.0,        clamp [-1.0, 1.0]   — moderate
    - step ≥ 4000        : mix=1.0,        clamp [-2.0, 2.0]   — full range
    """
    if step < 2000:
        return -0.5, 0.5, max(0.0, step / 2000.0)
    elif step < 4000:
        return -1.0, 1.0, 1.0
    else:
        return -2.0, 2.0, 1.0


def metal_z_buckets(z_anchors_tensor: torch.Tensor) -> torch.Tensor:
    """Map (K,) z-anchor tensor → (K,) int64 bucket indices.

    Mirrors src/data/calibration_solver.py:make_layer_bucket_map.
    Buckets: 0=pre_M1, 1=M1, 2=M2, 3=M3, 4=M4, 5=M5, 6=M6,
             7=upper, 8=top, 9=others.
    """
    z = z_anchors_tensor.cpu().tolist() if z_anchors_tensor.dim() else [float(z_anchors_tensor)]
    def _b(zv: float) -> int:
        if zv < 0.40: return 0
        if zv < 0.60: return 1
        if zv < 0.75: return 2
        if zv < 0.90: return 3
        if zv < 1.05: return 4
        if zv < 1.20: return 5
        if zv < 1.45: return 6
        if zv < 4.50: return 7
        if zv < 6.00: return 8
        return 9
    return torch.tensor([_b(float(v)) for v in z], dtype=torch.int64,
                        device=z_anchors_tensor.device)


def build_per_net_features(
    cuboids: torch.Tensor,         # (B, N, 10) per-tile per-cuboid tensor
    A_tgt: torch.Tensor,           # (B, N) per-tile target wire mask
    core_ratios: torch.Tensor,     # (B, N) per-cuboid core_ratio
    w_cpl_base_per_tile: torch.Tensor | None,  # (B,) Σ w_cpl_base × core_ratio per tile
    batch_net_ids: torch.Tensor,   # (B,) per-tile net id (in [0, num_nets))
    num_nets: int,
    z_anchors: torch.Tensor,       # (K,) flux_router metal_z_anchors
    fringe_init: torch.Tensor,     # (K,) sigmoid(gnd_fringe_scale init)
    n_buckets: int = 10,
) -> torch.Tensor:
    """Build (num_nets, 14) feature tensor from batch tensors.

    Mirrors the geometric aggregation used by src/data/calibration_extractor
    (channel-7 mask for target wires, fringe-corrected gnd_area, core_ratios).

    Returns: (num_nets, INPUT_DIM=14) float32 tensor on cuboids.device.
    """
    device = cuboids.device
    B, N = cuboids.shape[0], cuboids.shape[1]
    K = int(z_anchors.numel())

    # Per-cuboid: target wire mask (channel 7 == 1.0) AND name-match (A_tgt).
    # Use is_target == ch7 to match flux_head's actual contribution mask.
    is_target = ((cuboids[..., 7] == 1.0).float() * A_tgt).clamp(0.0, 1.0)  # (B, N)

    # gnd_area = bottom + fringe[layer]*sidewall (mirror flux_head:286-291)
    w = cuboids[..., 3]; h = cuboids[..., 4]; d = cuboids[..., 5]
    z_abs = cuboids[..., 2]
    bottom_area    = torch.clamp(w * h, min=1e-6)
    sidewall_area  = 2.0 * (w + h) * d
    z_idx = torch.argmin(torch.abs(z_abs.unsqueeze(-1) - z_anchors), dim=-1)  # (B, N) ∈ [0, K)
    ff = fringe_init[z_idx]                                                   # (B, N)
    gnd_area = bottom_area + ff * sidewall_area                                # (B, N)
    gnd_area_eff = (gnd_area * is_target * core_ratios).float()                # (B, N)

    # Bucket z_idx → bucket idx (per-cuboid)
    bucket_lut = metal_z_buckets(z_anchors).to(device)                         # (K,)
    bucket_per_cub = bucket_lut[z_idx]                                         # (B, N)

    # Per-tile aggregates (we'll scatter these into per-net buckets).
    tile_n_target = is_target.sum(dim=1)                                       # (B,)
    tile_total_area = gnd_area_eff.sum(dim=1)                                  # (B,)
    # Per-tile per-bucket area: scatter into (B, n_buckets)
    tile_area_per_bucket = torch.zeros(B, n_buckets, dtype=torch.float32,
                                        device=device)
    flat_b = torch.arange(B, device=device).unsqueeze(-1).expand(B, N).reshape(-1)
    flat_bucket = bucket_per_cub.reshape(-1)
    flat_area = gnd_area_eff.reshape(-1)
    flat_idx = flat_b * n_buckets + flat_bucket
    tile_area_per_bucket = tile_area_per_bucket.view(-1).index_add(
        0, flat_idx, flat_area
    ).view(B, n_buckets)

    # Aggregate to per-net via batch_net_ids.
    net_n_target  = torch.zeros(num_nets, dtype=torch.float32, device=device)
    net_total_area = torch.zeros(num_nets, dtype=torch.float32, device=device)
    net_w_cpl_base = torch.zeros(num_nets, dtype=torch.float32, device=device)
    net_area_per_bucket = torch.zeros(num_nets, n_buckets, dtype=torch.float32, device=device)

    net_n_target.index_add_(0, batch_net_ids, tile_n_target)
    net_total_area.index_add_(0, batch_net_ids, tile_total_area)
    if w_cpl_base_per_tile is not None:
        net_w_cpl_base.index_add_(0, batch_net_ids, w_cpl_base_per_tile.float())
    net_area_per_bucket.index_add_(0, batch_net_ids, tile_area_per_bucket)

    # n_layers_present: count of buckets with > 0 area.
    n_layers_present = (net_area_per_bucket > 0).sum(dim=1).float()             # (num_nets,)
    # Area distribution: normalize per net.
    total = net_area_per_bucket.sum(dim=1, keepdim=True).clamp(min=1e-6)
    area_dist = net_area_per_bucket / total                                     # (num_nets, n_buckets)

    feats = torch.cat([
        torch.log1p(net_n_target).unsqueeze(-1),       # 1
        torch.log1p(net_total_area).unsqueeze(-1),     # 1
        torch.log1p(net_w_cpl_base).unsqueeze(-1),     # 1
        n_layers_present.unsqueeze(-1),                # 1
        area_dist,                                     # 10
    ], dim=-1)                                          # (num_nets, 14)
    return feats
