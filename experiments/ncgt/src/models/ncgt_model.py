"""
NCGT model — Net-Centric Graph Transformer (Plan v4 §2).

Architecture:
1. Heterogeneous typed encoder: per-type MLPs + shared geometry MLP + layer embedding.
2. Sparse 3D attention backbone: target self-attention + edge-driven cross-attention
   (target ← aggressor) + readout-only global token.
3. Physics-guided residual heads (ResCap): C = C_base · (1 + residual).
4. (Phase 2.3+) Bin-specialized heads: GND/CPL each split into 5 magnitude bins with
   focal-loss classifier and per-bin residual MLPs.

Phase 1.0 mode: bins disabled (single GND/CPL head).
Phase 2.3 mode: bins enabled.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.ncgt.src.data.ncgt_dataset import N_TYPES
from experiments.ncgt.src.data.physics_base import (
    cpl_base_per_edge,
    compose_with_residual,
    compute_segment_geometry,
    edge_overlap_length,
    gnd_base_per_segment,
)


# Bin edges (fF) per Plan v4 §2.6. Phase 0 audit will refine after running on
# actual gnd/cpl distributions; placeholder values here.
GND_BIN_EDGES_FF = (0.0, 0.01, 0.1, 1.0, 10.0, float("inf"))
CPL_BIN_EDGES_FF = (0.0, 0.01, 0.1, 1.0, 10.0, float("inf"))
N_BINS = 5


def bin_assign(values: torch.Tensor, edges: tuple) -> torch.Tensor:
    """Assign values to bin indices using edges (last edge is +inf).

    edges = (e0, e1, e2, e3, e4, e5=inf): bin k = where edges[k] ≤ v < edges[k+1].
    """
    out = torch.zeros_like(values, dtype=torch.long)
    for k in range(len(edges) - 1):
        lo = edges[k]
        hi = edges[k + 1]
        mask = (values >= lo) & (values < hi)
        out = torch.where(mask, torch.full_like(out, k), out)
    return out

# Layer index range for embedding (intel22 LAYERS_INFO max ~250 in centimicron z*100).
N_LAYER_BUCKETS = 256


@dataclass
class NCGTConfig:
    d_model: int = 128
    d_geom: int = 64
    d_type: int = 64
    d_layer: int = 16
    n_blocks: int = 4
    n_heads: int = 8
    dropout: float = 0.1
    use_bins: bool = False  # Phase 2.3+
    eps_default: float = 3.0
    d_default: float = 0.2  # μm metal-to-layer-stack distance fallback
    t_default: float = 0.144  # μm metal thickness fallback
    n_layer_buckets: int = N_LAYER_BUCKETS
    n_bins: int = N_BINS
    bin_edges_fF: tuple = GND_BIN_EDGES_FF  # for routing physics base → bin
    boundary_smooth_pct: float = 0.10  # 10% of boundary → soft-route across two bins
    log_range: float = 2.3  # tanh(±∞) × 2.3 → exp(±2.3) ≈ ×0.1 .. ×10
    # Phase D: pex_v3 curriculum residual (hard clamp + progressive widening).
    use_curriculum: bool = False
    clamp_bound: float = math.log(1.5)  # initial; widened by trainer per schedule


class HeterogeneousEncoder(nn.Module):
    """Type-specific MLP + shared geometry MLP + layer embedding."""

    def __init__(self, cfg: NCGTConfig):
        super().__init__()
        self.cfg = cfg
        # Per-type MLPs (one per type id).
        self.type_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(12, cfg.d_type),
                    nn.GELU(),
                    nn.Linear(cfg.d_type, cfg.d_type),
                )
                for _ in range(N_TYPES)
            ]
        )
        # Shared geometry MLP.
        self.geom_mlp = nn.Sequential(
            nn.Linear(12, cfg.d_geom),
            nn.GELU(),
            nn.Linear(cfg.d_geom, cfg.d_geom),
        )
        # Layer embedding (z*100 bucket).
        self.layer_emb = nn.Embedding(cfg.n_layer_buckets, cfg.d_layer)
        # Final projection to d_model.
        self.proj = nn.Linear(cfg.d_geom + cfg.d_type + cfg.d_layer, cfg.d_model)

    def forward(self, feats: torch.Tensor, type_ids: torch.Tensor) -> torch.Tensor:
        """feats: (N, 12), type_ids: (N,) → (N, d_model)."""
        N = feats.shape[0]
        device = feats.device
        z_geom = self.geom_mlp(feats)  # (N, d_geom)

        # Per-type pass: route by type_id.
        z_type = torch.zeros(N, self.cfg.d_type, device=device, dtype=feats.dtype)
        for tid in range(N_TYPES):
            mask = type_ids == tid
            if mask.any():
                z_type[mask] = self.type_mlps[tid](feats[mask])

        # Layer index from feats[:, 7] (layer_idx column per ncgt_dataset.segment_to_features).
        layer_idx = feats[:, 7].clamp(0, self.cfg.n_layer_buckets - 1).long()
        z_layer = self.layer_emb(layer_idx)  # (N, d_layer)

        z = torch.cat([z_geom, z_type, z_layer], dim=-1)
        return self.proj(z)


class TransformerBlock(nn.Module):
    """One block: self-attention + edge-driven cross-attention + FFN.

    No batch normalization (NAS-Cap finding).
    """

    def __init__(self, cfg: NCGTConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.self_norm = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d)
        self.cross_qkv = nn.Linear(d, 3 * d)
        self.cross_out = nn.Linear(d, d)
        self.ffn_norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(4 * d, d)
        )

    def edge_cross_attention(
        self,
        z_t: torch.Tensor,         # (T, d)
        z_a: torch.Tensor,         # (A, d)
        edge_index: torch.Tensor,  # (2, E)
    ) -> torch.Tensor:
        """For each target segment, gather edge-connected aggressors and attend.

        Vectorized via segment-by-edge gathering then per-target softmax.
        """
        if edge_index.shape[1] == 0:
            return torch.zeros_like(z_t)

        T, d = z_t.shape
        H = self.cfg.n_heads
        dh = d // H

        # Compute QKV.
        qkv_t = self.cross_qkv(z_t).view(T, 3, H, dh)  # split per target
        q_t = qkv_t[:, 0]                              # (T, H, dh)
        # For aggressors, only K, V are needed.
        qkv_a = self.cross_qkv(z_a).view(z_a.shape[0], 3, H, dh)
        k_a = qkv_a[:, 1]
        v_a = qkv_a[:, 2]

        # For each edge, gather (target_idx, aggr_idx) → q_t[ti], k_a[ai], v_a[ai].
        ti = edge_index[0]
        ai = edge_index[1]
        q = q_t[ti]   # (E, H, dh)
        k = k_a[ai]   # (E, H, dh)
        v = v_a[ai]   # (E, H, dh)

        # Compute scaled-dot scores.
        score = (q * k).sum(dim=-1) / math.sqrt(dh)  # (E, H)

        # Per-target softmax over edges. Use scatter for normalization.
        score_max = torch.full((T, H), -1e9, device=z_t.device, dtype=score.dtype)
        score_max.scatter_reduce_(0, ti.unsqueeze(-1).expand(-1, H), score, reduce="amax", include_self=False)
        # Subtract max for stability.
        score_centered = score - score_max[ti]
        score_exp = score_centered.exp()
        denom = torch.zeros((T, H), device=z_t.device, dtype=score.dtype)
        denom.scatter_add_(0, ti.unsqueeze(-1).expand(-1, H), score_exp)
        denom = denom.clamp(min=1e-9)
        attn = score_exp / denom[ti]  # (E, H)

        # Weighted sum of values per target.
        weighted = attn.unsqueeze(-1) * v  # (E, H, dh)
        out_t = torch.zeros((T, H, dh), device=z_t.device, dtype=z_t.dtype)
        idx = ti.view(-1, 1, 1).expand(-1, H, dh)
        out_t.scatter_add_(0, idx, weighted)

        return self.cross_out(out_t.view(T, d))

    def forward(
        self,
        z_t: torch.Tensor,
        z_a: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Target self-attention (full; T typically ≤ 1K).
        zt_norm = self.self_norm(z_t.unsqueeze(0))
        sa_out, _ = self.self_attn(zt_norm, zt_norm, zt_norm, need_weights=False)
        z_t = z_t + sa_out.squeeze(0)

        # Edge-driven cross-attention (target ← aggressors).
        zt_norm = self.cross_norm(z_t)
        za_norm = self.cross_norm(z_a)
        ca_out = self.edge_cross_attention(zt_norm, za_norm, edge_index)
        z_t = z_t + ca_out

        # FFN on targets only.
        z_t = z_t + self.ffn(self.ffn_norm(z_t))
        # Aggressors only get layer-norm (no self-attention this block — they refresh
        # via target's evolved z_t in the next block? for Phase 1.0, leave aggressors static).
        return z_t, z_a


class NCGTModel(nn.Module):
    """End-to-end NCGT: encoder + L blocks + global readout + heads."""

    def __init__(self, cfg: Optional[NCGTConfig] = None):
        super().__init__()
        self.cfg = cfg or NCGTConfig()
        self.register_buffer("_clamp_bound", torch.tensor(float(self.cfg.clamp_bound)))
        self.encoder = HeterogeneousEncoder(self.cfg)
        self.blocks = nn.ModuleList([TransformerBlock(self.cfg) for _ in range(self.cfg.n_blocks)])
        self.global_token = nn.Parameter(torch.zeros(1, self.cfg.d_model))
        self.global_readout_attn = nn.MultiheadAttention(
            self.cfg.d_model, self.cfg.n_heads, dropout=self.cfg.dropout, batch_first=True
        )

        # GND head — produces residual logit per target segment.
        # Input: [z_t (d), geom (12), area (1), z_global (d)] → 1 logit (Phase 1.0)
        # Phase 2.3: 5 bin logits + 5 residual logits.
        gnd_in = self.cfg.d_model + 12 + 1 + self.cfg.d_model
        self.gnd_residual_head = nn.Sequential(
            nn.Linear(gnd_in, 128), nn.GELU(), nn.Linear(128, 1),
        )
        # Zero-init last layer → initial residual_logit = 0 → tanh(0) = 0 →
        # correction = (clamp_hi + clamp_lo) / 2 = 0.25 for [-0.5, +1.0].
        # We want initial C ≈ base, so set last bias to a value that makes
        # tanh(bias) * half_range + center = 0:
        #   bias = arctanh(-center / half_range) = arctanh(-1/3) ≈ -0.3466
        # Zero-init last layer: residual_logit = 0 → log_correction = 0 → C = base.
        nn.init.zeros_(self.gnd_residual_head[-1].weight)
        nn.init.zeros_(self.gnd_residual_head[-1].bias)
        if self.cfg.use_bins:
            self.gnd_bin_classifier = nn.Linear(gnd_in, self.cfg.n_bins)
            self.gnd_bin_residuals = nn.ModuleList(
                [nn.Sequential(nn.Linear(gnd_in, 64), nn.GELU(), nn.Linear(64, 1))
                 for _ in range(self.cfg.n_bins)]
            )
            # Zero-init each per-bin residual so initial pred = base (matches §2.5).
            for h in self.gnd_bin_residuals:
                nn.init.zeros_(h[-1].weight)
                nn.init.zeros_(h[-1].bias)

        # CPL head — per edge.
        # Input: [z_t (d), z_a (d), |Δr| (1), parallel_overlap (1), broadside_flag (1),
        #         layer_pair_emb (16), z_global (d), physics_base (1)] → 1 logit
        cpl_in = self.cfg.d_model + self.cfg.d_model + 3 + 16 + self.cfg.d_model + 1
        self.cpl_layer_pair_emb = nn.Embedding(self.cfg.n_layer_buckets * 2, 16)
        self.cpl_residual_head = nn.Sequential(
            nn.Linear(cpl_in, 128), nn.GELU(), nn.Linear(128, 1),
        )
        # Zero-init: initial CPL pred = base.
        nn.init.zeros_(self.cpl_residual_head[-1].weight)
        nn.init.zeros_(self.cpl_residual_head[-1].bias)
        if self.cfg.use_bins:
            self.cpl_bin_classifier = nn.Linear(cpl_in, self.cfg.n_bins)
            self.cpl_bin_residuals = nn.ModuleList(
                [nn.Sequential(nn.Linear(cpl_in, 64), nn.GELU(), nn.Linear(64, 1))
                 for _ in range(self.cfg.n_bins)]
            )
            for h in self.cpl_bin_residuals:
                nn.init.zeros_(h[-1].weight)
                nn.init.zeros_(h[-1].bias)

    def set_clamp_bound(self, value: float) -> None:
        """Update RES_CLAMP for pex_v3 curriculum (call per epoch / step interval)."""
        with torch.no_grad():
            self._clamp_bound.fill_(float(value))

    def get_clamp_bound(self) -> float:
        return float(self._clamp_bound.item())

    def compute_gnd_base(self, sample: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Differentiable parallel-plate + Sakurai-Tamaru GND base per target segment.

        Uses layer_table (passed via sample) when available; else placeholder.
        """
        ps = sample["target_p_start"]
        pe = sample["target_p_end"]
        feats = sample["target_feats"]
        width = feats[:, 5]

        layer_table = sample.get("_layer_table")
        if layer_table is not None:
            layer_idxs = feats[:, 7].long()
            phys = layer_table.build_seg_tensors(layer_idxs)
            thickness = phys["t_metal"].clamp(min=0.05)
            area, perim = compute_segment_geometry(ps, pe, width, thickness)
            return gnd_base_per_segment(
                seg_area_top=area, seg_area_bot=area,
                seg_perimeter=perim, seg_thickness=thickness,
                d_top=phys["d_above"], d_bot=phys["d_below"],
                eps_top=phys["eps_above"], eps_bot=phys["eps_below"],
            )
        # Placeholder.
        thickness = feats[:, 6].clamp(min=0.05)
        area, perim = compute_segment_geometry(ps, pe, width, thickness)
        eps = torch.full_like(area, self.cfg.eps_default)
        d = torch.full_like(area, self.cfg.d_default)
        return gnd_base_per_segment(
            seg_area_top=area, seg_area_bot=area,
            seg_perimeter=perim, seg_thickness=thickness,
            d_top=d, d_bot=d,
            eps_top=eps, eps_bot=eps,
        )

    def compute_cpl_base_and_meta(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Differentiable CPL base per edge; also returns meta tensors used by head."""
        ti = sample["edge_index"][0]
        ai = sample["edge_index"][1]
        E = ti.shape[0]
        if E == 0:
            zero = torch.zeros(0, device=sample["target_feats"].device)
            return {
                "cpl_base": zero,
                "rel_pose": torch.zeros(0, 3, device=zero.device),
                "layer_pair_idx": torch.zeros(0, dtype=torch.long, device=zero.device),
            }

        t_feats = sample["target_feats"][ti]
        a_feats = sample["aggr_feats"][ai]
        t_layer = t_feats[:, 7].long()
        a_layer = a_feats[:, 7].long()
        same_layer = (t_layer == a_layer)
        layer_table = sample.get("_layer_table")

        # Geometry: midpoint distances.
        t_mid = t_feats[:, :3]
        a_mid = a_feats[:, :3]
        diff = t_mid - a_mid
        d_xy = torch.linalg.norm(diff[:, :2], dim=-1)
        d_z = diff[:, 2].abs()
        d_total = torch.linalg.norm(diff, dim=-1)

        # Proper parallel-projection overlap (Risk 3 fix).
        t_ps = sample["target_p_start"][ti]
        t_pe = sample["target_p_end"][ti]
        a_ps = sample["aggr_p_start"][ai]
        a_pe = sample["aggr_p_end"][ai]
        overlap_len = edge_overlap_length(t_ps, t_pe, a_ps, a_pe)
        # Cross-layer broadside area: aggressor face × min(target_length, aggr_length)
        # gated by xy-projected overlap (re-using same parallel projection result).
        overlap_area = overlap_len * a_feats[:, 5]

        if layer_table is not None:
            pair = layer_table.build_pair_tensors(t_layer, a_layer)
            thick = pair["t_pair"].clamp(min=0.05)
            eps_pair = pair["eps_pair"]
        else:
            thick = t_feats[:, 6].clamp(min=0.05)
            eps_pair = torch.full_like(thick, self.cfg.eps_default)
        cpl_base = cpl_base_per_edge(
            same_layer=same_layer,
            overlap_length=overlap_len,
            overlap_area=overlap_area,
            lateral_distance=d_xy,
            vertical_distance=d_z,
            metal_thickness=thick,
            eps_pair=eps_pair,
        )

        # Layer pair index for embedding (pack t_layer, a_layer into single int).
        n_l = self.cfg.n_layer_buckets
        layer_pair_idx = (t_layer.clamp(0, n_l - 1) * 2 + same_layer.long()).clamp(0, n_l * 2 - 1)
        rel_pose = torch.stack([d_total, d_xy, d_z], dim=-1)  # (E, 3)

        return {
            "cpl_base": cpl_base,
            "rel_pose": rel_pose,
            "layer_pair_idx": layer_pair_idx,
            "same_layer": same_layer,
        }

    def global_readout(self, z_t: torch.Tensor) -> torch.Tensor:
        """Aggregate target representations into a single global token (readout-only).

        Returns: z_global (d_model,) — broadcasted to per-target/per-edge inputs.
        """
        cls = self.global_token.expand(1, 1, -1)  # (1, 1, d)
        z_t_b = z_t.unsqueeze(0)                   # (1, T, d)
        out, _ = self.global_readout_attn(cls, z_t_b, z_t_b, need_weights=False)
        return out.squeeze(0).squeeze(0)           # (d,)

    def forward(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """One sample (one net) → predictions.

        With use_bins=False: single residual head per task (Phase 1.0/2.0).
        With use_bins=True: 5-bin classifier + per-bin residual MLPs, soft-routed
        via softmax during training (gradient to all bins) and argmax at eval.
        """
        # Encoder.
        z_t = self.encoder(sample["target_feats"], sample["target_type_ids"])
        z_a = self.encoder(sample["aggr_feats"], sample["aggr_type_ids"])

        # Backbone.
        for blk in self.blocks:
            z_t, z_a = blk(z_t, z_a, sample["edge_index"])

        # Global readout (per-net summary).
        z_global = self.global_readout(z_t)  # (d,)

        # ----- GND head -----
        gnd_base = self.compute_gnd_base(sample)  # (T,)
        T = z_t.shape[0]
        z_global_t = z_global.unsqueeze(0).expand(T, -1)
        gnd_in = torch.cat([
            z_t,
            sample["target_feats"],
            gnd_base.unsqueeze(-1),
            z_global_t,
        ], dim=-1)

        gnd_bin_logits = None
        gnd_bin_target = None
        if self.cfg.use_bins:
            gnd_bin_logits = self.gnd_bin_classifier(gnd_in)  # (T, K)
            # 5 per-bin residual logits.
            gnd_bin_resids = torch.stack(
                [head(gnd_in).squeeze(-1) for head in self.gnd_bin_residuals], dim=-1
            )  # (T, K)
            # Soft routing: weighted sum of tanh(bin_resid) by softmax of bin classifier.
            gnd_bin_probs = torch.softmax(gnd_bin_logits, dim=-1)
            tanh_resid = torch.tanh(gnd_bin_resids)
            gnd_resid_effective = (gnd_bin_probs * tanh_resid).sum(dim=-1)  # (T,)
            log_corr = gnd_resid_effective * self.cfg.log_range
            pred_gnd_per_seg = gnd_base * torch.exp(log_corr)
            # Bin target derived from physics base magnitude (no per-seg gt available).
            gnd_bin_target = bin_assign(gnd_base, self.cfg.bin_edges_fF)
        else:
            gnd_resid = self.gnd_residual_head(gnd_in).squeeze(-1)
            pred_gnd_per_seg = compose_with_residual(
                gnd_base, gnd_resid,
                log_range=self.cfg.log_range,
                use_hard_clamp=self.cfg.use_curriculum,
                clamp_bound=float(self._clamp_bound.item()) if self.cfg.use_curriculum else None,
            )

        # ----- CPL head -----
        cpl_meta = self.compute_cpl_base_and_meta(sample)
        cpl_base = cpl_meta["cpl_base"]
        E = cpl_base.shape[0]
        cpl_bin_logits = None
        cpl_bin_target = None
        if E > 0:
            ti = sample["edge_index"][0]
            ai = sample["edge_index"][1]
            z_global_e = z_global.unsqueeze(0).expand(E, -1)
            layer_pair_emb = self.cpl_layer_pair_emb(cpl_meta["layer_pair_idx"])
            cpl_in = torch.cat([
                z_t[ti],
                z_a[ai],
                cpl_meta["rel_pose"],
                layer_pair_emb,
                z_global_e,
                cpl_base.unsqueeze(-1),
            ], dim=-1)
            if self.cfg.use_bins:
                cpl_bin_logits = self.cpl_bin_classifier(cpl_in)  # (E, K)
                cpl_bin_resids = torch.stack(
                    [head(cpl_in).squeeze(-1) for head in self.cpl_bin_residuals], dim=-1
                )
                cpl_bin_probs = torch.softmax(cpl_bin_logits, dim=-1)
                tanh_resid = torch.tanh(cpl_bin_resids)
                cpl_resid_effective = (cpl_bin_probs * tanh_resid).sum(dim=-1)
                log_corr = cpl_resid_effective * self.cfg.log_range
                pred_cpl_per_edge = cpl_base * torch.exp(log_corr)
                cpl_bin_target = bin_assign(cpl_base, self.cfg.bin_edges_fF)
            else:
                cpl_resid = self.cpl_residual_head(cpl_in).squeeze(-1)
                pred_cpl_per_edge = compose_with_residual(
                    cpl_base, cpl_resid,
                    log_range=self.cfg.log_range,
                    use_hard_clamp=self.cfg.use_curriculum,
                    clamp_bound=float(self._clamp_bound.item()) if self.cfg.use_curriculum else None,
                )
        else:
            pred_cpl_per_edge = torch.zeros(0, device=z_t.device)

        # Aggregate.
        pred_gnd_total = pred_gnd_per_seg.sum()
        pred_cpl_total = pred_cpl_per_edge.sum()
        pred_total = pred_gnd_total + pred_cpl_total

        # Phase B — per-aggressor-net CPL aggregation (denser supervision).
        n_aggr_nets = int(sample.get("n_aggr_nets", torch.tensor(0)).item()) if "n_aggr_nets" in sample else 0
        if E > 0 and n_aggr_nets > 0:
            ai = sample["edge_index"][1]
            edge_aggr_net_id = sample["aggr_net_ids"][ai]  # (E,)
            pred_cpl_per_aggr_net = torch.zeros(n_aggr_nets, device=z_t.device)
            pred_cpl_per_aggr_net.scatter_add_(0, edge_aggr_net_id, pred_cpl_per_edge)
        else:
            pred_cpl_per_aggr_net = torch.zeros(max(1, n_aggr_nets), device=z_t.device)

        out = {
            "pred_gnd_per_seg": pred_gnd_per_seg,
            "pred_gnd_total": pred_gnd_total,
            "pred_cpl_per_edge": pred_cpl_per_edge,
            "pred_cpl_per_aggr_net": pred_cpl_per_aggr_net,
            "pred_cpl_total": pred_cpl_total,
            "pred_total": pred_total,
            "gnd_base": gnd_base,
            "cpl_base": cpl_base,
            "z_global": z_global,
        }
        if self.cfg.use_bins:
            out["gnd_bin_logits"] = gnd_bin_logits
            out["gnd_bin_target"] = gnd_bin_target
            out["cpl_bin_logits"] = cpl_bin_logits
            out["cpl_bin_target"] = cpl_bin_target
        return out


# ---------------------------------------------------------------------------
# Smoke test.
# ---------------------------------------------------------------------------
def _smoke_test() -> None:
    """1-net forward pass with synthetic data."""
    cfg = NCGTConfig()
    model = NCGTModel(cfg)

    T = 12
    A = 200
    E = 800
    sample = {
        "target_feats": torch.randn(T, 12),
        "aggr_feats": torch.randn(A, 12),
        "target_type_ids": torch.zeros(T, dtype=torch.long),
        "aggr_type_ids": torch.randint(1, N_TYPES, (A,)),
        "target_p_start": torch.randn(T, 3),
        "target_p_end": torch.randn(T, 3),
        "aggr_p_start": torch.randn(A, 3),
        "aggr_p_end": torch.randn(A, 3),
        "edge_index": torch.stack([
            torch.randint(0, T, (E,)),
            torch.randint(0, A, (E,)),
        ]),
        "edge_band": torch.zeros(E, dtype=torch.long),
        "aggr_net_ids": torch.randint(0, 5, (A,)),
        "gnd_total": torch.tensor(0.5),
        "cpl_total": torch.tensor(0.3),
        "edge_gt": torch.rand(E) * 0.01,
        "edge_supervised": torch.ones(E, dtype=torch.bool),
    }
    # Make features positive-ish (layer index column needs valid bucket).
    sample["target_feats"][:, 5] = 0.05
    sample["target_feats"][:, 6] = 0.144
    sample["target_feats"][:, 7] = torch.randint(1, 30, (T,)).float()
    sample["aggr_feats"][:, 5] = 0.05
    sample["aggr_feats"][:, 6] = 0.144
    sample["aggr_feats"][:, 7] = torch.randint(1, 30, (A,)).float()

    out = model(sample)
    print(f"pred_gnd_per_seg: {out['pred_gnd_per_seg'].shape} = {out['pred_gnd_per_seg'].sum().item():.4f}")
    print(f"pred_cpl_per_edge: {out['pred_cpl_per_edge'].shape} = {out['pred_cpl_per_edge'].sum().item():.4f}")
    print(f"pred_total: {out['pred_total'].item():.4f}")
    assert out["pred_total"].requires_grad
    out["pred_total"].backward()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.2f}M, gradient OK")
    print("[ncgt_model smoke] OK")


if __name__ == "__main__":
    _smoke_test()
