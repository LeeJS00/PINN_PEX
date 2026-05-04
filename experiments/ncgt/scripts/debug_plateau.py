#!/usr/bin/env python3
"""
Debug NCGT training plateau (Phase 2.0 issue).

Real training plateaued exactly at step 300 with byte-identical TRAIN/VALID
metrics afterwards. Hypothesis ranked:
  1. Residual head saturating at clamp [-0.5, +1.0] → grad zero through clamp
  2. Gradient explosion → clipped → effective lr collapses
  3. Cosine LR decay too aggressive (lr 5e-4 → 5e-5 over 3000 steps)
  4. Adam v_hat saturation (large oscillating gradients shrink effective LR)
  5. Loss surface local minimum

Diagnostic instruments:
  - Per-step gradient norm (before clip)
  - Per-step residual logit distribution (mean, std, saturation %)
  - Per-step parameter L2 norm delta
  - Per-step actual loss components

Run: python3 -m experiments.ncgt.scripts.debug_plateau
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ncgt.src.data.layer_physics import LayerPhysicsTable  # noqa: E402
from experiments.ncgt.src.models.ncgt_model import NCGTConfig, NCGTModel  # noqa: E402
from experiments.ncgt.src.trainers.train_ncgt import (  # noqa: E402
    build_samples_from_design,
    compute_losses,
    hand_tuned_combined,
    _setup_pdk,
    _resolve_design_paths,
)


def param_l2(model) -> float:
    return float(sum(p.detach().pow(2).sum().item() for p in model.parameters()) ** 0.5)


def grad_l2(model) -> float:
    g2 = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g2 += p.grad.detach().pow(2).sum().item()
    return float(g2 ** 0.5)


def residual_diagnostics(model, sample) -> dict:
    """Forward pass; intercept residual logits before clamp; report distribution."""
    z_t = model.encoder(sample["target_feats"], sample["target_type_ids"])
    z_a = model.encoder(sample["aggr_feats"], sample["aggr_type_ids"])
    for blk in model.blocks:
        z_t, z_a = blk(z_t, z_a, sample["edge_index"])
    z_global = model.global_readout(z_t)

    # GND head input
    gnd_base = model.compute_gnd_base(sample)
    T = z_t.shape[0]
    z_global_t = z_global.unsqueeze(0).expand(T, -1)
    gnd_in = torch.cat([z_t, sample["target_feats"], gnd_base.unsqueeze(-1), z_global_t], dim=-1)
    gnd_resid = model.gnd_residual_head(gnd_in).squeeze(-1)

    # CPL head input
    cpl_meta = model.compute_cpl_base_and_meta(sample)
    cpl_base = cpl_meta["cpl_base"]
    if cpl_base.shape[0] > 0:
        ti = sample["edge_index"][0]
        ai = sample["edge_index"][1]
        E = cpl_base.shape[0]
        z_global_e = z_global.unsqueeze(0).expand(E, -1)
        layer_pair_emb = model.cpl_layer_pair_emb(cpl_meta["layer_pair_idx"])
        broadside = (~cpl_meta["same_layer"]).float().unsqueeze(-1)  # noqa: F841
        cpl_in = torch.cat([
            z_t[ti], z_a[ai],
            cpl_meta["rel_pose"], layer_pair_emb, z_global_e,
            cpl_base.unsqueeze(-1),
        ], dim=-1)
        cpl_resid = model.cpl_residual_head(cpl_in).squeeze(-1)
    else:
        cpl_resid = torch.zeros(0, device=z_t.device)

    def stat(x: torch.Tensor, lo=-0.5, hi=1.0) -> dict:
        if x.numel() == 0:
            return {"n": 0}
        with torch.no_grad():
            return {
                "n": int(x.numel()),
                "mean": float(x.mean().item()),
                "std": float(x.std().item()),
                "min": float(x.min().item()),
                "max": float(x.max().item()),
                "frac_at_lo": float((x <= lo).float().mean().item()),
                "frac_at_hi": float((x >= hi).float().mean().item()),
                "frac_in_range": float(((x > lo) & (x < hi)).float().mean().item()),
            }

    return {
        "gnd_resid": stat(gnd_resid),
        "cpl_resid": stat(cpl_resid),
    }


def main() -> int:
    cfg, layer_info, tech_lef, cell_lib, layer_table = _setup_pdk()
    def_path, spef_path = _resolve_design_paths(cfg, "intel22_gcd_f3")

    samples = build_samples_from_design(
        def_path=def_path,
        spef_path=spef_path,
        layer_info=layer_info,
        tech_lef=tech_lef,
        cell_lib=cell_lib,
        n_nets=50,
        r_aggr=12.0,
        max_aggr_per_net=2000,
        max_target_per_net=1000,
        min_target_segs=5,
        rng_seed=0,
    )
    print(f"[debug] {len(samples)} samples built")
    if len(samples) < 20:
        return 1

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    samples_dev = []
    for s in samples:
        sd = {k: v.to(device) if torch.is_tensor(v) else v for k, v in s.items()}
        sd["_layer_table"] = layer_table
        samples_dev.append(sd)

    cfg_m = NCGTConfig(use_bins=False, dropout=0.1)
    model = NCGTModel(cfg_m).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)

    rng = np.random.default_rng(0)
    last_param_l2 = param_l2(model)

    print("\nstep | grad_norm | param_l2_delta | gnd_resid mean/std/frac_lo/frac_hi | cpl_resid mean/std/frac_lo/frac_hi | total_loss | gnd_mape cpl_mape")
    print("-" * 175)

    for step in range(500):
        s = samples_dev[rng.integers(0, len(samples_dev))]
        opt.zero_grad()
        pred = model(s)
        losses = compute_losses(pred, s)
        total = hand_tuned_combined(losses, edge_weight=0.05)
        total.backward()

        gn_pre = grad_l2(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        new_l2 = param_l2(model)
        delta = new_l2 - last_param_l2
        last_param_l2 = new_l2

        if step in (0, 1, 5, 10, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 499):
            with torch.no_grad():
                rd = residual_diagnostics(model, s)
            gn = rd["gnd_resid"]
            cn = rd["cpl_resid"]
            print(
                f"{step:4d} | {gn_pre:9.4f} | {delta:+.6f} | "
                f"{gn.get('mean', 0):+6.3f}/{gn.get('std', 0):5.3f}/"
                f"{gn.get('frac_at_lo', 0):4.0%}/{gn.get('frac_at_hi', 0):4.0%} | "
                f"{cn.get('mean', 0):+6.3f}/{cn.get('std', 0):5.3f}/"
                f"{cn.get('frac_at_lo', 0):4.0%}/{cn.get('frac_at_hi', 0):4.0%} | "
                f"{float(total.item()):.4f} | "
                f"{float(losses['loss_gnd'].item()):.4f} {float(losses['loss_cpl_total'].item()):.4f}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
