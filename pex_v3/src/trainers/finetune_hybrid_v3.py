"""
finetune_hybrid_v3.py — Phase 1 Tier 2 (NEW). Direct real-BEOL fine-tune.

After K3 canary fired (synthetic pretrain useless for zero-init residual),
Phase 1 simplifies to direct fine-tune on real v3 features.

β-strategy gate (A2 [ROLE PASS]):
    On v3 valid (5-seed, last-step checkpoint, manifest-hashed):
      - gnd MAPE < 8%   (B1 baseline: 20.6%)
      - cpl MAPE < 8%   (B1 baseline: 12.4%)
      - total MAPE < 4% (B1 baseline: 4.66%)

Curriculum (per A5 [ROLE PASS]):
    Epoch 0-50:    RES_CLAMP = log(1.5) — bounded ≤ ±50%
    Epoch 50-150:  RES_CLAMP = log(2.5)
    Epoch 150+:    RES_CLAMP = log(4.0)

Training time estimate: ~5-10 GPU-h per seed (vs B3's 4.5h; hybrid is 30K
params, no AL loop). Multi-GPU 5-seed parallel ≈ same wall-clock as one.
"""
from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from ..models.hybrid_v3 import (
    HybridPexV3,
    DEFAULT_SELF_FEATURE_DIM,
    DEFAULT_PAIR_FEATURE_DIM,
    per_channel_mape_loss,
)
from ..models.residual_head_v3 import res_clamp_for_epoch


# ============================================================================
# Real v3 feature → tensors
# ============================================================================


_SELF_FEATURE_COLS = [
    "compact_gnd_estimate_fF",          # analytic prior (informative)
    "total_wire_length_um",
    "total_metal_area_um2",
    "n_cuboids",
    "bbox_xy_um2",
    "bbox_z_um",
    "n_layers_present",
    "eps_mean",
    "vss_shield_M1_M3",
    "vss_shield_M4_M5",
    "vss_shield_M6_plus",
    "density_M1_M3",
    "density_M4_M5",
    "density_M6_plus",
    "fanout",
    "aspect_ratio",
]

_PAIR_FEATURE_COLS = [
    "compact_cpl_estimate_total_fF",     # analytic cpl prior
    "n_aggressor_nets",
    "broadside_overlap_total_um2",
    "broadside_overlap_p95_um2",
    "lateral_overlap_total_um2",
    "lateral_overlap_p95_um2",
    "spacing_min_um",
    "spacing_p25_um",
    "spacing_p50_um",
    "spacing_p95_um",
    "n_edges_lt_1um",
    "n_edges_1_to_3um",
    "n_edges_3_to_4um",
    "compact_gnd_estimate_fF",
    "vss_shield_M1_M3",
    "vss_shield_M4_M5",
    "vss_shield_M6_plus",
    "n_layers_present",
    "eps_mean",
    "total_metal_area_um2",
    "n_aggressor_nets",                  # repeat ok; padding
    "fanout",
    "n_cuboids",
    "density_M1_M3",
]


def _safe_log1p_columns(df: pd.DataFrame, cols: list[str], dim: int) -> torch.Tensor:
    """Project DF cols to (n, dim) tensor with log1p compression on positive vals."""
    out = torch.zeros((len(df), dim), dtype=torch.float32)
    for i, col in enumerate(cols[:dim]):
        if col in df.columns:
            v = df[col].fillna(0.0).to_numpy(dtype=np.float32)
            v = np.log1p(np.clip(v, 0, None))
            out[:, i] = torch.from_numpy(v)
    return out


def df_to_tensors(df: pd.DataFrame) -> dict:
    """Convert one v3 features DF slice to model-ready tensors.

    Returns:
        {
            'analytic_gnd':   (B,) tensor  — compact_gnd_estimate_fF
            'analytic_cpl':   (B,) tensor  — compact_cpl_estimate_total_fF
            'self_features':  (B, 16)
            'pair_features':  (B, 24)
            'golden_gnd':     (B,)
            'golden_cpl':     (B,)
            'design_name':    np.array(B,)
            'net_name':       np.array(B,)
        }
    """
    return {
        "analytic_gnd": torch.tensor(
            df["compact_gnd_estimate_fF"].fillna(0.0).to_numpy(dtype=np.float32),
            dtype=torch.float32,
        ),
        "analytic_cpl": torch.tensor(
            df["compact_cpl_estimate_total_fF"].fillna(0.0).to_numpy(dtype=np.float32),
            dtype=torch.float32,
        ),
        "self_features": _safe_log1p_columns(df, _SELF_FEATURE_COLS, DEFAULT_SELF_FEATURE_DIM),
        "pair_features": _safe_log1p_columns(df, _PAIR_FEATURE_COLS, DEFAULT_PAIR_FEATURE_DIM),
        "golden_gnd": torch.tensor(
            df["c_gnd_fF"].fillna(0.0).to_numpy(dtype=np.float32),
            dtype=torch.float32,
        ),
        "golden_cpl": torch.tensor(
            df["c_cpl_total_fF"].fillna(0.0).to_numpy(dtype=np.float32),
            dtype=torch.float32,
        ),
        "design_name": df["design_name"].to_numpy(),
        "net_name": df["net_name"].to_numpy(),
    }


def split_by_manifest_column(df: pd.DataFrame) -> tuple:
    """Partition by 'split' column from H1 hash."""
    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_valid = df[df["split"] == "valid"].reset_index(drop=True)
    df_test = df[df["split"] == "test"].reset_index(drop=True)
    return df_train, df_valid, df_test


# ============================================================================
# Training loop
# ============================================================================


@dataclass
class FinetuneConfig:
    n_epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    seed: int = 42
    eps_fF: float = 1e-3
    w_gnd: float = 1.0
    w_cpl: float = 1.0
    log_every_n_steps: int = 100
    eval_every_n_epochs: int = 1
    curriculum_enabled: bool = True
    early_stop_patience: int = 5      # epochs without valid improvement


@dataclass
class FinetuneHistory:
    step: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    valid_total_mape: list[float] = field(default_factory=list)
    valid_gnd_mape: list[float] = field(default_factory=list)
    valid_cpl_mape: list[float] = field(default_factory=list)
    epoch_complete: list[int] = field(default_factory=list)
    best_valid_total_mape: float = float("inf")
    best_valid_gnd_mape: float = float("inf")
    best_valid_cpl_mape: float = float("inf")
    best_epoch: int = -1


def evaluate_per_channel(
    model: HybridPexV3,
    tensors: dict,
    device: str,
    eps_fF: float = 1e-3,
) -> dict:
    """Compute per-net gnd/cpl/total MAPE on a single batch (fits in memory)."""
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        analytic_gnd = tensors["analytic_gnd"].to(device)
        analytic_cpl = tensors["analytic_cpl"].to(device)
        self_feats = tensors["self_features"].to(device)
        pair_feats = tensors["pair_features"].to(device)
        gold_gnd = tensors["golden_gnd"].to(device)
        gold_cpl = tensors["golden_cpl"].to(device)

        pred_gnd = model.predict_gnd(analytic_gnd, self_feats)
        pred_cpl = model.predict_cpl(analytic_cpl, pair_feats)

        gold_safe_gnd = gold_gnd.clamp(min=eps_fF)
        gold_safe_cpl = gold_cpl.clamp(min=eps_fF)
        gnd_rel = (pred_gnd - gold_gnd).abs() / gold_safe_gnd
        cpl_rel = (pred_cpl - gold_cpl).abs() / gold_safe_cpl
        pred_total = pred_gnd + pred_cpl
        gold_total = gold_gnd + gold_cpl
        total_rel = (pred_total - gold_total).abs() / gold_total.clamp(min=eps_fF)

        return {
            "gnd_mape_median": float(gnd_rel.median().item()),
            "gnd_mape_mean": float(gnd_rel.mean().item()),
            "cpl_mape_median": float(cpl_rel.median().item()),
            "cpl_mape_mean": float(cpl_rel.mean().item()),
            "total_mape_median": float(total_rel.median().item()),
            "total_mape_mean": float(total_rel.mean().item()),
            "n_nets": int(len(gold_gnd)),
        }


def finetune_hybrid(
    model: HybridPexV3,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    config: FinetuneConfig,
    device: str = "cpu",
) -> FinetuneHistory:
    """Train HybridPexV3 with per-channel MAPE loss on real v3 features.

    Reports per-channel valid metrics every epoch. Returns history with
    best-valid-epoch tracking for last-step-vs-best comparison.
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    train_tensors = df_to_tensors(train_df)
    valid_tensors = df_to_tensors(valid_df)

    model = model.to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    history = FinetuneHistory()
    n_train = len(train_df)
    rng = np.random.default_rng(config.seed)
    global_step = 0
    t0 = time.time()
    epochs_without_improvement = 0

    for epoch in range(config.n_epochs):
        if config.curriculum_enabled:
            clamp = res_clamp_for_epoch(epoch)
            model.set_clamp_bounds(clamp)

        model.train()
        # Single-batch shuffle per epoch (since data fits in memory)
        idx = rng.permutation(n_train)
        for start in range(0, n_train, config.batch_size):
            batch_idx = idx[start:start + config.batch_size]
            if len(batch_idx) == 0:
                continue

            analytic_gnd = train_tensors["analytic_gnd"][batch_idx].to(device)
            analytic_cpl = train_tensors["analytic_cpl"][batch_idx].to(device)
            self_feats = train_tensors["self_features"][batch_idx].to(device)
            pair_feats = train_tensors["pair_features"][batch_idx].to(device)
            gold_gnd = train_tensors["golden_gnd"][batch_idx].to(device)
            gold_cpl = train_tensors["golden_cpl"][batch_idx].to(device)

            pred_gnd = model.predict_gnd(analytic_gnd, self_feats)
            pred_cpl = model.predict_cpl(analytic_cpl, pair_feats)

            losses = per_channel_mape_loss(
                pred_gnd, gold_gnd, pred_cpl, gold_cpl,
                eps_fF=config.eps_fF,
                w_gnd=config.w_gnd, w_cpl=config.w_cpl,
            )
            optimizer.zero_grad()
            losses["total_loss"].backward()
            optimizer.step()

            if global_step % config.log_every_n_steps == 0:
                history.step.append(global_step)
                history.train_loss.append(float(losses["total_loss"].item()))
            global_step += 1

        # End of epoch — eval
        if (epoch + 1) % config.eval_every_n_epochs == 0:
            v = evaluate_per_channel(model, valid_tensors, device)
            history.valid_total_mape.append(v["total_mape_median"])
            history.valid_gnd_mape.append(v["gnd_mape_median"])
            history.valid_cpl_mape.append(v["cpl_mape_median"])
            history.epoch_complete.append(epoch)
            elapsed = time.time() - t0
            print(
                f"  epoch {epoch}/{config.n_epochs}: "
                f"clamp={model.gnd_residual.get_clamp_bound():.3f}  "
                f"train_loss={losses['total_loss'].item():.4f}  "
                f"valid mape: gnd={v['gnd_mape_median']*100:.2f}%  "
                f"cpl={v['cpl_mape_median']*100:.2f}%  "
                f"total={v['total_mape_median']*100:.2f}%  "
                f"({elapsed:.1f}s)",
                flush=True,
            )
            # Best tracking (per A1: use last-step, but track best for diagnostic)
            if v["total_mape_median"] < history.best_valid_total_mape:
                history.best_valid_total_mape = v["total_mape_median"]
                history.best_valid_gnd_mape = v["gnd_mape_median"]
                history.best_valid_cpl_mape = v["cpl_mape_median"]
                history.best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.early_stop_patience:
                    print(f"  early stop at epoch {epoch}")
                    break

    return history


def evaluate_beta_gate(
    model: HybridPexV3,
    valid_df: pd.DataFrame,
    config: FinetuneConfig,
    device: str = "cpu",
    gnd_threshold: float = 0.08,
    cpl_threshold: float = 0.08,
    total_threshold: float = 0.04,
) -> dict:
    """β-strategy gate: gnd<8% AND cpl<8% AND total<4%."""
    valid_tensors = df_to_tensors(valid_df)
    metrics = evaluate_per_channel(model, valid_tensors, device, eps_fF=config.eps_fF)
    gate_gnd = metrics["gnd_mape_median"] < gnd_threshold
    gate_cpl = metrics["cpl_mape_median"] < cpl_threshold
    gate_total = metrics["total_mape_median"] < total_threshold
    passed = gate_gnd and gate_cpl and gate_total
    return {
        **metrics,
        "gate_gnd": gate_gnd,
        "gate_cpl": gate_cpl,
        "gate_total": gate_total,
        "beta_passed": passed,
        "verdict": "β-PASS" if passed else "β-FAIL",
    }
