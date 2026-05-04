"""
transfer_canary_v3.py — K3 hard kill criterion (Codex round 2).

Decides whether the synthetic pretrain (Stage 1 + Stage 2 Mode A) provides
a useful Bayesian prior for real-BEOL fine-tuning.

Protocol (per `pex_v3/docs/PHASE1_HYBRID_ARCH_SPEC.md` §6 + Codex round 2 P1):

    Two models, identical architecture:
        A. CONTROL — fresh random init
        B. PRETRAINED — initialized from synthetic-pretrained checkpoint

    Both fine-tuned for `n_finetune_steps` (default 1000) on a small slice
    of real v3 valid data (default 500 nets), same seeds, same data, same
    optimizer.

    Verdict:
        If pretrained_loss(after 1000 steps) ≤ control_loss(after 1000 steps)
        × (1 - speedup_threshold), the canary PASSES.

        speedup_threshold default = 0.50 (per Codex round 2 mandate).

    Hard kill K3: if canary FAILS, abort the synthetic strategy and fall
    back to direct real-data fine-tuning from random init.

This is the gate that decides whether to commit Q3D oracle GPU-months
(Stage 3+) or scope down to Stage 1+2 Mode A only.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from src.models.hybrid_v3 import (
    HybridPexV3,
    DEFAULT_SELF_FEATURE_DIM,
    DEFAULT_PAIR_FEATURE_DIM,
)


# ============================================================================
# Real-BEOL slice loader
# ============================================================================


def load_real_v3_canary_slice(
    features_csv: Path,
    n_nets: int,
    seed: int = 0,
    split: str = "valid",
) -> pd.DataFrame:
    """Load a fixed slice of v3 features for canary fine-tune.

    Same `n_nets` are deterministically selected per `seed` so control vs
    pretrained see IDENTICAL data — only the model init differs.
    """
    df = pd.read_csv(features_csv)
    df = df[df["split"] == split].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    if n_nets >= len(df):
        return df.copy()
    idx = rng.choice(len(df), size=n_nets, replace=False)
    return df.iloc[sorted(idx)].reset_index(drop=True)


# ============================================================================
# Per-net "geometry-only" analytic baseline for the canary
# ============================================================================


def compute_per_net_analytic_baseline(df: pd.DataFrame) -> torch.Tensor:
    """A coarse analytic prior for each real net.

    For the canary, we use a simple geometry-only proxy as the analytic
    baseline that the residual will multiply. The hand-engineered features
    `compact_gnd_estimate_fF` (Sakurai-Tamaru-class summed estimate from
    feature_dataset) is the right anchor — it captures the analytic prior
    on the same per-net basis as the residual-multiplier mechanic expects.

    Returns: (B,) torch tensor of analytic baseline values.
    """
    if "compact_gnd_estimate_fF" not in df.columns:
        raise KeyError(
            "feature_dataset must include 'compact_gnd_estimate_fF' column. "
            "Run pex_v3/scripts/04_build_feature_dataset.py with the "
            "post-A3-fix feature_dataset.py."
        )
    return torch.tensor(
        df["compact_gnd_estimate_fF"].fillna(0.0).to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )


def real_features_to_self_features(
    df: pd.DataFrame,
    self_feature_dim: int = DEFAULT_SELF_FEATURE_DIM,
) -> torch.Tensor:
    """Map real-BEOL hand-engineered features → 16-dim residual head input.

    Using a deterministic projection of the most-informative columns.
    Rest are zeros (the residual must learn from limited info).
    """
    out = torch.zeros((len(df), self_feature_dim), dtype=torch.float32)
    cols_to_use = [
        "total_wire_length_um",
        "total_metal_area_um2",
        "n_cuboids",
        "bbox_xy_um2",
        "n_aggressor_nets",
        "broadside_overlap_total_um2",
        "lateral_overlap_total_um2",
        "spacing_min_um",
        "n_layers_present",
        "eps_mean",
        "vss_shield_M1_M3",
        "density_M1_M3",
        "compact_gnd_estimate_fF",
        "compact_cpl_estimate_total_fF",
        "fanout",
    ]
    for i, col in enumerate(cols_to_use[:self_feature_dim]):
        if col in df.columns:
            v = df[col].fillna(0.0).to_numpy(dtype=np.float32)
            # log1p compress positive features
            v = np.log1p(np.clip(v, 0, None))
            out[:, i] = torch.from_numpy(v)
    return out


# ============================================================================
# Canary loop
# ============================================================================


@dataclass
class CanaryConfig:
    n_nets: int = 500
    n_finetune_steps: int = 1000
    batch_size: int = 64
    lr: float = 1e-3
    seed: int = 42
    speedup_threshold: float = 0.50
    log_every_n_steps: int = 50


@dataclass
class CanaryHistory:
    method: str = "control"
    step: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    final_loss: float = float("nan")


def _run_finetune(
    model: HybridPexV3,
    analytic: torch.Tensor,
    features: torch.Tensor,
    golden: torch.Tensor,
    config: CanaryConfig,
    method_label: str,
    device: str,
) -> CanaryHistory:
    """Run `n_finetune_steps` of fine-tuning on the canary slice."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    model = model.to(device)
    analytic = analytic.to(device)
    features = features.to(device)
    golden = golden.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    history = CanaryHistory(method=method_label)
    n = analytic.shape[0]
    rng = np.random.default_rng(config.seed)

    for step in range(config.n_finetune_steps):
        idx = rng.choice(n, size=config.batch_size, replace=True)
        idx_t = torch.tensor(idx, device=device, dtype=torch.long)
        a_b = analytic.index_select(0, idx_t)
        f_b = features.index_select(0, idx_t)
        g_b = golden.index_select(0, idx_t)

        pred = model.predict_gnd(a_b, f_b)
        rel_err = (pred - g_b).abs() / g_b.clamp(min=1e-3)
        loss = rel_err.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % config.log_every_n_steps == 0 or step == config.n_finetune_steps - 1:
            history.step.append(step)
            history.loss.append(float(loss.item()))

    history.final_loss = history.loss[-1]
    return history


def run_transfer_canary(
    pretrained_state_dict: dict,
    features_csv: Path,
    config: CanaryConfig,
    device: str = "cpu",
) -> dict:
    """Run the K3 canary protocol.

    Returns:
        {
            "verdict":              "PASS" | "FAIL",
            "control_final_loss":   float,
            "pretrained_final_loss": float,
            "speedup":               float (negative = pretrained worse),
            "k3_fired":              bool (True = abort synthetic strategy),
            "rationale":             str,
            "control_history":       list[(step, loss)],
            "pretrained_history":    list[(step, loss)],
        }
    """
    df = load_real_v3_canary_slice(features_csv, n_nets=config.n_nets, seed=config.seed)
    if len(df) == 0:
        raise RuntimeError("canary slice is empty — check features_csv")

    analytic = compute_per_net_analytic_baseline(df)
    features = real_features_to_self_features(df)
    golden = torch.tensor(
        df["c_gnd_fF"].fillna(0.0).to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )

    # CONTROL — fresh init
    torch.manual_seed(config.seed + 1)
    control = HybridPexV3()
    h_ctrl = _run_finetune(
        control, analytic, features, golden, config,
        method_label="control", device=device,
    )

    # PRETRAINED — load synthetic-pretrained ckpt
    torch.manual_seed(config.seed + 1)
    pretrained = HybridPexV3()
    pretrained.load_state_dict(pretrained_state_dict)
    h_pre = _run_finetune(
        pretrained, analytic, features, golden, config,
        method_label="pretrained", device=device,
    )

    speedup = (h_ctrl.final_loss - h_pre.final_loss) / max(h_ctrl.final_loss, 1e-12)
    pass_threshold = config.speedup_threshold
    passed = speedup >= pass_threshold
    return {
        "verdict": "PASS" if passed else "FAIL",
        "control_final_loss": h_ctrl.final_loss,
        "pretrained_final_loss": h_pre.final_loss,
        "speedup": float(speedup),
        "speedup_threshold": pass_threshold,
        "k3_fired": not passed,
        "rationale": (
            f"pretrained {h_pre.final_loss:.4f} vs control {h_ctrl.final_loss:.4f} "
            f"({speedup*100:.1f}% speedup vs threshold {pass_threshold*100:.0f}%)"
        ),
        "control_history": list(zip(h_ctrl.step, h_ctrl.loss)),
        "pretrained_history": list(zip(h_pre.step, h_pre.loss)),
        "n_nets": config.n_nets,
        "n_finetune_steps": config.n_finetune_steps,
    }
