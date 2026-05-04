"""
pretrain_synthetic_v3.py — Phase 1 synthetic pretraining harness.

Trains `hybrid_v3.HybridPexV3` on Stage 1 (parallel plate) + Stage 2 Mode A
(stacked dielectric). NOT Mode B per A4 audit.

Sanity invariant (verified by tests):
    On synthetic data, analytic == golden ⇒ MAPE loss directly equals
    `mean(|multiplier - 1|)`. Pretrain converges multiplier → 1.0 across
    the synthetic distribution. This establishes the day-1 prior:
    "trust analytic baseline; deviate only when real-BEOL data forces it."

Why pretrain at all if multiplier converges to a constant?
    1. Establishes a Bayesian prior. Real-BEOL fine-tune starts with
       residual head warmed-up at "1.0 across the geometry distribution",
       not random.
    2. Verifies the autograd path through analytic + residual is wired
       correctly (kill-signal for any later dataloader/feature-encoding
       bug — if pretrain doesn't converge, something is broken before
       real data is ever loaded).
    3. K3 canary: measures whether this prior accelerates real-BEOL
       fine-tuning vs from-scratch. If not, the synthetic strategy is
       cheap to abort.

Per A5 mandate (Tier 1 next deliverable).
"""
from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from src.models.hybrid_v3 import (
    HybridPexV3,
    DEFAULT_SELF_FEATURE_DIM,
    DEFAULT_PAIR_FEATURE_DIM,
    per_channel_mape_loss,
)
from src.synthetic.ground_truth import (
    parallel_plate_capacitance_fF,
    stacked_dielectric_capacitance_fF,
)


# ============================================================================
# Synthetic dataset — Stage 1 + Stage 2 Mode A
# ============================================================================


@dataclass
class SyntheticSample:
    """Single synthetic sample. Geometry + analytic = golden capacitance."""
    geometry: dict           # raw geometric inputs (w, h, d, eps_r, ... )
    analytic_C_gnd_fF: float # closed-form analytic prediction (Mode A or B)
    analytic_C_cpl_fF: float # for Stage 1+2 we set this to 0.0 (no aggressors)
    self_features: np.ndarray   # (DEFAULT_SELF_FEATURE_DIM,) for residual head
    pair_features: np.ndarray   # (DEFAULT_PAIR_FEATURE_DIM,) — placeholder zeros


class SyntheticPretrainDataset(Dataset):
    """Generates Stage 1 (parallel plate) + Stage 2 Mode A samples on demand.

    Each sample: features map to analytic capacitance via closed-form. Since
    the residual head will be asked to output multiplier ≈ 1.0 (because
    analytic == truth here), this is the regularization-prior pretrain.

    Args:
        n_samples:        total samples to generate
        seed:             RNG seed (deterministic generation)
        stage_2_fraction: fraction of samples drawn from Stage 2 Mode A
                          (0.0 = pure Stage 1; A5 spec recommends mixing)
        self_feature_dim, pair_feature_dim: vector lengths matching hybrid_v3
    """

    def __init__(
        self,
        n_samples: int = 10_000,
        seed: int = 42,
        stage_2_fraction: float = 0.5,
        self_feature_dim: int = DEFAULT_SELF_FEATURE_DIM,
        pair_feature_dim: int = DEFAULT_PAIR_FEATURE_DIM,
    ):
        self.n_samples = int(n_samples)
        self.rng = np.random.default_rng(seed)
        self.stage_2_fraction = float(stage_2_fraction)
        self.self_feature_dim = self_feature_dim
        self.pair_feature_dim = pair_feature_dim
        # Pre-decide which samples are Stage 1 vs Stage 2 for reproducibility
        self._is_stage_2 = self.rng.uniform(0, 1, n_samples) < stage_2_fraction
        # Precompute everything for fast __getitem__ (memory cost: ~10K × 16 × 4 = 640KB)
        self._cache: list[SyntheticSample] = []
        for i in range(n_samples):
            if self._is_stage_2[i]:
                self._cache.append(self._make_stage_2_sample(i))
            else:
                self._cache.append(self._make_stage_1_sample(i))

    def _make_stage_1_sample(self, idx: int) -> SyntheticSample:
        """Parallel plate, single dielectric."""
        rng = np.random.default_rng(idx ^ 0xCAFE)
        d = float(10.0 ** rng.uniform(-2, 0))
        w = float(10.0 ** rng.uniform(-1, 1))
        h = float(10.0 ** rng.uniform(-1, 1))
        eps_r = float(rng.uniform(1.0, 10.0))
        c = parallel_plate_capacitance_fF(w, h, d, eps_r)
        # Encode geometry as features (informative subset + decoy noise)
        feats = np.zeros(self.self_feature_dim, dtype=np.float32)
        feats[0] = math.log(max(c, 1e-12))      # log(analytic_C) — informative
        feats[1] = math.log(w)
        feats[2] = math.log(h)
        feats[3] = math.log(d)
        feats[4] = eps_r
        feats[5] = w * h                        # area
        feats[6] = math.log(w * h / d)          # geometric factor
        feats[7] = 1.0                          # stage-1 indicator
        # Decoy features (residual must learn to ignore)
        feats[8:] = rng.standard_normal(self.self_feature_dim - 8).astype(np.float32) * 0.5

        return SyntheticSample(
            geometry={"stage": 1, "w_um": w, "h_um": h, "d_um": d, "eps_r": eps_r},
            analytic_C_gnd_fF=c,
            analytic_C_cpl_fF=0.0,
            self_features=feats,
            pair_features=np.zeros(self.pair_feature_dim, dtype=np.float32),
        )

    def _make_stage_2_sample(self, idx: int) -> SyntheticSample:
        """Stacked dielectric (Stage 2 Mode A). 1-5 layers between plates."""
        rng = np.random.default_rng(idx ^ 0xBEEF)
        n_layers = int(rng.integers(1, 6))
        thicknesses = (10.0 ** rng.uniform(-2, -0.5, n_layers)).tolist()
        eps_layers = rng.uniform(1.5, 8.0, n_layers).tolist()
        w = float(rng.uniform(0.5, 10.0))
        h = float(rng.uniform(0.5, 10.0))
        c = stacked_dielectric_capacitance_fF(w, h, thicknesses, eps_layers)
        total_d = sum(thicknesses)
        eff_eps = total_d / sum(t / e for t, e in zip(thicknesses, eps_layers))
        feats = np.zeros(self.self_feature_dim, dtype=np.float32)
        feats[0] = math.log(max(c, 1e-12))
        feats[1] = math.log(w)
        feats[2] = math.log(h)
        feats[3] = math.log(total_d)
        feats[4] = eff_eps                      # effective ε from series formula
        feats[5] = w * h
        feats[6] = math.log(w * h / total_d)
        feats[7] = 0.0                          # Stage 2 indicator
        feats[8] = float(n_layers) / 5.0
        feats[9] = max(eps_layers) / 8.0        # ε spread
        feats[10] = min(eps_layers) / 8.0
        feats[11:] = rng.standard_normal(self.self_feature_dim - 11).astype(np.float32) * 0.5

        return SyntheticSample(
            geometry={"stage": 2, "w_um": w, "h_um": h, "thicknesses": thicknesses,
                      "eps_layers": eps_layers, "n_layers": n_layers,
                      "total_d_um": total_d, "eff_eps": eff_eps},
            analytic_C_gnd_fF=c,
            analytic_C_cpl_fF=0.0,
            self_features=feats,
            pair_features=np.zeros(self.pair_feature_dim, dtype=np.float32),
        )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        s = self._cache[idx]
        return {
            "analytic_gnd": torch.tensor(s.analytic_C_gnd_fF, dtype=torch.float32),
            "self_features": torch.from_numpy(s.self_features),
            "golden_gnd": torch.tensor(s.analytic_C_gnd_fF, dtype=torch.float32),
        }


def _collate_pretrain(batch: list[dict]) -> dict:
    return {
        "analytic_gnd": torch.stack([b["analytic_gnd"] for b in batch]),
        "self_features": torch.stack([b["self_features"] for b in batch]),
        "golden_gnd": torch.stack([b["golden_gnd"] for b in batch]),
    }


# ============================================================================
# Pretrain loop
# ============================================================================


@dataclass
class PretrainConfig:
    n_samples: int = 10_000
    n_epochs: int = 5
    batch_size: int = 256
    lr: float = 1e-3
    seed: int = 42
    stage_2_fraction: float = 0.5
    eps_fF: float = 1e-3
    log_every_n_steps: int = 50
    convergence_threshold: float = 0.05  # multiplier within 5% of 1.0


@dataclass
class PretrainHistory:
    step: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    multiplier_mean: list[float] = field(default_factory=list)
    multiplier_max_dev: list[float] = field(default_factory=list)
    epoch_complete: list[int] = field(default_factory=list)


def pretrain_hybrid(
    model: HybridPexV3,
    config: PretrainConfig,
    device: str = "cpu",
) -> PretrainHistory:
    """Pretrain `hybrid_v3` on synthetic Stage 1 + Mode A.

    Sanity gate: residual `multiplier` should converge toward 1.0 in mean
    AND max-deviation. If max-deviation exceeds `convergence_threshold`
    after epoch 1, the residual is overfitting noise.

    Returns history of (step, loss, multiplier_mean, multiplier_max_dev,
    epoch_complete) for diagnostic plotting.
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    dataset = SyntheticPretrainDataset(
        n_samples=config.n_samples,
        seed=config.seed,
        stage_2_fraction=config.stage_2_fraction,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate_pretrain,
    )

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    history = PretrainHistory()
    global_step = 0
    t0 = time.time()

    for epoch in range(config.n_epochs):
        model.train()
        for batch in loader:
            analytic = batch["analytic_gnd"].to(device)
            features = batch["self_features"].to(device)
            golden = batch["golden_gnd"].to(device)

            pred = model.predict_gnd(analytic, features)

            # MAPE loss on gnd channel only (cpl is zero in pretrain)
            golden_safe = golden.clamp(min=config.eps_fF)
            rel_err = (pred - golden).abs() / golden_safe
            loss = rel_err.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track multiplier behavior
            with torch.no_grad():
                multiplier = pred / analytic.clamp(min=1e-12)
                mul_mean = float(multiplier.mean().item())
                mul_max_dev = float((multiplier - 1.0).abs().max().item())

            if global_step % config.log_every_n_steps == 0:
                history.step.append(global_step)
                history.loss.append(float(loss.item()))
                history.multiplier_mean.append(mul_mean)
                history.multiplier_max_dev.append(mul_max_dev)
            global_step += 1

        history.epoch_complete.append(epoch)
        # Quick epoch summary print
        elapsed = time.time() - t0
        print(
            f"  epoch {epoch}/{config.n_epochs}: "
            f"loss={loss.item():.5f}  "
            f"mul_mean={mul_mean:.4f}  "
            f"mul_max_dev={mul_max_dev:.4f}  "
            f"({elapsed:.1f}s elapsed)",
            flush=True,
        )

    return history


# ============================================================================
# Sanity gate
# ============================================================================


def check_pretrain_converged(history: PretrainHistory, threshold: float = 0.05) -> dict:
    """Verify the final multiplier_mean ≈ 1.0 AND max_dev < threshold.

    A5 sanity invariant: on synthetic where analytic = truth, the
    residual head MUST converge to multiplier=1.0. Failure = the model
    is overfitting noise, fix data/model before fine-tune.
    """
    if not history.multiplier_mean:
        return {"converged": False, "reason": "no history recorded"}
    final_mean = history.multiplier_mean[-1]
    final_max_dev = history.multiplier_max_dev[-1]
    final_loss = history.loss[-1]
    converged = (
        abs(final_mean - 1.0) < threshold
        and final_max_dev < threshold * 5  # max-dev allowed to be 5× larger
    )
    return {
        "converged": bool(converged),
        "final_loss": final_loss,
        "final_mul_mean": final_mean,
        "final_mul_max_dev": final_max_dev,
        "threshold": threshold,
        "reason": (
            "OK" if converged
            else f"mul_mean={final_mean:.4f}, max_dev={final_max_dev:.4f} not within bounds"
        ),
    }
