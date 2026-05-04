"""
gam_baseline.py — Phase 0.5 B4.

Analytic compact-model (Sakurai-Tamaru class) + GAM / GBDT residual.
Closest paradigm match to ResCap (ASPDAC 2025). Doubles as:
    (a) Phase 0.5 baseline — physics-floor anchor
    (b) Phase 1 candidate initialization — start the hybrid model from
        a known-good GAM residual

Body deferred. Stub locks API.
"""
from __future__ import annotations
from pathlib import Path


def compute_compact_predictions(
    manifest_subset,
):
    """Run pure analytic compact model on each net. Return predicted cap."""
    raise NotImplementedError("Phase 0.5 scaffold")


def train_gam_residual(
    train_features,           # iterable of NetFeatureVector
    compact_predictions,      # output of compute_compact_predictions
    train_targets,            # golden cap
    output_dir: Path,
    seed: int,
):
    """Fit GAM (or GBDT) on residual = golden - compact. Save model + provenance."""
    raise NotImplementedError("Phase 0.5 scaffold")


def evaluate_gam_compact(
    model_dir: Path,
    eval_features,
    eval_compact_predictions,
    eval_targets,
    output_csv: Path,
) -> dict:
    raise NotImplementedError("Phase 0.5 scaffold")
