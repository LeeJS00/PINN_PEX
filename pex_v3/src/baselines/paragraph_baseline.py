"""
paragraph_baseline.py — Phase 0.5 B2.

ParaGraph (DAC 2020) reproduction on the v3 manifest with v3 split.
Reference: arxiv 2007.00514

Architecture skeleton:
    - Heterogeneous graph: nets ↔ devices, edges = electrical connections
    - Node features: device-type embedding, geometric features
    - 3-5 GNN layers (GraphSAGE / GIN / R-GCN — pick the one closest to paper)
    - Per-net cap regression head

Body deferred. This stub locks the public API.
"""
from __future__ import annotations
from pathlib import Path


def build_heterogeneous_graph(
    design_name: str,
    manifest_subset,  # pd.DataFrame restricted to one design
):
    """Build a hetero graph for one design from DEF + v3 manifest."""
    raise NotImplementedError("Phase 0.5 scaffold")


def train_paragraph(
    train_designs: list[str],
    output_dir: Path,
    seed: int,
    n_layers: int = 4,
):
    """Train ParaGraph-style relation-GNN. Save best ckpt + provenance."""
    raise NotImplementedError("Phase 0.5 scaffold")


def evaluate_paragraph(
    model_path: Path,
    eval_designs: list[str],
    output_csv: Path,
) -> dict:
    raise NotImplementedError("Phase 0.5 scaffold")
