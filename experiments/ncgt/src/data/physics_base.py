"""
NCGT physics-guided base capacitance models (ResCap-style, Plan v3 §2.5).

All formulas are differentiable (torch ops only), used as the base prediction
that the NN residual head modulates: C_predicted = C_base * (1 + softplus_residual).

References:
- Sakurai-Tamaru 1983: empirical formula for parallel-coupled wire capacitance.
- Wong-Salama-Shieh: refinement on lateral fringe.
- Standard parallel-plate: A/d for broadside coupling.

Physics-base coverage targets (Phase 1 sanity gate per PLAN.md §5):
- If pure physics base (residual=0) yields net MAPE > 50%, formula is wrong.
- Expected: 30-50% MAPE pure base, 4-15% with residual (matches ResCap).
"""
from __future__ import annotations

import math
from typing import Tuple

import torch

EPS_0 = 8.854187817e-3  # fF / μm (so output naturally in fF when distances are μm)
TWO_OVER_PI = 0.6366197723675814


def parallel_plate(
    area: torch.Tensor,
    distance: torch.Tensor,
    eps_eff: torch.Tensor,
    eps_floor: float = 1e-3,
) -> torch.Tensor:
    """C = ε₀ · ε_eff · A / d, clamped distance to avoid singularity."""
    d = distance.clamp(min=eps_floor)
    return EPS_0 * eps_eff * area / d


def sakurai_tamaru_lateral(
    perimeter: torch.Tensor,
    thickness: torch.Tensor,
    distance: torch.Tensor,
    eps_eff: torch.Tensor,
    eps_floor: float = 1e-3,
) -> torch.Tensor:
    """
    Sakurai-Tamaru lateral fringe term:
        C_fringe = ε₀ · ε_eff · P · log1p(t / d) · (2/π)
    where P is wire perimeter overlap (or full perimeter for GND), t is metal
    thickness, d is gap distance.
    """
    d = distance.clamp(min=eps_floor)
    return EPS_0 * eps_eff * perimeter * torch.log1p(thickness / d) * TWO_OVER_PI


def gnd_base_per_segment(
    *,
    seg_area_top: torch.Tensor,    # (N,) projected area facing layer above (μm²)
    seg_area_bot: torch.Tensor,    # (N,) projected area facing layer below (μm²)
    seg_perimeter: torch.Tensor,   # (N,) lateral fringe perimeter (μm)
    seg_thickness: torch.Tensor,   # (N,) metal thickness (μm)
    d_top: torch.Tensor,           # (N,) distance to layer above (μm)
    d_bot: torch.Tensor,           # (N,) distance to layer below (μm)
    eps_top: torch.Tensor,         # (N,) effective permittivity above
    eps_bot: torch.Tensor,         # (N,) effective permittivity below
) -> torch.Tensor:
    """Per-segment GND capacitance base. All inputs broadcast-aligned (N,)."""
    cap_top = parallel_plate(seg_area_top, d_top, eps_top)
    cap_bot = parallel_plate(seg_area_bot, d_bot, eps_bot)
    eps_avg = 0.5 * (eps_top + eps_bot)
    cap_fringe = sakurai_tamaru_lateral(seg_perimeter, seg_thickness, 0.5 * (d_top + d_bot), eps_avg)
    return cap_top + cap_bot + cap_fringe


def cpl_base_per_edge(
    *,
    same_layer: torch.Tensor,    # (E,) bool mask
    overlap_length: torch.Tensor,  # (E,) parallel-overlap length for same-layer pair (μm)
    overlap_area: torch.Tensor,    # (E,) projected overlap area for cross-layer pair (μm²)
    lateral_distance: torch.Tensor,    # (E,) edge-to-edge gap for same-layer (μm)
    vertical_distance: torch.Tensor,   # (E,) z-gap for cross-layer (μm)
    metal_thickness: torch.Tensor,     # (E,) average of two segments' metal thickness (μm)
    eps_pair: torch.Tensor,            # (E,) layer-pair-dependent permittivity
    eps_floor: float = 1e-3,
) -> torch.Tensor:
    """Per-edge CPL capacitance base.

    Same-layer pair: Sakurai-Tamaru lateral coupling.
    Cross-layer pair: parallel-plate broadside coupling.
    """
    # Same-layer: ε₀ · ε_pair · L_overlap · log1p(t / d_lat) · (2/π)
    d_lat = lateral_distance.clamp(min=eps_floor)
    same_layer_cap = EPS_0 * eps_pair * overlap_length * torch.log1p(metal_thickness / d_lat) * TWO_OVER_PI

    # Cross-layer: ε₀ · ε_pair · A_overlap / d_vert
    d_vert = vertical_distance.clamp(min=eps_floor)
    cross_layer_cap = EPS_0 * eps_pair * overlap_area / d_vert

    return torch.where(same_layer, same_layer_cap, cross_layer_cap)


def compose_with_residual(
    base: torch.Tensor,
    residual_logit: torch.Tensor,
    log_range: float = 2.3,
    clamp_bound: float = None,
    use_hard_clamp: bool = False,
    clamp_lo: float = None,  # deprecated
    clamp_hi: float = None,
) -> torch.Tensor:
    """
    Log-space residual composition: C = base * exp(bounded_logit).

    Two modes:
      use_hard_clamp=False (default, tanh soft):
          bounded_logit = tanh(logit) * log_range
          → smooth, gradient nonzero everywhere
          → range exp(±log_range), e.g. log_range=2.3 → ×0.1..×10

      use_hard_clamp=True (pex_v3 curriculum mode):
          bounded_logit = clamp(logit, -clamp_bound, +clamp_bound)
          → hard clamp, gradient zero outside band
          → curriculum schedule progressively widens clamp_bound:
                Phase 0: log(1.5) ≈ 0.405  → mul ∈ [0.67, 1.50]
                Phase 1: log(2.5) ≈ 0.916  → mul ∈ [0.40, 2.50]
                Phase 2: log(4.0) ≈ 1.386  → mul ∈ [0.25, 4.00]
          Why hard clamp + curriculum works (pex_v3 6.26%):
            - Tight initial bound forces model to use physics base
            - Residual heads (zero-init) start at 0 → mul=1 → C=base
            - Gradual expansion lets model learn small corrections first
            - Avoids saturation trap of always-loose tanh

    With zero-init last layer of residual head, initial residual_logit = 0 →
    bounded_logit = 0 → C = base (physics-only prediction).
    """
    if use_hard_clamp:
        cb = clamp_bound if clamp_bound is not None else math.log(1.5)
        bounded = torch.clamp(residual_logit, -cb, +cb)
    else:
        bounded = torch.tanh(residual_logit) * log_range
    return base * torch.exp(bounded)


def compute_segment_geometry(
    p_start: torch.Tensor,    # (N, 3)
    p_end: torch.Tensor,      # (N, 3)
    width: torch.Tensor,      # (N,)
    thickness: torch.Tensor,  # (N,)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (face_area_per_side, perimeter) for a wire segment.

    face_area_per_side: top OR bottom projected area = length × width  (μm²)
    perimeter: 2 × length (lateral edges contribute fringe both sides)  (μm)
    """
    delta = p_end - p_start
    length = torch.linalg.norm(delta, dim=-1)
    area = length * width
    perimeter = 2.0 * length
    return area, perimeter


def edge_overlap_length(
    p_start_a: torch.Tensor, p_end_a: torch.Tensor,
    p_start_b: torch.Tensor, p_end_b: torch.Tensor,
    parallel_cos_threshold: float = 0.9,
) -> torch.Tensor:
    """Parallel-projection overlap length for two wire segments.

    Used for Sakurai-Tamaru lateral coupling between two same-layer parallel wires.
    Returns:
        - For parallel pairs (|cos(θ)| ≥ threshold): the length of the parallel
          projection of B onto A's axis intersected with [0, |A|].
        - For non-parallel pairs: 0 (Sakurai-Tamaru assumes parallel coupling).
    """
    dir_a = p_end_a - p_start_a
    dir_b = p_end_b - p_start_b
    len_a = torch.linalg.norm(dir_a, dim=-1).clamp(min=1e-6)
    len_b = torch.linalg.norm(dir_b, dim=-1).clamp(min=1e-6)
    u_a = dir_a / len_a.unsqueeze(-1)
    u_b = dir_b / len_b.unsqueeze(-1)

    # Parallelism gate: |cos(angle between A and B)| ≥ threshold.
    cos_angle = (u_a * u_b).sum(dim=-1).abs()
    parallel_mask = cos_angle >= parallel_cos_threshold

    # Project B endpoints onto A's axis (origin = p_start_a).
    t_as = torch.zeros_like(len_a)
    t_ae = len_a
    t_bs = ((p_start_b - p_start_a) * u_a).sum(dim=-1)
    t_be = ((p_end_b - p_start_a) * u_a).sum(dim=-1)
    t_b_lo = torch.minimum(t_bs, t_be)
    t_b_hi = torch.maximum(t_bs, t_be)

    overlap_parallel = (torch.minimum(t_ae, t_b_hi) - torch.maximum(t_as, t_b_lo)).clamp(min=0.0)
    return torch.where(parallel_mask, overlap_parallel, torch.zeros_like(overlap_parallel))


# ---------------------------------------------------------------------------
# Self-test (smoke verification of formula sanity).
# ---------------------------------------------------------------------------
def _smoke_test() -> None:
    """Verifies physics base produces order-of-magnitude correct cap values.

    Reference ranges for intel22-class BEOL:
        - Single M4 wire 10μm long, 0.044μm wide, 4.4μm² area, d=0.2μm to layer
          above with ε≈3.0:
              C_top = 8.85e-3 · 3.0 · 4.4 / 0.2 ≈ 0.58 fF
        - Same wire fringe with t=0.144μm, d=0.2μm, ε=3.0, P=20μm:
              C_fringe = 8.85e-3 · 3.0 · 20 · log1p(0.72) · 0.6366 ≈ 0.18 fF
        - Total: ~0.8 fF for a ~10μm M4 wire — matches typical SPEF order.
    """
    p_start = torch.tensor([[0.0, 0.0, 1.0]])
    p_end = torch.tensor([[10.0, 0.0, 1.0]])
    width = torch.tensor([0.044])
    thickness = torch.tensor([0.144])
    area, perim = compute_segment_geometry(p_start, p_end, width, thickness)
    print(f"area={area.item():.4f} μm², perimeter={perim.item():.4f} μm")

    cap = gnd_base_per_segment(
        seg_area_top=area,
        seg_area_bot=area,
        seg_perimeter=perim,
        seg_thickness=thickness,
        d_top=torch.tensor([0.2]),
        d_bot=torch.tensor([0.2]),
        eps_top=torch.tensor([3.0]),
        eps_bot=torch.tensor([3.0]),
    )
    print(f"GND base for M4 10μm wire: {cap.item():.4f} fF")
    # M4 22nm thin wire 10μm → expect ~0.1-1.0 fF (top + bottom + fringe).
    assert 0.1 < cap.item() < 2.0, f"GND base out of expected range: {cap.item()}"

    # CPL same-layer: two parallel M4 wires, 10μm long, 0.1μm gap, ε=3.0
    cpl = cpl_base_per_edge(
        same_layer=torch.tensor([True]),
        overlap_length=torch.tensor([10.0]),
        overlap_area=torch.tensor([0.0]),
        lateral_distance=torch.tensor([0.1]),
        vertical_distance=torch.tensor([0.0]),
        metal_thickness=thickness,
        eps_pair=torch.tensor([3.0]),
    )
    print(f"CPL base same-layer 10μm parallel: {cpl.item():.4f} fF")
    assert 0.05 < cpl.item() < 1.0, f"Out of expected range: {cpl.item()}"

    print("[physics_base smoke] OK — formulas produce reasonable order of magnitude.")


if __name__ == "__main__":
    _smoke_test()
