"""
stage2_layered_image.py — Stage 2 of synthetic pretraining curriculum.

A conductor near a layered dielectric stack. Two analytic regimes captured
here without numerical Sommerfeld quadrature:

  Mode A — Stacked dielectric series (target ⇄ ground sandwich)
      Two parallel plates with N layers of distinct ε between them.
      Series capacitance:  C = ε₀ · A / Σ_i (d_i / ε_i)

  Mode B — Single-interface image charge correction (half-space below)
      A conductor over a planar dielectric interface; lower half-space
      has ε_below ≠ ε_between. Capacitance picks up a leading-order
      correction proportional to k = (ε_between − ε_below)/(ε_between + ε_below).
      For ε_below = ε_between this reduces exactly to parallel plate.

Mode C — Full Sommerfeld quadrature (general layered) is **deferred**
to Phase 1 optimization. Direct quadrature is O(10⁻³ s/eval), prohibitive
at 10⁷ scale. Vector Fitting / complex-image rational approximation is the
target there. We don't need this for the curriculum to be useful — Modes A
and B together cover the dominant BEOL phenomenology (multi-ILD stack +
asymmetric ε above vs below).

Generation cost: instantaneous for Mode A; O(1) for Mode B (closed-form
correction).
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, Sequence, Tuple

import numpy as np

from src.synthetic.ground_truth import (
    stacked_dielectric_capacitance_fF,
    interface_corrected_capacitance_fF,
    half_space_image_correction_factor,
)


# ----------------------------------------------------------------------
# Mode A — stacked dielectric (parallel plates, N layers)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class StackedDielectricSample:
    """Two parallel plates with a stack of N dielectric layers between."""
    w_um: float
    h_um: float
    n_layers: int
    layer_thicknesses_um: Tuple[float, ...]
    layer_eps_r: Tuple[float, ...]
    total_thickness_um: float       # = sum(layer_thicknesses_um)
    c_fF: float


def generate_stacked_dielectric_stream(
    n_samples: int,
    seed: int,
    n_layers_range: Tuple[int, int] = (1, 5),
    layer_thickness_range_um: Tuple[float, float] = (0.01, 0.5),
    eps_range: Tuple[float, float] = (1.5, 8.0),
    plate_xy_range_um: Tuple[float, float] = (0.5, 10.0),
) -> Iterator[StackedDielectricSample]:
    """Generate `n_samples` stacked-dielectric samples with closed-form C."""
    if n_samples <= 0:
        return
    rng = np.random.default_rng(seed)

    n_low, n_high = n_layers_range
    log_t_low, log_t_high = (
        np.log10(layer_thickness_range_um[0]),
        np.log10(layer_thickness_range_um[1]),
    )

    for _ in range(n_samples):
        n_layers = int(rng.integers(n_low, n_high + 1))
        thicknesses = (10.0 ** rng.uniform(log_t_low, log_t_high, n_layers)).tolist()
        eps_layers = rng.uniform(eps_range[0], eps_range[1], n_layers).tolist()
        w = float(rng.uniform(plate_xy_range_um[0], plate_xy_range_um[1]))
        h = float(rng.uniform(plate_xy_range_um[0], plate_xy_range_um[1]))

        c = stacked_dielectric_capacitance_fF(w, h, thicknesses, eps_layers)
        yield StackedDielectricSample(
            w_um=w,
            h_um=h,
            n_layers=n_layers,
            layer_thicknesses_um=tuple(thicknesses),
            layer_eps_r=tuple(eps_layers),
            total_thickness_um=float(sum(thicknesses)),
            c_fF=c,
        )


# ----------------------------------------------------------------------
# Mode B — single-interface image charge correction
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class SingleInterfaceSample:
    """A finite plate over a planar dielectric interface.

    Geometry:
        - Top plate at z = +d/2, bottom plate at z = -d/2 (between-region: eps_between)
        - Below bottom plate: half-space of eps_below (≠ eps_between in general)

    The capacitance with the half-space ground is approximated by
    the leading-order image-charge correction (see ground_truth.py).
    """
    w_um: float
    h_um: float
    d_um: float
    eps_between: float
    eps_below: float
    image_k: float                 # = (eps_between - eps_below) / (eps_between + eps_below)
    c_fF: float


def generate_single_interface_stream(
    n_samples: int,
    seed: int,
    d_range_um: Tuple[float, float] = (0.01, 1.0),
    plate_xy_range_um: Tuple[float, float] = (0.5, 10.0),
    eps_between_range: Tuple[float, float] = (2.0, 6.0),
    eps_below_range: Tuple[float, float] = (1.0, 8.0),
) -> Iterator[SingleInterfaceSample]:
    """Generate single-interface samples with image-charge correction."""
    if n_samples <= 0:
        return
    rng = np.random.default_rng(seed)

    log_d_low, log_d_high = np.log10(d_range_um[0]), np.log10(d_range_um[1])

    for _ in range(n_samples):
        d = float(10.0 ** rng.uniform(log_d_low, log_d_high))
        w = float(rng.uniform(plate_xy_range_um[0], plate_xy_range_um[1]))
        h = float(rng.uniform(plate_xy_range_um[0], plate_xy_range_um[1]))
        eps_between = float(rng.uniform(eps_between_range[0], eps_between_range[1]))
        eps_below = float(rng.uniform(eps_below_range[0], eps_below_range[1]))

        c = interface_corrected_capacitance_fF(
            w_um=w, h_um=h, d_um=d,
            eps_r_between=eps_between, eps_r_below=eps_below,
        )
        k = half_space_image_correction_factor(eps_between, eps_below)
        yield SingleInterfaceSample(
            w_um=w, h_um=h, d_um=d,
            eps_between=eps_between, eps_below=eps_below,
            image_k=k, c_fF=c,
        )


# ----------------------------------------------------------------------
# materialization
# ----------------------------------------------------------------------


def materialize_stage2_dataset(
    out_dir: Path,
    n_stacked: int,
    n_interface: int,
    seed: int,
) -> dict:
    """Generate Mode A and Mode B datasets, write to disk.

    Returns dict with paths and sample counts for provenance logging.
    """
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)

    if n_stacked > 0:
        rows_a = []
        for s in generate_stacked_dielectric_stream(n_samples=n_stacked, seed=seed):
            d = asdict(s)
            d["layer_thicknesses_um"] = list(d["layer_thicknesses_um"])
            d["layer_eps_r"] = list(d["layer_eps_r"])
            rows_a.append(d)
        df_a = pd.DataFrame(rows_a)
        # variable-length list cols → JSON encode for safe CSV
        df_a["layer_thicknesses_um"] = df_a["layer_thicknesses_um"].apply(str)
        df_a["layer_eps_r"] = df_a["layer_eps_r"].apply(str)
        path_a = out_dir / "stage2_stacked_dielectric.csv"
        df_a.to_csv(path_a, index=False)
    else:
        path_a = None

    if n_interface > 0:
        rows_b = [asdict(s) for s in generate_single_interface_stream(
            n_samples=n_interface, seed=seed + 1
        )]
        df_b = pd.DataFrame(rows_b)
        path_b = out_dir / "stage2_single_interface.csv"
        df_b.to_csv(path_b, index=False)
    else:
        path_b = None

    return {
        "stacked_dielectric_path": str(path_a) if path_a else None,
        "single_interface_path": str(path_b) if path_b else None,
        "n_stacked_samples": n_stacked,
        "n_interface_samples": n_interface,
        "seed": seed,
    }


# ----------------------------------------------------------------------
# verification (Phase 1 sanity gates)
# ----------------------------------------------------------------------


def verify_uniform_halfspace_limit(model_callable, tolerance_rel: float = 1e-3) -> Tuple[bool, dict]:
    """When ε_above = ε_below, Mode B reduces exactly to parallel plate.

    Verifies that the model produces this collapse (which is the most
    important physics-correctness invariant for layered Green's function).
    """
    rng = np.random.default_rng(31)
    rels = []
    n = 50
    for _ in range(n):
        w = float(rng.uniform(0.5, 10.0))
        h = float(rng.uniform(0.5, 10.0))
        d = float(10.0 ** rng.uniform(-2, 0))
        eps = float(rng.uniform(1.5, 8.0))
        # Both half-spaces same ε
        from src.synthetic.ground_truth import parallel_plate_capacitance_fF
        gold = parallel_plate_capacitance_fF(w, h, d, eps)
        pred = model_callable(w, h, d, eps, eps)  # eps_between=eps_below
        rels.append(abs(pred - gold) / gold)
    rels_arr = np.asarray(rels)
    passed = bool(rels_arr.max() < tolerance_rel)
    return passed, {
        "n_test": n,
        "max_rel_err": float(rels_arr.max()),
        "mean_rel_err": float(rels_arr.mean()),
        "tolerance_rel": tolerance_rel,
    }
