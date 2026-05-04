"""
stage1_parallel_plate.py — Stage 1 of synthetic pretraining curriculum.

Two parallel plates of width w, height h, separation d, with relative
permittivity ε_r between them. The capacitance has a closed-form analytic
expression:

    C_pp = ε₀ · ε_r · w · h / d           (parallel-plate, no fringe)

Generation cost: instantaneous (closed-form, no oracle, no integration).

Sample range (default):
    d ∈ [0.01 μm, 1 μm]    (log-uniform)
    w ∈ [0.1 μm, 10 μm]    (log-uniform)
    h ∈ [0.1 μm, 10 μm]    (log-uniform)
    ε_r ∈ [1.0, 10.0]      (uniform)

Default n_samples = 1,000,000.

Sanity invariant: when called as `(w, h, d, ε_r)`, the analytic formula
above must hold exactly. This is asserted in the unit tests.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np

from src.synthetic.ground_truth import parallel_plate_capacitance_fF


@dataclass(frozen=True)
class ParallelPlateSample:
    """One synthetic parallel-plate sample.

    All length values are in μm; capacitance in fF.
    """
    w_um: float       # plate width
    h_um: float       # plate height
    d_um: float       # plate separation
    eps_r: float      # dielectric relative permittivity (between plates)
    c_fF: float       # analytic capacitance, fF


def generate_parallel_plate_stream(
    n_samples: int,
    seed: int,
    d_range: Tuple[float, float] = (0.01, 1.0),    # μm
    w_range: Tuple[float, float] = (0.1, 10.0),    # μm
    h_range: Tuple[float, float] = (0.1, 10.0),    # μm
    eps_range: Tuple[float, float] = (1.0, 10.0),
) -> Iterator[ParallelPlateSample]:
    """Generate `n_samples` parallel-plate samples with analytic capacitance.

    Sampling:
      - d, w, h: log-uniform across the requested range (covers BEOL pitches
        equally across decades)
      - eps_r: uniform across the range (BEOL ε is roughly uniform 2-8)

    Args:
        n_samples: number of samples to yield
        seed: RNG seed for reproducibility
        d_range / w_range / h_range / eps_range: (low, high) inclusive

    Yields:
        ParallelPlateSample
    """
    if n_samples <= 0:
        return
    rng = np.random.default_rng(seed)

    log_d_low, log_d_high = np.log10(d_range[0]), np.log10(d_range[1])
    log_w_low, log_w_high = np.log10(w_range[0]), np.log10(w_range[1])
    log_h_low, log_h_high = np.log10(h_range[0]), np.log10(h_range[1])

    for _ in range(n_samples):
        d = float(10.0 ** rng.uniform(log_d_low, log_d_high))
        w = float(10.0 ** rng.uniform(log_w_low, log_w_high))
        h = float(10.0 ** rng.uniform(log_h_low, log_h_high))
        eps_r = float(rng.uniform(eps_range[0], eps_range[1]))
        c = parallel_plate_capacitance_fF(w, h, d, eps_r)
        yield ParallelPlateSample(
            w_um=w, h_um=h, d_um=d, eps_r=eps_r, c_fF=c,
        )


def materialize_parallel_plate_dataset(
    n_samples: int,
    seed: int,
    out_path: Path,
    **range_kwargs,
) -> Path:
    """Generate the dataset and write to disk as a Parquet (or CSV fallback).

    The output is a single tabular file with columns:
        w_um, h_um, d_um, eps_r, c_fF

    Args:
        n_samples: row count
        seed: RNG seed
        out_path: destination file (.parquet or .csv)
        **range_kwargs: forwarded to generate_parallel_plate_stream

    Returns:
        out_path
    """
    import pandas as pd

    rows = [asdict(s) for s in generate_parallel_plate_stream(
        n_samples=n_samples, seed=seed, **range_kwargs
    )]
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".parquet":
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
    return out_path


def verify_parallel_plate_limit(
    model_fn,
    tolerance_rel: float = 1e-3,
    n_test: int = 50,
    seed: int = 17,
) -> Tuple[bool, dict]:
    """Wrapper around `ground_truth.verify_module_against_parallel_plate`.

    Run before Phase 1 architecture is trusted on any other data. Used as
    a CI gate for the analytic baseline implementation.
    """
    from src.synthetic.ground_truth import verify_module_against_parallel_plate
    return verify_module_against_parallel_plate(
        module_fn=model_fn,
        tolerance_rel=tolerance_rel,
        n_test=n_test,
        seed=seed,
    )
