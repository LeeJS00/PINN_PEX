"""Analytic resistance computation per net from cuboids + layer stack.

For each cuboid in a net's routing geometry:
  R_seg = ρ_layer × length / (width × thickness)

where:
  ρ_layer  : sheet resistance from layers.info (ohm/sq) or layer_info dict
             — actually we use full bulk resistivity ρ via sheet × thickness,
             but with our cuboid model where w, h, d are width/length/thickness
             at the per-cuboid level, the formula is:
             R = ρ_bulk × L / A   with A = W × T
             where T (thickness) is the cuboid 'd' dimension.
  length   : longest dimension of the cuboid (max(w, h)) — minor axis is "width"
  width    : the perpendicular in-plane dimension (min(w, h))
  thickness: cuboid 'd' — usually the layer's metal thickness.

For a multi-cuboid net we sum R_seg over all cuboids — this is the
"total wire resistance" lumped value, ignoring the parallel/series
network topology. For SPEF compatibility and aggregate-MAPE evaluation
this is sufficient.

Layer stack lookup:
  layer_info_dict: {layer_name: {'rho_per_sq': X, 'thickness': T, 'z_pos': Z, ...}}

If sheet R is not provided, we use a default per-layer table for intel22.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# Default sheet resistance (ohm/sq) per layer for intel22-class technology.
# These are illustrative values typical for 22nm BEOL Cu metal lines.
# Actual values should come from layers.info but most layers.info we have
# do not include sheet resistance, only ε.
DEFAULT_SHEET_R_INTEL22 = {
    "m1": 1.5,
    "m2": 0.42,
    "m3": 0.42,
    "m4": 0.42,
    "m5": 0.42,
    "m6": 0.32,
    "m7": 0.32,
    "m8": 0.32,
    "m9": 0.18,
    # vias (per via)
    "v0": 5.0, "v1": 5.0, "v2": 5.0, "v3": 5.0,
    "v4": 5.0, "v5": 5.0, "v6": 5.0, "v7": 5.0, "v8": 5.0,
}

# Calibration factor — sum of cuboid R underestimates StarRC R by ~3-4x
# (vias contribute extra R we're not modeling per-via). Calibrated on train
# designs (aes/gcd/ibex) — see docs in pex_pipeline/__init__.py.
R_CALIBRATION_SCALE = 3.5


# Layer mapping from z position to layer name (intel22).
# z ranges from feat_extract_v3 LAYER_Z_RANGES + an explicit layer for via.
LAYER_Z_TO_NAME = [
    (0.0, 0.62, "m1"),
    (0.62, 0.78, "m2"),
    (0.78, 0.92, "m3"),
    (0.92, 1.07, "m4"),
    (1.07, 1.22, "m5"),
    (1.22, 1.40, "m6"),
    (1.40, 2.20, "m7"),
    (2.20, 5.00, "m8"),
    (5.00, 999.0, "m9"),
]


def z_to_layer_name(z: float) -> str:
    for lo, hi, name in LAYER_Z_TO_NAME:
        if lo <= z < hi:
            return name
    return "m1"


def cuboid_resistance(w: float, h: float, d: float, z: float,
                       sheet_r: Optional[Dict[str, float]] = None) -> float:
    """Compute analytic R for a single cuboid (μm units)."""
    sheet_r = sheet_r or DEFAULT_SHEET_R_INTEL22
    layer = z_to_layer_name(z)
    R_sheet = sheet_r.get(layer, 0.5)  # ohm/sq
    # length is the longer in-plane dimension; width is the shorter.
    L = max(w, h)
    W = min(w, h) + 1e-6
    n_squares = L / W
    return R_sheet * n_squares


def total_resistance_for_net(target_cuboids: np.ndarray,
                              n_target: int,
                              sheet_r: Optional[Dict[str, float]] = None) -> float:
    """Sum cuboid R for a net.

    target_cuboids: (max_pad, 10) channel array, layout:
        [x_rel, y_rel, z_abs, w, h, d, semantic_type, logic_flag, eps, net_type]
    n_target: number of valid (non-padding) cuboids.
    """
    if n_target <= 0:
        return 0.0
    cubs = target_cuboids[:n_target]
    z = cubs[:, 2]
    w = cubs[:, 3]
    h = cubs[:, 4]
    d = cubs[:, 5]
    R = 0.0
    for i in range(n_target):
        R += cuboid_resistance(float(w[i]), float(h[i]), float(d[i]), float(z[i]), sheet_r)
    # Calibration vs StarRC golden:
    #   - On train designs (aes/gcd/ibex), our raw cuboid sum underestimates
    #     median R by ~4.3x and mean R by ~2.0x (vias contribute extra R that
    #     we're not modeling explicitly per-via).
    #   - We apply a global scale factor R_CALIBRATION_SCALE to roughly
    #     match the golden distribution.
    return R_CALIBRATION_SCALE * R


def total_resistance_for_design(cuboid_arr_npz_path: Path,
                                  sheet_r: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Convenience: compute per-net total R for an entire design."""
    npz = np.load(cuboid_arr_npz_path, allow_pickle=True)
    target = npz["target"]
    n_target = npz["n_target"]
    net_names = npz["net_names"]
    out = {}
    for i, name in enumerate(net_names):
        out[str(name)] = total_resistance_for_net(target[i], int(n_target[i]), sheet_r)
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        npz_path = Path(sys.argv[1])
    else:
        npz_path = Path(__file__).resolve().parent.parent / "cache" / "cuboid_arr" / "intel22_tv80s_f3.npz"
    res = total_resistance_for_design(npz_path)
    vals = np.array(list(res.values()))
    print(f"n_nets: {len(res)}")
    print(f"R distribution: mean={vals.mean():.2f} median={np.median(vals):.2f} "
          f"p25={np.percentile(vals, 25):.2f} p75={np.percentile(vals, 75):.2f} "
          f"p99={np.percentile(vals, 99):.2f}")
