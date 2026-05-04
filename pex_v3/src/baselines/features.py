"""
features.py — hand-engineered per-net physical features for B1 (XGBoost) and B4 (GAM).

Phase 0.5 dependency. Owned by `classical-baseline-owner`.

Designed as PURE functions: input a `NetGeometry` dataclass (no IO),
output a `NetFeatureVector`. The IO orchestrator that scans DEFs/SPEFs
and builds the feature dataset is in `feature_dataset.py` (separate
concern, can iterate on it independently).

Schema is locked here so consumers (XGBoost, ParaGraph, GAM) can be
written against a stable feature contract while the orchestrator catches up.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, fields
from typing import Sequence, Optional

import numpy as np


# ============================================================================
# Input geometry struct (consumed by `extract_features_from_geometry`)
# ============================================================================


@dataclass(frozen=True)
class CuboidArr:
    """A bundle of cuboid arrays representing one net's conductors.

    Each array has shape (N,) or (N, 3)/(N, 6) depending on field. All N's
    must agree.

        x, y, z          — center coordinates (μm)
        w, h, d          — extents (μm)
        layer_idx        — int per cuboid (1..M9 for intel22)

    A "cuboid" here means an axis-aligned box covering one wire segment.
    """
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    w: np.ndarray
    h: np.ndarray
    d: np.ndarray
    layer_idx: np.ndarray

    @property
    def n(self) -> int:
        return int(len(self.x))


@dataclass(frozen=True)
class CouplingEdge:
    """A potential coupling edge between target and aggressor net."""
    aggressor_net: str
    target_layer: int
    aggressor_layer: int
    surface_dist_um: float       # closest surface-to-surface distance
    broadside_overlap_um2: float # xy overlap (target plate above aggressor plate)
    lateral_overlap_um2: float   # max(xz, yz) lateral side-by-side overlap


@dataclass(frozen=True)
class NetGeometry:
    """All geometry-derived info needed to compute hand-engineered features for one net.

    Fields:
        net_name:           the net's string name
        design_name:        the source design
        target_cuboids:     CuboidArr for target net's conductors
        coupling_edges:     list of CouplingEdge to neighbor signal aggressors
        vss_cuboids:        VSS/VDD power-net cuboids in the same window (CuboidArr)
        layer_stack:        per-layer ε; index aligned with layer_idx values
                            (e.g., layer_stack_eps[3] = ε of M3)
        fanout:             integer; how many sinks the net drives (from netlist)
        n_layers_total:     M1..MN max metal layer count for the PDK
        ground_plane_layer: layer where dominant ground is
        local_density_window_um2: window xy area used for density features
        local_metal_area_per_layer_um2: per-layer metal coverage in the window
                                        (length n_layers_total)
    """
    net_name: str
    design_name: str
    target_cuboids: CuboidArr
    coupling_edges: Sequence[CouplingEdge]
    vss_cuboids: CuboidArr
    layer_stack_eps: Sequence[float]    # indexed by layer; 1-based or 0-based per PDK
    fanout: int
    n_layers_total: int
    ground_plane_layer: int
    local_density_window_um2: float
    local_metal_area_per_layer_um2: Sequence[float]


# ============================================================================
# Output feature struct
# ============================================================================


@dataclass(frozen=True)
class NetFeatureVector:
    """Per-net feature vector. All values are floats; missing values use NaN.

    Length and order are LOCKED. Consumers (XGBoost, GAM) treat this as a
    contract.
    """
    # ---- Geometric ----------------------------------------------------
    total_wire_length_um: float
    total_metal_area_um2: float
    n_cuboids: float
    bbox_xy_um2: float
    bbox_z_um: float
    aspect_ratio: float
    # Layer histogram up to M9 (extra layers folded into 'plus')
    layer_hist_M1: float
    layer_hist_M2: float
    layer_hist_M3: float
    layer_hist_M4: float
    layer_hist_M5: float
    layer_hist_M6: float
    layer_hist_M7: float
    layer_hist_M8: float
    layer_hist_M9_plus: float
    # ---- Coupling-relevant -------------------------------------------
    n_aggressor_nets: float
    broadside_overlap_total_um2: float
    broadside_overlap_p95_um2: float
    lateral_overlap_total_um2: float
    lateral_overlap_p95_um2: float
    spacing_min_um: float
    spacing_p25_um: float
    spacing_p50_um: float
    spacing_p95_um: float
    n_edges_lt_1um: float
    n_edges_1_to_3um: float
    n_edges_3_to_4um: float
    # ---- Power-net context -------------------------------------------
    vss_n_cuboids: float
    vss_total_metal_area_um2: float
    vss_shield_M1_M3: float       # vss area in same xy as target with z below
    vss_shield_M4_M5: float
    vss_shield_M6_plus: float
    # ---- Topology ----------------------------------------------------
    fanout: float
    # ---- Layer-stack -------------------------------------------------
    eps_min: float
    eps_max: float
    eps_mean: float
    n_layers_present: float
    # ---- Local density -----------------------------------------------
    density_M1_M3: float
    density_M4_M5: float
    density_M6_plus: float
    # ---- Compact-model intermediates ---------------------------------
    compact_gnd_estimate_fF: float
    compact_cpl_estimate_total_fF: float

    def to_array(self) -> np.ndarray:
        """Flatten to a 1-D numpy array in the locked field order."""
        return np.array([getattr(self, f.name) for f in fields(self)],
                        dtype=np.float64)

    @classmethod
    def field_names(cls) -> list[str]:
        return [f.name for f in fields(cls)]


# ============================================================================
# Helpers: small numerical primitives used inside feature extraction
# ============================================================================


def _wire_length_um(cuboids: CuboidArr) -> float:
    """Approximate wire length = sum of max(w, h, d) per cuboid (longest extent).

    Manhattan routing makes one of (w, h, d) dominant per segment.
    """
    if cuboids.n == 0:
        return 0.0
    max_extents = np.maximum.reduce([cuboids.w, cuboids.h, cuboids.d])
    return float(max_extents.sum())


def _metal_area_um2(cuboids: CuboidArr) -> float:
    """Total xy footprint area of all cuboids (overlapping not deduped)."""
    if cuboids.n == 0:
        return 0.0
    return float((cuboids.w * cuboids.h).sum())


def _bbox(cuboids: CuboidArr) -> tuple[float, float, float, float, float, float]:
    """Net's axis-aligned bounding box. Returns (xmin, xmax, ymin, ymax, zmin, zmax)."""
    if cuboids.n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    x_min = float((cuboids.x - cuboids.w / 2).min())
    x_max = float((cuboids.x + cuboids.w / 2).max())
    y_min = float((cuboids.y - cuboids.h / 2).min())
    y_max = float((cuboids.y + cuboids.h / 2).max())
    z_min = float((cuboids.z - cuboids.d / 2).min())
    z_max = float((cuboids.z + cuboids.d / 2).max())
    return x_min, x_max, y_min, y_max, z_min, z_max


def _layer_histogram(cuboids: CuboidArr, n_layers: int = 9) -> np.ndarray:
    """Count of cuboids per layer index (1..n_layers).

    Returns shape (n_layers,). Layers above n_layers are folded into the last bin.
    """
    if cuboids.n == 0:
        return np.zeros(n_layers, dtype=np.float64)
    layer_idx = np.asarray(cuboids.layer_idx, dtype=np.int64)
    layer_idx = np.clip(layer_idx, 1, n_layers)
    hist, _ = np.histogram(layer_idx, bins=np.arange(1, n_layers + 2))
    return hist.astype(np.float64)


def _spacing_distribution(edges: Sequence[CouplingEdge]) -> dict:
    """Aggregate the surface_dist_um across all coupling edges."""
    if not edges:
        return {
            "spacing_min_um": np.nan,
            "spacing_p25_um": np.nan,
            "spacing_p50_um": np.nan,
            "spacing_p95_um": np.nan,
            "n_edges_lt_1um": 0.0,
            "n_edges_1_to_3um": 0.0,
            "n_edges_3_to_4um": 0.0,
        }
    arr = np.array([e.surface_dist_um for e in edges], dtype=np.float64)
    return {
        "spacing_min_um": float(arr.min()),
        "spacing_p25_um": float(np.percentile(arr, 25)),
        "spacing_p50_um": float(np.percentile(arr, 50)),
        "spacing_p95_um": float(np.percentile(arr, 95)),
        "n_edges_lt_1um": float((arr < 1.0).sum()),
        "n_edges_1_to_3um": float(((arr >= 1.0) & (arr < 3.0)).sum()),
        "n_edges_3_to_4um": float(((arr >= 3.0) & (arr < 4.0)).sum()),
    }


def _overlap_stats(edges: Sequence[CouplingEdge]) -> dict:
    """Aggregate broadside + lateral overlaps across edges."""
    if not edges:
        return {
            "broadside_overlap_total_um2": 0.0,
            "broadside_overlap_p95_um2": 0.0,
            "lateral_overlap_total_um2": 0.0,
            "lateral_overlap_p95_um2": 0.0,
        }
    bs = np.array([e.broadside_overlap_um2 for e in edges], dtype=np.float64)
    lat = np.array([e.lateral_overlap_um2 for e in edges], dtype=np.float64)
    return {
        "broadside_overlap_total_um2": float(bs.sum()),
        "broadside_overlap_p95_um2": float(np.percentile(bs, 95)),
        "lateral_overlap_total_um2": float(lat.sum()),
        "lateral_overlap_p95_um2": float(np.percentile(lat, 95)),
    }


def _vss_shielding(
    target_cuboids: CuboidArr,
    vss_cuboids: CuboidArr,
) -> dict:
    """Estimate VSS/VDD shield area in same xy as target, by layer bucket.

    Buckets: M1-M3, M4-M5, M6+
    """
    out = {"vss_shield_M1_M3": 0.0, "vss_shield_M4_M5": 0.0, "vss_shield_M6_plus": 0.0}
    if target_cuboids.n == 0 or vss_cuboids.n == 0:
        return out
    # Compute target bbox in xy
    txmin = (target_cuboids.x - target_cuboids.w / 2).min()
    txmax = (target_cuboids.x + target_cuboids.w / 2).max()
    tymin = (target_cuboids.y - target_cuboids.h / 2).min()
    tymax = (target_cuboids.y + target_cuboids.h / 2).max()
    # Find VSS cuboids whose xy intersects target bbox
    vxmin = vss_cuboids.x - vss_cuboids.w / 2
    vxmax = vss_cuboids.x + vss_cuboids.w / 2
    vymin = vss_cuboids.y - vss_cuboids.h / 2
    vymax = vss_cuboids.y + vss_cuboids.h / 2
    intersects = (vxmax >= txmin) & (vxmin <= txmax) & (vymax >= tymin) & (vymin <= tymax)
    if not intersects.any():
        return out
    # For intersecting VSS cuboids, accumulate area by layer bucket
    vw = vss_cuboids.w[intersects]
    vh = vss_cuboids.h[intersects]
    vlayer = vss_cuboids.layer_idx[intersects]
    areas = vw * vh
    out["vss_shield_M1_M3"] = float(areas[(vlayer >= 1) & (vlayer <= 3)].sum())
    out["vss_shield_M4_M5"] = float(areas[(vlayer >= 4) & (vlayer <= 5)].sum())
    out["vss_shield_M6_plus"] = float(areas[vlayer >= 6].sum())
    return out


def _compact_gnd_estimate_fF(
    target_cuboids: CuboidArr,
    layer_stack_eps: Sequence[float],
    ground_plane_layer: int,
) -> float:
    """Sakurai-Tamaru-style estimate of self-capacitance to ground.

    Per-cuboid: parallel-plate over distance to ground plane, with ε from layer stack.
    Returns total in fF.
    """
    if target_cuboids.n == 0:
        return 0.0
    from src.synthetic.ground_truth import EPS0_FF_UM
    total = 0.0
    eps = layer_stack_eps  # 1-based or 0-based per caller — caller handles indexing
    for i in range(target_cuboids.n):
        layer = int(target_cuboids.layer_idx[i])
        # Distance from cuboid bottom to ground plane (rough)
        d = abs(layer - ground_plane_layer)
        if d == 0:
            d = 1
        d_um = max(0.05, d * 0.1)  # 100nm per layer placeholder
        eps_r = float(eps[layer]) if 0 <= layer < len(eps) else 4.0
        A = float(target_cuboids.w[i] * target_cuboids.h[i])
        total += EPS0_FF_UM * eps_r * A / d_um
    return float(total)


def _compact_cpl_estimate_total_fF(
    edges: Sequence[CouplingEdge],
    layer_stack_eps: Sequence[float],
) -> float:
    """Sakurai-Tamaru-style sum of all aggressor coupling capacitances."""
    if not edges:
        return 0.0
    from src.synthetic.ground_truth import EPS0_FF_UM
    total = 0.0
    for e in edges:
        # Use lateral overlap with surface distance
        d_um = max(0.05, e.surface_dist_um)
        # Average ε of the two layers involved
        l1, l2 = int(e.target_layer), int(e.aggressor_layer)
        eps1 = float(layer_stack_eps[l1]) if 0 <= l1 < len(layer_stack_eps) else 4.0
        eps2 = float(layer_stack_eps[l2]) if 0 <= l2 < len(layer_stack_eps) else 4.0
        eps_avg = 0.5 * (eps1 + eps2)
        A = e.lateral_overlap_um2 + e.broadside_overlap_um2
        total += EPS0_FF_UM * eps_avg * A / d_um
    return float(total)


# ============================================================================
# Main entrypoint
# ============================================================================


def extract_features_from_geometry(geo: NetGeometry) -> NetFeatureVector:
    """Compute the full per-net feature vector from a NetGeometry input.

    Pure function — no IO, no globals. Suitable for unit testing with
    synthetic geometry.

    Args:
        geo: NetGeometry input

    Returns:
        NetFeatureVector
    """
    cuboids = geo.target_cuboids
    edges = geo.coupling_edges
    vss = geo.vss_cuboids
    eps = geo.layer_stack_eps

    # Geometric
    wire_len = _wire_length_um(cuboids)
    metal_area = _metal_area_um2(cuboids)
    xmin, xmax, ymin, ymax, zmin, zmax = _bbox(cuboids)
    bbox_xy = (xmax - xmin) * (ymax - ymin)
    bbox_z = zmax - zmin
    layer_hist = _layer_histogram(cuboids, n_layers=9)
    aspect = (xmax - xmin) / max(ymax - ymin, 1e-6) if (xmax > xmin) else 1.0

    # Coupling
    overlap = _overlap_stats(edges)
    spacing = _spacing_distribution(edges)
    n_aggr = len({e.aggressor_net for e in edges})

    # Power
    vss_metal_area = _metal_area_um2(vss)
    vss_shield = _vss_shielding(cuboids, vss)

    # Layer stack
    eps_arr = np.asarray(eps, dtype=np.float64)
    eps_arr_pos = eps_arr[eps_arr > 0]
    eps_min = float(eps_arr_pos.min()) if len(eps_arr_pos) > 0 else 1.0
    eps_max = float(eps_arr_pos.max()) if len(eps_arr_pos) > 0 else 1.0
    eps_mean = float(eps_arr_pos.mean()) if len(eps_arr_pos) > 0 else 1.0
    n_layers_present = float((layer_hist > 0).sum())

    # Density
    win = max(geo.local_density_window_um2, 1e-6)
    metal_per_layer = np.asarray(geo.local_metal_area_per_layer_um2, dtype=np.float64)
    if len(metal_per_layer) >= 9:
        d_M1_M3 = float(metal_per_layer[1:4].sum() / win)
        d_M4_M5 = float(metal_per_layer[4:6].sum() / win)
        d_M6_plus = float(metal_per_layer[6:].sum() / win)
    else:
        d_M1_M3 = d_M4_M5 = d_M6_plus = float("nan")

    # Compact-model intermediates
    compact_gnd = _compact_gnd_estimate_fF(cuboids, eps, geo.ground_plane_layer)
    compact_cpl = _compact_cpl_estimate_total_fF(edges, eps)

    return NetFeatureVector(
        total_wire_length_um=wire_len,
        total_metal_area_um2=metal_area,
        n_cuboids=float(cuboids.n),
        bbox_xy_um2=float(bbox_xy),
        bbox_z_um=float(bbox_z),
        aspect_ratio=float(aspect),
        layer_hist_M1=float(layer_hist[0]),
        layer_hist_M2=float(layer_hist[1]),
        layer_hist_M3=float(layer_hist[2]),
        layer_hist_M4=float(layer_hist[3]),
        layer_hist_M5=float(layer_hist[4]),
        layer_hist_M6=float(layer_hist[5]),
        layer_hist_M7=float(layer_hist[6]),
        layer_hist_M8=float(layer_hist[7]),
        layer_hist_M9_plus=float(layer_hist[8]),
        n_aggressor_nets=float(n_aggr),
        broadside_overlap_total_um2=overlap["broadside_overlap_total_um2"],
        broadside_overlap_p95_um2=overlap["broadside_overlap_p95_um2"],
        lateral_overlap_total_um2=overlap["lateral_overlap_total_um2"],
        lateral_overlap_p95_um2=overlap["lateral_overlap_p95_um2"],
        spacing_min_um=spacing["spacing_min_um"],
        spacing_p25_um=spacing["spacing_p25_um"],
        spacing_p50_um=spacing["spacing_p50_um"],
        spacing_p95_um=spacing["spacing_p95_um"],
        n_edges_lt_1um=spacing["n_edges_lt_1um"],
        n_edges_1_to_3um=spacing["n_edges_1_to_3um"],
        n_edges_3_to_4um=spacing["n_edges_3_to_4um"],
        vss_n_cuboids=float(vss.n),
        vss_total_metal_area_um2=vss_metal_area,
        vss_shield_M1_M3=vss_shield["vss_shield_M1_M3"],
        vss_shield_M4_M5=vss_shield["vss_shield_M4_M5"],
        vss_shield_M6_plus=vss_shield["vss_shield_M6_plus"],
        fanout=float(geo.fanout),
        eps_min=eps_min,
        eps_max=eps_max,
        eps_mean=eps_mean,
        n_layers_present=n_layers_present,
        density_M1_M3=d_M1_M3,
        density_M4_M5=d_M4_M5,
        density_M6_plus=d_M6_plus,
        compact_gnd_estimate_fF=compact_gnd,
        compact_cpl_estimate_total_fF=compact_cpl,
    )


# ============================================================================
# Convenience: helper for callers building empty CuboidArr
# ============================================================================


def empty_cuboid_arr() -> CuboidArr:
    """Construct an empty CuboidArr (all zero-length arrays)."""
    z = np.zeros(0, dtype=np.float64)
    zi = np.zeros(0, dtype=np.int64)
    return CuboidArr(x=z, y=z, z=z, w=z, h=z, d=z, layer_idx=zi)
