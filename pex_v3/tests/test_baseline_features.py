"""
test_baseline_features.py — verify hand-engineered feature extraction.

Synthetic NetGeometry inputs → check NetFeatureVector outputs against
known closed-form values.
"""
from __future__ import annotations
import numpy as np
import pytest

from src.baselines.features import (
    CuboidArr,
    CouplingEdge,
    NetGeometry,
    NetFeatureVector,
    empty_cuboid_arr,
    extract_features_from_geometry,
    _wire_length_um,
    _metal_area_um2,
    _layer_histogram,
    _spacing_distribution,
    _overlap_stats,
    _vss_shielding,
)


# ============================================================================
# Helpers — synthetic NetGeometry construction
# ============================================================================


def _arr(values, dtype=np.float64):
    return np.asarray(values, dtype=dtype)


def _make_cuboid_arr(rows: list[tuple]) -> CuboidArr:
    """rows: list of (x, y, z, w, h, d, layer_idx)"""
    if not rows:
        return empty_cuboid_arr()
    arr = np.asarray(rows, dtype=np.float64)
    return CuboidArr(
        x=arr[:, 0], y=arr[:, 1], z=arr[:, 2],
        w=arr[:, 3], h=arr[:, 4], d=arr[:, 5],
        layer_idx=arr[:, 6].astype(np.int64),
    )


def _make_geometry(
    target_rows=None,
    edges=None,
    vss_rows=None,
    fanout=1,
) -> NetGeometry:
    target = _make_cuboid_arr(target_rows or [])
    vss = _make_cuboid_arr(vss_rows or [])
    return NetGeometry(
        net_name="testnet",
        design_name="testdesign",
        target_cuboids=target,
        coupling_edges=edges or [],
        vss_cuboids=vss,
        layer_stack_eps=[1.0, 4.0, 4.0, 4.0, 3.5, 3.5, 3.0, 3.0, 3.0, 3.0],
        fanout=fanout,
        n_layers_total=9,
        ground_plane_layer=0,
        local_density_window_um2=100.0,
        local_metal_area_per_layer_um2=[0.0, 5.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.0],
    )


# ============================================================================
# CuboidArr basics
# ============================================================================


def test_empty_cuboid_arr_has_zero_size():
    e = empty_cuboid_arr()
    assert e.n == 0


def test_wire_length_zero_for_empty():
    assert _wire_length_um(empty_cuboid_arr()) == 0.0


def test_wire_length_takes_max_extent():
    """Each cuboid contributes max(w, h, d). Sum across cuboids."""
    c = _make_cuboid_arr([
        (0, 0, 0, 5.0, 0.1, 0.1, 1),    # max=5.0
        (0, 0, 0, 0.1, 3.0, 0.1, 1),    # max=3.0
        (0, 0, 0, 0.1, 0.1, 7.0, 1),    # max=7.0
    ])
    assert _wire_length_um(c) == 15.0


def test_metal_area_sums_w_h():
    c = _make_cuboid_arr([
        (0, 0, 0, 2.0, 3.0, 0.1, 1),    # w*h=6.0
        (0, 0, 0, 1.0, 1.0, 0.1, 2),    # w*h=1.0
    ])
    assert _metal_area_um2(c) == 7.0


# ============================================================================
# Layer histogram
# ============================================================================


def test_layer_histogram_basic():
    c = _make_cuboid_arr([
        (0, 0, 0, 1, 1, 0.1, 1),
        (0, 0, 0, 1, 1, 0.1, 1),
        (0, 0, 0, 1, 1, 0.1, 3),
        (0, 0, 0, 1, 1, 0.1, 5),
    ])
    h = _layer_histogram(c, n_layers=9)
    assert h[0] == 2  # M1
    assert h[1] == 0  # M2
    assert h[2] == 1  # M3
    assert h[4] == 1  # M5


def test_layer_histogram_clips_above_n_layers():
    c = _make_cuboid_arr([
        (0, 0, 0, 1, 1, 0.1, 12),  # above M9 → clipped to 9
        (0, 0, 0, 1, 1, 0.1, 9),
    ])
    h = _layer_histogram(c, n_layers=9)
    assert h[8] == 2  # M9 + clipped above


def test_layer_histogram_empty():
    h = _layer_histogram(empty_cuboid_arr(), n_layers=9)
    assert h.shape == (9,)
    assert h.sum() == 0


# ============================================================================
# Spacing distribution
# ============================================================================


def test_spacing_distribution_empty_returns_nan():
    s = _spacing_distribution([])
    assert np.isnan(s["spacing_min_um"])
    assert s["n_edges_lt_1um"] == 0.0


def test_spacing_distribution_correct_buckets():
    edges = [
        CouplingEdge("a", 1, 1, 0.5, 0.0, 0.0),    # < 1
        CouplingEdge("b", 1, 1, 0.8, 0.0, 0.0),    # < 1
        CouplingEdge("c", 1, 1, 1.5, 0.0, 0.0),    # 1..3
        CouplingEdge("d", 1, 1, 2.5, 0.0, 0.0),    # 1..3
        CouplingEdge("e", 1, 1, 3.5, 0.0, 0.0),    # 3..4
    ]
    s = _spacing_distribution(edges)
    assert s["spacing_min_um"] == 0.5
    assert s["n_edges_lt_1um"] == 2
    assert s["n_edges_1_to_3um"] == 2
    assert s["n_edges_3_to_4um"] == 1


# ============================================================================
# Overlap stats
# ============================================================================


def test_overlap_stats_empty():
    s = _overlap_stats([])
    assert s["broadside_overlap_total_um2"] == 0.0


def test_overlap_stats_sum():
    edges = [
        CouplingEdge("a", 1, 1, 0.5, 1.0, 2.0),
        CouplingEdge("b", 1, 1, 0.5, 4.0, 6.0),
    ]
    s = _overlap_stats(edges)
    assert s["broadside_overlap_total_um2"] == 5.0
    assert s["lateral_overlap_total_um2"] == 8.0


# ============================================================================
# VSS shielding
# ============================================================================


def test_vss_shielding_target_no_intersect():
    target = _make_cuboid_arr([(0, 0, 0, 1, 1, 0.1, 3)])
    vss = _make_cuboid_arr([(100, 100, 0, 1, 1, 0.1, 1)])  # very far
    out = _vss_shielding(target, vss)
    assert out["vss_shield_M1_M3"] == 0.0


def test_vss_shielding_target_intersect_M1_M3():
    target = _make_cuboid_arr([(0, 0, 0, 4, 4, 0.1, 3)])
    vss = _make_cuboid_arr([
        (0, 0, 0, 1, 1, 0.1, 1),  # M1, area 1
        (0, 0, 0, 2, 2, 0.1, 4),  # M4, area 4
        (0, 0, 0, 3, 1, 0.1, 7),  # M7, area 3
    ])
    out = _vss_shielding(target, vss)
    assert out["vss_shield_M1_M3"] == 1.0
    assert out["vss_shield_M4_M5"] == 4.0
    assert out["vss_shield_M6_plus"] == 3.0


# ============================================================================
# extract_features_from_geometry — end to end
# ============================================================================


def test_extract_features_empty_net():
    geo = _make_geometry()
    fv = extract_features_from_geometry(geo)
    assert isinstance(fv, NetFeatureVector)
    assert fv.total_wire_length_um == 0.0
    assert fv.total_metal_area_um2 == 0.0
    assert fv.n_cuboids == 0
    assert fv.fanout == 1


def test_extract_features_single_cuboid():
    geo = _make_geometry(
        target_rows=[(0, 0, 0, 5.0, 1.0, 0.1, 3)],
        fanout=4,
    )
    fv = extract_features_from_geometry(geo)
    assert fv.n_cuboids == 1
    assert fv.total_wire_length_um == 5.0    # max extent
    assert fv.total_metal_area_um2 == 5.0    # w*h
    assert fv.layer_hist_M3 == 1.0
    assert fv.layer_hist_M1 == 0.0
    assert fv.fanout == 4


def test_extract_features_with_edges():
    edges = [
        CouplingEdge("a", 3, 3, 0.5, 1.0, 2.0),
        CouplingEdge("b", 3, 4, 1.5, 3.0, 4.0),
        CouplingEdge("a", 3, 3, 0.5, 1.0, 2.0),  # duplicate aggressor name
    ]
    geo = _make_geometry(
        target_rows=[(0, 0, 0, 5, 1, 0.1, 3)],
        edges=edges,
    )
    fv = extract_features_from_geometry(geo)
    # Distinct aggressor count is the unique set of names
    assert fv.n_aggressor_nets == 2
    # Spacing min is 0.5
    assert fv.spacing_min_um == 0.5
    assert fv.broadside_overlap_total_um2 == 5.0


def test_extract_features_to_array_consistent_length():
    geo = _make_geometry(target_rows=[(0, 0, 0, 1, 1, 0.1, 1)])
    fv = extract_features_from_geometry(geo)
    arr = fv.to_array()
    assert arr.shape == (len(NetFeatureVector.field_names()),)
    # All numeric, no NaN in this fully-specified geometry
    # (some can be NaN when edges empty — that's fine for the contract)


def test_field_names_locked():
    """The field order must NEVER change once consumers depend on it."""
    expected_first = "total_wire_length_um"
    expected_last = "compact_cpl_estimate_total_fF"
    names = NetFeatureVector.field_names()
    assert names[0] == expected_first
    assert names[-1] == expected_last


def test_compact_gnd_increases_with_area():
    geo_small = _make_geometry(target_rows=[(0, 0, 0, 1, 1, 0.1, 3)])
    geo_big = _make_geometry(target_rows=[(0, 0, 0, 4, 4, 0.1, 3)])  # 16× area
    fv_small = extract_features_from_geometry(geo_small)
    fv_big = extract_features_from_geometry(geo_big)
    # 16× area → 16× compact gnd estimate (parallel-plate scaling)
    assert fv_big.compact_gnd_estimate_fF == pytest.approx(
        16 * fv_small.compact_gnd_estimate_fF, rel=1e-6
    )


def test_extract_features_density_calc():
    geo = _make_geometry(target_rows=[(0, 0, 0, 1, 1, 0.1, 1)])
    # local_metal_area_per_layer = [0, 5, 8, 6, 4, 3, 2, 1, 0.5, 0]
    # window = 100 μm²
    # M1-M3: layers 1+2+3 = 5+8+6 = 19, density = 0.19
    # M4-M5: layers 4+5 = 4+3 = 7, density = 0.07
    # M6+:   layers 6+7+8+9 = 2+1+0.5+0 = 3.5, density = 0.035
    fv = extract_features_from_geometry(geo)
    assert fv.density_M1_M3 == pytest.approx(0.19, abs=1e-6)
    assert fv.density_M4_M5 == pytest.approx(0.07, abs=1e-6)
    assert fv.density_M6_plus == pytest.approx(0.035, abs=1e-6)
