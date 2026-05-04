"""
feature_dataset.py — DEF/SPEF → NetFeatureVector pipeline for B1/B4 baselines.

The orchestrator missing piece. Reads:
  - v3 manifest (`cfg.MANIFEST_PATH_V3`)
  - DEF files (`cfg.TRAIN_DEFS + cfg.TEST_DEFS`)
  - Golden SPEFs (`cfg.SPEF_DIR`)
  - Layer stack (`cfg.LAYERS_INFO_PATH`)

Outputs per-design feature parquet files at:
  `cfg.PROCESSED_DIR_V3 / features / <design>.parquet`

Each parquet contains one row per net in the v3 manifest with:
  - Identity:   design_name, net_name, split
  - Targets:    c_gnd_fF, c_cpl_total_fF, total_cap_fF, total_res_ohm
  - Features:   all 43 NetFeatureVector fields (locked schema)

Phase 0.5 v1 keeps it simple:
  - One-pass SPEF parser that yields ALL nets in a single file traversal
    (vs legacy per-net which reopens the file each call → O(N²))
  - Coupling edges enumerated via simple bbox+distance check, no SpatialGrid
    (correct, slower; feature extraction is one-time per design)
  - Per-design fanout = number of distinct (target, aggressor) pairs in SPEF
    coupled_caps (proxy; real fanout from netlist deferred)

Cost: ~5-10 min per design on intel22 sizes. ~60 min total for 11 designs.
Ouput size: ~few MB per design (parquet, ~1KB per net).

Run:
    python3 pex_v3/scripts/04_build_feature_dataset.py
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "pex_v3"))


def _legacy_import(qualified_name: str):
    """Load a module from the legacy src/ tree by file path.

    Why: `pex_v3/src/__init__.py` makes `src` resolve to pex_v3 first when
    both are on sys.path. We bypass the import system to load legacy modules
    directly from disk, side-stepping the namespace conflict.
    """
    import importlib.util
    parts = qualified_name.split(".")
    fpath = _PROJECT_ROOT.joinpath("src", *parts).with_suffix(".py")
    if not fpath.exists():
        raise ImportError(f"legacy module not at {fpath}")
    spec = importlib.util.spec_from_file_location(
        f"_pex_legacy_{qualified_name.replace('.', '_')}", str(fpath)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_legacy_def = _legacy_import("preprocessing.def_parser")
_legacy_layer = _legacy_import("preprocessing.layer_parser")
_legacy_lef = _legacy_import("preprocessing.lef_parser")
_legacy_cell = _legacy_import("preprocessing.cell_parser")
DefStreamParser = _legacy_def.DefStreamParser
LayerInfoParser = _legacy_layer.LayerInfoParser
LefParser = _legacy_lef.LefParser
CellLibParser = _legacy_cell.CellLibParser

from .features import (  # noqa: E402
    CuboidArr,
    CouplingEdge,
    NetGeometry,
    NetFeatureVector,
    extract_features_from_geometry,
    empty_cuboid_arr,
)


# ============================================================================
# One-pass SPEF parser
# ============================================================================


def _normalize_name(s: str) -> str:
    return s.replace("\\", "").strip() if s else ""


def _extract_net_name(node: str) -> str:
    return _normalize_name(node.split(":")[0])


def stream_spef_nets(spef_path: Path) -> Iterator[dict]:
    """Yield one record per *D_NET in the SPEF.

    Record keys:
        net_name, total_cap_fF, ground_cap_fF, c_cpl_total_fF, total_res_ohm,
        coupled_caps (dict aggressor -> fF)

    SPEF dialect handled: StarRC standard format. Notation:
        *D_NET <name> <total_cap>
        *CAP
            N node val           # ground cap
            N node1 node2 val    # coupling cap
        *RES
            N node1 node2 res
        *END
    """
    p_dnet = re.compile(r"\*D_NET\s+(\S+)\s+([0-9\.\+eE\-]+)")
    p_res_with_anno = re.compile(
        r"^\d+\s+\S+\s+\S+\s+([0-9\.\+eE\-]+)\s*\/.*$"
    )
    p_res_simple = re.compile(r"^\d+\s+\S+\s+\S+\s+([0-9\.\+eE\-]+)\s*$")

    with open(spef_path, "r") as f:
        in_cap = False
        in_res = False
        cur = None  # the in-flight net record

        for line in f:
            line = line.rstrip()
            if not line:
                continue

            m = p_dnet.match(line)
            if m:
                # Emit previous net (if any) and start a new one
                if cur is not None:
                    yield cur
                cur = {
                    "net_name": _normalize_name(m.group(1)),
                    "total_cap_fF": float(m.group(2)),
                    "ground_cap_fF": 0.0,
                    "c_cpl_total_fF": 0.0,
                    "total_res_ohm": 0.0,
                    "coupled_caps": {},
                }
                in_cap = False
                in_res = False
                continue

            if cur is None:
                continue

            stripped = line.strip()
            if stripped.startswith("*CAP"):
                in_cap = True
                in_res = False
                continue
            if stripped.startswith("*RES"):
                in_res = True
                in_cap = False
                continue
            if stripped.startswith("*END"):
                yield cur
                cur = None
                in_cap = False
                in_res = False
                continue

            if in_cap:
                tokens = stripped.split()
                # Ground cap: 3 tokens = "id node val"
                if len(tokens) == 3:
                    try:
                        cur["ground_cap_fF"] += float(tokens[2])
                    except ValueError:
                        pass
                # Coupling cap: 4+ tokens = "id node1 node2 val"
                elif len(tokens) >= 4:
                    n1 = _extract_net_name(tokens[1])
                    n2 = _extract_net_name(tokens[2])
                    try:
                        val = float(tokens[3])
                    except ValueError:
                        continue
                    aggr = n2 if n1 == cur["net_name"] else n1
                    cur["coupled_caps"][aggr] = (
                        cur["coupled_caps"].get(aggr, 0.0) + val
                    )
                    cur["c_cpl_total_fF"] += val

            elif in_res:
                m2 = p_res_with_anno.match(stripped) or p_res_simple.match(stripped)
                if m2:
                    try:
                        cur["total_res_ohm"] += float(m2.group(1))
                    except ValueError:
                        pass

        # End-of-file: emit any in-flight net
        if cur is not None:
            yield cur


def parse_spef_to_dict(spef_path: Path) -> dict:
    """One-pass SPEF parse → {net_name: record}.

    Wrapper around stream_spef_nets() for caller convenience when the
    full corpus fits in memory (we expect ~50K nets/design × ~200 bytes = ~10 MB).
    """
    out = {}
    for rec in stream_spef_nets(spef_path):
        out[rec["net_name"]] = rec
    return out


# ============================================================================
# Per-design DEF → NetGeometry
# ============================================================================


_POWER_NAMES = {"vss", "vdd", "vcc", "gnd", "vssx", "vccx", "vddx"}


def _layer_z_position(layer_idx: int, layer_eps: list[float]) -> float:
    """Approximate z-center of a metal layer. Used only for distance
    estimation in coupling-edge enumeration; not for capacitance physics."""
    # 100 nm per layer placeholder
    return 0.1 * layer_idx


def _bbox_from_cuboids(arr: np.ndarray) -> tuple[float, float, float, float]:
    """(xmin, xmax, ymin, ymax) bounding box from cuboid array (n, 7)."""
    if len(arr) == 0:
        return 0.0, 0.0, 0.0, 0.0
    xmin = float((arr[:, 0] - arr[:, 3] / 2).min())
    xmax = float((arr[:, 0] + arr[:, 3] / 2).max())
    ymin = float((arr[:, 1] - arr[:, 4] / 2).min())
    ymax = float((arr[:, 1] + arr[:, 4] / 2).max())
    return xmin, xmax, ymin, ymax


def _scan_design_geometry(def_path: Path, layer_map) -> dict:
    """Parse one DEF; return per-net cuboid arrays + global VSS array.

    Returns:
        {
            'nets':       {net_name: ndarray (n_i, 7) — x,y,z,w,h,d,layer_idx},
            'vss':        ndarray (n_vss, 7),
            'all_cuboids': ndarray (N, 7) — concatenated for fast spatial
                            query during coupling enumeration,
            'all_owner':  ndarray (N,) — net_name string per row,
            'design_name': stem,
        }
    """
    tech_lef_path = _PROJECT_ROOT / "tool" / "pdk" / "22nm" / "tech_lef" / "p1222_js.lef"
    cell_lef_path = _PROJECT_ROOT / "tool" / "pdk" / "22nm" / "cell_lef" / "b15_nn.lef"
    tech_lef = LefParser(tech_lef_path).parse()
    cell_lib = CellLibParser(cell_lef_path).parse()

    parser = DefStreamParser(str(def_path), layer_map, tech_lef, cell_lib)
    nets: dict[str, list] = {}
    vss_rows: list = []

    # FIX (Phase C audit A3): segments carry `"layer"` STRING (e.g. "m3"),
    # not `"layer_idx"`. Old code's `.get("layer_idx", 0)` silently routed
    # ALL cuboids to layer 0, dead-weighting 15 of 43 features. Parse the
    # string to an integer index.
    _LAYER_RE = re.compile(r"[mM](\d+)")

    def _layer_str_to_idx(name) -> int:
        if name is None:
            return 0
        m = _LAYER_RE.match(str(name))
        if m:
            return int(m.group(1))
        return 0

    for net_name, cuboids, segments in parser.parse():
        if cuboids is None or cuboids.size == 0:
            continue
        if cuboids.shape[1] == 6:
            # Use first segment's layer as proxy for all cuboids of this net.
            # Manhattan routing on single-layer wires is dominant; via cuboids
            # would need per-cuboid layer assignment but are rare here.
            if segments and len(segments) > 0:
                layer = _layer_str_to_idx(segments[0].get("layer"))
            else:
                layer = 0
            layer_col = np.full((len(cuboids), 1), layer, dtype=np.float64)
            cuboids = np.hstack([cuboids, layer_col])

        if net_name.lower() in _POWER_NAMES:
            vss_rows.append(cuboids)
        else:
            nets[net_name] = cuboids

    vss_arr = np.vstack(vss_rows) if vss_rows else np.zeros((0, 7), dtype=np.float64)
    all_rows = []
    all_owners = []
    for n, arr in nets.items():
        all_rows.append(arr)
        all_owners.extend([n] * len(arr))

    if all_rows:
        all_cuboids = np.vstack(all_rows)
        all_owner = np.asarray(all_owners, dtype=object)
    else:
        all_cuboids = np.zeros((0, 7), dtype=np.float64)
        all_owner = np.array([], dtype=object)

    return {
        "nets": nets,
        "vss": vss_arr,
        "all_cuboids": all_cuboids,
        "all_owner": all_owner,
        "design_name": def_path.stem,
    }


def _np_to_cuboid_arr(arr: np.ndarray) -> CuboidArr:
    """Convert (k, 7) ndarray to CuboidArr."""
    if len(arr) == 0:
        return empty_cuboid_arr()
    return CuboidArr(
        x=arr[:, 0], y=arr[:, 1], z=arr[:, 2],
        w=arr[:, 3], h=arr[:, 4], d=arr[:, 5],
        layer_idx=arr[:, 6].astype(np.int64),
    )


class _SpatialGrid:
    """Phase C audit A3 #9 fix: O(N log N) hash grid for coupling enumeration.
    Lifted from `scripts/build_dataset.py:169`. Bin global cuboids by their
    xy bbox cells; query returns indices whose cells overlap a window.
    """
    __slots__ = ("bx", "by", "grid")

    def __init__(self, bin_size_x: float = 4.0, bin_size_y: float = 4.0):
        from collections import defaultdict
        self.bx = bin_size_x
        self.by = bin_size_y
        self.grid = defaultdict(list)

    def build_from_cuboids(self, cuboids: np.ndarray) -> None:
        if len(cuboids) == 0:
            return
        mins = cuboids[:, :2] - cuboids[:, 3:5] / 2
        maxs = cuboids[:, :2] + cuboids[:, 3:5] / 2
        min_idx = np.floor(mins / [self.bx, self.by]).astype(np.int32)
        max_idx = np.floor(maxs / [self.bx, self.by]).astype(np.int32)
        for i in range(len(cuboids)):
            for x in range(min_idx[i, 0], max_idx[i, 0] + 1):
                for y in range(min_idx[i, 1], max_idx[i, 1] + 1):
                    self.grid[(x, y)].append(i)

    def query_bbox(self, xmin: float, xmax: float, ymin: float, ymax: float) -> np.ndarray:
        min_idx_x = int(np.floor(xmin / self.bx))
        max_idx_x = int(np.floor(xmax / self.bx))
        min_idx_y = int(np.floor(ymin / self.by))
        max_idx_y = int(np.floor(ymax / self.by))
        seen = set()
        for x in range(min_idx_x, max_idx_x + 1):
            for y in range(min_idx_y, max_idx_y + 1):
                if (x, y) in self.grid:
                    seen.update(self.grid[(x, y)])
        if not seen:
            return np.empty(0, dtype=np.int64)
        return np.fromiter(seen, dtype=np.int64)


def _enumerate_coupling_edges(
    target_arr: np.ndarray,
    all_cuboids: np.ndarray,
    all_owner: np.ndarray,
    target_net_name: str,
    cutoff_um: float,
    max_edges: int = 768,
    spatial_grid: "_SpatialGrid | None" = None,
) -> list[CouplingEdge]:
    """Find aggressor edges within cutoff distance of any target cuboid.

    Phase C A3 #9 fix: SpatialGrid query + numpy broadcast replaces the
    O(N²) Python loop. Speedup ≈ 10-100× on dense designs.

    NOTE (Phase C A3 #10): Still aggregates one CouplingEdge per
    (target, aggressor) pair via `closest_dist` semantics. Full H4
    pairwise emit is deferred to Phase 1. B1 baseline runs on degraded
    semantics until then.
    """
    if len(target_arr) == 0 or len(all_cuboids) == 0:
        return []

    txmin, txmax, tymin, tymax = _bbox_from_cuboids(target_arr)
    txmin -= cutoff_um; txmax += cutoff_um
    tymin -= cutoff_um; tymax += cutoff_um

    if spatial_grid is not None:
        cand_idx = spatial_grid.query_bbox(txmin, txmax, tymin, tymax)
        if len(cand_idx) == 0:
            return []
        cand_arr = all_cuboids[cand_idx]
        cand_owners = all_owner[cand_idx]
    else:
        axmin = all_cuboids[:, 0] - all_cuboids[:, 3] / 2
        axmax = all_cuboids[:, 0] + all_cuboids[:, 3] / 2
        aymin = all_cuboids[:, 1] - all_cuboids[:, 4] / 2
        aymax = all_cuboids[:, 1] + all_cuboids[:, 4] / 2
        inside = (axmax >= txmin) & (axmin <= txmax) & (aymax >= tymin) & (aymin <= tymax)
        cand_arr = all_cuboids[inside]
        cand_owners = all_owner[inside]

    drop_self = cand_owners != target_net_name
    cand_owners = cand_owners[drop_self]
    cand_arr = cand_arr[drop_self]
    if len(cand_arr) == 0:
        return []

    # Vectorized pairwise surface distance: (N_t, N_a)
    tx = target_arr[:, 0:1]; ty = target_arr[:, 1:2]
    tw = target_arr[:, 3:4]; th = target_arr[:, 4:5]
    ax = cand_arr[:, 0]; ay = cand_arr[:, 1]
    aw = cand_arr[:, 3]; ah = cand_arr[:, 4]
    dx = np.maximum(np.abs(tx - ax) - (tw + aw) / 2, 0)
    dy = np.maximum(np.abs(ty - ay) - (th + ah) / 2, 0)
    d = np.sqrt(dx * dx + dy * dy)

    closest_t = d.argmin(axis=0)
    closest_d = d.min(axis=0)
    in_range = closest_d <= cutoff_um
    if not in_range.any():
        return []

    sel_cuboids = cand_arr[in_range]
    sel_owners = cand_owners[in_range]
    sel_dist = closest_d[in_range]
    sel_tidx = closest_t[in_range]
    matched_t = target_arr[sel_tidx]

    bs_x = np.maximum(
        np.minimum(matched_t[:, 0] + matched_t[:, 3] / 2,
                   sel_cuboids[:, 0] + sel_cuboids[:, 3] / 2)
        - np.maximum(matched_t[:, 0] - matched_t[:, 3] / 2,
                     sel_cuboids[:, 0] - sel_cuboids[:, 3] / 2),
        0,
    )
    bs_y = np.maximum(
        np.minimum(matched_t[:, 1] + matched_t[:, 4] / 2,
                   sel_cuboids[:, 1] + sel_cuboids[:, 4] / 2)
        - np.maximum(matched_t[:, 1] - matched_t[:, 4] / 2,
                     sel_cuboids[:, 1] - sel_cuboids[:, 4] / 2),
        0,
    )
    broadside = bs_x * bs_y
    lateral = matched_t[:, 5] * np.maximum(bs_x, bs_y)

    aggr_to_closest: dict[str, dict] = {}
    for k in range(len(sel_owners)):
        a_owner = str(sel_owners[k])
        d_k = float(sel_dist[k])
        prior = aggr_to_closest.get(a_owner)
        if prior is None or d_k < prior["dist"]:
            aggr_to_closest[a_owner] = {
                "dist": d_k,
                "broadside": float(broadside[k]),
                "lateral": float(lateral[k]),
                "aggr_layer": int(sel_cuboids[k, 6]),
                "tgt_layer": int(matched_t[k, 6]),
            }

    edges = [
        CouplingEdge(
            aggressor_net=a,
            target_layer=info["tgt_layer"],
            aggressor_layer=info["aggr_layer"],
            surface_dist_um=info["dist"],
            broadside_overlap_um2=info["broadside"],
            lateral_overlap_um2=info["lateral"],
        )
        for a, info in aggr_to_closest.items()
    ]
    edges.sort(key=lambda e: -(e.broadside_overlap_um2 + e.lateral_overlap_um2))
    return edges[:max_edges]


def _layer_eps_array(layer_map: dict, n_layers: int = 10) -> list[float]:
    """Build a 0..n_layers ε array from the legacy layer_map.

    Layer 0 is air (ε=1.0). Layers 1..N follow legacy ordering. Missing layers
    default to 4.0 (typical ILD).
    """
    out = [1.0] * (n_layers + 1)
    # Heuristic: legacy layer_map keys are layer names; we look up by index
    # by matching keys that contain "M<i>".
    for k, v in layer_map.items():
        eps = None
        if isinstance(v, dict):
            eps = v.get("epsilon") or v.get("eps") or v.get("eps_r")
        if eps is None or not isinstance(eps, (int, float)):
            continue
        for i in range(1, n_layers + 1):
            if f"M{i}" in str(k).upper() or f"METAL{i}" in str(k).upper():
                out[i] = float(eps)
                break
    return out


# ============================================================================
# Public entrypoint
# ============================================================================


def build_feature_dataset_for_design(
    def_path: Path,
    spef_path: Path,
    manifest_subset: pd.DataFrame,
    cutoff_um: float = 4.0,
    max_aggr_per_net: int = 768,  # Phase C audit A3: 256 saturated 100% of aes; bump
) -> pd.DataFrame:
    """Build the feature DataFrame for a single design.

    `manifest_subset` is the v3 manifest filtered to this design (column 'split'
    propagates from manifest).
    """
    layer_map = LayerInfoParser(_PROJECT_ROOT / "tool" / "pdk" / "22nm" / "layers" / "layers.info").parse()
    layer_eps = _layer_eps_array(layer_map, n_layers=10)

    print(f"  parsing DEF: {def_path.name}")
    geo = _scan_design_geometry(def_path, layer_map)
    print(f"    {len(geo['nets']):,} nets, {len(geo['vss']):,} VSS cuboids")

    # Phase C A3 #9 fix: build SpatialGrid ONCE per design
    print(f"  building SpatialGrid for {len(geo['all_cuboids']):,} cuboids ...")
    spatial_grid = _SpatialGrid(bin_size_x=4.0, bin_size_y=4.0)
    spatial_grid.build_from_cuboids(geo["all_cuboids"])

    print(f"  parsing SPEF: {spef_path.name}")
    spef_dict = parse_spef_to_dict(spef_path)
    print(f"    {len(spef_dict):,} SPEF nets")

    # Index manifest by net_name → split
    manifest_by_net = {}
    for _, row in manifest_subset.iterrows():
        manifest_by_net.setdefault(str(row["net_name"]), str(row["split"]))

    # We iterate the union of (manifest nets) ∩ (SPEF nets) ∩ (DEF nets).
    common = set(manifest_by_net.keys()) & set(spef_dict.keys()) & set(geo["nets"].keys())
    print(f"    {len(common):,} nets present in all of (manifest, SPEF, DEF)")

    # Local density: for now use design-wide histogram as a proxy
    density = np.zeros(11, dtype=np.float64)
    for arr in geo["nets"].values():
        for i in range(1, 10):
            mask = arr[:, 6] == i
            density[i] += float((arr[mask, 3] * arr[mask, 4]).sum())
    # Window: design bbox area
    if len(geo["all_cuboids"]) > 0:
        xmin, xmax, ymin, ymax = _bbox_from_cuboids(geo["all_cuboids"])
        density_window = max(1.0, (xmax - xmin) * (ymax - ymin))
    else:
        density_window = 1.0

    rows = []
    for i, net_name in enumerate(sorted(common)):
        if i % 5000 == 0:
            print(f"    extracting features ... {i:,}/{len(common):,}")
        target_arr = geo["nets"][net_name]
        spef_rec = spef_dict[net_name]

        edges = _enumerate_coupling_edges(
            target_arr=target_arr,
            all_cuboids=geo["all_cuboids"],
            all_owner=geo["all_owner"],
            target_net_name=net_name,
            cutoff_um=cutoff_um,
            max_edges=max_aggr_per_net,
            spatial_grid=spatial_grid,
        )

        # VSS subset relevant to this net (reuse same enumeration)
        if len(geo["vss"]) > 0:
            txmin, txmax, tymin, tymax = _bbox_from_cuboids(target_arr)
            cutoff = cutoff_um
            txmin -= cutoff; txmax += cutoff; tymin -= cutoff; tymax += cutoff
            vxmin = geo["vss"][:, 0] - geo["vss"][:, 3] / 2
            vxmax = geo["vss"][:, 0] + geo["vss"][:, 3] / 2
            vymin = geo["vss"][:, 1] - geo["vss"][:, 4] / 2
            vymax = geo["vss"][:, 1] + geo["vss"][:, 4] / 2
            inside = (vxmax >= txmin) & (vxmin <= txmax) & (vymax >= tymin) & (vymin <= tymax)
            vss_subset = geo["vss"][inside]
        else:
            vss_subset = np.zeros((0, 7), dtype=np.float64)

        net_geo = NetGeometry(
            net_name=net_name,
            design_name=def_path.stem,
            target_cuboids=_np_to_cuboid_arr(target_arr),
            coupling_edges=edges,
            vss_cuboids=_np_to_cuboid_arr(vss_subset),
            layer_stack_eps=layer_eps,
            fanout=len(spef_rec["coupled_caps"]),  # proxy
            n_layers_total=10,
            ground_plane_layer=0,
            local_density_window_um2=density_window,
            local_metal_area_per_layer_um2=density.tolist(),
        )

        fv = extract_features_from_geometry(net_geo)

        row = {
            "design_name": def_path.stem,
            "net_name": net_name,
            "split": manifest_by_net[net_name],
            "total_cap_fF": spef_rec["total_cap_fF"],
            "c_gnd_fF": spef_rec["ground_cap_fF"],
            "c_cpl_total_fF": spef_rec["c_cpl_total_fF"],
            "total_res_ohm": spef_rec["total_res_ohm"],
            **{f.name: getattr(fv, f.name) for f in NetFeatureVector.__dataclass_fields__.values()},  # noqa: E501
        }
        rows.append(row)

    return pd.DataFrame(rows)


def write_feature_dataset_for_design(
    def_path: Path,
    spef_path: Path,
    manifest_subset: pd.DataFrame,
    out_path: Path,
    cutoff_um: float = 4.0,
) -> int:
    """Build features and write to parquet. Returns row count."""
    df = build_feature_dataset_for_design(
        def_path=def_path,
        spef_path=spef_path,
        manifest_subset=manifest_subset,
        cutoff_um=cutoff_um,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".parquet":
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
    return len(df)
