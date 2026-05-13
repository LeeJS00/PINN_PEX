"""pex_cold.py — TreePEX cold-start: DEF/LEF/layer.info → features → SPEF.

Treats each design as unseen: builds the 67-D feature vector (41 V3 base +
26 V4 H3) on the fly, then runs the 5-seed XGBoost ensemble, writes SPEF,
and compares against golden StarRC. Designs run in parallel.

Feature pipeline:
  V3 (41-D, signal-net hand features): built from raw DEF + LEF + layer.info
    via the same coupling-enumeration logic the training data used
    (archive/pex_v3/src/baselines/feature_dataset.py). Per-net parallelized
    with mp.Pool. **Includes PIN/INST_PORT pseudo-nets in all_owner array**
    to match training-time aggressor counts.

  V4 H3 (26-D, top-K aggressor): the training distribution was computed on
    tile-aggregated cuboids (overlapping tile windows → cuboid duplication
    inflates target_n_cuboids_check and top-K scores by 4-100×). To match
    that distribution we read the per-design tile pkl.gz cache and aggregate
    target/aggressor cuboids per net, then run the V4 _net_features kernel.
    Tile cache is treated as a pre-cached raw-geometry asset; for a brand-new
    design it would be built once via build_dataset.py (cost reported
    separately).

Usage:
    python3 TreePEX/scripts/pex_cold.py                     # tv80s + nova
    python3 TreePEX/scripts/pex_cold.py --design intel22_tv80s_f3
    python3 TreePEX/scripts/pex_cold.py --workers 16
"""
from __future__ import annotations

import argparse
import gc
import gzip
import json
import multiprocessing as mp
import os
import pickle
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.preprocessing.def_parser import DefStreamParser
from src.preprocessing.lef_parser import LefParser
from src.preprocessing.cell_parser import CellLibParser
from src.preprocessing.layer_parser import LayerInfoParser
from src.physics.materials import BEOLMaterialStack

TREEPEX = ROOT / "TreePEX"
MODELS_DIR = TREEPEX / "models"
PRED_DIR = TREEPEX / "outputs" / "predictions"
SPEF_DIR_OUT = TREEPEX / "outputs" / "spef"
COLD_REPORT_DIR = TREEPEX / "outputs" / "cold_reports"
for _p in (PRED_DIR, SPEF_DIR_OUT, COLD_REPORT_DIR):
    _p.mkdir(parents=True, exist_ok=True)

TECH_LEF_PATH = ROOT / "tool" / "pdk" / "22nm" / "tech_lef" / "p1222_js.lef"
CELL_LEF_PATH = ROOT / "tool" / "pdk" / "22nm" / "cell_lef" / "b15_nn.lef"
LAYERS_INFO_PATH = ROOT / "tool" / "pdk" / "22nm" / "layers" / "layers.info"

GOLDEN_SPEF_DIR = ROOT / "golden_data" / "spef_data" / "intel22"
TILE_CACHE_ROOT = Path("/data/PINNPEX/data/processed_v3/intel22")

DESIGNS = {
    "intel22_tv80s_f3": Path("/home/jslee/projects/PEX_SSL/data/raw/def/intel22/intel22_tv80s_f3.def"),
    "intel22_nova_f3":  Path("/home/jslee/projects/PEX_SSL/data/raw/def/intel22/intel22_nova_f3.def"),
}

SEEDS = [42, 0, 1, 2, 3]
POWER_NAMES = {"vss", "vdd", "vcc", "gnd", "vssx", "vccx", "vddx"}
EPS0_FF_UM = 8.8541878128e-3
EPS_Z_V4 = 0.05
TOP_K = 3
MAX_TARGET_CUBS_V4 = 256
MAX_TARGET_CUBS_V3 = 512   # V3-A: broadcast-only target sub-sample cap
CUTOFF_UM = 4.0
SPATIAL_BIN_UM = 4.0
SLACK_UM_V4 = 5.0
MAX_AGGR_PER_NET = 768
N_LAYERS_HIST = 9
N_LAYERS_EPS = 10

# V4 cuboid tensor column indices (FeatureTensorizer output)
CB_X, CB_Y, CB_Z, CB_W, CB_H, CB_D, CB_SEM, CB_LOG, CB_EPS = range(9)

_LAYER_RE = re.compile(r"[mM](\d+)")


def _layer_str_to_idx(name) -> int:
    if name is None:
        return 0
    m = _LAYER_RE.match(str(name))
    return int(m.group(1)) if m else 0


def _bbox_xy(arr: np.ndarray):
    if len(arr) == 0:
        return 0.0, 0.0, 0.0, 0.0
    xmin = float((arr[:, 0] - arr[:, 3] / 2).min())
    xmax = float((arr[:, 0] + arr[:, 3] / 2).max())
    ymin = float((arr[:, 1] - arr[:, 4] / 2).min())
    ymax = float((arr[:, 1] + arr[:, 4] / 2).max())
    return xmin, xmax, ymin, ymax


# ============================================================================
# DEF parse — match training behavior:
#   - PIN_/INST_PORT_ pseudo-nets kept in all_owner (training-time
#     `_scan_design_geometry` did not filter them; they appear as additional
#     aggressor identities)
#   - duplicate net names: overwrite (training used dict assignment)
# Signal target nets = nets[*] excluding POWER_NAMES.
# ============================================================================

def scan_design(def_path: Path, layer_map, tech_lef, cell_lib) -> dict:
    parser = DefStreamParser(str(def_path), layer_map, tech_lef, cell_lib)
    nets: Dict[str, np.ndarray] = {}
    vss_rows: List[np.ndarray] = []

    for net_name, cuboids, segments in parser.parse():
        if cuboids is None or cuboids.size == 0:
            continue
        if cuboids.shape[1] == 6:
            layer = _layer_str_to_idx(segments[0].get("layer")) if segments else 0
            layer_col = np.full((len(cuboids), 1), layer, dtype=np.float64)
            cuboids = np.hstack([cuboids, layer_col])
        if net_name.lower() in POWER_NAMES:
            vss_rows.append(cuboids)
        else:
            nets[net_name] = cuboids  # overwrite to match training

    vss = np.vstack(vss_rows) if vss_rows else np.zeros((0, 7), dtype=np.float64)

    # all_owner = union of every per-net cuboid (signal + pin pseudo)
    all_rows = []
    all_owner_list: List[str] = []
    target_names: List[str] = []
    for n, arr in nets.items():
        all_rows.append(arr)
        all_owner_list.extend([n] * len(arr))
        if "PIN_" in n.upper() or "INST_PORT_" in n.upper():
            continue
        target_names.append(n)
    if all_rows:
        all_cuboids = np.vstack(all_rows)
        all_owner = np.asarray(all_owner_list, dtype=object)
    else:
        all_cuboids = np.zeros((0, 7), dtype=np.float64)
        all_owner = np.array([], dtype=object)
    target_set = set(target_names)
    return {
        "nets": nets,
        "vss": vss,
        "all_cuboids": all_cuboids,
        "all_owner": all_owner,
        "target_set": target_set,
    }


# ============================================================================
# Spatial grid (xy bbox bucketing for O(1) candidate query)
# ============================================================================

class SpatialGrid:
    __slots__ = ("bx", "by", "grid")

    def __init__(self, bx: float = SPATIAL_BIN_UM, by: float = SPATIAL_BIN_UM):
        self.bx = bx
        self.by = by
        self.grid: Dict[Tuple[int, int], List[int]] = defaultdict(list)

    def build(self, cuboids: np.ndarray) -> None:
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

    def query_bbox(self, xmin, xmax, ymin, ymax) -> np.ndarray:
        min_x = int(np.floor(xmin / self.bx))
        max_x = int(np.floor(xmax / self.bx))
        min_y = int(np.floor(ymin / self.by))
        max_y = int(np.floor(ymax / self.by))
        seen = set()
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                bkt = self.grid.get((x, y))
                if bkt is not None:
                    seen.update(bkt)
        if not seen:
            return np.empty(0, dtype=np.int64)
        return np.fromiter(seen, dtype=np.int64)


# ============================================================================
# V3 41-D feature extraction (DEF-only)
# Fork-shared globals filled by init_worker_v3.
# ============================================================================

_V3_NETS = None
_V3_VSS = None
_V3_ALL_CUBS = None
_V3_ALL_OWNER = None
_V3_GRID: SpatialGrid = None
_V3_EPS_BY_LAYER = None
_V3_DENSITY_PER_LAYER = None
_V3_DENSITY_WINDOW = None


def init_worker_v3(geo, grid, eps_by_layer, density_per_layer, density_window):
    global _V3_NETS, _V3_VSS, _V3_ALL_CUBS, _V3_ALL_OWNER, _V3_GRID, _V3_EPS_BY_LAYER
    global _V3_DENSITY_PER_LAYER, _V3_DENSITY_WINDOW
    _V3_NETS = geo["nets"]
    _V3_VSS = geo["vss"]
    _V3_ALL_CUBS = geo["all_cuboids"]
    _V3_ALL_OWNER = geo["all_owner"]
    _V3_GRID = grid
    _V3_EPS_BY_LAYER = eps_by_layer
    _V3_DENSITY_PER_LAYER = density_per_layer
    _V3_DENSITY_WINDOW = density_window


def _layer_eps_array(layer_map: dict, n_layers: int = N_LAYERS_EPS) -> List[float]:
    out = [1.0] * (n_layers + 1)
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


# Lazy torch handle and persistent CUDA device — only initialized when the
# user opts into the GPU path. Keeping it module-level + lazy avoids forcing
# every cold-start invocation to import torch.
_V3_GPU_DEVICE = None
_V3_GPU_TORCH = None
_V3_GPU_ALL_CUBS_XYWH = None   # persistent torch tensor: every cuboid's (x,y,w,h)
                               # on GPU. Lets per-net dispatch fall back to a
                               # ~kB gather index transfer instead of shipping
                               # `cand_arr` (10s-100s of MB) every call.


def _v3_gpu_init(device: str = "cuda:0"):
    """Idempotent torch + CUDA init for the V3 closest-pair kernel."""
    global _V3_GPU_DEVICE, _V3_GPU_TORCH
    if _V3_GPU_TORCH is not None:
        return _V3_GPU_TORCH
    import torch as _torch
    if not _torch.cuda.is_available():
        raise SystemExit("V3 GPU path requested but torch reports no CUDA device")
    _V3_GPU_TORCH = _torch
    _V3_GPU_DEVICE = _torch.device(device)
    return _V3_GPU_TORCH


def _v3_gpu_upload_all_cubs(all_cubs: np.ndarray):
    """One-time upload of every cuboid's xy + wh to GPU. Per-net broadcast
    later gathers from this tensor by index, avoiding O(N_c × bytes) transfers
    on every net.
    """
    global _V3_GPU_ALL_CUBS_XYWH
    _torch = _V3_GPU_TORCH
    arr = np.ascontiguousarray(all_cubs[:, [0, 1, 3, 4]], dtype=np.float32)
    _V3_GPU_ALL_CUBS_XYWH = _torch.from_numpy(arr).to(_V3_GPU_DEVICE)


def _v3_compute_closest_gpu(target_arr_bc: np.ndarray, cand_arr: np.ndarray,
                            cutoff_um: float,
                            cand_idx_into_all: np.ndarray = None):
    """torch-on-CUDA replacement for the (N_t × N_c) numpy broadcast.
    Matches `_v3_compute_closest_cpu` modulo fp32 rounding.

    When `cand_idx_into_all` is provided and the persistent all-cubs tensor
    has been uploaded, the candidate side is gathered on GPU (only int64
    index ships across PCIe) — this is the round-2.2 batched-dispatch
    optimization, ≈10× faster per net than the legacy fresh-transfer path.
    """
    _torch = _V3_GPU_TORCH
    dev = _V3_GPU_DEVICE
    t_np = target_arr_bc[:, [0, 1, 3, 4]].astype(np.float32, copy=False)
    t_t = _torch.from_numpy(t_np).to(dev, non_blocking=True)
    if cand_idx_into_all is not None and _V3_GPU_ALL_CUBS_XYWH is not None:
        idx_t = _torch.as_tensor(cand_idx_into_all, dtype=_torch.int64, device=dev)
        c_t = _V3_GPU_ALL_CUBS_XYWH.index_select(0, idx_t)
    else:
        c_np = cand_arr[:, [0, 1, 3, 4]].astype(np.float32, copy=False)
        c_t = _torch.from_numpy(c_np).to(dev, non_blocking=True)
    dx = _torch.clamp(
        _torch.abs(t_t[:, 0:1] - c_t[None, :, 0])
        - (t_t[:, 2:3] + c_t[None, :, 2]) / 2, min=0)
    dy = _torch.clamp(
        _torch.abs(t_t[:, 1:2] - c_t[None, :, 1])
        - (t_t[:, 3:4] + c_t[None, :, 3]) / 2, min=0)
    d_mat = _torch.sqrt(dx * dx + dy * dy)
    closest_d_t, closest_t_t = d_mat.min(dim=0)
    closest_d = closest_d_t.cpu().numpy().astype(np.float64, copy=False)
    closest_t = closest_t_t.cpu().numpy().astype(np.int64, copy=False)
    in_range = closest_d <= cutoff_um
    return closest_t, closest_d, in_range


def _v3_compute_closest_batched_gpu(jobs, cutoff_um: float,
                                    mem_budget_bytes: int = 4_000_000_000):
    """Run the per-net (N_t × N_c) broadcast for several nets in one GPU
    launch. `jobs` is a list of dicts {'target': float32[N_t,4],
    'cand_idx_into_all': int64[N_c]}.

    Greedy memory-budget batcher: appends jobs to the current batch while
    B × max_t × max_c × 4B (the broadcast tensor size) stays under
    `mem_budget_bytes`. Yields per-job (closest_t, closest_d, in_range)
    in the same order as `jobs`.
    """
    _torch = _V3_GPU_TORCH
    dev = _V3_GPU_DEVICE
    all_cubs_t = _V3_GPU_ALL_CUBS_XYWH
    results = [None] * len(jobs)

    def _run_batch(batch_indices, max_t, max_c):
        B = len(batch_indices)
        # Pad targets to (B, max_t, 4) and cand index to (B, max_c).
        t_padded = np.zeros((B, max_t, 4), dtype=np.float32)
        c_padded = np.zeros((B, max_c), dtype=np.int64)
        n_t_per = np.zeros(B, dtype=np.int64)
        n_c_per = np.zeros(B, dtype=np.int64)
        for k, j_idx in enumerate(batch_indices):
            j = jobs[j_idx]
            t = j["target"]
            ci = j["cand_idx_into_all"]
            t_padded[k, :t.shape[0]] = t
            c_padded[k, :ci.shape[0]] = ci
            n_t_per[k] = t.shape[0]
            n_c_per[k] = ci.shape[0]
        t_t = _torch.from_numpy(t_padded).to(dev)
        ci_t = _torch.from_numpy(c_padded).to(dev)
        c_gather = all_cubs_t.index_select(0, ci_t.reshape(-1)).reshape(B, max_c, 4)
        dx = _torch.clamp(
            (t_t[:, :, None, 0] - c_gather[:, None, :, 0]).abs()
            - (t_t[:, :, None, 2] + c_gather[:, None, :, 2]) / 2, min=0)
        dy = _torch.clamp(
            (t_t[:, :, None, 1] - c_gather[:, None, :, 1]).abs()
            - (t_t[:, :, None, 3] + c_gather[:, None, :, 3]) / 2, min=0)
        d_mat = _torch.sqrt(dx * dx + dy * dy)
        # Mask padding rows on the target axis to +inf so argmin ignores them.
        t_active = (_torch.arange(max_t, device=dev)[None, :]
                    < _torch.as_tensor(n_t_per, device=dev)[:, None])
        d_mat = _torch.where(t_active[:, :, None], d_mat,
                             _torch.tensor(float("inf"), device=dev))
        closest_d_t, closest_t_t = d_mat.min(dim=1)  # both (B, max_c)
        cd_np = closest_d_t.cpu().numpy()
        ct_np = closest_t_t.cpu().numpy()
        for k, j_idx in enumerate(batch_indices):
            n_c = int(n_c_per[k])
            cd_k = cd_np[k, :n_c].astype(np.float64, copy=False)
            ct_k = ct_np[k, :n_c].astype(np.int64, copy=False)
            in_range_k = cd_k <= cutoff_um
            results[j_idx] = (ct_k, cd_k, in_range_k)

    cur: List[int] = []
    cur_max_t = 0
    cur_max_c = 0
    for j_idx, j in enumerate(jobs):
        n_t = j["target"].shape[0]
        n_c = j["cand_idx_into_all"].shape[0]
        if n_t == 0 or n_c == 0:
            results[j_idx] = (np.zeros(0, dtype=np.int64),
                              np.full(n_c, np.inf, dtype=np.float64),
                              np.zeros(n_c, dtype=bool))
            continue
        new_max_t = max(cur_max_t, n_t)
        new_max_c = max(cur_max_c, n_c)
        cost = (len(cur) + 1) * new_max_t * new_max_c * 4
        if cur and cost > mem_budget_bytes:
            _run_batch(cur, cur_max_t, cur_max_c)
            cur = [j_idx]
            cur_max_t = n_t
            cur_max_c = n_c
        else:
            cur.append(j_idx)
            cur_max_t = new_max_t
            cur_max_c = new_max_c
    if cur:
        _run_batch(cur, cur_max_t, cur_max_c)
    return results


def _v3_compute_closest_cpu(target_arr_bc: np.ndarray, cand_arr: np.ndarray,
                            cutoff_um: float):
    tx2 = target_arr_bc[:, 0:1]; ty2 = target_arr_bc[:, 1:2]
    tw2 = target_arr_bc[:, 3:4]; th2 = target_arr_bc[:, 4:5]
    ax = cand_arr[:, 0]; ay = cand_arr[:, 1]
    aw = cand_arr[:, 3]; ah = cand_arr[:, 4]
    dx = np.maximum(np.abs(tx2 - ax) - (tw2 + aw) / 2, 0)
    dy = np.maximum(np.abs(ty2 - ay) - (th2 + ah) / 2, 0)
    d_mat = np.sqrt(dx * dx + dy * dy)
    closest_t = d_mat.argmin(axis=0).astype(np.int64, copy=False)
    closest_d = d_mat.min(axis=0)
    in_range = closest_d <= cutoff_um
    return closest_t, closest_d, in_range


# Per-net broadcast backend selector. Default = CPU; set to True in
# `extract_v3_features` when the GPU path is enabled. Lives at module scope
# because `_v3_per_net` is a top-level worker entry point.
_V3_USE_GPU = False
# When the batched-GPU precompute is active, this dict maps net_name to a
# pre-computed (closest_t, closest_d, in_range) tuple, and `_v3_per_net`
# skips its own broadcast and consults the cache.
_V3_CLOSEST_CACHE: Dict[str, tuple] = None

# Round 3 (V3 algorithmic redesign): replace the global (N_t × N_c) numpy
# broadcast with a per-target-cuboid SpatialGrid query + inline per-aggressor
# closest reduction. Avoids materializing the (N_t × N_c) distance matrix.
# Gate: only the long-tail nets (>= _V3_PER_TARGET_PAIR_THRESHOLD pairs)
# switch to the new path; tiny nets stay on the well-tuned numpy broadcast
# (Python grid_query overhead per-target wins only when the avoided pair
# count dwarfs the broadcast).
#
# Threshold 30 M ≥ tv80s max pair count (512 × 52,662 ≈ 27 M) so tv80s
# stays on legacy entirely (avoids the 30 % V3 regression measured at
# threshold 10 M). nova long-tail (N_c ≥ 120 k) → pair ≥ 61 M, still
# triggers per-target. Measured nova win at threshold 10 M: pipeline
# 7,182 → 5,346 s (1.67 × on V3, MAPE drift +0.007 pp tot ✅). Lifting
# to 30 M only declassifies the marginal tv80s long-tail; nova benefit
# preserved.
_V3_PER_TARGET_PAIR_THRESHOLD = 30_000_000
# Mode: "auto" → threshold-gated, "per_target" → always new, "legacy" → always old.
_V3_PER_TARGET_MODE: str = "auto"


def _v3_aggregate_per_target(net_name: str, target_arr_bc: np.ndarray,
                             cutoff_um: float) -> List[dict]:
    """Per-target-cuboid alternative to `_v3_compute_closest_cpu` + the
    per-aggressor argmin aggregator. Produces the same `edges` list shape
    as the legacy path. Two semantic guarantees:

    1. **Equivalence**: aggregator output (per-aggressor closest pair) is
       `min over (t, c) pairs within CUTOFF`; numpy broadcast reduced over
       target axis first then over candidates yields the same minimum.
    2. **Tie-break parity**: SpatialGrid `seen` is a Python set (iteration
       order is impl-defined). We `sub_idx.sort()` so per-target candidate
       ordering is deterministic; the per-aggressor update uses strict `<`
       so a first-found pair survives equal-distance follow-ups, matching
       the legacy `argmin(axis=0)` first-index tie rule on the target side.
    """
    grid = _V3_GRID
    all_cubs = _V3_ALL_CUBS
    all_owner = _V3_ALL_OWNER
    n_t = len(target_arr_bc)
    aggr_to_closest: Dict[str, dict] = {}

    for ti in range(n_t):
        tx = float(target_arr_bc[ti, 0]); ty = float(target_arr_bc[ti, 1])
        tw = float(target_arr_bc[ti, 3]); th = float(target_arr_bc[ti, 4])
        tl = int(target_arr_bc[ti, 6])
        t_lat_scale = float(target_arr_bc[ti, 5])

        local_idx = grid.query_bbox(
            tx - tw / 2 - cutoff_um, tx + tw / 2 + cutoff_um,
            ty - th / 2 - cutoff_um, ty + th / 2 + cutoff_um,
        )
        if len(local_idx) == 0:
            continue
        local_owners = all_owner[local_idx]
        keep = local_owners != net_name
        if not keep.any():
            continue
        sub_idx = local_idx[keep]
        sub_idx.sort()
        sub_owners = all_owner[sub_idx]
        sub_cubs = all_cubs[sub_idx]

        dx = np.maximum(np.abs(sub_cubs[:, 0] - tx) - (sub_cubs[:, 3] + tw) / 2, 0.0)
        dy = np.maximum(np.abs(sub_cubs[:, 1] - ty) - (sub_cubs[:, 4] + th) / 2, 0.0)
        dist = np.sqrt(dx * dx + dy * dy)
        in_range = dist <= cutoff_um
        if not in_range.any():
            continue
        sel_cubs = sub_cubs[in_range]
        sel_owners = sub_owners[in_range]
        sel_dist = dist[in_range]
        bs_x = np.maximum(
            np.minimum(tx + tw / 2, sel_cubs[:, 0] + sel_cubs[:, 3] / 2)
            - np.maximum(tx - tw / 2, sel_cubs[:, 0] - sel_cubs[:, 3] / 2),
            0.0,
        )
        bs_y = np.maximum(
            np.minimum(ty + th / 2, sel_cubs[:, 1] + sel_cubs[:, 4] / 2)
            - np.maximum(ty - th / 2, sel_cubs[:, 1] - sel_cubs[:, 4] / 2),
            0.0,
        )
        broadside = bs_x * bs_y
        lateral = t_lat_scale * np.maximum(bs_x, bs_y)
        for j in range(sel_owners.shape[0]):
            a_owner = str(sel_owners[j])
            d_j = float(sel_dist[j])
            prior = aggr_to_closest.get(a_owner)
            if prior is None or d_j < prior["dist"]:
                aggr_to_closest[a_owner] = {
                    "dist": d_j,
                    "broadside": float(broadside[j]),
                    "lateral": float(lateral[j]),
                    "aggr_layer": int(sel_cubs[j, 6]),
                    "tgt_layer": tl,
                }

    edges = [
        {
            "aggressor_net": a,
            "tgt_layer": info["tgt_layer"],
            "aggr_layer": info["aggr_layer"],
            "surface_dist_um": info["dist"],
            "broadside_overlap_um2": info["broadside"],
            "lateral_overlap_um2": info["lateral"],
        }
        for a, info in aggr_to_closest.items()
    ]
    edges.sort(key=lambda e: -(e["broadside_overlap_um2"] + e["lateral_overlap_um2"]))
    return edges[:MAX_AGGR_PER_NET]


def _v3_phase_a_for_net(net_name: str):
    """Run the Phase-A (spatial-query + V3-A target subsample) prep work
    that the batched GPU precompute needs. Returns
    (target_arr_bc_xywh, cand_idx_into_all) or None if there is no work.
    """
    target_arr = _V3_NETS.get(net_name)
    if target_arr is None or len(target_arr) == 0:
        return None
    x_min = float((target_arr[:, 0] - target_arr[:, 3] / 2).min())
    x_max = float((target_arr[:, 0] + target_arr[:, 3] / 2).max())
    y_min = float((target_arr[:, 1] - target_arr[:, 4] / 2).min())
    y_max = float((target_arr[:, 1] + target_arr[:, 4] / 2).max())
    cand_idx = _V3_GRID.query_bbox(x_min - CUTOFF_UM, x_max + CUTOFF_UM,
                                   y_min - CUTOFF_UM, y_max + CUTOFF_UM)
    if len(cand_idx) == 0:
        return None
    cand_owners = _V3_ALL_OWNER[cand_idx]
    keep = cand_owners != net_name
    cand_idx_into_all = cand_idx[keep]
    if len(cand_idx_into_all) == 0:
        return None
    if len(target_arr) > MAX_TARGET_CUBS_V3:
        rng_t = np.random.RandomState(hash(net_name) & 0xFFFFFFFF)
        sub_idx = rng_t.choice(len(target_arr), MAX_TARGET_CUBS_V3, replace=False)
        target_arr_bc = target_arr[sub_idx]
    else:
        target_arr_bc = target_arr
    target_xywh = target_arr_bc[:, [0, 1, 3, 4]].astype(np.float32, copy=False)
    return target_xywh, cand_idx_into_all


def _v3_precompute_closest_chunk(chunk_nets: List[str], mem_budget_bytes: int = 8_000_000_000):
    """Run Phase-A on every net in the chunk, then a single batched GPU
    call. Returns a dict {net_name: dict(...)} containing both the
    closest-pair result AND the Phase-A intermediates so `_v3_per_net`
    can skip its own spatial query when the cache is hot.
    """
    jobs = []
    job_meta: List[tuple] = []  # (net_name, target_arr_bc_indices_into_target, cand_idx_into_all)
    for nm in chunk_nets:
        prep = _v3_phase_a_for_net(nm)
        if prep is None:
            continue
        target_xywh, cand_idx_into_all = prep
        jobs.append({"target": target_xywh, "cand_idx_into_all": cand_idx_into_all})
        job_meta.append((nm, target_xywh, cand_idx_into_all))
    if not jobs:
        return {}
    results = _v3_compute_closest_batched_gpu(jobs, CUTOFF_UM, mem_budget_bytes)
    out = {}
    for (nm, txywh, cidx), (ct, cd, ir) in zip(job_meta, results):
        out[nm] = {
            "closest_t": ct,
            "closest_d": cd,
            "in_range": ir,
            "cand_idx_into_all": cidx,
        }
    return out


def _v3_per_net(net_name: str) -> Dict[str, float]:
    target_arr = _V3_NETS.get(net_name)
    if target_arr is None or len(target_arr) == 0:
        return {}
    feats: Dict[str, float] = {"net_name": net_name}

    n = len(target_arr)
    max_ext = np.maximum.reduce([target_arr[:, 3], target_arr[:, 4], target_arr[:, 5]])
    feats["total_wire_length_um"] = float(max_ext.sum())
    feats["total_metal_area_um2"] = float((target_arr[:, 3] * target_arr[:, 4]).sum())
    feats["n_cuboids"] = float(n)

    x_min = float((target_arr[:, 0] - target_arr[:, 3] / 2).min())
    x_max = float((target_arr[:, 0] + target_arr[:, 3] / 2).max())
    y_min = float((target_arr[:, 1] - target_arr[:, 4] / 2).min())
    y_max = float((target_arr[:, 1] + target_arr[:, 4] / 2).max())
    z_min = float((target_arr[:, 2] - target_arr[:, 5] / 2).min())
    z_max = float((target_arr[:, 2] + target_arr[:, 5] / 2).max())
    feats["bbox_xy_um2"] = (x_max - x_min) * (y_max - y_min)
    feats["bbox_z_um"] = z_max - z_min
    feats["aspect_ratio"] = (x_max - x_min) / max(y_max - y_min, 1e-6) if x_max > x_min else 1.0

    layer_idx = np.clip(target_arr[:, 6].astype(np.int64), 1, N_LAYERS_HIST)
    hist, _ = np.histogram(layer_idx, bins=np.arange(1, N_LAYERS_HIST + 2))
    for i in range(8):
        feats[f"layer_hist_M{i+1}"] = float(hist[i])
    feats["layer_hist_M9_plus"] = float(hist[8])

    # Spatial query: target bbox + cutoff
    cached_entry = _V3_CLOSEST_CACHE.get(net_name) if _V3_CLOSEST_CACHE is not None else None
    if cached_entry is not None:
        # Reuse the spatial query already done by `_v3_phase_a_for_net`.
        cand_idx_into_all = cached_entry["cand_idx_into_all"]
        cand_arr = _V3_ALL_CUBS[cand_idx_into_all]
        cand_owners = _V3_ALL_OWNER[cand_idx_into_all]
    else:
        cand_idx = _V3_GRID.query_bbox(x_min - CUTOFF_UM, x_max + CUTOFF_UM,
                                       y_min - CUTOFF_UM, y_max + CUTOFF_UM)
        if len(cand_idx) == 0:
            cand_arr = np.zeros((0, 7), dtype=np.float64)
            cand_owners = np.array([], dtype=object)
            cand_idx_into_all = None
        else:
            cand_owners = _V3_ALL_OWNER[cand_idx]
            keep = cand_owners != net_name
            cand_idx_into_all = cand_idx[keep]
            cand_arr = _V3_ALL_CUBS[cand_idx_into_all]
            cand_owners = cand_owners[keep]
        # V3-B (cand cap) was withdrawn after the first patch demolished
        # n_aggressor_nets / fanout (R² = -5.72). Aggressor identities are
        # carried by ALL cuboid rows; any count-based cap loses entire
        # aggressor nets at the tail. Speedup comes from V3-A alone (target
        # subsample) until a V3-D-style spatial-bucket sweep replaces this.

    edges: List[dict] = []
    if len(cand_arr) > 0:
        # V3-A: broadcast-only sub-sample of target rows. Sum / scalar
        # features above used the full target_arr; closest-pair stats only
        # need a representative subset.
        if len(target_arr) > MAX_TARGET_CUBS_V3:
            rng_t = np.random.RandomState(hash(net_name) & 0xFFFFFFFF)
            sub_idx = rng_t.choice(len(target_arr), MAX_TARGET_CUBS_V3, replace=False)
            target_arr_bc = target_arr[sub_idx]
        else:
            target_arr_bc = target_arr

        # Round 3 gate: per-target-cuboid grid path for long-tail nets.
        # Skipped when batched-GPU cache is hot (it already has the result).
        n_pairs = len(target_arr_bc) * len(cand_arr)
        use_per_target = (
            cached_entry is None
            and (_V3_PER_TARGET_MODE == "per_target"
                 or (_V3_PER_TARGET_MODE == "auto"
                     and n_pairs >= _V3_PER_TARGET_PAIR_THRESHOLD))
        )

        if use_per_target:
            edges = _v3_aggregate_per_target(net_name, target_arr_bc, CUTOFF_UM)
        else:
            if cached_entry is not None:
                closest_t = cached_entry["closest_t"]
                closest_d = cached_entry["closest_d"]
                in_range = cached_entry["in_range"]
            elif _V3_USE_GPU and len(target_arr_bc) * len(cand_arr) >= 4096:
                closest_t, closest_d, in_range = _v3_compute_closest_gpu(
                    target_arr_bc, cand_arr, CUTOFF_UM,
                    cand_idx_into_all=cand_idx_into_all)
            else:
                closest_t, closest_d, in_range = _v3_compute_closest_cpu(
                    target_arr_bc, cand_arr, CUTOFF_UM)
            if in_range.any():
                sel_cuboids = cand_arr[in_range]
                sel_owners = cand_owners[in_range]
                sel_dist = closest_d[in_range]
                sel_tidx = closest_t[in_range]
                matched_t = target_arr_bc[sel_tidx]
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
                aggr_to_closest: Dict[str, dict] = {}
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
                    {
                        "aggressor_net": a,
                        "tgt_layer": info["tgt_layer"],
                        "aggr_layer": info["aggr_layer"],
                        "surface_dist_um": info["dist"],
                        "broadside_overlap_um2": info["broadside"],
                        "lateral_overlap_um2": info["lateral"],
                    }
                    for a, info in aggr_to_closest.items()
                ]
                edges.sort(key=lambda e: -(e["broadside_overlap_um2"] + e["lateral_overlap_um2"]))
                edges = edges[:MAX_AGGR_PER_NET]

    feats["n_aggressor_nets"] = float(len({e["aggressor_net"] for e in edges}))
    if edges:
        bs = np.array([e["broadside_overlap_um2"] for e in edges], dtype=np.float64)
        lat = np.array([e["lateral_overlap_um2"] for e in edges], dtype=np.float64)
        dist = np.array([e["surface_dist_um"] for e in edges], dtype=np.float64)
        feats["broadside_overlap_total_um2"] = float(bs.sum())
        feats["broadside_overlap_p95_um2"] = float(np.percentile(bs, 95))
        feats["lateral_overlap_total_um2"] = float(lat.sum())
        feats["lateral_overlap_p95_um2"] = float(np.percentile(lat, 95))
        feats["spacing_min_um"] = float(dist.min())
        feats["spacing_p25_um"] = float(np.percentile(dist, 25))
        feats["spacing_p50_um"] = float(np.percentile(dist, 50))
        feats["spacing_p95_um"] = float(np.percentile(dist, 95))
        feats["n_edges_lt_1um"] = float((dist < 1.0).sum())
        feats["n_edges_1_to_3um"] = float(((dist >= 1.0) & (dist < 3.0)).sum())
        feats["n_edges_3_to_4um"] = float(((dist >= 3.0) & (dist < 4.0)).sum())
    else:
        feats["broadside_overlap_total_um2"] = 0.0
        feats["broadside_overlap_p95_um2"] = 0.0
        feats["lateral_overlap_total_um2"] = 0.0
        feats["lateral_overlap_p95_um2"] = 0.0
        feats["spacing_min_um"] = float("nan")
        feats["spacing_p25_um"] = float("nan")
        feats["spacing_p50_um"] = float("nan")
        feats["spacing_p95_um"] = float("nan")
        feats["n_edges_lt_1um"] = 0.0
        feats["n_edges_1_to_3um"] = 0.0
        feats["n_edges_3_to_4um"] = 0.0

    # VSS subset
    if len(_V3_VSS) > 0:
        txmin, txmax, tymin, tymax = x_min, x_max, y_min, y_max
        vxmin = _V3_VSS[:, 0] - _V3_VSS[:, 3] / 2
        vxmax = _V3_VSS[:, 0] + _V3_VSS[:, 3] / 2
        vymin = _V3_VSS[:, 1] - _V3_VSS[:, 4] / 2
        vymax = _V3_VSS[:, 1] + _V3_VSS[:, 4] / 2
        inter = (vxmax >= txmin - CUTOFF_UM) & (vxmin <= txmax + CUTOFF_UM) \
                & (vymax >= tymin - CUTOFF_UM) & (vymin <= tymax + CUTOFF_UM)
        vss_subset = _V3_VSS[inter]
    else:
        vss_subset = np.zeros((0, 7), dtype=np.float64)

    feats["vss_n_cuboids"] = float(len(vss_subset))
    feats["vss_total_metal_area_um2"] = float((vss_subset[:, 3] * vss_subset[:, 4]).sum()) if len(vss_subset) else 0.0

    s13 = s45 = s6p = 0.0
    if len(vss_subset) > 0:
        vxmin2 = vss_subset[:, 0] - vss_subset[:, 3] / 2
        vxmax2 = vss_subset[:, 0] + vss_subset[:, 3] / 2
        vymin2 = vss_subset[:, 1] - vss_subset[:, 4] / 2
        vymax2 = vss_subset[:, 1] + vss_subset[:, 4] / 2
        inter2 = (vxmax2 >= x_min) & (vxmin2 <= x_max) & (vymax2 >= y_min) & (vymin2 <= y_max)
        if inter2.any():
            vw = vss_subset[inter2, 3]; vh = vss_subset[inter2, 4]
            vl = vss_subset[inter2, 6].astype(np.int64)
            areas = vw * vh
            s13 = float(areas[(vl >= 1) & (vl <= 3)].sum())
            s45 = float(areas[(vl >= 4) & (vl <= 5)].sum())
            s6p = float(areas[vl >= 6].sum())
    feats["vss_shield_M1_M3"] = s13
    feats["vss_shield_M4_M5"] = s45
    feats["vss_shield_M6_plus"] = s6p

    feats["fanout"] = float(len({e["aggressor_net"] for e in edges}))

    eps_arr = np.asarray(_V3_EPS_BY_LAYER, dtype=np.float64)
    eps_pos = eps_arr[eps_arr > 0]
    feats["eps_min"] = float(eps_pos.min()) if len(eps_pos) > 0 else 1.0
    feats["eps_max"] = float(eps_pos.max()) if len(eps_pos) > 0 else 1.0
    feats["eps_mean"] = float(eps_pos.mean()) if len(eps_pos) > 0 else 1.0
    feats["n_layers_present"] = float((hist > 0).sum())

    win = max(_V3_DENSITY_WINDOW, 1e-6)
    mpl = _V3_DENSITY_PER_LAYER
    if len(mpl) >= 9:
        feats["density_M1_M3"] = float(mpl[1:4].sum() / win)
        feats["density_M4_M5"] = float(mpl[4:6].sum() / win)
        feats["density_M6_plus"] = float(mpl[6:].sum() / win)
    else:
        feats["density_M1_M3"] = float("nan")
        feats["density_M4_M5"] = float("nan")
        feats["density_M6_plus"] = float("nan")

    compact_gnd = 0.0
    for i in range(n):
        li = int(target_arr[i, 6])
        d_layers = max(1, li)
        d_um = max(0.05, d_layers * 0.1)
        eps_r = float(_V3_EPS_BY_LAYER[li]) if 0 <= li < len(_V3_EPS_BY_LAYER) else 4.0
        A = float(target_arr[i, 3] * target_arr[i, 4])
        compact_gnd += EPS0_FF_UM * eps_r * A / d_um
    feats["compact_gnd_estimate_fF"] = compact_gnd

    compact_cpl = 0.0
    for e in edges:
        d_um = max(0.05, e["surface_dist_um"])
        l1 = int(e["tgt_layer"]); l2 = int(e["aggr_layer"])
        e1 = float(_V3_EPS_BY_LAYER[l1]) if 0 <= l1 < len(_V3_EPS_BY_LAYER) else 4.0
        e2 = float(_V3_EPS_BY_LAYER[l2]) if 0 <= l2 < len(_V3_EPS_BY_LAYER) else 4.0
        A = e["lateral_overlap_um2"] + e["broadside_overlap_um2"]
        compact_cpl += EPS0_FF_UM * 0.5 * (e1 + e2) * A / d_um
    feats["compact_cpl_estimate_total_fF"] = compact_cpl

    return feats


# ============================================================================
# V4 H3 26-D from tile pkl.gz cache (matches training distribution)
# Lifted from archive/pex_v4/scripts/29_extract_new_features.py
# ============================================================================

def _load_tile(tile_path: Path):
    try:
        with gzip.open(tile_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _v4_net_features(target_cubs: np.ndarray, agg_groups: Dict[str, np.ndarray]) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    if target_cubs.shape[0] > MAX_TARGET_CUBS_V4:
        idx = np.random.RandomState(42).choice(target_cubs.shape[0], MAX_TARGET_CUBS_V4, replace=False)
        target_cubs = target_cubs[idx]
    feats["target_n_cuboids_check"] = int(target_cubs.shape[0])

    if target_cubs.shape[0] == 0:
        for k in range(1, TOP_K + 1):
            for col in ("score", "overlap_um2", "min_xy_dist_um", "mean_dz_um",
                        "agg_size_um2", "layer_diff_flag"):
                feats[f"top{k}_{col}"] = 0.0
        for col in ("agg_count_above_target_z", "agg_count_below_target_z",
                    "agg_n_distinct", "topk_score_concentration"):
            feats[col] = 0.0
        for r in (1.0, 3.0, 5.0):
            feats[f"agg_count_within_{int(r)}um_xyz"] = 0.0
        return feats

    target_z_mean = float(target_cubs[:, CB_Z].mean())
    tx = target_cubs[:, None, CB_X]
    ty = target_cubs[:, None, CB_Y]
    tz = target_cubs[:, None, CB_Z]
    tw = target_cubs[:, None, CB_W]
    th = target_cubs[:, None, CB_H]
    teps = target_cubs[:, None, CB_EPS]

    txc = target_cubs[:, CB_X]; tyc = target_cubs[:, CB_Y]
    tw_arr = target_cubs[:, CB_W]; th_arr = target_cubs[:, CB_H]
    t_bb_xmin = float((txc - tw_arr / 2).min())
    t_bb_xmax = float((txc + tw_arr / 2).max())
    t_bb_ymin = float((tyc - th_arr / 2).min())
    t_bb_ymax = float((tyc + th_arr / 2).max())

    agg_count_above = 0
    agg_count_below = 0
    agg_scores: List[Tuple[str, float, float, float, float, float, int]] = []
    all_agg_for_density: List[np.ndarray] = []

    for agg_name, agg_cubs in agg_groups.items():
        if agg_cubs.shape[0] == 0:
            continue
        a_xc = agg_cubs[:, CB_X]; a_yc = agg_cubs[:, CB_Y]
        a_w = agg_cubs[:, CB_W]; a_h = agg_cubs[:, CB_H]
        a_xmin = (a_xc - a_w / 2).min(); a_xmax = (a_xc + a_w / 2).max()
        a_ymin = (a_yc - a_h / 2).min(); a_ymax = (a_yc + a_h / 2).max()
        if (a_xmax + SLACK_UM_V4 < t_bb_xmin
            or a_xmin - SLACK_UM_V4 > t_bb_xmax
            or a_ymax + SLACK_UM_V4 < t_bb_ymin
            or a_ymin - SLACK_UM_V4 > t_bb_ymax):
            continue
        all_agg_for_density.append(agg_cubs)
        agg_count_above += int((agg_cubs[:, CB_Z] > target_z_mean).sum())
        agg_count_below += int((agg_cubs[:, CB_Z] < target_z_mean).sum())
        ax = agg_cubs[None, :, CB_X]; ay = agg_cubs[None, :, CB_Y]; az = agg_cubs[None, :, CB_Z]
        aw = agg_cubs[None, :, CB_W]; ah = agg_cubs[None, :, CB_H]
        aeps = agg_cubs[None, :, CB_EPS]
        ovx = np.maximum(0.0, np.minimum(tx + tw / 2, ax + aw / 2) - np.maximum(tx - tw / 2, ax - aw / 2))
        ovy = np.maximum(0.0, np.minimum(ty + th / 2, ay + ah / 2) - np.maximum(ty - th / 2, ay - ah / 2))
        overlap = ovx * ovy
        dz_mat = np.abs(tz - az)
        eps_avg = 0.5 * (teps + aeps)
        score = float((eps_avg * overlap / np.maximum(dz_mat, EPS_Z_V4)).sum())
        total_xy_ov = float(overlap.sum())
        xy_dist = np.hypot(tx - ax, ty - ay)
        min_xy_dist = float(xy_dist.min())
        mean_dz = float(dz_mat.mean())
        agg_size = float(np.sum(agg_cubs[:, CB_W] * agg_cubs[:, CB_H]))
        layer_diff_flag = 1.0 if abs(float(agg_cubs[:, CB_Z].mean()) - target_z_mean) > 0.3 else 0.0
        agg_scores.append((agg_name, score, total_xy_ov, min_xy_dist,
                           mean_dz, agg_size, int(layer_diff_flag)))

    feats["agg_count_above_target_z"] = float(agg_count_above)
    feats["agg_count_below_target_z"] = float(agg_count_below)
    feats["agg_n_distinct"] = float(len(agg_scores))
    agg_scores.sort(key=lambda x: -x[1])
    total_score_all = float(sum(x[1] for x in agg_scores))
    for k in range(1, TOP_K + 1):
        if k <= len(agg_scores):
            _, sc, ov, min_d, mean_dz, asz, ldf = agg_scores[k - 1]
            feats[f"top{k}_score"] = sc
            feats[f"top{k}_overlap_um2"] = ov
            feats[f"top{k}_min_xy_dist_um"] = min_d
            feats[f"top{k}_mean_dz_um"] = mean_dz
            feats[f"top{k}_agg_size_um2"] = asz
            feats[f"top{k}_layer_diff_flag"] = float(ldf)
        else:
            for col in ("score", "overlap_um2", "min_xy_dist_um", "mean_dz_um",
                        "agg_size_um2", "layer_diff_flag"):
                feats[f"top{k}_{col}"] = 0.0
    top_k_score_sum = float(sum(x[1] for x in agg_scores[:TOP_K]))
    feats["topk_score_concentration"] = top_k_score_sum / total_score_all if total_score_all > 0 else 0.0

    if all_agg_for_density:
        all_agg = np.vstack(all_agg_for_density)
        tc_xyz = target_cubs[:, [CB_X, CB_Y, CB_Z]]
        t_centroid = tc_xyz.mean(axis=0)
        d_agg = np.sqrt(((all_agg[:, [CB_X, CB_Y, CB_Z]] - t_centroid) ** 2).sum(axis=-1))
        for r in (1.0, 3.0, 5.0):
            feats[f"agg_count_within_{int(r)}um_xyz"] = float((d_agg <= r).sum())
    else:
        for r in (1.0, 3.0, 5.0):
            feats[f"agg_count_within_{int(r)}um_xyz"] = 0.0

    return feats


def _v4_process_net(args: Tuple[str, List[Path]]) -> Dict[str, float]:
    """Aggregate tile pkl.gz files for one net then compute V4 H3 features."""
    net_name, tile_paths = args
    target_chunks: List[np.ndarray] = []
    agg_groups: Dict[str, List[np.ndarray]] = defaultdict(list)
    for tp in tile_paths:
        tile = _load_tile(tp)
        if tile is None:
            continue
        cubs = tile.get("cuboids")
        names = tile.get("cuboid_net_names")
        if cubs is None or names is None or len(names) == 0:
            continue
        cubs = np.asarray(cubs, dtype=np.float32)
        names_arr = np.asarray([str(n) for n in names])
        t_mask = (names_arr == net_name)
        if t_mask.any():
            target_chunks.append(cubs[t_mask])
        a_mask = ~t_mask
        if a_mask.any():
            agg_cubs = cubs[a_mask]
            agg_names = names_arr[a_mask]
            for an in np.unique(agg_names):
                agg_groups[an].append(agg_cubs[agg_names == an])
    if not target_chunks:
        return {}
    target_cubs = np.concatenate(target_chunks, axis=0)
    agg_groups_np = {k: np.concatenate(v, axis=0) for k, v in agg_groups.items()}
    feats = _v4_net_features(target_cubs, agg_groups_np)
    feats["net_name"] = net_name
    return feats


# ============================================================================
# V4-A indexed per-design cache (Round 2.1). Eliminates per-net tile-load.
# Fork-shared globals populated by `init_worker_v4cache`.
# ============================================================================

_V4C_CUBS = None        # ndarray (N_total, 10) fp32, mmap-able
_V4C_OWNER = None       # ndarray (N_total,) int32
_V4C_TILE_OFFSETS = None  # ndarray (n_tiles + 1,) int64
_V4C_TILE_SET = None    # ndarray (sum_target_tiles,) int32, flat CSR
_V4C_TILE_SET_OFF = None  # ndarray (n_nets + 1,) int64
_V4C_NET_NAMES = None   # list[str], int id -> str
_V4C_NAME_TO_ID = None  # dict[str, int]


def init_worker_v4cache(cache: dict):
    global _V4C_CUBS, _V4C_OWNER, _V4C_TILE_OFFSETS
    global _V4C_TILE_SET, _V4C_TILE_SET_OFF
    global _V4C_NET_NAMES, _V4C_NAME_TO_ID
    _V4C_CUBS = cache["cubs"]
    _V4C_OWNER = cache["owner"]
    _V4C_TILE_OFFSETS = cache["tile_offsets"]
    _V4C_TILE_SET = cache["tile_set"]
    _V4C_TILE_SET_OFF = cache["tile_set_off"]
    _V4C_NET_NAMES = cache["net_names"]
    _V4C_NAME_TO_ID = cache["_name_to_id"]


def _v4_process_net_from_cache(net_name: str) -> Dict[str, float]:
    """Same output as `_v4_process_net` but reads from the indexed per-design
    cache instead of unpacking tile pkl.gz files. We iterate this net's
    `_map.csv`-declared target tiles and split cuboids by `owner == nid`
    per tile (same rule as the tile pkl.gz path).
    """
    nid = _V4C_NAME_TO_ID.get(net_name)
    if nid is None:
        return {}
    set_s = _V4C_TILE_SET_OFF[nid]; set_e = _V4C_TILE_SET_OFF[nid + 1]
    if set_e <= set_s:
        return {}
    tile_set = _V4C_TILE_SET[set_s:set_e]
    target_chunks: List[np.ndarray] = []
    agg_chunks: List[np.ndarray] = []
    agg_owner_chunks: List[np.ndarray] = []
    for tid in tile_set:
        s = _V4C_TILE_OFFSETS[tid]
        e = _V4C_TILE_OFFSETS[tid + 1]
        if e <= s:
            continue
        tile_owners = _V4C_OWNER[s:e]
        t_mask = tile_owners == nid
        if t_mask.any():
            target_chunks.append(_V4C_CUBS[s:e][t_mask])
        a_mask = ~t_mask
        if a_mask.any():
            agg_chunks.append(_V4C_CUBS[s:e][a_mask])
            agg_owner_chunks.append(tile_owners[a_mask])
    if not target_chunks:
        return {}
    target_cubs = np.concatenate(target_chunks, axis=0)
    agg_groups: Dict[str, np.ndarray] = {}
    if agg_chunks:
        agg_cubs_flat = np.concatenate(agg_chunks, axis=0)
        agg_owners_int = np.concatenate(agg_owner_chunks)
        for aid in np.unique(agg_owners_int):
            m = agg_owners_int == aid
            agg_groups[_V4C_NET_NAMES[int(aid)]] = agg_cubs_flat[m]
    feats = _v4_net_features(target_cubs, agg_groups)
    feats["net_name"] = net_name
    return feats


# ============================================================================
# Per-design driver
# ============================================================================

FEATURE_COLS_67 = (MODELS_DIR / "FEATURE_ORDER.txt").read_text().strip().split("\n")

# Pre-fit XGBoost-Tweedie proxy for `fanout`. Training-time `fanout` was the
# count of SPEF coupled_caps keys — labeled, not derivable from DEF alone.
# Single-tree XGBoost-Tweedie on 8 DEF-only structural features beats the
# Ridge baseline (tv80s OOS MAPE_med 31% → 12%). cpl XGBoost has 0.81
# feature_importance on `fanout`, so proxy quality directly impacts cpl MAPE.
_FANOUT_PROXY_META_PATH = MODELS_DIR / "fanout_proxy_meta.json"
_FANOUT_PROXY_RIDGE_PATH = MODELS_DIR / "fanout_proxy_ridge.json"
_FANOUT_PROXY_KIND = None
_FANOUT_PROXY_XGB = None
_FANOUT_PROXY_RIDGE = None
_FANOUT_PROXY_FEATS = None

if _FANOUT_PROXY_META_PATH.exists():
    meta = json.loads(_FANOUT_PROXY_META_PATH.read_text())
    _FANOUT_PROXY_FEATS = meta["feats"]
    _FANOUT_PROXY_KIND = meta["kind"]
    if _FANOUT_PROXY_KIND == "xgb_tweedie":
        import xgboost as _xgb_proxy
        _FANOUT_PROXY_XGB = _xgb_proxy.XGBRegressor()
        _FANOUT_PROXY_XGB.load_model(str(MODELS_DIR / meta["model_file"]))
elif _FANOUT_PROXY_RIDGE_PATH.exists():
    _FANOUT_PROXY_RIDGE = json.loads(_FANOUT_PROXY_RIDGE_PATH.read_text())
    _FANOUT_PROXY_FEATS = _FANOUT_PROXY_RIDGE["feats"]
    _FANOUT_PROXY_KIND = "ridge"


def apply_fanout_proxy(df: pd.DataFrame) -> pd.Series:
    if _FANOUT_PROXY_KIND is None:
        return df["fanout"]
    X = df[_FANOUT_PROXY_FEATS].fillna(0.0).values
    if _FANOUT_PROXY_KIND == "xgb_tweedie":
        pred = _FANOUT_PROXY_XGB.predict(X.astype(np.float32))
    else:  # ridge
        coef = np.asarray(_FANOUT_PROXY_RIDGE["coef"], dtype=np.float64)
        intercept = float(_FANOUT_PROXY_RIDGE["intercept"])
        X_log = np.log1p(X)
        pred_log = X_log @ coef + intercept
        pred = np.expm1(pred_log)
    return pd.Series(np.maximum(pred, 1.0), index=df.index)


def extract_v3_features(geo, layer_map, n_workers: int,
                        use_gpu: bool = False,
                        algo: str = "auto") -> pd.DataFrame:
    """Run V3 feature extraction. When `use_gpu` is True, the per-net
    closest-pair broadcast (`pex_cold.py:_v3_compute_closest_gpu`) runs on
    CUDA and the dispatch falls back to single-process — torch CUDA contexts
    do not survive fork-Pool workers.

    `algo` controls the closest-pair backend selector that `_v3_per_net`
    consults: `"auto"` (default, threshold-gated per-target for long-tail),
    `"per_target"` (always per-target), or `"legacy"` (always numpy broadcast).
    """
    global _V3_USE_GPU, _V3_PER_TARGET_MODE
    if algo not in {"auto", "per_target", "legacy"}:
        raise ValueError(f"unknown --v3-algo {algo!r}")
    _V3_PER_TARGET_MODE = algo
    eps_by_layer = _layer_eps_array(layer_map, N_LAYERS_EPS)
    grid = SpatialGrid()
    grid.build(geo["all_cuboids"])
    density_per_layer = np.zeros(N_LAYERS_EPS + 2, dtype=np.float64)
    if len(geo["all_cuboids"]) > 0:
        for li in range(1, N_LAYERS_EPS + 1):
            mask = geo["all_cuboids"][:, 6] == li
            density_per_layer[li] = float(
                (geo["all_cuboids"][mask, 3] * geo["all_cuboids"][mask, 4]).sum())
        xmin, xmax, ymin, ymax = _bbox_xy(geo["all_cuboids"])
        density_window = max(1.0, (xmax - xmin) * (ymax - ymin))
    else:
        density_window = 1.0

    # V3-C: dispatch largest nets first. Adaptive chunksize: small designs
    # (tv80s, 3.4k nets) need chunksize=1 so few-second tail nets distribute
    # across workers immediately; large designs (nova, 119k nets) need
    # chunksize~7 so the IPC cost per task is amortized over a non-trivial
    # batch.
    target_nets = sorted(geo["target_set"],
                         key=lambda n: -len(geo["nets"].get(n, ())))
    rows = []
    total = len(target_nets)
    t_progress = time.time()

    if use_gpu:
        global _V3_CLOSEST_CACHE
        _v3_gpu_init()
        _v3_gpu_upload_all_cubs(geo["all_cuboids"])
        _V3_USE_GPU = True
        init_worker_v3(geo, grid, eps_by_layer, density_per_layer, density_window)
        CHUNK = 1024
        done = 0
        for chunk_start in range(0, total, CHUNK):
            chunk = target_nets[chunk_start:chunk_start + CHUNK]
            _V3_CLOSEST_CACHE = _v3_precompute_closest_chunk(chunk)
            for nm in chunk:
                r = _v3_per_net(nm)
                if r:
                    rows.append(r)
            _V3_CLOSEST_CACHE = None
            done += len(chunk)
            elapsed = time.time() - t_progress
            rate = done / max(elapsed, 1e-3)
            eta = (total - done) / max(rate, 1e-3)
            print(f"  V3(gpu-batched) progress {done}/{total} "
                  f"elapsed={elapsed:.0f}s rate={rate:.0f}/s eta={eta:.0f}s "
                  f"chunk_size={CHUNK}", flush=True)
        _V3_USE_GPU = False
        return pd.DataFrame(rows)

    chunksize = max(1, total // (n_workers * 1000))
    if n_workers <= 1:
        init_worker_v3(geo, grid, eps_by_layer, density_per_layer, density_window)
        for nm in target_nets:
            r = _v3_per_net(nm)
            if r:
                rows.append(r)
    else:
        with mp.Pool(processes=n_workers, initializer=init_worker_v3,
                     initargs=(geo, grid, eps_by_layer, density_per_layer, density_window)) as pool:
            for i, r in enumerate(pool.imap_unordered(
                    _v3_per_net, target_nets, chunksize=chunksize), 1):
                if r:
                    rows.append(r)
                if i % 5000 == 0 or i == total:
                    elapsed = time.time() - t_progress
                    rate = i / max(elapsed, 1e-3)
                    eta = (total - i) / max(rate, 1e-3)
                    print(f"  V3 progress {i}/{total} elapsed={elapsed:.0f}s "
                          f"rate={rate:.0f}/s eta={eta:.0f}s "
                          f"chunk={chunksize}", flush=True)
    return pd.DataFrame(rows)


def _load_v4_pernet_cache(design: str):
    """Load the Round 2.1 v3 per-design V4 cache if it exists on disk.
    Returns a dict of mmap'd / loaded numpy arrays + name map, or None.
    """
    prefix = TILE_CACHE_ROOT / f"{design}_v4_pernet"
    meta_path = prefix.with_suffix(".meta.pkl")
    if not meta_path.exists():
        return None
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    if meta.get("schema_version") != 4:
        raise SystemExit(f"unexpected v4 cache schema {meta.get('schema_version')} "
                         f"at {meta_path}")
    cache = {
        "cubs":          np.load(prefix.with_suffix(".cubs.npy"), mmap_mode="r"),
        "owner":         np.load(prefix.with_suffix(".owner.npy"), mmap_mode="r"),
        "tile_offsets":  np.load(prefix.with_suffix(".tile_offsets.npy")),
        "tile_set":      np.load(prefix.with_suffix(".tile_set.npy")),
        "tile_set_off":  np.load(prefix.with_suffix(".tile_set_off.npy")),
        "net_names":     meta["net_names"],
    }
    cache["_name_to_id"] = {n: i for i, n in enumerate(cache["net_names"])}
    return cache


def extract_v4_h3_from_pernet_cache(design: str, target_nets: set, n_workers: int,
                                    cache: dict) -> pd.DataFrame:
    """V4 H3 feature extraction via the pre-built per-net indexed cache."""
    name_to_id = cache["_name_to_id"]
    tile_set_off = cache["tile_set_off"]

    def _net_tile_count(nm: str) -> int:
        nid = name_to_id.get(nm)
        if nid is None:
            return 0
        return int(tile_set_off[nid + 1] - tile_set_off[nid])
    job_args = sorted(target_nets, key=lambda n: -_net_tile_count(n))
    rows = []
    total = len(job_args)
    chunksize = max(1, total // (n_workers * 1000))
    t_progress = time.time()
    if n_workers <= 1:
        init_worker_v4cache(cache)
        for nm in job_args:
            f = _v4_process_net_from_cache(nm)
            if f:
                rows.append(f)
    else:
        with mp.Pool(processes=n_workers,
                     initializer=init_worker_v4cache, initargs=(cache,)) as pool:
            for i, f in enumerate(pool.imap_unordered(
                    _v4_process_net_from_cache, job_args, chunksize=chunksize), 1):
                if f:
                    rows.append(f)
                if i % 5000 == 0 or i == total:
                    elapsed = time.time() - t_progress
                    rate = i / max(elapsed, 1e-3)
                    eta = (total - i) / max(rate, 1e-3)
                    print(f"  V4(cached) progress {i}/{total} elapsed={elapsed:.0f}s "
                          f"rate={rate:.0f}/s eta={eta:.0f}s "
                          f"chunk={chunksize}", flush=True)
    return pd.DataFrame(rows)


def extract_v4_h3_from_tile_cache(design: str, target_nets: set, n_workers: int) -> pd.DataFrame:
    """Build (net → list of tile paths) from <design>_map.csv and run V4 H3.

    If a Round 2.1 indexed cache (`<design>_v4_pernet.pkl`) exists, use it
    instead — same features, ~10-30× wall reduction by eliminating per-net
    tile pkl.gz reads.
    """
    cache = _load_v4_pernet_cache(design)
    if cache is not None:
        return extract_v4_h3_from_pernet_cache(design, target_nets, n_workers, cache)

    tile_dir = TILE_CACHE_ROOT / design
    map_csv = TILE_CACHE_ROOT / f"{design}_map.csv"
    if not tile_dir.exists() or not map_csv.exists():
        raise SystemExit(
            f"[{design}] tile cache missing at {tile_dir}\n"
            "  Build it via: python3 scripts/build_dataset_multi.py "
            f"(or build_dataset.py --def_path {DESIGNS[design]})"
        )
    df = pd.read_csv(map_csv)
    df = df[df["net_name"].isin(target_nets)].reset_index(drop=True)
    grp: Dict[str, List[Path]] = defaultdict(list)
    for r in df.itertuples(index=False):
        grp[r.net_name].append(tile_dir / r.sample_filename)
    # V3-C (V4 arm): largest-tile nets first, adaptive chunksize.
    job_args = sorted(grp.items(), key=lambda kv: -len(kv[1]))
    rows = []
    total = len(job_args)
    chunksize = max(1, total // (n_workers * 1000))
    t_progress = time.time()
    if n_workers <= 1:
        for ja in job_args:
            f = _v4_process_net(ja)
            if f:
                rows.append(f)
    else:
        with mp.Pool(processes=n_workers) as pool:
            for i, f in enumerate(pool.imap_unordered(
                    _v4_process_net, job_args, chunksize=chunksize), 1):
                if f:
                    rows.append(f)
                if i % 5000 == 0 or i == total:
                    elapsed = time.time() - t_progress
                    rate = i / max(elapsed, 1e-3)
                    eta = (total - i) / max(rate, 1e-3)
                    print(f"  V4 progress {i}/{total} elapsed={elapsed:.0f}s "
                          f"rate={rate:.0f}/s eta={eta:.0f}s "
                          f"chunk={chunksize}", flush=True)
    return pd.DataFrame(rows)


# ============================================================================
# Inference + SPEF write + compare
# ============================================================================

def load_models():
    import xgboost as xgb
    g_models, c_models = [], []
    for s in SEEDS:
        mg = xgb.XGBRegressor(); mg.load_model(str(MODELS_DIR / f"tweedie_gnd_seed{s}.json"))
        mc = xgb.XGBRegressor(); mc.load_model(str(MODELS_DIR / f"tweedie_cpl_seed{s}.json"))
        g_models.append(mg); c_models.append(mc)
    return g_models, c_models


def predict_ensemble(models, X):
    preds = np.stack([m.predict(X).clip(0.0) for m in models], axis=0)
    return preds.mean(axis=0)


SPEF_HEADER_TMPL = """*SPEF "IEEE 1481-1999"
*DESIGN "{design}"
*DATE "{date}"
*VENDOR "TreePEX cold-start (5-seed Tweedie XGBoost ensemble)"
*PROGRAM "TreePEX cold-start PEX tool"
*VERSION "1.0"
*DESIGN_FLOW "PIN_CAP NONE" "NAME_SCOPE LOCAL"
*DIVIDER /
*DELIMITER :
*BUS_DELIMITER []
*T_UNIT 1.0 NS
*C_UNIT 1.0 FF
*R_UNIT 1.0 OHM
*L_UNIT 1.0 HENRY

"""


def write_spef(design: str, df: pd.DataFrame, out_path: Path):
    stem = design.split("intel22_")[-1].replace("_f3", "")
    date_str = datetime.now().strftime("%a %b %d %H:%M:%S %Y")
    header = SPEF_HEADER_TMPL.format(design=stem, date=date_str)
    lines = [header]
    for _, row in df.iterrows():
        net = str(row["net_name"]).strip()
        c_tot = float(row["pred_total"])
        c_gnd = float(row["pred_gnd"])
        lines.append(f"*D_NET {net} {c_tot:.5f}")
        lines.append("*CONN")
        lines.append("*CAP")
        lines.append(f"1 {net}:0 {c_gnd:.5f}")
        lines.append("*END")
        lines.append("")
    out_path.write_text("\n".join(lines))


def parse_spef_full(path: Path) -> dict:
    out = {}
    name_map = {}
    text = path.read_text(errors="replace")
    m_unit = re.search(r"\*C_UNIT\s+([\d.]+)\s+(PF|FF)", text, re.IGNORECASE)
    unit_mult = 1.0
    if m_unit:
        val = float(m_unit.group(1)); unit = m_unit.group(2).upper()
        unit_mult = (val * 1000.0) if unit == "PF" else val
    in_namemap = False
    for ln in text.split("\n"):
        s = ln.strip()
        if s.startswith("*NAME_MAP"):
            in_namemap = True; continue
        if in_namemap:
            if not s or (s.startswith("*") and not s[1:].split()[0].lstrip("-").isdigit()):
                break
            parts = s.split()
            if len(parts) >= 2 and parts[0].startswith("*"):
                name_map[parts[0]] = parts[1]
    cur = None; in_cap = False
    for ln in text.split("\n"):
        s = ln.rstrip()
        if not s:
            continue
        m = re.match(r"\*D_NET\s+(\S+)\s+([\d.eE+-]+)", s)
        if m:
            if cur is not None:
                out[cur["name"]] = cur
            nid = m.group(1)
            cur = {"name": name_map.get(nid, nid),
                   "total": float(m.group(2)) * unit_mult, "gnd": 0.0, "cpl": 0.0}
            in_cap = False; continue
        stripped = s.strip()
        if stripped.startswith("*CAP"):
            in_cap = True; continue
        if stripped.startswith("*RES") or stripped.startswith("*CONN"):
            in_cap = False; continue
        if stripped.startswith("*END"):
            if cur is not None:
                out[cur["name"]] = cur; cur = None
            in_cap = False; continue
        if in_cap and cur is not None:
            toks = stripped.split()
            if len(toks) == 3:
                try: cur["gnd"] += float(toks[2]) * unit_mult
                except ValueError: pass
            elif len(toks) >= 4:
                try: cur["cpl"] += float(toks[3]) * unit_mult
                except ValueError: pass
    if cur is not None:
        out[cur["name"]] = cur
    return out


def r2(p, g):
    g = np.asarray(g); p = np.asarray(p)
    ss_res = float(((g - p) ** 2).sum())
    ss_tot = float(((g - g.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def mape_med(p, g):
    g = np.asarray(g); p = np.asarray(p)
    return float(np.median(np.abs(p - g) / np.maximum(np.abs(g), 1e-3) * 100))


def mape_mean(p, g):
    g = np.asarray(g); p = np.asarray(p)
    return float(np.mean(np.abs(p - g) / np.maximum(np.abs(g), 1e-3) * 100))


# ============================================================================
# Per-design pipeline
# ============================================================================

def run_design(design: str, def_path: Path, n_workers: int,
               v3_gpu: bool = False, v3_algo: str = "auto") -> dict:
    print(f"\n========== COLD-START ▶ {design} ==========", flush=True)
    timings = {"design": design, "n_workers_per_design": n_workers}
    t_design_start = time.time()

    # 1) PDK once
    t0 = time.time()
    layer_map = LayerInfoParser(LAYERS_INFO_PATH).parse()
    tech_lef = LefParser(TECH_LEF_PATH).parse()
    cell_lib = CellLibParser(CELL_LEF_PATH).parse()
    timings["t_pdk_parse_s"] = round(time.time() - t0, 3)
    print(f"[{design}] PDK parse: {timings['t_pdk_parse_s']}s", flush=True)

    # 2) DEF parse
    t0 = time.time()
    geo = scan_design(def_path, layer_map, tech_lef, cell_lib)
    timings["t_def_parse_s"] = round(time.time() - t0, 3)
    timings["n_target_nets"] = len(geo["target_set"])
    timings["n_pin_pseudo_nets"] = len(geo["nets"]) - len(geo["target_set"])
    timings["n_total_cuboids"] = int(len(geo["all_cuboids"]))
    print(f"[{design}] DEF parse: {timings['t_def_parse_s']}s  "
          f"target_nets={timings['n_target_nets']:,}  "
          f"pin_pseudo={timings['n_pin_pseudo_nets']:,}  "
          f"cuboids={timings['n_total_cuboids']:,}", flush=True)

    # 3) V3 feature extraction (DEF-only, parallel)
    t0 = time.time()
    df_v3 = extract_v3_features(geo, layer_map, n_workers,
                                use_gpu=v3_gpu, algo=v3_algo)
    timings["t_v3_features_s"] = round(time.time() - t0, 3)
    timings["n_v3_rows"] = len(df_v3)
    print(f"[{design}] V3 features: {timings['t_v3_features_s']}s  rows={timings['n_v3_rows']:,}",
          flush=True)

    # Free DEF geometry before V4 phase
    target_set = geo["target_set"]
    del geo
    gc.collect()

    # 4) V4 H3 features (tile-cache aggregation, parallel)
    t0 = time.time()
    df_v4 = extract_v4_h3_from_tile_cache(design, target_set, n_workers)
    timings["t_v4_h3_features_s"] = round(time.time() - t0, 3)
    timings["n_v4_rows"] = len(df_v4)
    print(f"[{design}] V4 H3 features: {timings['t_v4_h3_features_s']}s  rows={timings['n_v4_rows']:,}",
          flush=True)

    # 5) Merge + apply fanout proxy (label-free regression) + inference
    t0 = time.time()
    feat_df = df_v3.merge(df_v4, on="net_name", how="left")
    for c in FEATURE_COLS_67:
        if c not in feat_df.columns:
            feat_df[c] = 0.0
    feat_df = feat_df.dropna(subset=FEATURE_COLS_67).reset_index(drop=True)
    # Override fanout (training-time SPEF-derived) with offline-fit proxy
    feat_df["fanout"] = apply_fanout_proxy(feat_df)
    # Cache cold-start features so other models (B1, mesh-PINN, ResMLP, ...)
    # can be evaluated without re-running the heavy feature pipeline.
    feat_cache_path = COLD_REPORT_DIR / f"{design}_cold_features.parquet"
    try:
        feat_df.to_parquet(feat_cache_path, index=False)
    except Exception:
        feat_df.to_csv(feat_cache_path.with_suffix(".csv"), index=False)
    timings["n_feature_rows"] = len(feat_df)
    g_models, c_models = load_models()
    X = feat_df[FEATURE_COLS_67].astype(np.float32).values
    pred_g = predict_ensemble(g_models, X)
    pred_c = predict_ensemble(c_models, X)
    pred_df = feat_df[["net_name"]].copy()
    pred_df["pred_gnd"] = pred_g
    pred_df["pred_cpl"] = pred_c
    pred_df["pred_total"] = pred_g + pred_c
    pred_df["design_name"] = design
    pred_csv = PRED_DIR / f"{design}_cold_pred.csv"
    pred_df.to_csv(pred_csv, index=False)
    timings["t_inference_s"] = round(time.time() - t0, 3)
    print(f"[{design}] inference: {timings['t_inference_s']}s  rows={len(pred_df):,}", flush=True)

    # 6) SPEF write
    t0 = time.time()
    spef_out = SPEF_DIR_OUT / f"{design}_cold_pred.spef"
    write_spef(design, pred_df, spef_out)
    timings["t_spef_write_s"] = round(time.time() - t0, 3)
    timings["spef_size_mb"] = round(spef_out.stat().st_size / 1024 / 1024, 2)
    print(f"[{design}] SPEF write: {timings['t_spef_write_s']}s "
          f"({timings['spef_size_mb']} MB)", flush=True)

    # 7) Golden compare
    t0 = time.time()
    golden_path = GOLDEN_SPEF_DIR / f"{design}_starrc.spef"
    gold = parse_spef_full(golden_path)
    pred_g_map = dict(zip(pred_df["net_name"].astype(str), pred_df["pred_gnd"]))
    pred_c_map = dict(zip(pred_df["net_name"].astype(str), pred_df["pred_cpl"]))
    pred_t_map = dict(zip(pred_df["net_name"].astype(str), pred_df["pred_total"]))
    common = set(pred_t_map.keys()) & set(gold.keys())
    rows = []
    for net in common:
        g = gold[net]
        rows.append({
            "net": net,
            "pred_gnd": pred_g_map[net], "pred_cpl": pred_c_map[net],
            "pred_total": pred_t_map[net],
            "gold_gnd": g["gnd"], "gold_cpl": g["cpl"], "gold_total": g["total"],
        })
    cmp = pd.DataFrame(rows)
    if len(cmp) > 0:
        timings["n_nets_compared"] = int(len(cmp))
        timings["MAPE_tot_med"] = round(mape_med(cmp["pred_total"], cmp["gold_total"]), 4)
        timings["MAPE_tot_mean"] = round(mape_mean(cmp["pred_total"], cmp["gold_total"]), 4)
        timings["MAPE_gnd_med"] = round(mape_med(cmp["pred_gnd"], cmp["gold_gnd"]), 4)
        timings["MAPE_cpl_med"] = round(mape_med(cmp["pred_cpl"], cmp["gold_cpl"]), 4)
        timings["R2_tot"] = round(r2(cmp["pred_total"], cmp["gold_total"]), 6)
        timings["R2_gnd"] = round(r2(cmp["pred_gnd"], cmp["gold_gnd"]), 6)
        timings["R2_cpl"] = round(r2(cmp["pred_cpl"], cmp["gold_cpl"]), 6)
        timings["pred_chip_total_fF"] = round(float(cmp["pred_total"].sum()), 3)
        timings["pred_chip_gnd_fF"] = round(float(cmp["pred_gnd"].sum()), 3)
        timings["pred_chip_cpl_fF"] = round(float(cmp["pred_cpl"].sum()), 3)
        timings["gold_chip_total_fF"] = round(float(cmp["gold_total"].sum()), 3)
        timings["gold_chip_gnd_fF"] = round(float(cmp["gold_gnd"].sum()), 3)
        timings["gold_chip_cpl_fF"] = round(float(cmp["gold_cpl"].sum()), 3)
        cmp.to_csv(COLD_REPORT_DIR / f"{design}_per_net.csv", index=False)
    timings["t_compare_s"] = round(time.time() - t0, 3)
    print(f"[{design}] compare: {timings['t_compare_s']}s  "
          f"MAPE_tot={timings.get('MAPE_tot_med', 'NA')}%", flush=True)

    timings["t_design_total_s"] = round(time.time() - t_design_start, 3)
    timings["t_user_pipeline_s"] = round(
        timings["t_pdk_parse_s"] + timings["t_def_parse_s"]
        + timings["t_v3_features_s"] + timings["t_v4_h3_features_s"]
        + timings["t_inference_s"] + timings["t_spef_write_s"], 3)
    return timings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", type=str, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--serial", action="store_true")
    ap.add_argument("--v3-gpu", action="store_true",
                    help="Run V3 closest-pair on torch + CUDA (single-process, "
                         "no fork-Pool). Round 2.2.")
    ap.add_argument("--v3-algo", type=str, default="auto",
                    choices=["auto", "per_target", "legacy"],
                    help="V3 closest-pair backend: auto (threshold-gated, "
                         "default), per_target (always new per-target grid path), "
                         "legacy (always numpy broadcast). Round 3.")
    args = ap.parse_args()
    if args.design:
        if args.design not in DESIGNS:
            raise SystemExit(f"unknown design {args.design}")
        targets = [(args.design, DESIGNS[args.design])]
    else:
        targets = list(DESIGNS.items())

    t_wall_start = time.time()
    summaries: List[dict] = []
    if args.serial or len(targets) == 1:
        for d, p in targets:
            summaries.append(run_design(d, p, args.workers,
                                         v3_gpu=args.v3_gpu, v3_algo=args.v3_algo))
    else:
        if args.v3_gpu:
            raise SystemExit("--v3-gpu requires --serial or a single --design "
                             "(CUDA state is per-process)")
        with ProcessPoolExecutor(max_workers=len(targets)) as ex:
            futs = {ex.submit(run_design, d, p, args.workers,
                                False, args.v3_algo): d
                    for d, p in targets}
            for fut in as_completed(futs):
                d = futs[fut]
                try:
                    summaries.append(fut.result())
                except Exception as e:
                    summaries.append({"design": d, "error": str(e)})

    wall_total = round(time.time() - t_wall_start, 3)
    out = {"wall_total_s_parallel": wall_total, "per_design": summaries}
    out_path = COLD_REPORT_DIR / "cold_summary.json"
    out_path.write_text(json.dumps(out, indent=2))

    print("\n========== COLD-START SUMMARY ==========")
    for s in summaries:
        if "error" in s and "MAPE_tot_med" not in s:
            print(f"  {s['design']}: ERROR — {s['error']}")
            continue
        print(f"  {s['design']:24s} | n={s.get('n_nets_compared', 0):>6,} | "
              f"MAPE tot={s.get('MAPE_tot_med'):.3f}% gnd={s.get('MAPE_gnd_med'):.2f}% "
              f"cpl={s.get('MAPE_cpl_med'):.2f}% | R²_tot={s.get('R2_tot'):.4f}")
        print(f"     pdk={s['t_pdk_parse_s']:.2f}s  def={s['t_def_parse_s']:.2f}s  "
              f"v3={s['t_v3_features_s']:.2f}s  v4={s['t_v4_h3_features_s']:.2f}s  "
              f"infer={s['t_inference_s']:.2f}s  spef={s['t_spef_write_s']:.2f}s  "
              f"=> user_pipeline={s['t_user_pipeline_s']:.2f}s  "
              f"(compare {s['t_compare_s']:.2f}s, design_total {s['t_design_total_s']:.2f}s)")
        print(f"     pred chip total={s.get('pred_chip_total_fF', 0):,.1f} fF  "
              f"gnd={s.get('pred_chip_gnd_fF', 0):,.1f}  cpl={s.get('pred_chip_cpl_fF', 0):,.1f}  "
              f"vs gold total={s.get('gold_chip_total_fF', 0):,.1f}  "
              f"gnd={s.get('gold_chip_gnd_fF', 0):,.1f}  cpl={s.get('gold_chip_cpl_fF', 0):,.1f}")
    print(f"\n  wall total (parallel both designs): {wall_total:.2f} s")
    print(f">>> wrote {out_path}")


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
