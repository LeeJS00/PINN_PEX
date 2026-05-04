"""engine.py — exp_007 Sakurai-Tamaru gnd allocator on top of parallel pass-2.

Differences vs exp_006/engine.py:
  - per-net c_gnd estimate: Sakurai-Tamaru per-segment (top + bot plate, with
    layer-aware ε, distance to nearest conductor, and pre-fit per-layer NNLS
    multiplier) instead of `length × width × ε × 0.22`.
  - per-node c_gnd distribution: post-pass `redistribute_node_caps_inplace`
    that overrides legacy length-only weighting with Sakurai-Tamaru weighting
    derived from edge `$l/$w/$lvl` comments.
  - c_cpl path: untouched (still ratio 1.3 × c_gnd_total + geometric per-aggressor).

The parallel infrastructure (KD-tree, top_ports, by_net) is identical to
exp_006. We pass `LayerStackPlate` as a worker-init global so each worker
has cheap O(1) lookup tables for Sakurai-Tamaru constants.
"""
from __future__ import annotations

import gzip
import io
import multiprocessing as mp
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

# This file lives at pex_v3/joint_pareto/experiments/exp_007_sakurai_gnd/engine.py
# parents[4] = PINNPEX repo root.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[4])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from scipy.spatial import cKDTree

# Reuse legacy infrastructure unchanged.
from src.v3.utils.fast_spef_engine import (
    SegmentRecord,
    build_kdtree,
    compute_aggressor_weights,
    stream_index_pass,
)
from src.utils.spef_writer import NetCapWriter, RCTopologyBuilder, SPEFWriter

# Sakurai-Tamaru allocator (THIS variant's only delta).
from pex_v3.joint_pareto.allocators.gnd.sakurai_tamaru import (
    LayerStackPlate,
    analytic_per_net_cap_estimate as st_analytic_per_net_cap_estimate,
    redistribute_node_caps_inplace,
)


# ============================================================================
# Worker globals — populated once per child process via the pool initializer.
# ============================================================================
_W_RECORDS: list[SegmentRecord] | None = None
_W_TREE: cKDTree | None = None
_W_BY_NET: dict[str, list[SegmentRecord]] | None = None
_W_TOP_PORTS = None
_W_LAYER_INFO: dict | None = None
_W_TECH_LEF: dict | None = None
_W_PLATE: LayerStackPlate | None = None
_W_MAX_DIST: float = 5.0
_W_TOP_K: int = 20
_W_REDISTRIBUTE_NODES: bool = False


def _worker_init(records, by_net_dict, top_ports, layer_info, tech_lef,
                  plate, max_dist, top_k, redistribute_nodes):
    """Pool initializer — rebuild KD-tree locally and bind globals."""
    global _W_RECORDS, _W_TREE, _W_BY_NET, _W_TOP_PORTS
    global _W_LAYER_INFO, _W_TECH_LEF, _W_PLATE, _W_MAX_DIST, _W_TOP_K
    global _W_REDISTRIBUTE_NODES
    _W_RECORDS = records
    _W_TREE = build_kdtree(records)
    _W_BY_NET = by_net_dict
    _W_TOP_PORTS = top_ports
    _W_LAYER_INFO = layer_info
    _W_TECH_LEF = tech_lef
    _W_PLATE = plate
    _W_MAX_DIST = max_dist
    _W_TOP_K = top_k
    _W_REDISTRIBUTE_NODES = redistribute_nodes


def _process_one_net(task: tuple[int, str, str]) -> tuple[int, str, str, bool]:
    """Worker function: returns (idx, net_name, spef_body, ok)."""
    idx, net_name, path_str = task
    path = Path(path_str)
    try:
        with gzip.open(path, "rb") as f:
            topo = pickle.load(f)
    except Exception:
        return idx, net_name, "", False
    global_segments = topo.get("global_segments", [])
    del topo
    if not global_segments:
        return idx, net_name, "", False

    target_segs = _W_BY_NET.get(net_name, []) if _W_BY_NET else []
    # ---- DELTA #1: Sakurai-Tamaru per-net cap estimate ----
    c_gnd_total, c_cpl_total = st_analytic_per_net_cap_estimate(target_segs, _W_PLATE)
    if c_gnd_total <= 0 and c_cpl_total <= 0:
        return idx, net_name, "", False

    aggr_weights = compute_aggressor_weights(
        target_segs, _W_RECORDS, _W_TREE,
        target_net_name=net_name,
        max_dist_um=_W_MAX_DIST, top_k=_W_TOP_K,
    )
    c_cpl_dict = {n: c_cpl_total * w for n, w in aggr_weights.items()}

    try:
        rc = RCTopologyBuilder(
            net_name=net_name,
            global_segments=global_segments,
            top_ports=_W_TOP_PORTS,
            layer_info=_W_LAYER_INFO,
            tech_lef=_W_TECH_LEF,
        )
    except Exception:
        return idx, net_name, "", False

    try:
        writer = NetCapWriter(rc, c_gnd_total, c_cpl_dict)
        # ---- DELTA #2 (configurable): Sakurai-Tamaru per-node c_gnd redistribution ----
        # Smoke-test seed 0 (full ST: per-net + per-node) showed
        # gnd_unmatched 35.59% (vs 21.50% baseline) and tot_mean 8.09%
        # (vs 7.04%). The per-node Sakurai weighting collides with the
        # legacy distribute_net_caps + 1e-5 fF truncation in a way that
        # hurts unmatched. Per-net c_gnd total via ST is the only delta
        # we keep for now; per-node redistribution is gated off.
        if _W_REDISTRIBUTE_NODES:
            redistribute_node_caps_inplace(writer, _W_PLATE)

        buf = io.StringIO()
        shim = SPEFWriter(buf, design_name="__shim__", top_ports=[])
        shim.stream_net_cap_writer(writer)
        return idx, net_name, buf.getvalue(), True
    except Exception:
        return idx, net_name, "", False


def write_sakurai_spef_parallel(
    design_name: str,
    topology_dir: Path,
    layer_info: dict,
    tech_lef: dict | None,
    out_spef_path: Path,
    max_dist_um: float = 5.0,
    top_k: int = 20,
    progress: bool = True,
    n_workers_pass1: int = 1,
    n_workers_pass2: int = 16,
    chunksize: int | None = None,
    redistribute_nodes: bool = False,
) -> dict:
    """Sakurai-Tamaru-gnd parallel-pass-2 SPEF write.

    Args:
        redistribute_nodes: if True, override the legacy length-only per-node
            c_gnd distribution with Sakurai-Tamaru per-edge weighting. Disabled
            by default — smoke-test showed it regresses unmatched-net MAPE
            because the legacy 1e-5 fF cap-line truncation interacts badly
            with the new spatial weighting. When False, only the per-net
            c_gnd total estimate is upgraded to Sakurai-Tamaru.
    """
    timings: dict[str, float] = {}

    # ---- Build LayerStackPlate once on parent (cheap; pickled to workers) ----
    t0 = time.perf_counter()
    plate = LayerStackPlate(layer_info)
    timings["plate_build_s"] = time.perf_counter() - t0

    # ---- Pass 1: identical to baseline ----
    t0 = time.perf_counter()
    records, nets_metadata, top_ports = stream_index_pass(
        topology_dir, n_workers=n_workers_pass1,
    )
    timings["index_pass_s"] = time.perf_counter() - t0

    if not nets_metadata:
        raise SystemExit(f"No topologies found under {topology_dir}")

    t0 = time.perf_counter()
    by_net: dict[str, list[SegmentRecord]] = defaultdict(list)
    for r in records:
        by_net[r.net_name].append(r)
    _ = build_kdtree(records)  # parity with baseline; workers rebuild
    timings["kdtree_build_s"] = time.perf_counter() - t0

    # ---- Pass 2 parallel ----
    t0 = time.perf_counter()
    tasks = [
        (idx, net_name, str(path)) for idx, (net_name, path) in enumerate(nets_metadata)
    ]
    if chunksize is None:
        chunksize = max(1, min(64, len(tasks) // (n_workers_pass2 * 4)))

    n_written = 0
    n_skipped = 0
    out_spef_path.parent.mkdir(parents=True, exist_ok=True)

    ctx = mp.get_context("spawn")
    by_net_dict = dict(by_net)
    init_args = (records, by_net_dict, top_ports, layer_info, tech_lef,
                  plate, max_dist_um, top_k, redistribute_nodes)

    with open(out_spef_path, "w") as fh:
        spef_header_writer = SPEFWriter(fh, design_name=design_name, top_ports=top_ports)
        spef_header_writer.write_header()

        with ctx.Pool(
            processes=n_workers_pass2,
            initializer=_worker_init,
            initargs=init_args,
        ) as pool:
            for ret_idx, ret_net, body, ok in pool.imap(
                _process_one_net, tasks, chunksize=chunksize,
            ):
                if not ok or not body:
                    n_skipped += 1
                    continue
                fh.write(body)
                n_written += 1
                if progress and n_written % 500 == 0 and n_written > 0:
                    elapsed = time.perf_counter() - t0
                    print(f"  ... {n_written} nets written in {elapsed:.1f}s", flush=True)

    timings["spef_write_s"] = time.perf_counter() - t0
    timings["nets_written"] = n_written
    timings["nets_skipped"] = n_skipped
    timings["nets_total"] = len(nets_metadata)
    timings["n_workers_pass2"] = n_workers_pass2
    timings["chunksize_pass2"] = chunksize
    timings["redistribute_nodes"] = redistribute_nodes
    return timings
