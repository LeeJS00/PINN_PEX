"""engine.py — exp_006 parallel pass-2 SPEF engine variant.

Mirrors `pex_v3/src/utils/fast_spef_engine.write_fast_autonomous_spef` exactly
(same analytic placeholder, same aggressor weights, same RCTopologyBuilder,
same SPEFWriter), with one change: pass-2 (per-net assembly + write) is
parallelized via `multiprocessing.Pool.imap` (ordered) over per-net tasks.

Determinism: each worker reads its own topology pkl.gz, computes its caps and
writes its body string into an in-memory `io.StringIO`, and returns the body
string. Parent appends in original net order. The same KD-tree records,
`by_net`, and `top_ports` are constructed by the parent and passed to workers
once via the pool initializer (avoids pickling them per-task).

Numerical equivalence to baseline: aggressor weights, NetCapWriter, and
SPEFWriter are imported unchanged from the legacy paths; the parallel split
only changes *who* runs the body, not *what* is computed.
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

# IMPORTANT: spawn workers re-import this module from disk and do NOT inherit
# sys.path mutations from the parent driver. We must inject the project root
# here so workers can find `src.utils.spef_writer` etc. The path is computed
# from this file's location: <root>/pex_v3/joint_pareto/experiments/<exp>/engine.py
_PROJECT_ROOT = str(Path(__file__).resolve().parents[4])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from scipy.spatial import cKDTree

# Import the EXACT baseline helpers — no reimplementation, no semantic drift.
from src.v3.utils.fast_spef_engine import (
    SegmentRecord,
    analytic_per_net_cap_estimate,
    build_kdtree,
    compute_aggressor_weights,
    stream_index_pass,
)
from src.utils.spef_writer import NetCapWriter, RCTopologyBuilder, SPEFWriter


# ============================================================================
# Worker globals — populated once per child process via the pool initializer.
# Avoids re-pickling 200 K SegmentRecord objects per task.
# ============================================================================
_W_RECORDS: list[SegmentRecord] | None = None
_W_TREE: cKDTree | None = None
_W_BY_NET: dict[str, list[SegmentRecord]] | None = None
_W_TOP_PORTS = None
_W_LAYER_INFO: dict | None = None
_W_TECH_LEF: dict | None = None
_W_MAX_DIST: float = 5.0
_W_TOP_K: int = 20


def _worker_init(records, by_net_dict, top_ports, layer_info, tech_lef, max_dist, top_k):
    """Pool initializer — rebuild KD-tree locally (cheap) and bind globals."""
    global _W_RECORDS, _W_TREE, _W_BY_NET, _W_TOP_PORTS
    global _W_LAYER_INFO, _W_TECH_LEF, _W_MAX_DIST, _W_TOP_K
    _W_RECORDS = records
    # Rebuilding the KD-tree in each worker (cheap, ~0.025 s for tv80s) is
    # cleaner than pickling the cKDTree object across the spawn boundary —
    # cKDTree pickling is supported but slower and version-fragile.
    _W_TREE = build_kdtree(records)
    _W_BY_NET = by_net_dict
    _W_TOP_PORTS = top_ports
    _W_LAYER_INFO = layer_info
    _W_TECH_LEF = tech_lef
    _W_MAX_DIST = max_dist
    _W_TOP_K = top_k


def _process_one_net(task: tuple[int, str, str]) -> tuple[int, str, str, bool]:
    """Worker function: returns (idx, net_name, spef_body, ok).

    `idx` lets the parent recover original order even under unordered streaming
    (we use the ordered `imap` so the idx is mostly cosmetic, but we keep it
    for defensive ordering verification).
    """
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
    c_gnd_total, c_cpl_total = analytic_per_net_cap_estimate(target_segs)
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
        # Write to in-memory buffer using the SAME SPEFWriter.stream_net_cap_writer
        # method the baseline uses, so byte-for-byte the per-net body block
        # is identical to the serial path.
        buf = io.StringIO()
        # Construct a SPEFWriter shim that only consumes stream_net_cap_writer().
        # We don't call write_header on this per-net writer — header is written
        # once by the parent.
        shim = SPEFWriter(buf, design_name="__shim__", top_ports=[])
        shim.stream_net_cap_writer(writer)
        return idx, net_name, buf.getvalue(), True
    except Exception:
        return idx, net_name, "", False


def write_fast_autonomous_spef_parallel(
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
) -> dict:
    """Parallel-pass-2 variant of write_fast_autonomous_spef.

    Behaviour identical to the baseline except for `n_workers_pass2` workers
    in the per-net loop. Returns a stats dict with timing breakdown.
    """
    timings: dict[str, float] = {}

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
    # Also build the KD-tree in the parent for stat reporting parity; workers
    # rebuild their own from `records` in `_worker_init`.
    _ = build_kdtree(records)
    timings["kdtree_build_s"] = time.perf_counter() - t0

    # ---- Pass 2 parallel ----
    t0 = time.perf_counter()
    tasks = [
        (idx, net_name, str(path)) for idx, (net_name, path) in enumerate(nets_metadata)
    ]
    if chunksize is None:
        # Match the imap_unordered heuristic from baseline pass-1.
        chunksize = max(1, min(64, len(tasks) // (n_workers_pass2 * 4)))

    n_written = 0
    n_skipped = 0
    out_spef_path.parent.mkdir(parents=True, exist_ok=True)

    ctx = mp.get_context("spawn")
    by_net_dict = dict(by_net)  # convert defaultdict → plain dict for pickling
    init_args = (records, by_net_dict, top_ports, layer_info, tech_lef, max_dist_um, top_k)

    with open(out_spef_path, "w") as fh:
        spef_header_writer = SPEFWriter(fh, design_name=design_name, top_ports=top_ports)
        spef_header_writer.write_header()

        with ctx.Pool(
            processes=n_workers_pass2,
            initializer=_worker_init,
            initargs=init_args,
        ) as pool:
            # `imap` preserves task submission order ⇒ output SPEF body
            # ordering is identical to the serial baseline.
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
    return timings
