#!/usr/bin/env python3
"""profile_pass2.py — instrumented baseline run to confirm pass-2 sub-stage costs.

This is read-only against the baseline engine: we replicate its loop in this
file with `time.perf_counter()` per sub-stage so we can attribute the 52.4 s
pass-2 cost across (topology pkl.gz reload, aggressor KD-tree query,
RCTopologyBuilder, NetCapWriter+SPEFWriter).

Output: profile_pass2.json next to this script.
"""
from __future__ import annotations

import gzip
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]  # PINNPEX repo root
sys.path.insert(0, str(_ROOT))

from configs import config_v3 as cfg  # noqa: E402
from src.preprocessing.layer_parser import LayerInfoParser  # noqa: E402
from src.preprocessing.lef_parser import LefParser  # noqa: E402
from src.utils.spef_writer import NetCapWriter, RCTopologyBuilder, SPEFWriter  # noqa: E402
from src.v3.utils.fast_spef_engine import (  # noqa: E402
    analytic_per_net_cap_estimate,
    build_kdtree,
    compute_aggressor_weights,
    stream_index_pass,
)


DESIGN = "intel22_tv80s_f3"
TOPO_DIR = Path("/data/PINNPEX/data/processed_v3/intel22") / DESIGN / "topology"
OUT_DIR = Path(__file__).parent / "profile_runs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    layer_info = LayerInfoParser(cfg.LAYERS_INFO_PATH).parse()
    tech_lef = LefParser(cfg.TECH_LEF_PATH).parse()

    timings = {
        "index_pass_s": 0.0,
        "kdtree_build_s": 0.0,
        "topo_reload_s": 0.0,
        "analytic_estimate_s": 0.0,
        "aggressor_weights_s": 0.0,
        "rc_topology_build_s": 0.0,
        "spef_write_s": 0.0,
    }

    t0 = time.perf_counter()
    records, nets_metadata, top_ports = stream_index_pass(TOPO_DIR, n_workers=1)
    timings["index_pass_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    tree = build_kdtree(records)
    by_net: dict[str, list] = defaultdict(list)
    for r in records:
        by_net[r.net_name].append(r)
    timings["kdtree_build_s"] = time.perf_counter() - t0

    n_written = 0
    out_spef = OUT_DIR / f"{DESIGN}_profiled.spef"
    t_pass2_start = time.perf_counter()
    with open(out_spef, "w") as fh:
        spef = SPEFWriter(fh, design_name=DESIGN, top_ports=top_ports)
        spef.write_header()
        for net_name, path in nets_metadata:
            ts = time.perf_counter()
            try:
                with gzip.open(path, "rb") as f:
                    topo = pickle.load(f)
            except Exception:
                continue
            global_segments = topo.get("global_segments", [])
            del topo
            timings["topo_reload_s"] += time.perf_counter() - ts
            if not global_segments:
                continue

            ts = time.perf_counter()
            target_segs = by_net.get(net_name, [])
            c_gnd_total, c_cpl_total = analytic_per_net_cap_estimate(target_segs)
            timings["analytic_estimate_s"] += time.perf_counter() - ts
            if c_gnd_total <= 0 and c_cpl_total <= 0:
                continue

            ts = time.perf_counter()
            aggr_weights = compute_aggressor_weights(
                target_segs, records, tree, target_net_name=net_name,
                max_dist_um=5.0, top_k=20,
            )
            c_cpl_dict = {n: c_cpl_total * w for n, w in aggr_weights.items()}
            timings["aggressor_weights_s"] += time.perf_counter() - ts

            ts = time.perf_counter()
            try:
                rc = RCTopologyBuilder(
                    net_name=net_name,
                    global_segments=global_segments,
                    top_ports=top_ports,
                    layer_info=layer_info,
                    tech_lef=tech_lef,
                )
            except Exception:
                continue
            timings["rc_topology_build_s"] += time.perf_counter() - ts

            ts = time.perf_counter()
            try:
                writer = NetCapWriter(rc, c_gnd_total, c_cpl_dict)
                spef.stream_net_cap_writer(writer)
                n_written += 1
            except Exception:
                continue
            timings["spef_write_s"] += time.perf_counter() - ts

    pass2_total = time.perf_counter() - t_pass2_start
    timings["pass2_total_s"] = pass2_total
    timings["nets_written"] = n_written

    out = {
        "design": DESIGN,
        "n_nets_meta": len(nets_metadata),
        "n_workers_index_pass": 1,
        "timings_s": timings,
        "pass2_subtotal_s": (
            timings["topo_reload_s"]
            + timings["analytic_estimate_s"]
            + timings["aggressor_weights_s"]
            + timings["rc_topology_build_s"]
            + timings["spef_write_s"]
        ),
    }
    out_json = Path(__file__).parent / "profile_pass2.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
