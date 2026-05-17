"""00_build_design_parallel.py — generic 16-worker parallel V3 feature builder.

Clone of 00_build_ldpc_parallel.py with --design CLI arg, so any ASAP7 design
can be rebuilt at ldpc-style throughput (11+ nets/s) instead of single-process
(2 nets/s). Use for vga_enh after wb_conmax finishes in main script.

Usage:
    python3 00_build_design_parallel.py --design asap7_vga_enh_top_x1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import multiprocessing as mp
import numpy as np
import pandas as pd

ROOT = Path("/home/jslee/projects/PINNPEX")
EXP = ROOT / "experiments" / "tv80s_autonomous_2026_05_02"
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP / "src"))

from baselines.feature_dataset import (
    _scan_design_geometry, parse_spef_to_dict, _bbox_from_cuboids,
    _enumerate_coupling_edges, _layer_eps_array, _np_to_cuboid_arr,
    LayerInfoParser,
)
from baselines.features import NetFeatureVector, NetGeometry, extract_features_from_geometry

DATA_ROOT = Path("/data/PINNPEX/data/processed_v3/asap7")
DEF_DIR = Path("/home/jslee/projects/PINNPEX/data/raw/def/asap7")
SPEF_DIR = Path("/home/jslee/projects/PINNPEX/golden_data/spef_data/asap7")
PDK = ROOT / "tool" / "pdk" / "7nm"
LAYERS = PDK / "layers" / "layers.info"
TECH = PDK / "lef" / "asap7_tech_1x_201209_JS.lef"
CELL = PDK / "lef" / "asap7sc7p5t_28_R_1x_220121a.lef"
CUTOFF = 4.0
MAX_AGGR = 768  # L9 2026-05-16: match intel22 cap
N_WORKERS = 16

_GEO = None
_SPEF = None
_MANIFEST_BY_NET = None
_LAYER_EPS = None
_DENSITY = None
_DENSITY_WINDOW = None
_DEF_STEM = None


def _init_worker(geo, spef_dict, manifest_by_net, layer_eps, density, density_window, def_stem):
    global _GEO, _SPEF, _MANIFEST_BY_NET, _LAYER_EPS, _DENSITY, _DENSITY_WINDOW, _DEF_STEM
    _GEO = geo
    _SPEF = spef_dict
    _MANIFEST_BY_NET = manifest_by_net
    _LAYER_EPS = layer_eps
    _DENSITY = density
    _DENSITY_WINDOW = density_window
    _DEF_STEM = def_stem


def _process_net(net_name: str) -> dict:
    target_arr = _GEO["nets"][net_name]
    spef_rec = _SPEF[net_name]
    edges = _enumerate_coupling_edges(
        target_arr=target_arr,
        all_cuboids=_GEO["all_cuboids"],
        all_owner=_GEO["all_owner"],
        target_net_name=net_name,
        cutoff_um=CUTOFF,
        max_edges=MAX_AGGR,
    )
    if len(_GEO["vss"]) > 0:
        txmin, txmax, tymin, tymax = _bbox_from_cuboids(target_arr)
        c = CUTOFF
        txmin -= c; txmax += c; tymin -= c; tymax += c
        vxmin = _GEO["vss"][:, 0] - _GEO["vss"][:, 3] / 2
        vxmax = _GEO["vss"][:, 0] + _GEO["vss"][:, 3] / 2
        vymin = _GEO["vss"][:, 1] - _GEO["vss"][:, 4] / 2
        vymax = _GEO["vss"][:, 1] + _GEO["vss"][:, 4] / 2
        inside = (vxmax >= txmin) & (vxmin <= txmax) & (vymax >= tymin) & (vymin <= tymax)
        vss_subset = _GEO["vss"][inside]
    else:
        vss_subset = np.zeros((0, 7), dtype=np.float64)
    net_geo = NetGeometry(
        net_name=net_name,
        design_name=_DEF_STEM,
        target_cuboids=_np_to_cuboid_arr(target_arr),
        coupling_edges=edges,
        vss_cuboids=_np_to_cuboid_arr(vss_subset),
        layer_stack_eps=_LAYER_EPS,
        fanout=len(spef_rec["coupled_caps"]),
        n_layers_total=10,
        ground_plane_layer=0,
        local_density_window_um2=_DENSITY_WINDOW,
        local_metal_area_per_layer_um2=_DENSITY.tolist(),
    )
    fv = extract_features_from_geometry(net_geo)
    row = {
        "design_name": _DEF_STEM,
        "net_name": net_name,
        "split": _MANIFEST_BY_NET[net_name],
        "total_cap_fF": spef_rec["total_cap_fF"],
        "c_gnd_fF": spef_rec["ground_cap_fF"],
        "c_cpl_total_fF": spef_rec["c_cpl_total_fF"],
        "total_res_ohm": spef_rec["total_res_ohm"],
        **{f.name: getattr(fv, f.name) for f in NetFeatureVector.__dataclass_fields__.values()},
    }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True, help="e.g. asap7_vga_enh_top_x1")
    ap.add_argument("--n_workers", type=int, default=N_WORKERS)
    args = ap.parse_args()
    design = args.design
    n_workers = args.n_workers

    def_path = DEF_DIR / f"{design}.def"
    spef_path = SPEF_DIR / f"{design}_starrc.spef"
    out_path = DATA_ROOT / "features" / f"{design}.csv"

    if not def_path.exists():
        raise FileNotFoundError(f"DEF not found: {def_path}")
    if not spef_path.exists():
        raise FileNotFoundError(f"SPEF not found: {spef_path}")

    print(f"=== Parallel V3 feature extraction (design={design}, N_WORKERS={n_workers}) ===", flush=True)
    print(f"  DEF:  {def_path}", flush=True)
    print(f"  SPEF: {spef_path}", flush=True)
    print(f"  OUT:  {out_path}", flush=True)
    print(f"  MAX_AGGR cap: {MAX_AGGR}", flush=True)

    layer_map = LayerInfoParser(LAYERS).parse()
    layer_eps = _layer_eps_array(layer_map, n_layers=10)

    print(f"parsing DEF: {def_path.name}", flush=True)
    t0 = time.time()
    geo = _scan_design_geometry(def_path, layer_map, tech_lef_path=TECH, cell_lef_path=CELL)
    print(f"  {len(geo['nets']):,} nets, {len(geo['vss']):,} VSS cuboids ({time.time()-t0:.0f}s)", flush=True)

    print(f"parsing SPEF: {spef_path.name}", flush=True)
    t0 = time.time()
    spef_dict = parse_spef_to_dict(spef_path)
    print(f"  {len(spef_dict):,} SPEF nets ({time.time()-t0:.0f}s)", flush=True)

    manifest = pd.read_csv(DATA_ROOT / "dataset_manifest.csv")
    manifest_subset = manifest[manifest["design_name"] == design]
    manifest_by_net = {}
    for _, row in manifest_subset.iterrows():
        manifest_by_net.setdefault(str(row["net_name"]), str(row["split"]))
    common = sorted(set(manifest_by_net.keys()) & set(spef_dict.keys()) & set(geo["nets"].keys()))
    print(f"common nets: {len(common):,}", flush=True)

    density = np.zeros(11, dtype=np.float64)
    for arr in geo["nets"].values():
        for i in range(1, 10):
            mask = arr[:, 6] == i
            density[i] += float((arr[mask, 3] * arr[mask, 4]).sum())
    if len(geo["all_cuboids"]) > 0:
        xmin, xmax, ymin, ymax = _bbox_from_cuboids(geo["all_cuboids"])
        density_window = max(1.0, (xmax - xmin) * (ymax - ymin))
    else:
        density_window = 1.0

    print(f">>> dispatching {len(common):,} nets across {n_workers} workers ...", flush=True)
    rows = []
    t0 = time.time()
    chunksize = max(1, len(common) // (n_workers * 100))
    def_stem = def_path.stem
    with mp.Pool(processes=n_workers, initializer=_init_worker,
                 initargs=(geo, spef_dict, manifest_by_net, layer_eps,
                           density, density_window, def_stem)) as pool:
        for i, row in enumerate(pool.imap_unordered(_process_net, common,
                                                     chunksize=chunksize), 1):
            rows.append(row)
            if i % 2000 == 0 or i == len(common):
                el = time.time() - t0
                rate = i / max(el, 1e-3)
                eta = (len(common) - i) / max(rate, 1e-3)
                print(f"  {i:,}/{len(common):,} elapsed={el:.0f}s rate={rate:.1f}/s "
                      f"eta={eta:.0f}s ({eta/60:.1f}min) chunksize={chunksize}", flush=True)

    print(f"\n>>> writing {len(rows):,} rows → {out_path}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
