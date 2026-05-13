"""Round 2.1 V4-A — pre-aggregate per-tile pkl.gz cache into a per-design
indexed asset so cold-start V4 H3 stops doing N_tile gzip+pickle reads
per net.

Schema v3 (NPY arrays + small metadata pickle, no per-target duplication):

    `<design>_v4_pernet.cubs.npy`         fp32 (N_total, 10)  mmap-able
    `<design>_v4_pernet.owner.npy`        int32 (N_total,)
    `<design>_v4_pernet.tile_offsets.npy` int64 (N_tiles + 1,)
    `<design>_v4_pernet.tgt_idx.npy`      int64 (sum_per_net_targets,)
    `<design>_v4_pernet.tgt_off.npy`      int64 (n_nets + 1,)
    `<design>_v4_pernet.tile_set.npy`     int32 (sum_per_net_tiles,)
    `<design>_v4_pernet.tile_set_off.npy` int64 (n_nets + 1,)
    `<design>_v4_pernet.meta.pkl`         small dict (design, schema_version, net_names list)

The schema-v2 single 6.7 GB pickle for tv80s decoded for 20+ min in our
test because Python's pickle has to allocate a separate `PyArrayObject`
for each of the 3,384 per-net arrays (target_idx + tile_set). Flat CSR
arrays here keep the per-net lookup numpy-only and let np.load mmap the
big tensors.

Cuboids are stored in tile_id ASC order; a per-net target_idx is the
row indices into `cubs` that this net owns. Per-net aggressors are
materialized at read time by iterating the net's tile_set and filtering
by `owner != self_id` (≈ 1 ms / net).

Usage:
    python3 TreePEX/scripts/build_v4_pernet_cache.py --design intel22_tv80s_f3
    python3 TreePEX/scripts/build_v4_pernet_cache.py --design intel22_nova_f3 --workers 16
"""
from __future__ import annotations

import argparse
import gzip
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "TreePEX" / "scripts"))

from pex_cold import TILE_CACHE_ROOT, DESIGNS


def _load_tile_indexed(args: Tuple[int, Path]):
    tile_id, tile_path = args
    try:
        with gzip.open(tile_path, "rb") as f:
            tile = pickle.load(f)
    except Exception:
        return tile_id, None, None
    cubs = tile.get("cuboids")
    names = tile.get("cuboid_net_names")
    if cubs is None or names is None or len(names) == 0:
        return tile_id, None, None
    return tile_id, np.asarray(cubs, dtype=np.float32), [str(n) for n in names]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True, choices=list(DESIGNS.keys()))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out-prefix", type=Path, default=None,
                    help="output prefix (default: TILE_CACHE_ROOT/<design>_v4_pernet)")
    args = ap.parse_args()

    tile_dir = TILE_CACHE_ROOT / args.design
    map_csv = TILE_CACHE_ROOT / f"{args.design}_map.csv"
    if not tile_dir.exists() or not map_csv.exists():
        raise SystemExit(f"tile cache missing for {args.design}")

    df = pd.read_csv(map_csv)
    tile_files = sorted(set(df["sample_filename"]))
    tile_paths = [tile_dir / fn for fn in tile_files]
    tile_jobs = list(enumerate(tile_paths))
    n_tiles = len(tile_paths)
    n_target_in_map = df["net_name"].nunique()
    print(f"[build] {args.design}: {n_tiles} unique tile files, "
          f"{len(df)} (net,tile) mappings, {n_target_in_map} target nets",
          flush=True)

    # === Pass 1: parallel tile load + flatten ===
    t0 = time.time()
    cubs_per_tile: List[np.ndarray] = [None] * n_tiles
    names_per_tile: List[List[str]] = [None] * n_tiles
    chunksize = max(1, n_tiles // (args.workers * 1000))
    done = 0
    with mp.Pool(processes=args.workers) as pool:
        for tile_id, cubs, names in pool.imap_unordered(
                _load_tile_indexed, tile_jobs, chunksize=chunksize):
            cubs_per_tile[tile_id] = cubs
            names_per_tile[tile_id] = names
            done += 1
            if done % 50000 == 0 or done == n_tiles:
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-3)
                print(f"  tile-load {done}/{n_tiles} elapsed={elapsed:.0f}s "
                      f"rate={rate:.0f}/s chunk={chunksize}", flush=True)

    # === Flatten + name id map ===
    t1 = time.time()
    net_name_to_id: Dict[str, int] = {}
    total_rows = sum(c.shape[0] for c in cubs_per_tile if c is not None)
    cubs_flat = np.empty((total_rows, 10), dtype=np.float32)
    owner_flat = np.empty(total_rows, dtype=np.int32)
    tile_id_flat = np.empty(total_rows, dtype=np.int32)
    row_off = 0
    for t in range(n_tiles):
        cubs = cubs_per_tile[t]
        names = names_per_tile[t]
        if cubs is None:
            continue
        n = cubs.shape[0]
        cubs_flat[row_off:row_off + n] = cubs
        for j, nm in enumerate(names):
            nid = net_name_to_id.get(nm)
            if nid is None:
                nid = len(net_name_to_id)
                net_name_to_id[nm] = nid
            owner_flat[row_off + j] = nid
        tile_id_flat[row_off:row_off + n] = t
        row_off += n
    n_nets = len(net_name_to_id)
    print(f"  flatten: {time.time() - t1:.1f}s  N_total={total_rows:,} n_nets={n_nets:,}",
          flush=True)

    # Free per-tile lists
    cubs_per_tile = names_per_tile = None

    # === Sort by tile_id (so each tile is contiguous in cubs_sorted) ===
    t2 = time.time()
    sort_idx = np.argsort(tile_id_flat, kind="stable")
    cubs_sorted = cubs_flat[sort_idx]
    owner_sorted = owner_flat[sort_idx]
    tile_id_sorted = tile_id_flat[sort_idx]
    del cubs_flat, owner_flat, tile_id_flat, sort_idx
    tile_counts = np.bincount(tile_id_sorted, minlength=n_tiles)
    tile_offsets = np.concatenate(([0], np.cumsum(tile_counts))).astype(np.int64)
    print(f"  sort+CSR: {time.time() - t2:.1f}s", flush=True)

    # === Per-net tile_set as flat CSR — built from `_map.csv` so the cache
    # matches the original `_v4_process_net` semantics: a tile is "target"
    # for net X only if (X, tile) appears in the design's map. Just because
    # X owns cuboids inside another tile (as an aggressor) does NOT make
    # that tile part of X's target tile set.
    t3 = time.time()
    tile_filename_to_id = {fn: i for i, fn in enumerate(tile_files)}
    df["tile_id_int"] = df["sample_filename"].map(tile_filename_to_id).astype("int32")
    tile_set_chunks: List[np.ndarray] = []
    tile_set_off = np.zeros(n_nets + 1, dtype=np.int64)
    map_groups = df.groupby("net_name", sort=False)["tile_id_int"]
    by_net_tile_ids: Dict[str, np.ndarray] = {
        net: np.sort(np.unique(g.values.astype(np.int32, copy=False)))
        for net, g in map_groups
    }
    for nid in range(n_nets):
        nm = None  # will fill from net_names below
        pass
    net_names_so_far = [None] * n_nets
    for name, nid in net_name_to_id.items():
        net_names_so_far[nid] = name
    for nid in range(n_nets):
        nm = net_names_so_far[nid]
        ts = by_net_tile_ids.get(nm)
        if ts is None or ts.size == 0:
            ts = np.zeros(0, dtype=np.int32)
        tile_set_chunks.append(ts)
        tile_set_off[nid + 1] = tile_set_off[nid] + ts.shape[0]
    tile_set_flat = (np.concatenate(tile_set_chunks)
                     if tile_set_chunks else np.zeros(0, dtype=np.int32))
    del tile_set_chunks
    print(f"  per-net CSR: {time.time() - t3:.1f}s  "
          f"|tile_set|={tile_set_flat.size:,}", flush=True)

    # === Save NPY + meta pickle ===
    out_prefix = args.out_prefix or (TILE_CACHE_ROOT / f"{args.design}_v4_pernet")
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    t4 = time.time()
    np.save(out_prefix.with_suffix(".cubs.npy"), cubs_sorted)
    np.save(out_prefix.with_suffix(".owner.npy"), owner_sorted)
    np.save(out_prefix.with_suffix(".tile_offsets.npy"), tile_offsets)
    np.save(out_prefix.with_suffix(".tile_set.npy"), tile_set_flat)
    np.save(out_prefix.with_suffix(".tile_set_off.npy"), tile_set_off)
    net_names = sorted(net_name_to_id.keys(), key=lambda n: net_name_to_id[n])
    meta = {
        "design": args.design,
        "schema_version": 4,
        "net_names": net_names,
        "n_total_rows": total_rows,
        "n_tiles": n_tiles,
    }
    with open(out_prefix.with_suffix(".meta.pkl"), "wb") as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
    # total disk size
    total_mb = 0
    for suffix in (".cubs.npy", ".owner.npy", ".tile_offsets.npy",
                   ".tile_set.npy", ".tile_set_off.npy", ".meta.pkl"):
        total_mb += out_prefix.with_suffix(suffix).stat().st_size / 1024**2
    print(f"[build] wrote {out_prefix}.* total={total_mb:.1f} MB  "
          f"wall_save={time.time() - t4:.1f}s  wall_total={time.time() - t0:.1f}s",
          flush=True)


if __name__ == "__main__":
    main()
