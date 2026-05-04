#!/usr/bin/env python3
"""
18_extract_per_net_cuboids.py — Per-net cuboid tensor extraction.

For each (design, net) in the v3 manifest, aggregates ALL target cuboids
(filtered by `cuboid_net_names == net_name`) across all tiles for that
net. Saves a compact per-design npz with:
    cuboids: object array of (N_i, 10) float32 per net
    abs_geos: object array of (N_i, 6) float32 per net
    net_names: (M,) array
    splits: (M,) train/valid/test

Used by Path 2 cuboid set encoder MVP. Run once (~30min-1h).

Output structure:
    /data/PINNPEX/data/processed_v3/intel22/per_net_cuboids/
        <design_name>.npz   — one file per design

Each design has order ~10K-100K nets. We keep object arrays (variable N)
and let downstream pad-to-max in the dataloader.
"""
from __future__ import annotations
import argparse
import gzip
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "pex_v3"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Per-net cuboid extraction")
    p.add_argument(
        "--manifest", type=Path,
        default=Path("/data/PINNPEX/data/processed_v3/intel22/dataset_manifest_v3.csv"),
    )
    p.add_argument(
        "--data-root", type=Path,
        default=Path("/data/PINNPEX/data/processed_v3/intel22"),
    )
    p.add_argument(
        "--out-dir", type=Path,
        default=Path("/data/PINNPEX/data/processed_v3/intel22/per_net_cuboids"),
    )
    p.add_argument(
        "--designs", nargs="*", default=None,
        help="Limit to specific designs (default: all)",
    )
    p.add_argument(
        "--max-cuboids-per-net", type=int, default=512,
        help="Cap per-net cuboid count (random subsample if exceeded). "
             "512 covers >99%% of nets, prevents huge nets from dominating memory.",
    )
    return p.parse_args()


def _load_tile(tile_path: Path) -> dict:
    with gzip.open(tile_path, "rb") as f:
        return pickle.load(f)


def _process_design(
    design: str,
    design_df: pd.DataFrame,
    data_root: Path,
    out_dir: Path,
    max_cuboids_per_net: int,
) -> dict:
    """Aggregate all cuboids per net for one design."""
    t0 = time.time()
    print(f">>> {design}: {len(design_df):,} tiles, {design_df.net_name.nunique():,} nets")

    # Group tiles by net
    net_to_tiles = design_df.groupby("net_name")
    n_nets = len(net_to_tiles)

    rng = np.random.default_rng(42)
    per_net_cuboids: dict[str, np.ndarray] = {}
    per_net_abs_geos: dict[str, np.ndarray] = {}
    per_net_splits: dict[str, str] = {}

    n_skipped = 0
    n_truncated = 0

    for net_name, sub in net_to_tiles:
        cuboid_chunks = []
        abs_geo_chunks = []
        split = sub["split"].iloc[0]  # all tiles of a net share split (H1)

        for _, row in sub.iterrows():
            tile_path = data_root / row["rel_path"]
            if not tile_path.exists():
                continue
            try:
                tile = _load_tile(tile_path)
            except Exception:
                continue

            cuboids = tile.get("cuboids")
            cuboid_net_names = tile.get("cuboid_net_names")
            abs_geos = tile.get("abs_geometries")
            if cuboids is None or cuboid_net_names is None:
                continue

            # Filter to TARGET cuboids only (drop aggressors)
            mask = np.array([n == net_name for n in cuboid_net_names])
            if not mask.any():
                continue
            cuboid_chunks.append(cuboids[mask])
            if abs_geos is not None:
                abs_geo_chunks.append(abs_geos[mask])

        if not cuboid_chunks:
            n_skipped += 1
            continue

        all_cuboids = np.concatenate(cuboid_chunks, axis=0)
        all_abs_geos = (
            np.concatenate(abs_geo_chunks, axis=0)
            if abs_geo_chunks else np.zeros((len(all_cuboids), 6), dtype=np.float32)
        )

        # Optional truncation if too many cuboids
        if len(all_cuboids) > max_cuboids_per_net:
            idx = rng.choice(len(all_cuboids), max_cuboids_per_net, replace=False)
            idx.sort()
            all_cuboids = all_cuboids[idx]
            all_abs_geos = all_abs_geos[idx]
            n_truncated += 1

        per_net_cuboids[net_name] = all_cuboids.astype(np.float32)
        per_net_abs_geos[net_name] = all_abs_geos.astype(np.float32)
        per_net_splits[net_name] = split

    elapsed = time.time() - t0

    # Save as per-design npz (object arrays)
    net_names = list(per_net_cuboids.keys())
    cuboids_arr = np.empty(len(net_names), dtype=object)
    abs_geos_arr = np.empty(len(net_names), dtype=object)
    splits_arr = np.empty(len(net_names), dtype=object)
    n_cuboids_arr = np.zeros(len(net_names), dtype=np.int32)
    for i, n in enumerate(net_names):
        cuboids_arr[i] = per_net_cuboids[n]
        abs_geos_arr[i] = per_net_abs_geos[n]
        splits_arr[i] = per_net_splits[n]
        n_cuboids_arr[i] = len(per_net_cuboids[n])

    out_path = out_dir / f"{design}.npz"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        net_names=np.array(net_names),
        cuboids=cuboids_arr,
        abs_geos=abs_geos_arr,
        splits=splits_arr,
        n_cuboids=n_cuboids_arr,
    )

    print(
        f"    {design}: kept={len(net_names):,}  skipped={n_skipped}  "
        f"truncated={n_truncated}  "
        f"avg_cuboids={n_cuboids_arr.mean():.1f}  "
        f"max={n_cuboids_arr.max()}  "
        f"elapsed={elapsed:.1f}s  "
        f"out={out_path.name}"
    )

    return {
        "design": design,
        "n_nets": len(net_names),
        "n_skipped": n_skipped,
        "n_truncated": n_truncated,
        "avg_cuboids": float(n_cuboids_arr.mean()),
        "max_cuboids": int(n_cuboids_arr.max()),
        "elapsed_sec": elapsed,
    }


def main() -> None:
    args = parse_args()
    print(f">>> manifest: {args.manifest}")
    print(f">>> data:     {args.data_root}")
    print(f">>> out:      {args.out_dir}")
    print(f">>> max_cuboids_per_net: {args.max_cuboids_per_net}")

    df = pd.read_csv(args.manifest)
    print(f">>> manifest rows: {len(df):,}")

    designs = args.designs or sorted(df["design_name"].unique())
    print(f">>> processing {len(designs)} designs:")
    for d in designs:
        print(f"    - {d}")

    summaries = []
    t_total = time.time()
    for design in designs:
        sub = df[df["design_name"] == design]
        if sub.empty:
            print(f">>> {design}: no rows in manifest, skipping")
            continue
        summary = _process_design(
            design, sub, args.data_root, args.out_dir,
            args.max_cuboids_per_net,
        )
        summaries.append(summary)

    total_elapsed = time.time() - t_total
    print()
    print(f"=" * 60)
    print(f"✅ Done in {total_elapsed:.1f}s")
    print(f"  total nets extracted: {sum(s['n_nets'] for s in summaries):,}")
    print(f"  total truncated:      {sum(s['n_truncated'] for s in summaries):,}")
    print(f"  output dir:           {args.out_dir}")

    summary_csv = args.out_dir / "_extraction_summary.csv"
    pd.DataFrame(summaries).to_csv(summary_csv, index=False)
    print(f"  summary:              {summary_csv}")


if __name__ == "__main__":
    main()
