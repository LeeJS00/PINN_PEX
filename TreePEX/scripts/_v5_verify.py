"""Schema v5 correctness probe.

Pick one tile pkl.gz, extract its origin + cuboid set, then re-derive the
same cuboid set from the DEF parse (geo['all_cuboids'] + owner array) by a
14×14 µm xy bbox query around the origin. Report set diff so we know
whether schema-v5 reconstruction is bit-exact.
"""
from __future__ import annotations

import gzip
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "TreePEX" / "scripts"))

from pex_cold import (
    DESIGNS, TECH_LEF_PATH, CELL_LEF_PATH, LAYERS_INFO_PATH,
    SpatialGrid, scan_design,
)
from src.preprocessing.lef_parser import LefParser
from src.preprocessing.cell_parser import CellLibParser
from src.preprocessing.layer_parser import LayerInfoParser
from src.physics.materials import BEOLMaterialStack


WINDOW_HALF = 7.0  # the absolute-xy half-side measured from sample tile (max span 14)


def main():
    design = "intel22_tv80s_f3"
    tile_name = "intel22_tv80s_f3__A_0_tile16.pkl.gz"
    tile_path = Path("/data/PINNPEX/data/processed_v3/intel22") / design / tile_name

    with gzip.open(tile_path, "rb") as f:
        tile = pickle.load(f)
    origin = np.asarray(tile["origin"], dtype=np.float64)
    tile_cubs_local = np.asarray(tile["cuboids"], dtype=np.float64)
    tile_cubs_abs = np.asarray(tile["abs_geometries"], dtype=np.float64)
    tile_names = np.asarray([str(n) for n in tile["cuboid_net_names"]])
    print(f"tile {tile_name}: origin={origin} n_cubs={len(tile_cubs_local)}")
    print(f"abs xy bbox: x=[{tile_cubs_abs[:, 0].min():.3f}, {tile_cubs_abs[:, 0].max():.3f}] "
          f"y=[{tile_cubs_abs[:, 1].min():.3f}, {tile_cubs_abs[:, 1].max():.3f}]")

    # Parse DEF
    print("parsing DEF...")
    layer_map = LayerInfoParser(LAYERS_INFO_PATH).parse()
    tech_lef = LefParser(TECH_LEF_PATH).parse()
    cell_lib = CellLibParser(CELL_LEF_PATH).parse()
    geo = scan_design(DESIGNS[design], layer_map, tech_lef, cell_lib)

    # Include VSS (tiles do) and exclude INST_PORT_* / PIN_* (tiles don't have those).
    all_cubs_with_vss = np.vstack([geo["all_cuboids"], geo["vss"]])
    vss_owner = np.array(["VSS_RAIL"] * len(geo["vss"]), dtype=object)
    all_owner_full = np.concatenate([geo["all_owner"], vss_owner])
    # Strip pseudo nets the tile builder doesn't keep
    keep = np.array([("INST_PORT_" not in str(n)) and ("PIN_" not in str(n).upper())
                     for n in all_owner_full])
    all_cubs_with_vss = all_cubs_with_vss[keep]
    all_owner_full = all_owner_full[keep]

    grid = SpatialGrid()
    grid.build(all_cubs_with_vss)
    cx, cy = origin[0], origin[1]
    cand_idx = grid.query_bbox(cx - WINDOW_HALF, cx + WINDOW_HALF,
                               cy - WINDOW_HALF, cy + WINDOW_HALF)
    cand_cubs = all_cubs_with_vss[cand_idx]
    cand_owners = all_owner_full[cand_idx]
    # SpatialGrid already returns OVERLAP (bbox of cuboid overlaps query bbox);
    # use it directly as the tile inclusion rule.
    inside = np.ones(len(cand_cubs), dtype=bool)
    print(f"  grid query: {len(cand_idx)} candidates")
    rec_cubs = cand_cubs[inside]
    rec_owners = cand_owners[inside]
    rec_owners_set = set(str(o) for o in rec_owners)
    tile_owners_set = set(tile_names)
    print(f"  tile unique owners: {len(tile_owners_set)}, reconstructed owners: {len(rec_owners_set)}")
    print(f"  owners only in tile: {len(tile_owners_set - rec_owners_set)}")
    print(f"  owners only in reconstruction: {len(rec_owners_set - tile_owners_set)}")
    # Show a few mismatched names
    only_tile = list(tile_owners_set - rec_owners_set)[:10]
    only_rec = list(rec_owners_set - tile_owners_set)[:10]
    print(f"    tile only sample: {only_tile}")
    print(f"    rec only sample:  {only_rec}")
    # Cuboid count by owner: choose a sample owner present in both
    common = list(tile_owners_set & rec_owners_set)
    print(f"  common owners: {len(common)} (showing per-owner cuboid count for first 8):")
    for owner in common[:8]:
        n_tile = int((tile_names == owner).sum())
        n_rec = int((rec_owners == owner).sum())
        print(f"    {owner[:60]:60s}  tile={n_tile:5d}  rec={n_rec:5d}  Δ={n_rec - n_tile:+d}")


if __name__ == "__main__":
    main()
