"""
Online context builder for inference-time per-net tiling without disk tiles.

Replaces build_dataset.py's spatial-grid preprocessing for inference:
  DEF → OnlineContextBuilder.build() → per-net tile tensors → model

Usage:
    ctx = OnlineContextBuilder(layer_info, mat_stack, window_size=(4,4,20), stride=1.0)
    ctx.build(def_parser)   # two-pass over all nets
    for net_name, tiles in ctx.iter_net_tiles():
        # tiles: list of {'tensor': (N,9), 'core_ratios': (N,), 'center': (3,), 'net_name': str}
        ...
"""

import numpy as np
from collections import defaultdict
from typing import Dict, Generator, List, Tuple

import configs.config as cfg


class _SpatialGrid:
    def __init__(self, bin_x: float, bin_y: float):
        self.bx = bin_x
        self.by = bin_y
        self.grid: Dict[Tuple[int, int], List[int]] = defaultdict(list)

    def build(self, geo: np.ndarray):
        """geo: (M, 7+) with [cx, cy, cz, w, h, d, net_id]."""
        mins = geo[:, :2] - geo[:, 3:5] / 2.0
        maxs = geo[:, :2] + geo[:, 3:5] / 2.0
        min_idx = np.floor(mins / [self.bx, self.by]).astype(np.int32)
        max_idx = np.floor(maxs / [self.bx, self.by]).astype(np.int32)
        for i in range(len(geo)):
            for x in range(min_idx[i, 0], max_idx[i, 0] + 1):
                for y in range(min_idx[i, 1], max_idx[i, 1] + 1):
                    self.grid[(x, y)].append(i)

    def query(self, center, window_size) -> List[int]:
        cx, cy = center[0], center[1]
        wx, wy = window_size[0], window_size[1]
        lo_x = int(np.floor((cx - wx / 2) / self.bx))
        hi_x = int(np.floor((cx + wx / 2) / self.bx))
        lo_y = int(np.floor((cy - wy / 2) / self.by))
        hi_y = int(np.floor((cy + wy / 2) / self.by))
        indices: set = set()
        for x in range(lo_x, hi_x + 1):
            for y in range(lo_y, hi_y + 1):
                if (x, y) in self.grid:
                    indices.update(self.grid[(x, y)])
        return list(indices)


class OnlineContextBuilder:
    """
    Two-pass online context builder for inference.

    Pass 1 (build):  Parse all nets → flat global_geo (M, 7) + spatial grid
    Pass 2 (tiles):  For each target net, slide windows and return (N, 9) tensors
    """

    HALO_MARGIN = 1.5   # um — core zone margin (matches build_dataset.py)
    CONTEXT_MARGIN = 2.0  # um — extra context beyond window

    def __init__(self, tensorizer, window_size=None, max_fracture_len=9.1):
        self.tensorizer = tensorizer
        self.win = np.array(window_size or cfg.WINDOW_SIZE, dtype=np.float64)
        self.stride = self.win[:2] - 2.0 * self.HALO_MARGIN
        self.ctx_size = np.array([
            self.win[0] + 2 * self.CONTEXT_MARGIN,
            self.win[1] + 2 * self.CONTEXT_MARGIN,
            self.win[2]
        ])

        self._global_geo = None   # (M, 7): [cx,cy,cz,w,h,d, net_id]
        self._net_data: Dict[int, dict] = {}   # net_id → {name, cuboids, segments}
        self._grid = _SpatialGrid(self.win[0], self.win[1])
        self._built = False

    # ------------------------------------------------------------------
    # Pass 1 — collect all geometry

    def build(self, def_parser_results):
        """
        Accepts either:
          - a DefStreamParser instance (calls .parse() internally), or
          - an iterable of (net_name, cuboids, segments) tuples.

        Builds the spatial index over ALL cuboids.
        """
        all_geo: List[np.ndarray] = []
        net_id = 0

        it = def_parser_results.parse() if hasattr(def_parser_results, 'parse') else def_parser_results

        for net_name, cuboids, segments in it:
            if cuboids is None or len(cuboids) == 0:
                continue
            ids = np.full((len(cuboids), 1), net_id, dtype=np.float32)
            row = np.hstack([cuboids.astype(np.float32), ids])
            all_geo.append(row)
            self._net_data[net_id] = {
                'name': net_name,
                'cuboids': cuboids,
                'segments': segments,
            }
            net_id += 1

        if not all_geo:
            self._global_geo = np.zeros((0, 7), dtype=np.float32)
            self._built = True
            return

        self._global_geo = np.vstack(all_geo)
        print(f"[OnlineContextBuilder] {len(self._global_geo):,} cuboids from {net_id} nets — building spatial grid...")
        self._grid.build(self._global_geo)
        self._built = True
        print("[OnlineContextBuilder] Ready.")

    # ------------------------------------------------------------------
    # Pass 2 — tile a single net and yield tensors

    def iter_net_tiles(self) -> Generator[Tuple[str, List[dict]], None, None]:
        """Yield (net_name, tiles) for every net in build order."""
        assert self._built, "Call build() first"
        for nid, info in self._net_data.items():
            tiles = self._tile_net(nid, info)
            if tiles:
                yield info['name'], tiles

    def get_net_tiles(self, net_name: str) -> List[dict]:
        """Return tiles for a single net by name (linear scan — use iter_net_tiles for batch)."""
        assert self._built, "Call build() first"
        for nid, info in self._net_data.items():
            if info['name'] == net_name:
                return self._tile_net(nid, info)
        return []

    # ------------------------------------------------------------------
    # Internal

    def _tile_net(self, nid: int, info: dict) -> List[dict]:
        cuboids = info['cuboids']  # (N, 6)
        if len(cuboids) == 0:
            return []

        cx_vals = cuboids[:, 0]
        cy_vals = cuboids[:, 1]
        x_min, x_max = cx_vals.min(), cx_vals.max()
        y_min, y_max = cy_vals.min(), cy_vals.max()

        stride_x, stride_y = self.stride[0], self.stride[1]
        win_x, win_y = self.win[0], self.win[1]

        # Tile centers from bounding box
        starts_x = np.arange(x_min + win_x / 2.0, x_max + win_x / 2.0 + 1e-6, stride_x)
        starts_y = np.arange(y_min + win_y / 2.0, y_max + win_y / 2.0 + 1e-6, stride_y)
        if len(starts_x) == 0:
            starts_x = np.array([(x_min + x_max) / 2.0])
        if len(starts_y) == 0:
            starts_y = np.array([(y_min + y_max) / 2.0])

        tiles = []
        for cx in starts_x:
            for cy in starts_y:
                center = np.array([cx, cy, self.win[2] / 2.0])
                tile = self._make_tile(nid, info['name'], center)
                if tile is not None:
                    tiles.append(tile)
        return tiles

    def _make_tile(self, nid: int, net_name: str, center: np.ndarray) -> dict | None:
        # Query spatial grid
        nearby_idx = self._grid.query(center, self.ctx_size)
        if not nearby_idx:
            return None

        geo = self._global_geo[nearby_idx]
        net_ids = geo[:, 6]

        # Core zone bounds
        core_min = center[:2] - self.win[:2] / 2.0 + self.HALO_MARGIN
        core_max = center[:2] + self.win[:2] / 2.0 - self.HALO_MARGIN

        box_min = geo[:, :2] - geo[:, 3:5] / 2.0
        box_max = geo[:, :2] + geo[:, 3:5] / 2.0
        in_core = (
            (box_max[:, 0] > core_min[0]) & (box_min[:, 0] < core_max[0]) &
            (box_max[:, 1] > core_min[1]) & (box_min[:, 1] < core_max[1])
        )

        is_tgt = (net_ids == nid)
        type_ids = np.zeros(len(geo), dtype=np.int32)
        type_ids[is_tgt & in_core] = 1   # target wire in core
        type_ids[is_tgt & ~in_core] = 3  # target wire in halo (treated as context)
        type_ids[~is_tgt] = 3            # aggressor

        valid = type_ids > 0
        if not np.any(valid) or not np.any(type_ids[valid] == 1):
            return None

        v_geo = geo[valid, :6]
        v_types = type_ids[valid]

        # core_ratios: fraction of cuboid area inside core zone
        v_box_min = v_geo[:, :2] - v_geo[:, 3:5] / 2.0
        v_box_max = v_geo[:, :2] + v_geo[:, 3:5] / 2.0
        inter_min = np.maximum(v_box_min, core_min)
        inter_max = np.minimum(v_box_max, core_max)
        inter_dims = np.maximum(inter_max - inter_min, 0.0)
        inter_area = inter_dims[:, 0] * inter_dims[:, 1]
        total_area = (v_geo[:, 3] * v_geo[:, 4]).clip(min=1e-9)
        core_ratios = (inter_area / total_area).astype(np.float32)

        # is_target_mask (1.0 for target-core cuboids)
        is_target_mask = (v_types == 1).astype(np.float32)

        tensor = self.tensorizer.process(v_geo, v_types, center[:2]).astype(np.float32)

        return {
            'tensor': tensor,               # (N, 9)
            'core_ratios': core_ratios,     # (N,)
            'is_target': is_target_mask,    # (N,) — A_tgt equivalent
            'center': center,
            'net_name': net_name,
        }
