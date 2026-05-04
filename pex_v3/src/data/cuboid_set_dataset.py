"""
cuboid_set_dataset.py — per-net cuboid sequence loader for HybridPexV3Mesh.

Loads the per-design `.npz` files written by `18_extract_per_net_cuboids.py`
and indexes per-net by (design, net) string key. The dataset returns
variable-length cuboid arrays per net; the collate function pads to
batch-max and emits a padding mask.

Used together with the v3 features CSV (analytic + scalar features).
The dataset returns a dict augmenting `df_to_tensors` with:
    cuboids:      (N_max, in_dim) float32
    padding_mask: (N_max,)        float32 (1=valid)
    n_cuboids:    int
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PerNetCuboidStore:
    """Loads all per-design npz files and indexes by (design, net_name)."""

    def __init__(self, npz_dir: Path):
        self.npz_dir = Path(npz_dir)
        self._index: dict[tuple[str, str], int] = {}
        self._design_data: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        for npz_path in sorted(self.npz_dir.glob("intel22_*.npz")):
            design = npz_path.stem
            data = np.load(npz_path, allow_pickle=True)
            self._design_data[design] = {
                "net_names": data["net_names"],
                "cuboids": data["cuboids"],   # object array of (N_i, 10)
                "n_cuboids": data["n_cuboids"],
            }
            # Build (design, net) -> array index
            for i, n in enumerate(data["net_names"]):
                self._index[(design, str(n))] = i

    def get(self, design: str, net_name: str) -> Optional[np.ndarray]:
        idx = self._index.get((design, net_name))
        if idx is None:
            return None
        return self._design_data[design]["cuboids"][idx]

    def has(self, design: str, net_name: str) -> bool:
        return (design, net_name) in self._index

    def __len__(self) -> int:
        return len(self._index)


class CuboidAugmentedDataset(Dataset):
    """Wrap a v3 features DataFrame slice + cuboid store.

    Each item is a dict:
        analytic_gnd:  scalar
        analytic_cpl:  scalar
        self_features: (16,)
        pair_features: (24,)
        golden_gnd:    scalar
        golden_cpl:    scalar
        cuboids:       (N_i, in_dim)
        n_cuboids:     int
    """

    def __init__(
        self,
        df: pd.DataFrame,
        store: PerNetCuboidStore,
        self_feature_cols: list[str],
        pair_feature_cols: list[str],
        self_dim: int = 16,
        pair_dim: int = 24,
        max_cuboids_per_net: int = 512,
        cuboid_in_dim: int = 10,
    ):
        # Filter df to nets present in the store
        keep = df.apply(
            lambda r: store.has(r["design_name"], r["net_name"]),
            axis=1,
        )
        self.df = df[keep].reset_index(drop=True)
        self.store = store
        self.self_dim = self_dim
        self.pair_dim = pair_dim
        self.self_cols = self_feature_cols
        self.pair_cols = pair_feature_cols
        self.max_cuboids = max_cuboids_per_net
        self.cuboid_in_dim = cuboid_in_dim

        n_dropped = (~keep).sum()
        if n_dropped > 0:
            print(f"[CuboidAugmentedDataset] dropped {n_dropped} nets not in store")

        # Pre-extract scalar features as ndarray for speed
        self._self_feats = self._featurize(self.df, self.self_cols, self_dim)
        self._pair_feats = self._featurize(self.df, self.pair_cols, pair_dim)
        self._analytic_gnd = self.df["compact_gnd_estimate_fF"].fillna(0.0).to_numpy(dtype=np.float32)
        self._analytic_cpl = self.df["compact_cpl_estimate_total_fF"].fillna(0.0).to_numpy(dtype=np.float32)
        self._gold_gnd = self.df["c_gnd_fF"].fillna(0.0).to_numpy(dtype=np.float32)
        self._gold_cpl = self.df["c_cpl_total_fF"].fillna(0.0).to_numpy(dtype=np.float32)
        self._designs = self.df["design_name"].to_numpy()
        self._nets = self.df["net_name"].to_numpy()

    @staticmethod
    def _featurize(df: pd.DataFrame, cols: list[str], dim: int) -> np.ndarray:
        out = np.zeros((len(df), dim), dtype=np.float32)
        for i, c in enumerate(cols[:dim]):
            if c in df.columns:
                v = df[c].fillna(0.0).clip(lower=0).to_numpy(dtype=np.float32)
                v = np.log1p(v)
                out[:, i] = v
        return out

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        cuboids = self.store.get(self._designs[idx], self._nets[idx])
        # Already truncated at extraction time but defense-in-depth
        if cuboids is None:
            cuboids = np.zeros((1, self.cuboid_in_dim), dtype=np.float32)
        if len(cuboids) > self.max_cuboids:
            # Random subsample for in-batch consistency (shouldn't fire for v3 cap=512)
            keep_idx = np.random.choice(len(cuboids), self.max_cuboids, replace=False)
            cuboids = cuboids[keep_idx]
        return {
            "analytic_gnd": float(self._analytic_gnd[idx]),
            "analytic_cpl": float(self._analytic_cpl[idx]),
            "self_features": self._self_feats[idx],
            "pair_features": self._pair_feats[idx],
            "golden_gnd": float(self._gold_gnd[idx]),
            "golden_cpl": float(self._gold_cpl[idx]),
            "cuboids": cuboids,
            "n_cuboids": len(cuboids),
            "design_name": str(self._designs[idx]),
            "net_name": str(self._nets[idx]),
        }


def collate_cuboid_batch(items: list[dict]) -> dict:
    """Pad cuboids to batch-max + emit padding mask."""
    B = len(items)
    n_max = max(it["n_cuboids"] for it in items) if items else 1
    in_dim = items[0]["cuboids"].shape[1] if items else 10

    cuboids = np.zeros((B, n_max, in_dim), dtype=np.float32)
    mask = np.zeros((B, n_max), dtype=np.float32)
    for i, it in enumerate(items):
        n = it["n_cuboids"]
        cuboids[i, :n] = it["cuboids"]
        mask[i, :n] = 1.0

    return {
        "analytic_gnd": torch.tensor(
            np.array([it["analytic_gnd"] for it in items], dtype=np.float32)
        ),
        "analytic_cpl": torch.tensor(
            np.array([it["analytic_cpl"] for it in items], dtype=np.float32)
        ),
        "self_features": torch.tensor(
            np.array([it["self_features"] for it in items])
        ),
        "pair_features": torch.tensor(
            np.array([it["pair_features"] for it in items])
        ),
        "golden_gnd": torch.tensor(
            np.array([it["golden_gnd"] for it in items], dtype=np.float32)
        ),
        "golden_cpl": torch.tensor(
            np.array([it["golden_cpl"] for it in items], dtype=np.float32)
        ),
        "cuboids": torch.tensor(cuboids),
        "padding_mask": torch.tensor(mask),
        "design_name": [it["design_name"] for it in items],
        "net_name": [it["net_name"] for it in items],
    }
