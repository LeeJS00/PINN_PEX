"""
Per-net cuboid set dataset for the DeepSet/MLP baseline.

Loads each net's tile pkls on demand, deduplicates target cuboids by absolute
geometry hash, and emits:
    target_features: (T, F)   T = # target cuboids, F = 10
    aggressor_features: (A, F)
    pwr_features: (P, F)
    hand_features: (H,)        from the cached parquet
    y:               total_cap_fF (or 3-target vector)

For training we cap T,A,P at MAX_PER_KIND. Multiple seeds use different
random subsamples to reduce variance.
"""
from __future__ import annotations

import gzip
import os
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from configs import cfg


MAX_TARGET = 256
MAX_AGG    = 512
MAX_PWR    = 256


def _load_tile(p: Path) -> dict:
    with gzip.open(p, "rb") as f:
        return pickle.load(f)


class CuboidSetDataset(Dataset):
    """Walk the manifest grouping by (design, net), keep tile paths, lazy-load.

    `target` is a string column in `df` to predict (e.g., 'total_cap_fF').
    `df` already has features and targets from the parquet.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        manifest: pd.DataFrame,
        feature_cols,
        targets=None,
        max_target: int = MAX_TARGET,
        max_agg: int = MAX_AGG,
        max_pwr: int = MAX_PWR,
        train: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.targets = targets or [cfg.PRIMARY_TARGET]
        self.max_target = max_target
        self.max_agg    = max_agg
        self.max_pwr    = max_pwr
        self.train = train
        # Build a lookup: (design, net) → list[abs_path]
        m = manifest.copy()
        m["abs_path"] = str(cfg.DATA_ROOT) + "/" + m["rel_path"].astype(str)
        self.tile_lookup = m.groupby(["design_name", "net_name"])["abs_path"].apply(list).to_dict()

        # standardize hand features (will be filled by caller)
        self.hand_mean = None
        self.hand_std  = None

    def set_hand_normalizer(self, mean: np.ndarray, std: np.ndarray):
        self.hand_mean = mean.astype(np.float32)
        self.hand_std  = std.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def _gather_cuboids(self, design: str, net: str):
        paths = self.tile_lookup.get((design, net), [])
        target_rows = []
        agg_rows = []
        pwr_rows = []
        for p in paths:
            try:
                rec = _load_tile(Path(p))
            except Exception:
                continue
            c = rec["cuboids"]   # (M, 10)
            mask_t   = c[:, 7] == 1.0
            mask_agg = (c[:, 7] == 0.0) & (c[:, 9] < 0.6)
            mask_pwr = (c[:, 7] == 0.0) & (c[:, 9] >= 0.6)
            target_rows.append(c[mask_t])
            agg_rows.append(c[mask_agg])
            pwr_rows.append(c[mask_pwr])
        T = np.concatenate(target_rows) if target_rows else np.zeros((0, 10), dtype=np.float32)
        A = np.concatenate(agg_rows)    if agg_rows else np.zeros((0, 10), dtype=np.float32)
        P = np.concatenate(pwr_rows)    if pwr_rows else np.zeros((0, 10), dtype=np.float32)
        return T, A, P

    def _subsample(self, X: np.ndarray, k: int) -> Tuple[np.ndarray, int]:
        n = X.shape[0]
        if n == 0:
            return np.zeros((k, X.shape[1] if X.ndim == 2 else 10), dtype=np.float32), 0
        if n <= k:
            out = np.zeros((k, X.shape[1]), dtype=np.float32)
            out[:n] = X
            return out, n
        # subsample
        idx = np.random.choice(n, k, replace=False) if self.train else np.arange(k) * (n // k)
        return X[idx].astype(np.float32), k

    def __getitem__(self, i):
        row = self.df.iloc[i]
        design, net = row["design_name"], row["net_name"]
        T, A, P = self._gather_cuboids(design, net)

        T_pad, n_t = self._subsample(T, self.max_target)
        A_pad, n_a = self._subsample(A, self.max_agg)
        P_pad, n_p = self._subsample(P, self.max_pwr)

        hand = row[self.feature_cols].to_numpy(dtype=np.float32)
        if self.hand_mean is not None:
            hand = (hand - self.hand_mean) / (self.hand_std + 1e-6)
            hand = np.clip(hand, -8.0, 8.0)

        y = np.array([row[t] for t in self.targets], dtype=np.float32)

        return {
            "target": torch.from_numpy(T_pad),
            "aggressor": torch.from_numpy(A_pad),
            "power": torch.from_numpy(P_pad),
            "n_target": n_t, "n_agg": n_a, "n_pwr": n_p,
            "hand": torch.from_numpy(hand),
            "y": torch.from_numpy(y),
            "design": design, "net": net,
        }


def make_loaders(splits, manifest, feature_cols, batch_size: int = 64, num_workers: int = 4):
    dsets = {
        k: CuboidSetDataset(df, manifest, feature_cols, train=(k == "train"))
        for k, df in splits.items()
    }
    # Hand-feature normalizer fit on train only
    Xtr = splits["train"][feature_cols].to_numpy(dtype=np.float32)
    mean = np.nanmean(Xtr, axis=0)
    std  = np.nanstd(Xtr, axis=0)
    for d in dsets.values():
        d.set_hand_normalizer(mean, std)

    loaders = {}
    for k, d in dsets.items():
        loaders[k] = DataLoader(
            d,
            batch_size=batch_size,
            shuffle=(k == "train"),
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            drop_last=(k == "train"),
        )
    return loaders, dsets, mean, std
