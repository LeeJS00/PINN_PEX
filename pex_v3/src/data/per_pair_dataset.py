"""
per_pair_dataset.py — per-pair (target, aggressor) coupling dataset.

Strike #2: per-pair coupling supervision. Each batch item is a single
(target, aggressor) pair with full target + aggressor representation:
    - target_self_features  (16,)
    - target_cuboids        (Nt, 10) → encoded
    - target_analytic_gnd   scalar (for joint gnd loss)
    - aggressor_self_features (16,)
    - aggressor_cuboids     (Na, 10) → encoded
    - pair_features          (24,) — original pair-level engineered
    - analytic_pair_cpl     scalar (uniform: total/n_aggressors as prior)
    - golden_pair_cpl       scalar (target)
    - golden_target_gnd     scalar
    - golden_target_cpl_total scalar (sum of all aggressors, for joint loss)

Sampling: K aggressors per target per epoch (K configurable, default 5).
Aggressors that are NOT in the cuboid store are skipped.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PerPairDataset(Dataset):
    """Per-pair coupling supervision.

    On `__getitem__(idx)`, returns a target net + K randomly-sampled aggressors.
    Collate function pads to batch-max for both target + aggressor cuboids.
    """

    def __init__(
        self,
        target_df: pd.DataFrame,            # per-net features (after calibration)
        per_pair_df: pd.DataFrame,          # design_name + target_net + aggressor_net + c_pair_fF
        cuboid_store,                       # PerNetCuboidStore
        self_feature_cols: list[str],
        pair_feature_cols: list[str],       # placeholder (currently shared)
        self_dim: int = 16,
        pair_dim: int = 24,
        max_cuboids_per_net: int = 512,
        cuboid_in_dim: int = 10,
        k_aggressors: int = 5,              # per epoch sample size
        rng_seed: int = 0,
    ):
        # Filter target_df to nets in store
        target_df = target_df[target_df.apply(
            lambda r: cuboid_store.has(r["design_name"], r["net_name"]), axis=1
        )].reset_index(drop=True)
        self.target_df = target_df
        self.store = cuboid_store
        self.self_dim = self_dim
        self.pair_dim = pair_dim
        self.self_cols = self_feature_cols
        self.pair_cols = pair_feature_cols
        self.max_cuboids = max_cuboids_per_net
        self.cuboid_in_dim = cuboid_in_dim
        self.k = k_aggressors
        self._rng = np.random.default_rng(rng_seed)

        # Per-target lookup: design,net → row index
        self._target_lookup: dict[tuple[str, str], int] = {
            (r["design_name"], r["net_name"]): i
            for i, r in target_df.iterrows()
        }

        # Per-target feature cache (numpy arrays)
        self._self_feats = self._featurize(target_df, self_feature_cols, self_dim)
        self._pair_feats_self = self._featurize(target_df, pair_feature_cols, pair_dim)
        self._analytic_gnd = target_df["compact_gnd_estimate_fF"].fillna(0.0).to_numpy(dtype=np.float32)
        self._analytic_cpl = target_df["compact_cpl_estimate_total_fF"].fillna(0.0).to_numpy(dtype=np.float32)
        self._gold_gnd = target_df["c_gnd_fF"].fillna(0.0).to_numpy(dtype=np.float32)
        self._gold_cpl_total = target_df["c_cpl_total_fF"].fillna(0.0).to_numpy(dtype=np.float32)
        self._designs = target_df["design_name"].to_numpy()
        self._nets = target_df["net_name"].to_numpy()

        # Per-target aggressor list: design,net → list of (aggressor_net, c_pair_fF)
        # Filter pairs whose aggressor is in cuboid store (drop missing-aggressor pairs)
        per_pair_df = per_pair_df.copy()
        in_store = per_pair_df.apply(
            lambda r: cuboid_store.has(r["design_name"], r["aggressor_net"]), axis=1
        )
        n_before = len(per_pair_df)
        per_pair_df = per_pair_df[in_store].reset_index(drop=True)
        n_dropped = n_before - len(per_pair_df)
        if n_dropped > 0:
            print(f"[PerPairDataset] dropped {n_dropped:,} pairs (aggressor not in store)")

        self._aggr_lookup: dict[tuple[str, str], np.ndarray] = {}
        # group by target
        for (design, target), grp in per_pair_df.groupby(["design_name", "target_net"]):
            self._aggr_lookup[(design, target)] = grp[
                ["aggressor_net", "c_pair_fF"]
            ].to_numpy()  # (n_aggr, 2)

        # Drop targets with no aggressors in store
        keep = self.target_df.apply(
            lambda r: (r["design_name"], r["net_name"]) in self._aggr_lookup, axis=1
        )
        n_dropped_t = (~keep).sum()
        if n_dropped_t > 0:
            print(f"[PerPairDataset] dropped {n_dropped_t} targets without any aggressor in store")
            # Re-filter all caches
            mask = keep.to_numpy()
            self.target_df = self.target_df[keep].reset_index(drop=True)
            self._self_feats = self._self_feats[mask]
            self._pair_feats_self = self._pair_feats_self[mask]
            self._analytic_gnd = self._analytic_gnd[mask]
            self._analytic_cpl = self._analytic_cpl[mask]
            self._gold_gnd = self._gold_gnd[mask]
            self._gold_cpl_total = self._gold_cpl_total[mask]
            self._designs = self._designs[mask]
            self._nets = self._nets[mask]
            self._target_lookup = {
                (r["design_name"], r["net_name"]): i
                for i, r in self.target_df.iterrows()
            }

    @staticmethod
    def _featurize(df: pd.DataFrame, cols: list[str], dim: int) -> np.ndarray:
        out = np.zeros((len(df), dim), dtype=np.float32)
        for i, c in enumerate(cols[:dim]):
            if c in df.columns:
                v = df[c].fillna(0.0).clip(lower=0).to_numpy(dtype=np.float32)
                out[:, i] = np.log1p(v)
        return out

    def __len__(self) -> int:
        return len(self.target_df)

    def __getitem__(self, idx: int) -> dict:
        design = str(self._designs[idx])
        target = str(self._nets[idx])
        aggr_arr = self._aggr_lookup[(design, target)]  # (n, 2)

        # Sample K aggressors (with replacement if n < K)
        n_aggr = len(aggr_arr)
        if n_aggr <= self.k:
            sampled_idx = np.arange(n_aggr)
        else:
            sampled_idx = self._rng.choice(n_aggr, self.k, replace=False)
        sampled = aggr_arr[sampled_idx]
        sampled_aggr_names = sampled[:, 0]
        sampled_c_pair = sampled[:, 1].astype(np.float32)

        # Target cuboids
        target_cuboids = self.store.get(design, target)
        if target_cuboids is None or len(target_cuboids) == 0:
            target_cuboids = np.zeros((1, self.cuboid_in_dim), dtype=np.float32)

        # Aggressor data: get features (from target_lookup if exists) + cuboids
        aggr_self_feats_list = []
        aggr_cuboids_list = []
        for a_name in sampled_aggr_names:
            a_idx = self._target_lookup.get((design, str(a_name)))
            if a_idx is not None:
                aggr_self_feats_list.append(self._self_feats[a_idx])
            else:
                # Aggressor not in target_df (e.g., it's in test split or filtered)
                # Use zeros — its cuboids carry geometry signal
                aggr_self_feats_list.append(np.zeros(self.self_dim, dtype=np.float32))
            a_cuboids = self.store.get(design, str(a_name))
            if a_cuboids is None or len(a_cuboids) == 0:
                a_cuboids = np.zeros((1, self.cuboid_in_dim), dtype=np.float32)
            aggr_cuboids_list.append(a_cuboids)

        # Per-pair analytic baseline = uniform `compact_cpl_total / n_aggressors`
        n_aggr_total = len(aggr_arr)
        analytic_pair = float(self._analytic_cpl[idx]) / max(n_aggr_total, 1)

        return {
            "design_name": design,
            "target_net": target,
            "target_cuboids": target_cuboids,
            "target_self_features": self._self_feats[idx],
            "target_pair_features": self._pair_feats_self[idx],
            "target_analytic_gnd": float(self._analytic_gnd[idx]),
            "target_analytic_cpl_total": float(self._analytic_cpl[idx]),
            "target_golden_gnd": float(self._gold_gnd[idx]),
            "target_golden_cpl_total": float(self._gold_cpl_total[idx]),
            "n_aggr_total": n_aggr_total,
            "sampled_aggr_names": list(sampled_aggr_names),
            "sampled_aggr_self_features": np.stack(aggr_self_feats_list),  # (k_sampled, self_dim)
            "sampled_aggr_cuboids": aggr_cuboids_list,                     # list[(N_a, in_dim)]
            "sampled_c_pair_golden": sampled_c_pair,                        # (k_sampled,)
            "analytic_pair_baseline": float(analytic_pair),                 # scalar; same for all aggressors
        }


def collate_per_pair_batch(items: list[dict]) -> dict:
    """Pad target cuboids + aggressor cuboids to batch-max.

    Returns:
        target_cuboids:    (B, Nt_max, in_dim)
        target_mask:       (B, Nt_max)
        target_self_features: (B, self_dim)
        target_pair_features: (B, pair_dim)
        target_analytic_gnd: (B,)
        target_analytic_cpl_total: (B,)
        target_golden_gnd: (B,)
        target_golden_cpl_total: (B,)
        n_aggr_total: (B,)  number of aggressors per target (for prior)

        aggr_cuboids:      (B, K, Na_max, in_dim)
        aggr_mask:         (B, K, Na_max)
        aggr_self_features:(B, K, self_dim)
        c_pair_golden:     (B, K)
        sampled_mask:      (B, K)  1=valid sample (might be < k for nets with fewer aggressors)
        analytic_pair_baseline: (B,)  same value broadcast per pair
    """
    B = len(items)
    in_dim = items[0]["target_cuboids"].shape[1]
    self_dim = items[0]["target_self_features"].shape[0]
    pair_dim = items[0]["target_pair_features"].shape[0]

    Nt_max = max(len(it["target_cuboids"]) for it in items)
    K_max = max(len(it["sampled_aggr_cuboids"]) for it in items)
    Na_max = max(
        max(len(c) for c in it["sampled_aggr_cuboids"])
        for it in items
    )

    target_cuboids = np.zeros((B, Nt_max, in_dim), dtype=np.float32)
    target_mask = np.zeros((B, Nt_max), dtype=np.float32)
    aggr_cuboids = np.zeros((B, K_max, Na_max, in_dim), dtype=np.float32)
    aggr_mask = np.zeros((B, K_max, Na_max), dtype=np.float32)
    aggr_self = np.zeros((B, K_max, self_dim), dtype=np.float32)
    c_pair = np.zeros((B, K_max), dtype=np.float32)
    sampled_mask = np.zeros((B, K_max), dtype=np.float32)

    for b, it in enumerate(items):
        Nt = len(it["target_cuboids"])
        target_cuboids[b, :Nt] = it["target_cuboids"]
        target_mask[b, :Nt] = 1.0
        K = len(it["sampled_aggr_cuboids"])
        for j in range(K):
            Na = len(it["sampled_aggr_cuboids"][j])
            aggr_cuboids[b, j, :Na] = it["sampled_aggr_cuboids"][j]
            aggr_mask[b, j, :Na] = 1.0
        aggr_self[b, :K] = it["sampled_aggr_self_features"]
        c_pair[b, :K] = it["sampled_c_pair_golden"]
        sampled_mask[b, :K] = 1.0

    return {
        "target_cuboids": torch.tensor(target_cuboids),
        "target_mask": torch.tensor(target_mask),
        "target_self_features": torch.tensor(np.stack([it["target_self_features"] for it in items])),
        "target_pair_features": torch.tensor(np.stack([it["target_pair_features"] for it in items])),
        "target_analytic_gnd": torch.tensor(np.array([it["target_analytic_gnd"] for it in items], dtype=np.float32)),
        "target_analytic_cpl_total": torch.tensor(np.array([it["target_analytic_cpl_total"] for it in items], dtype=np.float32)),
        "target_golden_gnd": torch.tensor(np.array([it["target_golden_gnd"] for it in items], dtype=np.float32)),
        "target_golden_cpl_total": torch.tensor(np.array([it["target_golden_cpl_total"] for it in items], dtype=np.float32)),
        "n_aggr_total": torch.tensor(np.array([it["n_aggr_total"] for it in items], dtype=np.float32)),
        "aggr_cuboids": torch.tensor(aggr_cuboids),
        "aggr_mask": torch.tensor(aggr_mask),
        "aggr_self_features": torch.tensor(aggr_self),
        "c_pair_golden": torch.tensor(c_pair),
        "sampled_mask": torch.tensor(sampled_mask),
        "analytic_pair_baseline": torch.tensor(np.array([it["analytic_pair_baseline"] for it in items], dtype=np.float32)),
        "design_name": [it["design_name"] for it in items],
        "target_net": [it["target_net"] for it in items],
    }
