"""
NCGT per-net dataset (Plan v4 §3).

Each sample = one net's full graph: target segments + aggressor segments + edges
+ supervision targets (gnd, cpl_net, cpl_per_edge with mask). No tile-level
aggregation; no padding to a fixed N.

Phase 0-calibrated parameters:
- R_aggr = 12 μm (after via exclusion)
- Per-net target cap = 1K, signal aggressor cap = 4K, power aggressor cap = 2K
- L_subdiv = 4 μm
"""
from __future__ import annotations

import gzip
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from experiments.ncgt.src.data.segment_extractor import Segment, role_for


# Type id mapping (Plan v4 §2.2; same/cross-layer collapsed pending classifier fix).
TYPE_TO_ID = {
    "target": 0,
    "signal_aggr": 1,
    "power_VDD": 2,
    "power_VSS": 3,
    "pin": 4,
    "branch_node": 5,
}
N_TYPES = len(TYPE_TO_ID)


@dataclass
class NCGTSample:
    """Per-net packed sample."""
    net_name: str
    design_name: str
    # Segment features.
    target_feats: np.ndarray       # (T, 12) float32
    aggr_feats: np.ndarray         # (A, 12) float32
    # Type ids (heterogeneous embedding routing).
    target_type_ids: np.ndarray    # (T,) int8 — typically all 0 (target)
    aggr_type_ids: np.ndarray      # (A,) int8
    # Endpoints for physics base.
    target_p_start: np.ndarray     # (T, 3) float32
    target_p_end: np.ndarray       # (T, 3) float32
    aggr_p_start: np.ndarray       # (A, 3) float32
    aggr_p_end: np.ndarray         # (A, 3) float32
    # Edges (target_idx, aggr_idx).
    edge_index: np.ndarray         # (2, E) int64
    edge_band: np.ndarray          # (E,) int8 — 0=local, 1=mid, 2=long
    # Aggressor net ids (for grouping CPL output).
    aggr_net_ids: np.ndarray       # (A,) int64
    # Supervision targets.
    gnd_total: float               # net-total GND cap (fF)
    cpl_total: float               # net-total CPL cap (fF)
    cpl_per_aggr_net: Dict[int, float]   # {aggr_net_id: total_cpl}
    edge_gt: np.ndarray            # (E,) float32 — per-edge target (0 if unsupervised)
    edge_supervised: np.ndarray    # (E,) bool

    def to_torch(self, device: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """Convert to dict of tensors for model forward.

        Phase B: also packs gt_cpl_per_aggr_net as dense tensor for per-net CPL
        supervision (50-200× denser than net-total).
        """
        # Phase B: pack per-aggressor-net CPL targets.
        if self.cpl_per_aggr_net and self.aggr_net_ids.size > 0:
            n_aggr_nets = int(self.aggr_net_ids.max()) + 1
        else:
            n_aggr_nets = 0
        gt_cpl_per_aggr = np.zeros(max(1, n_aggr_nets), dtype=np.float32)
        for aid, val in self.cpl_per_aggr_net.items():
            if 0 <= aid < n_aggr_nets:
                gt_cpl_per_aggr[aid] = float(val)

        out = {
            "target_feats": torch.from_numpy(self.target_feats).float(),
            "aggr_feats": torch.from_numpy(self.aggr_feats).float(),
            "target_type_ids": torch.from_numpy(self.target_type_ids.astype(np.int64)),
            "aggr_type_ids": torch.from_numpy(self.aggr_type_ids.astype(np.int64)),
            "target_p_start": torch.from_numpy(self.target_p_start).float(),
            "target_p_end": torch.from_numpy(self.target_p_end).float(),
            "aggr_p_start": torch.from_numpy(self.aggr_p_start).float(),
            "aggr_p_end": torch.from_numpy(self.aggr_p_end).float(),
            "edge_index": torch.from_numpy(self.edge_index).long(),
            "edge_band": torch.from_numpy(self.edge_band.astype(np.int64)),
            "aggr_net_ids": torch.from_numpy(self.aggr_net_ids).long(),
            "gnd_total": torch.tensor(self.gnd_total, dtype=torch.float32),
            "cpl_total": torch.tensor(self.cpl_total, dtype=torch.float32),
            "edge_gt": torch.from_numpy(self.edge_gt).float(),
            "edge_supervised": torch.from_numpy(self.edge_supervised),
            "gt_cpl_per_aggr_net": torch.from_numpy(gt_cpl_per_aggr),
            "n_aggr_nets": torch.tensor(n_aggr_nets, dtype=torch.long),
        }
        if device:
            out = {k: v.to(device) for k, v in out.items()}
        return out


def segment_to_features(s: Segment, role_id: int) -> np.ndarray:
    """Pack a Segment into the 12D feature vector per Plan v4 §2.1."""
    return np.array(
        [
            s.x_mid, s.y_mid, s.z,
            s.dx, s.dy,
            s.w, s.h,
            float(s.layer_idx),
            s.semantic_type,
            float(role_id),  # role placeholder (also routed via type_id)
            {"signal": 0, "VDD": 1, "VSS": 2, "clock": 3}.get(s.net_class, 0),
            float(s.is_subdivision),
        ],
        dtype=np.float32,
    )


def role_to_type_id(role: str) -> int:
    if role == "target":
        return TYPE_TO_ID["target"]
    if role.startswith("signal_aggr"):
        return TYPE_TO_ID["signal_aggr"]
    if role == "power_VDD":
        return TYPE_TO_ID["power_VDD"]
    if role == "power_VSS":
        return TYPE_TO_ID["power_VSS"]
    if role == "pin":
        return TYPE_TO_ID["pin"]
    if role == "branch_node":
        return TYPE_TO_ID["branch_node"]
    # Vias should be excluded earlier; fallback to signal_aggr.
    return TYPE_TO_ID["signal_aggr"]


def build_sample(
    net_name: str,
    design_name: str,
    target_segments: List[Segment],
    aggressor_segments: List[Segment],
    aggr_net_names: List[str],
    edge_index: np.ndarray,
    edge_band: np.ndarray,
    gnd_total: float,
    cpl_total: float,
    cpl_per_aggr_net: Dict[str, float],
    edge_gt: np.ndarray,
    edge_supervised: np.ndarray,
) -> NCGTSample:
    """Pack raw extraction outputs into an NCGTSample."""
    T = len(target_segments)
    A = len(aggressor_segments)
    assert edge_index.shape == (2, edge_band.shape[0])
    assert edge_gt.shape == edge_supervised.shape == (edge_band.shape[0],)

    # Target features (all role=target).
    t_feats = np.empty((T, 12), dtype=np.float32)
    t_pstart = np.empty((T, 3), dtype=np.float32)
    t_pend = np.empty((T, 3), dtype=np.float32)
    target_type_ids = np.full(T, TYPE_TO_ID["target"], dtype=np.int8)
    for i, s in enumerate(target_segments):
        t_feats[i] = segment_to_features(s, TYPE_TO_ID["target"])
        t_pstart[i] = s.p_start
        t_pend[i] = s.p_end

    # Aggressor features. role_for needs target_layer_idx for same/cross-layer split,
    # but we collapsed to "signal_aggr" pending classifier fix.
    a_feats = np.empty((A, 12), dtype=np.float32)
    a_pstart = np.empty((A, 3), dtype=np.float32)
    a_pend = np.empty((A, 3), dtype=np.float32)
    aggr_type_ids = np.empty(A, dtype=np.int8)
    aggr_net_id_arr = np.empty(A, dtype=np.int64)
    name_to_id: Dict[str, int] = {}
    for i, s in enumerate(aggressor_segments):
        role = role_for(s, target_net=net_name)  # target_layer_idx omitted → no same/cross split
        type_id = role_to_type_id(role)
        a_feats[i] = segment_to_features(s, type_id)
        a_pstart[i] = s.p_start
        a_pend[i] = s.p_end
        aggr_type_ids[i] = type_id
        if aggr_net_names[i] not in name_to_id:
            name_to_id[aggr_net_names[i]] = len(name_to_id)
        aggr_net_id_arr[i] = name_to_id[aggr_net_names[i]]

    # Per-aggressor-net cpl total.
    cpl_per_id = {name_to_id[k]: v for k, v in cpl_per_aggr_net.items() if k in name_to_id}

    return NCGTSample(
        net_name=net_name,
        design_name=design_name,
        target_feats=t_feats,
        aggr_feats=a_feats,
        target_type_ids=target_type_ids,
        aggr_type_ids=aggr_type_ids,
        target_p_start=t_pstart,
        target_p_end=t_pend,
        aggr_p_start=a_pstart,
        aggr_p_end=a_pend,
        edge_index=edge_index.astype(np.int64),
        edge_band=edge_band.astype(np.int8),
        aggr_net_ids=aggr_net_id_arr,
        gnd_total=float(gnd_total),
        cpl_total=float(cpl_total),
        cpl_per_aggr_net=cpl_per_id,
        edge_gt=edge_gt.astype(np.float32),
        edge_supervised=edge_supervised.astype(bool),
    )


def save_sample(sample: NCGTSample, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(sample, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_sample(path: Path) -> NCGTSample:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


class NCGTDataset(Dataset):
    """Per-net dataset reading pickled NCGTSample objects."""

    def __init__(self, manifest_csv: Path, split: str = "train", augment: bool = False, allow_rot90: bool = False):
        import pandas as pd

        self.manifest = pd.read_csv(manifest_csv)
        self.manifest = self.manifest[self.manifest["split"] == split].reset_index(drop=True)
        self.augment = augment
        self.allow_rot90 = allow_rot90
        self._rng = np.random.default_rng(42)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.manifest.iloc[idx]
        sample = load_sample(Path(row["pkl_path"]))
        if self.augment:
            from experiments.ncgt.src.data.geometric_aug import (
                random_safe_transform,
                apply_to_segment_features,
                apply_to_endpoints,
            )

            t = random_safe_transform(self._rng, allow_rot90=self.allow_rot90)
            sample.target_feats = apply_to_segment_features(sample.target_feats, transform=t)
            sample.aggr_feats = apply_to_segment_features(sample.aggr_feats, transform=t)
            sample.target_p_start, sample.target_p_end = apply_to_endpoints(
                sample.target_p_start, sample.target_p_end, t
            )
            sample.aggr_p_start, sample.aggr_p_end = apply_to_endpoints(
                sample.aggr_p_start, sample.aggr_p_end, t
            )
        return sample.to_torch()
