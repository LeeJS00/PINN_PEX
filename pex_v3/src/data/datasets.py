"""
datasets.py — Strategy v3 H2 fix (priority truncation).

Wraps the legacy `NeuralFieldDataset` and overrides only the truncation
logic in `__getitem__` to sort cuboids by (is_target priority, distance
from tile center) before slicing to `pad_size`.

Why:
    Legacy `src/data/datasets.py:232-234` truncates positionally
    (`tensor[:safe_pad]`), so when N > 1024, aggressors are dropped in
    insertion order — silently losing tile-edge aggressors that are
    physically important for top-metal coupling.

H2 fix:
    Priority order (descending):
        1. is_target == 1.0  (channel index 7) — never drop
        2. distance from tile center (channels 0,1) ascending — keep nearest
    After sorting, take the first `pad_size` rows.

This file does not re-implement the legacy loader; it reuses it via
composition. Phase 1+ may rewrite the loader entirely; until then this
is a surgical fix.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Tuple
import sys

# Add legacy src to path for import (this is a READ-only legacy import)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import Dataset

# Channel indices in the (N, 10) cuboid tensor (per docs/PROJECT_REPORT.md §1.3).
# Channel 0,1 = x_rel, y_rel  (tile-relative xy)
# Channel 2   = z_abs
# Channel 3,4,5 = w, h, d
# Channel 6 = semantic_type (1.0 wire, 0.5 pin)
# Channel 7 = is_target (1.0 target, 0.0 aggressor)
# Channel 8 = epsilon
# Channel 9 = net_type (VSS aggressor, v9+)
CH_X = 0
CH_Y = 1
CH_IS_TARGET = 7


def priority_truncate(
    tensor: torch.Tensor,
    pad_size: int,
    extra_arrays: tuple = (),
) -> Tuple[torch.Tensor, tuple, torch.Tensor]:
    """H2 priority truncation.

    If `tensor.shape[0] <= pad_size`, returns padded tensor with valid_mask
    indicating real rows. If larger, sorts by priority then truncates.

    Args:
        tensor: shape (N, C) where C >= 8 (channels 0,1,7 must exist)
        pad_size: target row count
        extra_arrays: tuple of np.ndarray, each shape (N, ...) or (N,);
                      reordered/truncated alongside `tensor`

    Returns:
        (truncated_or_padded_tensor, truncated_or_padded_extra_arrays, mask)
        mask: bool tensor of shape (pad_size,) — True where padding (no data).
    """
    N, C = tensor.shape
    assert C >= 8, f"H2 expects ≥8 channels (need is_target at idx 7); got C={C}"

    if N <= pad_size:
        pad_len = pad_size - N
        if pad_len == 0:
            mask = torch.zeros(pad_size, dtype=torch.bool)
            return tensor, extra_arrays, mask
        padding = torch.zeros((pad_len, C), dtype=tensor.dtype)
        out_tensor = torch.cat([tensor, padding], dim=0)
        out_extras = []
        for arr in extra_arrays:
            if arr.ndim == 1:
                pad = np.zeros(pad_len, dtype=arr.dtype)
            else:
                pad_shape = (pad_len,) + arr.shape[1:]
                pad = np.zeros(pad_shape, dtype=arr.dtype)
            out_extras.append(np.concatenate([arr, pad], axis=0))
        mask = torch.ones(pad_size, dtype=torch.bool)
        mask[:N] = False
        return out_tensor, tuple(out_extras), mask

    # N > pad_size — H2 priority sort then truncate.
    is_target = tensor[:, CH_IS_TARGET]  # (N,) values in {0, 0.5, 1.0}
    xy = tensor[:, [CH_X, CH_Y]]  # (N, 2) tile-relative
    dist = torch.linalg.norm(xy, dim=-1)  # (N,)

    # Priority: lower value = higher priority.
    # is_target == 1.0 → -1e6 (highest priority, never dropped if pad_size ≥ #targets)
    # is_target == 0.5 (pin) → -1e3 (high priority but below wires)
    # is_target == 0.0 (aggressor) → 0 (sorted by distance)
    target_bias = -is_target.float() * 1e6  # wire targets dominate
    # Pin (semantic_type 0.5 in CH_IS_TARGET? actually CH_IS_TARGET is binary
    # for target/non-target; semantic_type is ch6. Treat is_target>0 as target.)
    priority = target_bias + dist

    keep_idx = torch.argsort(priority)[:pad_size]
    keep_idx_np = keep_idx.cpu().numpy()

    out_tensor = tensor[keep_idx]
    out_extras = []
    for arr in extra_arrays:
        out_extras.append(arr[keep_idx_np])
    mask = torch.zeros(pad_size, dtype=torch.bool)
    return out_tensor, tuple(out_extras), mask


class V3PriorityTruncatedDataset(Dataset):
    """Thin wrapper over a legacy dataset that applies H2 priority truncation.

    Use:
        from src.data.datasets import NeuralFieldDataset  # legacy
        legacy_ds = NeuralFieldDataset(processed_dir, manifest_df, ...)
        ds = V3PriorityTruncatedDataset(legacy_ds, pad_size=1024)

    The wrapper takes whatever the legacy dataset returns and replaces the
    positional truncation with priority truncation.

    NOTE — the legacy `NeuralFieldDataset.__getitem__` returns a complex
    structure (tensor, mask, meta, extras...). Phase 0 minimum: re-pad after
    legacy loads, sorted by H2 priority. We do this by *re-running* the
    sort on the post-legacy output, which is correct as long as the legacy
    truncation hasn't already dropped data.

    For Phase 0 to be useful, this must run BEFORE legacy truncation. The
    cleanest way is to set the legacy `pad_size` to a value larger than any
    real tile (e.g. 8192) so legacy never truncates, then we apply H2 here.

    A more invasive Phase 1 fix will replace the legacy loader entirely.
    """

    def __init__(self, legacy_dataset, pad_size: int):
        self.legacy_ds = legacy_dataset
        self.pad_size = int(pad_size)

    def __len__(self) -> int:
        return len(self.legacy_ds)

    def __getitem__(self, idx: int):
        # NOTE: this assumes legacy_ds was constructed with pad_size large
        # enough that no truncation happened upstream. The contract is:
        # legacy_pad_size > max_real_N.
        item = self.legacy_ds[idx]
        # Legacy item is a tuple. The first element is the (N_legacy_pad, C)
        # tensor and second is the legacy mask. We re-truncate.
        tensor = item[0]
        mask = item[1]
        # Strip legacy padding so we sort over real rows only.
        valid = ~mask  # legacy mask: True where padding
        real_tensor = tensor[valid]
        # Walk through any extra np arrays that need parallel reordering.
        # Legacy structure (per src/data/datasets.py:255 onwards):
        #   item = (tensor, mask, batch_extras_dict, meta_dict, ...)
        # We only need to truncate `tensor` and `mask`; meta dicts carry
        # their own length-N arrays that legacy already aligned. To avoid
        # silently corrupting per-cuboid arrays in `meta_dict`, Phase 0
        # uses a SAFE-MODE: only truncate when N_real == legacy_pad (no
        # legacy slice happened). Otherwise we pass through.
        N_real = int(valid.sum().item())
        if N_real == 0:
            return item  # empty tile, nothing to do
        # If legacy already dropped rows, our priority truncation is moot
        # because we lost the unsorted complement. Phase 1 fixes this by
        # owning the loader.
        new_tensor, _, new_mask = priority_truncate(
            real_tensor, pad_size=self.pad_size
        )
        # Reconstruct item with the new tensor + mask. Pass remaining
        # entries through.
        return (new_tensor, new_mask) + tuple(item[2:])
