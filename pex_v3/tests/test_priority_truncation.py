"""
test_priority_truncation.py — H2 invariants.

Verifies that when N > pad_size, the priority truncation:
  - Always retains target cuboids (channel 7 == 1.0)
  - Drops aggressors farthest from tile center first
  - Padding mask is correct
"""
from __future__ import annotations
import torch

from src.data.datasets import priority_truncate, CH_X, CH_Y, CH_IS_TARGET


def _make_cuboid(x: float, y: float, is_target: float, channels: int = 10) -> torch.Tensor:
    row = torch.zeros(channels, dtype=torch.float32)
    row[CH_X] = x
    row[CH_Y] = y
    row[CH_IS_TARGET] = is_target
    return row


def test_priority_keeps_targets_when_overflow():
    """If N > pad_size, all target cuboids must survive truncation."""
    # 5 targets at various distances + 100 aggressors at various distances
    rows = []
    for i in range(5):
        rows.append(_make_cuboid(x=i * 0.5, y=0.0, is_target=1.0))
    for i in range(100):
        rows.append(_make_cuboid(x=10.0 + i * 0.1, y=0.0, is_target=0.0))
    tensor = torch.stack(rows)
    assert tensor.shape == (105, 10)

    pad = 50
    out, _, mask = priority_truncate(tensor, pad_size=pad)
    assert out.shape == (pad, 10)
    # All 5 targets should be kept (channel 7 == 1.0 in 5 rows)
    target_kept = (out[:, CH_IS_TARGET] >= 0.99).sum().item()
    assert target_kept == 5, f"Lost a target! kept={target_kept}/5"
    # Mask should be all False (no padding when overflow)
    assert (~mask).all().item()


def test_priority_keeps_nearest_aggressors_when_overflow():
    """When truncating, nearest aggressors (smaller distance) survive over far ones."""
    rows = []
    rows.append(_make_cuboid(x=0.0, y=0.0, is_target=1.0))  # 1 target at center
    # Aggressors at increasing radii from center
    for i in range(100):
        radius = 1.0 + i * 0.5  # 1.0 .. 50.5 μm
        rows.append(_make_cuboid(x=radius, y=0.0, is_target=0.0))
    tensor = torch.stack(rows)

    pad = 20
    out, _, _ = priority_truncate(tensor, pad_size=pad)
    # Out must include the target + 19 nearest aggressors. Of the 100
    # aggressors, the 19 nearest are at radii 1.0, 1.5, ..., 10.0.
    # No surviving aggressor should have radius > 10.0 (i.e. x > 10.0).
    aggr_x = out[out[:, CH_IS_TARGET] < 0.5, CH_X]
    assert aggr_x.numel() == 19
    assert aggr_x.max().item() <= 10.0 + 1e-3, (
        f"Far aggressor survived: max_x={aggr_x.max().item():.3f} > 10.0"
    )


def test_priority_pads_when_underflow():
    """When N < pad_size, output is padded; mask marks padding rows."""
    rows = [_make_cuboid(x=0.0, y=0.0, is_target=1.0) for _ in range(10)]
    tensor = torch.stack(rows)

    pad = 50
    out, _, mask = priority_truncate(tensor, pad_size=pad)
    assert out.shape == (pad, 10)
    # First 10 rows are real
    assert (~mask[:10]).all().item()
    # Last 40 rows are padding
    assert mask[10:].all().item()
    # Padded rows should be all-zero
    assert torch.allclose(out[10:], torch.zeros(40, 10))


def test_priority_extra_arrays_aligned():
    """Extra parallel arrays are reordered/truncated alongside the tensor."""
    import numpy as np
    rows = []
    names = []
    # 3 targets + 30 aggressors
    for i in range(3):
        rows.append(_make_cuboid(x=i * 0.1, y=0.0, is_target=1.0))
        names.append(f"target_{i}")
    for i in range(30):
        rows.append(_make_cuboid(x=10.0 + i, y=0.0, is_target=0.0))
        names.append(f"aggr_{i}")
    tensor = torch.stack(rows)
    names_arr = np.array(names)

    pad = 10
    out, extras, _ = priority_truncate(tensor, pad_size=pad, extra_arrays=(names_arr,))
    out_names = extras[0]
    # The 3 targets should be in the output names
    assert all(f"target_{i}" in out_names.tolist() for i in range(3))
    # We have 7 aggressor slots after 3 targets; nearest 7 aggressors survive.
    # Aggressor x = 10.0, 11.0, ..., 39.0 → nearest 7 are 10..16.
    aggr_names_kept = [n for n in out_names if n.startswith("aggr_")]
    aggr_indices = [int(n.split("_")[1]) for n in aggr_names_kept]
    assert max(aggr_indices) <= 6, (
        f"Far aggressor survived. kept indices: {sorted(aggr_indices)}"
    )


def test_priority_no_op_when_exact_pad():
    """N == pad_size → no truncation, no padding."""
    rows = [_make_cuboid(x=i * 0.1, y=0.0, is_target=1.0) for i in range(50)]
    tensor = torch.stack(rows)
    pad = 50
    out, _, mask = priority_truncate(tensor, pad_size=pad)
    assert out.shape == tensor.shape
    assert (~mask).all().item()
