"""
NCGT geometric augmentation (Plan v3 §3.2, NAS-Cap technique).

Capacitance is invariant under symmetries that preserve the layer stack and BEOL
xy-isotropy. **xy-isotropy is NOT assumed without verification**: real BEOL has
preferred routing directions per layer (M_odd horizontal, M_even vertical). Phase
0 audit must verify before enabling 90°/270° rotations.

Default subset (6×, verified-safe):
    0: identity
    1: xy-rotation 180°
    2: x-reflection
    3: y-reflection
    4: xy-diagonal reflection (swap x ↔ y)
    5: xy-anti-diagonal reflection (swap x ↔ -y, y ↔ -x)

Phase-0-verified extra (8× total):
    6: xy-rotation 90° CCW
    7: xy-rotation 270° CCW

Apply randomly per-batch in collate. Storage: 1×; effective training data: 6× or 8×.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


SAFE_TRANSFORMS = ("identity", "rot180", "reflect_x", "reflect_y", "diag", "antidiag")
EXTRA_TRANSFORMS = ("rot90", "rot270")


def transform_xy(
    xy: np.ndarray,
    transform: str,
) -> np.ndarray:
    """Apply 2D point transform.

    xy: (N, 2) — last axis is [x, y]. Other leading axes are preserved.
    Returns same shape.
    """
    x = xy[..., 0]
    y = xy[..., 1]
    if transform == "identity":
        return xy.copy()
    if transform == "rot180":
        return np.stack([-x, -y], axis=-1)
    if transform == "reflect_x":
        return np.stack([x, -y], axis=-1)
    if transform == "reflect_y":
        return np.stack([-x, y], axis=-1)
    if transform == "diag":
        return np.stack([y, x], axis=-1)
    if transform == "antidiag":
        return np.stack([-y, -x], axis=-1)
    if transform == "rot90":
        return np.stack([-y, x], axis=-1)
    if transform == "rot270":
        return np.stack([y, -x], axis=-1)
    raise ValueError(f"Unknown transform: {transform}")


def transform_xy_extent(
    dxdy: np.ndarray,
    transform: str,
) -> np.ndarray:
    """Apply same transform to (dx, dy) extent vectors.

    Note: extents are signed direction vectors so they transform like positions
    under linear transforms (no center offset).
    """
    return transform_xy(dxdy, transform)


def apply_to_segment_features(
    feats: np.ndarray,        # (N, F) where F includes columns [x_mid, y_mid, ..., dx, dy, ...]
    *,
    x_idx: int = 0,
    y_idx: int = 1,
    dx_idx: int = 3,
    dy_idx: int = 4,
    transform: str = "identity",
) -> np.ndarray:
    """In-place-friendly transform on a feature matrix.

    Default column layout matches PLAN.md v3 §2.1:
        [x_mid, y_mid, z, dx, dy, w, h, layer_idx, semantic_type, role, net_class, is_subdivision]
    """
    out = feats.copy()
    xy = np.stack([feats[:, x_idx], feats[:, y_idx]], axis=-1)
    dxdy = np.stack([feats[:, dx_idx], feats[:, dy_idx]], axis=-1)

    xy_t = transform_xy(xy, transform)
    dxdy_t = transform_xy_extent(dxdy, transform)
    out[:, x_idx] = xy_t[..., 0]
    out[:, y_idx] = xy_t[..., 1]
    out[:, dx_idx] = dxdy_t[..., 0]
    out[:, dy_idx] = dxdy_t[..., 1]
    return out


def apply_to_endpoints(
    p_start: np.ndarray,  # (N, 3)
    p_end: np.ndarray,    # (N, 3)
    transform: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform xy endpoints; z preserved (layer stack untouched)."""
    xy_s = transform_xy(p_start[..., :2], transform)
    xy_e = transform_xy(p_end[..., :2], transform)
    p_start_t = np.concatenate([xy_s, p_start[..., 2:3]], axis=-1)
    p_end_t = np.concatenate([xy_e, p_end[..., 2:3]], axis=-1)
    return p_start_t, p_end_t


def random_safe_transform(rng: np.random.Generator, allow_rot90: bool = False) -> str:
    """Sample a random safe transform.

    Args:
        rng: numpy Generator for reproducibility.
        allow_rot90: if True, expands set to include rot90/rot270 (only after Phase 0 verification).
    """
    options = list(SAFE_TRANSFORMS)
    if allow_rot90:
        options += list(EXTRA_TRANSFORMS)
    idx = int(rng.integers(0, len(options)))
    return options[idx]


# ---------------------------------------------------------------------------
# Self-test.
# ---------------------------------------------------------------------------
def _smoke_test() -> None:
    """Verifies invariance: pairwise distances preserved under all transforms."""
    rng = np.random.default_rng(42)
    pts = rng.uniform(-10, 10, size=(30, 2))
    d_orig = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)

    for t in SAFE_TRANSFORMS + EXTRA_TRANSFORMS:
        pts_t = transform_xy(pts, t)
        d_t = np.linalg.norm(pts_t[:, None, :] - pts_t[None, :, :], axis=-1)
        rel_err = np.max(np.abs(d_orig - d_t)) / max(d_orig.max(), 1e-9)
        print(f"  {t:12s}  max distance error: {rel_err:.2e}")
        assert rel_err < 1e-6, f"distance not preserved under {t}"
    print("[geometric_aug smoke] OK — all 8 transforms preserve pairwise distances.")


if __name__ == "__main__":
    _smoke_test()
