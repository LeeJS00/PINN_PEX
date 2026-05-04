"""
manifest.py — Strategy v3 H1 fix (net-level deterministic hash split).

Replaces legacy `scripts/build_dataset_multi.py:91-95` random tile-level
split (which leaks 12.32% of nets across train/valid) with a hash of
(design_name, net_name) → uniform [0, 1) → split bucket.

Key invariants (enforced by `tests/test_split_invariants.py`):
  - Every (design, net) appears in exactly one of {train, valid, test}.
  - Test designs (TEST_DEFS) always go to `test`, regardless of hash.
  - Hash seed is fixed in config → split is deterministic, reproducible.
  - All tiles of a given net land in the same split.

This module does NOT touch the legacy manifest. It reads legacy as input
and writes a v3 manifest at `cfg.MANIFEST_PATH_V3`.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Set, Tuple

import pandas as pd


def net_split_bucket(
    design_name: str,
    net_name: str,
    valid_ratio: float,
    hash_seed: str,
) -> str:
    """Deterministic split bucket for a (design, net) pair.

    Returns 'train' or 'valid'. Test designs are not handled here — caller
    decides 'test' assignment based on TEST_DEFS membership.

    Hash function:
        SHA256( hash_seed || "::" || design_name || "::" || net_name )
        → first 16 hex chars → int → / 0xFFFF...F → uniform [0, 1)

    Args:
        design_name: e.g. 'intel22_aes_cipher_top_f3'
        net_name:    e.g. 'net_1234' (any string, including escaped slashes)
        valid_ratio: fraction sent to 'valid'; e.g. 0.10 for 10%
        hash_seed:   string baked into the hash to lock the split (config key)

    Returns:
        'train' if hash >= valid_ratio, else 'valid'.
    """
    key = f"{hash_seed}::{design_name}::{net_name}".encode("utf-8")
    h = hashlib.sha256(key).hexdigest()
    bucket = int(h[:16], 16) / 0xFFFFFFFFFFFFFFFF
    return "valid" if bucket < valid_ratio else "train"


def build_v3_manifest(
    legacy_manifest_path: Path,
    test_design_stems: Set[str],
    valid_ratio: float,
    hash_seed: str,
    schema_version: str,
) -> pd.DataFrame:
    """Read legacy manifest, recompute split column with H1 hash, return v3 manifest.

    Preserves all other columns (sample_filename, net_name, design_name,
    tile_idx, n_tiles, ...). Adds a new column `schema_version` so loaders
    can verify they loaded a v3 manifest.

    Net mixing in the legacy manifest is allowed (12.32% existed); this
    function simply rewrites split deterministically — the new manifest
    will contain zero net mixing by construction.
    """
    df = pd.read_csv(legacy_manifest_path)

    required_cols = {"design_name", "net_name", "split"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Legacy manifest missing required columns: {missing}. "
            f"Got: {list(df.columns)}"
        )

    def _assign(row) -> str:
        design = row["design_name"]
        if design in test_design_stems:
            return "test"
        return net_split_bucket(
            design_name=design,
            net_name=str(row["net_name"]),
            valid_ratio=valid_ratio,
            hash_seed=hash_seed,
        )

    df = df.copy()
    df["split"] = df.apply(_assign, axis=1)
    df["schema_version"] = schema_version
    return df


def assert_no_net_leak(manifest: pd.DataFrame) -> None:
    """Verify no (design, net) appears in more than one split.

    Raises AssertionError on violation. Phase 0 acceptance gate.
    """
    grouped = manifest.groupby(["design_name", "net_name"])["split"].nunique()
    leaks = grouped[grouped > 1]
    if len(leaks) > 0:
        sample = leaks.head(10)
        raise AssertionError(
            f"Net-level split leak detected: {len(leaks)} (design, net) pairs "
            f"appear in multiple splits. First 10:\n{sample.to_string()}"
        )


def manifest_summary(manifest: pd.DataFrame) -> dict:
    """Return summary statistics for logging / report."""
    n_tiles = len(manifest)
    n_nets = manifest.groupby(["design_name", "net_name"]).ngroups
    by_split_tiles = manifest.groupby("split").size().to_dict()
    by_split_nets = (
        manifest.groupby(["split", "design_name", "net_name"]).size()
        .reset_index()
        .groupby("split").size().to_dict()
    )
    by_design = manifest.groupby("design_name").size().to_dict()
    return {
        "total_tiles": int(n_tiles),
        "total_nets": int(n_nets),
        "tiles_by_split": {k: int(v) for k, v in by_split_tiles.items()},
        "nets_by_split": {k: int(v) for k, v in by_split_nets.items()},
        "tiles_by_design": {k: int(v) for k, v in by_design.items()},
    }


def write_v3_manifest(
    df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Write v3 manifest atomically (write to .tmp, rename)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(out_path)
