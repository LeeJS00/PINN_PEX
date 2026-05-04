"""
conftest.py — pytest fixtures shared across pex_v3 tests.

Runs from repo root via `python3 -m pytest pex_v3/tests/`. Path setup
ensures `pex_v3.src.*` and legacy `src.*` both resolve.
"""
from __future__ import annotations
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "pex_v3"))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def synthetic_legacy_manifest() -> pd.DataFrame:
    """Build a tiny synthetic 'legacy-style' manifest with a known H1 leak.

    Returns a DataFrame mimicking the legacy manifest schema with intentional
    net-level split mixing — used to test that H1 hash recompute eliminates it.

    Schema (legacy):
        sample_filename, net_name, design_name, split, tile_idx, ...
    """
    rows = []
    # Train design: 3 nets × 4 tiles each, intentionally split-mixed
    for net_idx in range(3):
        net_name = f"trainnet_{net_idx:03d}"
        for tile_idx in range(4):
            # Legacy random split: tile-level. Some tiles of the same net
            # land in train, others in valid → net mixing.
            split = "valid" if (net_idx + tile_idx) % 2 == 0 else "train"
            rows.append({
                "sample_filename": f"trainnet_{net_idx:03d}_t{tile_idx}.pkl.gz",
                "net_name": net_name,
                "design_name": "intel22_aes_cipher_top_f3",
                "split": split,
                "tile_idx": tile_idx,
            })
    # Test design: should always be 'test'
    for net_idx in range(2):
        net_name = f"testnet_{net_idx:03d}"
        for tile_idx in range(2):
            rows.append({
                "sample_filename": f"testnet_{net_idx:03d}_t{tile_idx}.pkl.gz",
                "net_name": net_name,
                "design_name": "intel22_nova_f3",
                "split": "test",
                "tile_idx": tile_idx,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def test_design_stems() -> set[str]:
    """Set of test-design stems matching cfg.TEST_DEFS."""
    return {"intel22_nova_f3", "intel22_tv80s_f3"}
