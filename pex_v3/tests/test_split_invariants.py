"""
test_split_invariants.py — H1 net-level split invariants.

Phase 0 acceptance gate. Run from repo root:

    python3 -m pytest pex_v3/tests/test_split_invariants.py -v
"""
from __future__ import annotations
import pytest

from src.data.manifest import (
    build_v3_manifest,
    net_split_bucket,
)
from src.data.leak_check import (
    check_no_net_overlap,
    check_test_designs_pure,
    check_train_designs_pure,
    check_split_coverage,
    check_schema_version,
    run_all_checks,
)


HASH_SEED = "test_seed_2026_05_01"
SCHEMA = "v3"


# -------------------------------------------------------------------- H1 hash


def test_net_split_bucket_deterministic():
    """Same (design, net) + same seed → same bucket on every call."""
    a = net_split_bucket("aes", "net_42", 0.10, HASH_SEED)
    b = net_split_bucket("aes", "net_42", 0.10, HASH_SEED)
    assert a == b


def test_net_split_bucket_seed_changes_outcome():
    """Different seeds can produce different buckets — proves seed lock works."""
    seeds = [f"seed_{i}" for i in range(50)]
    buckets = [net_split_bucket("aes", "net_42", 0.10, s) for s in seeds]
    assert len(set(buckets)) > 1, "Hash should not be seed-invariant"


def test_net_split_bucket_distribution():
    """Hash output approximates the requested valid_ratio over many nets."""
    n = 10_000
    valid_ratio = 0.10
    buckets = [
        net_split_bucket(
            design_name="aes",
            net_name=f"net_{i:06d}",
            valid_ratio=valid_ratio,
            hash_seed=HASH_SEED,
        )
        for i in range(n)
    ]
    valid_count = buckets.count("valid")
    train_count = buckets.count("train")
    assert valid_count + train_count == n
    # Allow ±2% drift around the target
    actual_ratio = valid_count / n
    assert abs(actual_ratio - valid_ratio) < 0.02, (
        f"Hash distribution skewed: got {actual_ratio:.3%}, expected {valid_ratio:.3%}"
    )


def test_net_split_bucket_per_design_independence():
    """Same net_name in different designs hash independently."""
    designs = [f"design_{i}" for i in range(20)]
    buckets = [
        net_split_bucket(d, "net_shared", 0.10, HASH_SEED) for d in designs
    ]
    # Should not all be the same bucket — different designs get fresh hashes
    assert len(set(buckets)) > 1


# -------------------------------------------------------------- end-to-end H1


def test_build_v3_manifest_no_leak(synthetic_legacy_manifest, test_design_stems, tmp_path):
    """End-to-end: build v3 manifest from leaky legacy → no net mixing."""
    legacy_path = tmp_path / "legacy_manifest.csv"
    synthetic_legacy_manifest.to_csv(legacy_path, index=False)

    df_v3 = build_v3_manifest(
        legacy_manifest_path=legacy_path,
        test_design_stems=test_design_stems,
        valid_ratio=0.10,
        hash_seed=HASH_SEED,
        schema_version=SCHEMA,
    )

    # All H1 invariants must pass
    run_all_checks(df_v3, test_design_stems, expected_schema=SCHEMA)


def test_build_v3_manifest_test_designs_pure(synthetic_legacy_manifest, test_design_stems, tmp_path):
    """All rows from TEST_DEFS designs land in 'test' regardless of hash."""
    legacy_path = tmp_path / "legacy_manifest.csv"
    synthetic_legacy_manifest.to_csv(legacy_path, index=False)

    df_v3 = build_v3_manifest(
        legacy_manifest_path=legacy_path,
        test_design_stems=test_design_stems,
        valid_ratio=0.50,  # Even at 50% valid, test designs stay 'test'
        hash_seed=HASH_SEED,
        schema_version=SCHEMA,
    )

    test_rows = df_v3[df_v3["design_name"].isin(test_design_stems)]
    assert (test_rows["split"] == "test").all()


def test_check_no_net_overlap_catches_leak(synthetic_legacy_manifest):
    """The legacy manifest fixture intentionally has net mixing — verify
    that `check_no_net_overlap` raises on it (i.e., the test detects leaks)."""
    with pytest.raises(AssertionError, match="multiple splits"):
        check_no_net_overlap(synthetic_legacy_manifest)


def test_schema_version_required():
    """Loaders refuse to load a manifest missing schema_version column."""
    import pandas as pd
    bad = pd.DataFrame({
        "design_name": ["a"], "net_name": ["n1"], "split": ["train"]
    })
    with pytest.raises(AssertionError, match="schema_version|Schema version"):
        check_schema_version(bad, "v3")


def test_schema_version_mismatch_rejected():
    """Loader refuses a manifest stamped with a different schema_version."""
    import pandas as pd
    bad = pd.DataFrame({
        "design_name": ["a"], "net_name": ["n1"], "split": ["train"],
        "schema_version": ["v9"],  # legacy version
    })
    with pytest.raises(AssertionError, match="Schema version mismatch"):
        check_schema_version(bad, "v3")
