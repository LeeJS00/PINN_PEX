"""
leak_check.py — H1 split-leak invariants.

Importable helper used by both `tests/test_split_invariants.py` (CI) and
`scripts/01_resplit_manifest.py` (post-split sanity).
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd


def check_no_net_overlap(manifest: pd.DataFrame) -> None:
    """Assert no (design_name, net_name) appears in multiple splits.

    Raises AssertionError with up to 10 violating examples.
    """
    grouped = manifest.groupby(["design_name", "net_name"])["split"].nunique()
    leaks = grouped[grouped > 1]
    if len(leaks) > 0:
        sample = leaks.head(10)
        raise AssertionError(
            f"H1 invariant violated: {len(leaks)} (design, net) pairs span "
            f"multiple splits.\nFirst 10:\n{sample.to_string()}"
        )


def check_test_designs_pure(
    manifest: pd.DataFrame, test_design_stems: set[str]
) -> None:
    """Assert all rows from TEST_DEFS designs are in split=='test'."""
    test_rows = manifest[manifest["design_name"].isin(test_design_stems)]
    bad = test_rows[test_rows["split"] != "test"]
    if len(bad) > 0:
        bad_designs = bad["design_name"].unique().tolist()
        raise AssertionError(
            f"H1 invariant violated: TEST_DEFS designs leaked out of "
            f"split='test'. Affected designs: {bad_designs}, "
            f"row count: {len(bad)}"
        )


def check_train_designs_pure(
    manifest: pd.DataFrame,
    test_design_stems: set[str],
) -> None:
    """Assert no train design rows landed in split=='test'."""
    non_test = manifest[~manifest["design_name"].isin(test_design_stems)]
    bad = non_test[non_test["split"] == "test"]
    if len(bad) > 0:
        raise AssertionError(
            f"H1 invariant violated: {len(bad)} train-design rows have "
            f"split=='test'. Sample design_names: "
            f"{bad['design_name'].unique()[:5].tolist()}"
        )


def check_split_coverage(manifest: pd.DataFrame) -> None:
    """Assert every row has a non-null split in {train, valid, test}."""
    valid_splits = {"train", "valid", "test"}
    null_count = manifest["split"].isna().sum()
    if null_count > 0:
        raise AssertionError(
            f"H1 invariant violated: {null_count} rows have null split."
        )
    bad_values = set(manifest["split"].unique()) - valid_splits
    if bad_values:
        raise AssertionError(
            f"H1 invariant violated: unexpected split values {bad_values}. "
            f"Allowed: {valid_splits}"
        )


def check_schema_version(manifest: pd.DataFrame, expected: str) -> None:
    """Assert manifest has schema_version column matching `expected`.

    A loader that expects v3 must REFUSE to load a manifest with a
    different schema_version. This is the experiment-systems-engineer
    discipline against silent format drift.
    """
    if "schema_version" not in manifest.columns:
        raise AssertionError(
            "Schema version column missing — refusing to load. "
            "This manifest may be a legacy v8/v9 file. Use the v3 manifest "
            "produced by `scripts/01_resplit_manifest.py` instead."
        )
    versions = set(manifest["schema_version"].unique())
    if versions != {expected}:
        raise AssertionError(
            f"Schema version mismatch: manifest contains {versions}, "
            f"expected {{'{expected}'}}."
        )


def run_all_checks(
    manifest: pd.DataFrame,
    test_design_stems: set[str],
    expected_schema: str,
) -> None:
    """Run all H1 invariants. Raises on first failure."""
    check_schema_version(manifest, expected_schema)
    check_split_coverage(manifest)
    check_test_designs_pure(manifest, test_design_stems)
    check_train_designs_pure(manifest, test_design_stems)
    check_no_net_overlap(manifest)
