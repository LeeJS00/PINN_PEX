"""
test_evaluation.py — metrics + stratified eval + seed aggregator tests.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    cap_mape, cap_mape_summary, delay_error, power_error,
    rc_percentile_metrics, build_metrics_row,
)
from src.evaluation.stratified_eval import (
    stratify_by_cap_quartile, stratify_by_design, full_stratified_report,
    write_stratified_report,
)
from src.evaluation.seed_aggregator import (
    cohens_d, cohens_d_label, bootstrap_median_ci,
    aggregate_per_method, aggregate_mwu_pairs,
    write_aggregation,
)


# ============================================================================
# metrics.py
# ============================================================================


def test_cap_mape_perfect_prediction_zero():
    pred = np.array([1.0, 2.0, 3.0])
    gold = np.array([1.0, 2.0, 3.0])
    err = cap_mape(pred, gold)
    assert np.allclose(err, 0.0)


def test_cap_mape_zero_target_returns_nan():
    pred = np.array([0.5])
    gold = np.array([0.0])
    err = cap_mape(pred, gold)
    assert np.isnan(err[0])


def test_cap_mape_summary_basic():
    pred = np.array([1.1, 2.2, 3.3])
    gold = np.array([1.0, 2.0, 3.0])
    s = cap_mape_summary(pred, gold)
    assert s["n_valid"] == 3
    assert s["median_mape"] > 0
    assert s["mean_mape"] > 0


def test_delay_error_zero_when_perfect():
    pred = np.array([1.0, 2.0])
    gold = np.array([1.0, 2.0])
    res = np.array([10.0, 20.0])
    d = delay_error(pred, gold, res)
    assert d["median_delay_err"] < 1e-10


def test_rc_percentile_ratio_one_when_perfect():
    rng = np.random.default_rng(0)
    gold = rng.uniform(0.1, 10.0, 200)
    pred = gold.copy()
    rc = rc_percentile_metrics(pred, gold)
    assert abs(rc["chip_ratio_p50"] - 1.0) < 1e-10


def test_build_metrics_row_no_resistance_handles_none():
    pred = np.array([1.0, 2.0, 3.0])
    gold = np.array([1.1, 2.2, 3.3])
    row = build_metrics_row("baseline", seed=0, pred_fF=pred, golden_fF=gold)
    assert row.method == "baseline"
    assert row.seed == 0
    assert row.cap_mape_mean > 0


# ============================================================================
# stratified_eval.py
# ============================================================================


@pytest.fixture
def synthetic_eval_df():
    """Synthetic per-net prediction frame."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        "design_name": rng.choice(["d1", "d2", "d3"], size=n),
        "net_name": [f"n_{i}" for i in range(n)],
        "golden_fF": np.abs(rng.normal(1.0, 5.0, n)) + 0.01,
        "pred_fF": np.abs(rng.normal(1.05, 5.0, n)) + 0.01,
        "layer_top": rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9], size=n),
        "length_um": rng.uniform(1.0, 1000.0, n),
        "net_class": rng.choice(["clock", "signal"], size=n),
    })


def test_stratify_by_cap_quartile_returns_4_buckets(synthetic_eval_df):
    out = stratify_by_cap_quartile(synthetic_eval_df)
    assert len(out) == 4
    assert all(c in out.columns for c in ["n_nets", "median_mape", "chip_ratio"])


def test_stratify_by_design_one_row_per_design(synthetic_eval_df):
    out = stratify_by_design(synthetic_eval_df)
    designs_in = set(synthetic_eval_df["design_name"].unique())
    designs_out = set(out["design_name"].unique())
    assert designs_in == designs_out


def test_full_stratified_report_has_all_axes(synthetic_eval_df):
    rep = full_stratified_report(synthetic_eval_df)
    assert "by_design" in rep
    assert "by_cap_quartile" in rep
    assert "by_layer" in rep
    assert "by_length" in rep
    assert "by_class" in rep
    assert "overall" in rep


def test_full_stratified_report_writes_files(synthetic_eval_df, tmp_path):
    rep = full_stratified_report(synthetic_eval_df)
    paths = write_stratified_report(rep, tmp_path)
    for k, p in paths.items():
        assert (tmp_path / f"stratified_{k}.csv").exists()


def test_full_stratified_report_missing_required_raises():
    bad = pd.DataFrame({"design_name": ["a"], "net_name": ["n"], "pred_fF": [1.0]})
    with pytest.raises(ValueError, match="missing required"):
        full_stratified_report(bad)


# ============================================================================
# seed_aggregator.py
# ============================================================================


def test_cohens_d_zero_for_identical_groups():
    a = np.array([1.0, 2.0, 3.0])
    d = cohens_d(a, a)
    assert d == 0.0 or np.isnan(d)  # zero pooled-stdev → nan


def test_cohens_d_label_thresholds():
    assert cohens_d_label(0.1) == "negligible"
    assert cohens_d_label(0.4) == "small"
    assert cohens_d_label(0.65) == "medium"
    assert cohens_d_label(1.0) == "large"


def test_bootstrap_median_ci_brackets_median():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    med, lo, hi = bootstrap_median_ci(values, n_resamples=500, seed=1)
    assert lo <= med <= hi


def test_aggregate_per_method_groups_correctly():
    per_run = pd.DataFrame({
        "method": ["a"] * 5 + ["b"] * 5,
        "seed": list(range(5)) * 2,
        "cap_mape_median": [0.1, 0.2, 0.15, 0.18, 0.17, 0.05, 0.06, 0.07, 0.08, 0.09],
    })
    out = aggregate_per_method(per_run)
    assert len(out) == 2
    assert set(out["method"]) == {"a", "b"}


def test_aggregate_mwu_pairs_one_pair_for_two_methods():
    per_run = pd.DataFrame({
        "method": ["a"] * 5 + ["b"] * 5,
        "seed": list(range(5)) * 2,
        "cap_mape_median": [0.1, 0.2, 0.15, 0.18, 0.17, 0.05, 0.06, 0.07, 0.08, 0.09],
    })
    out = aggregate_mwu_pairs(per_run)
    assert len(out) == 1
    assert out.iloc[0]["method_a"] == "a"
    assert out.iloc[0]["method_b"] == "b"


def test_write_aggregation_produces_3_csvs(tmp_path):
    per_run = pd.DataFrame({
        "method": ["a"] * 5 + ["b"] * 5,
        "seed": list(range(5)) * 2,
        "cap_mape_median": [0.10, 0.20, 0.15, 0.18, 0.17, 0.05, 0.06, 0.07, 0.08, 0.09],
    })
    paths = write_aggregation(per_run, tmp_path)
    assert (tmp_path / "per_run.csv").exists()
    assert (tmp_path / "per_method.csv").exists()
    assert (tmp_path / "mwu_pairs.csv").exists()
