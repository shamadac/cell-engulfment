"""Tests for replicate statistics and group-comparison selection."""

from __future__ import annotations

import pytest

from pipeline import SampleResult
from statistics import GroupComparisonResult, ReplicateStats, compute_replicate_stats, run_group_comparison


def _result(sample_name: str, replicate: str, technical_replicate: int, rate: float) -> SampleResult:
    return SampleResult(
        sample_name=sample_name,
        session_label="test-session",
        biological_replicate=replicate,
        technical_replicate=technical_replicate,
        hflu_count_before=2,
        hflu_count_after=2,
        scer_count_before=2,
        scer_count_after=2,
        hflu_mean_vol_before=0.3,
        hflu_median_vol_before=0.3,
        hflu_std_vol_before=0.0,
        hflu_mean_vol_after=0.3,
        hflu_median_vol_after=0.3,
        hflu_std_vol_after=0.0,
        scer_mean_vol_before=50.0,
        scer_median_vol_before=50.0,
        scer_std_vol_before=0.0,
        scer_mean_vol_after=50.0,
        scer_median_vol_after=50.0,
        scer_std_vol_after=0.0,
        engulfing_yeast_count=1,
        engulfment_rate=rate,
        engulfed_yeast_centroids=[],
    )


def test_compute_replicate_stats_known_values() -> None:
    results = [_result("A1", "A", 1, 10.0), _result("A2", "A", 2, 20.0), _result("A3", "A", 3, 30.0)]
    stats = compute_replicate_stats(results)

    assert len(stats) == 1
    assert stats[0].mean_engulfment_rate == pytest.approx(20.0)
    assert stats[0].sem_engulfment_rate == pytest.approx(5.77350269)
    assert stats[0].normality_test_applicable is True


def test_small_n_marks_normality_as_not_applicable() -> None:
    results = [_result("A1", "A", 1, 10.0), _result("A2", "A", 2, 20.0)]
    stats = compute_replicate_stats(results)

    assert stats[0].normality_test_applicable is False
    assert stats[0].shapiro_wilk_stat is None
    assert stats[0].shapiro_wilk_p is None


def test_run_group_comparison_selects_anova() -> None:
    results = [
        _result("A1", "A", 1, 10.0),
        _result("A2", "A", 2, 11.0),
        _result("A3", "A", 3, 12.0),
        _result("B1", "B", 1, 20.0),
        _result("B2", "B", 2, 21.0),
        _result("B3", "B", 3, 22.0),
        _result("C1", "C", 1, 30.0),
        _result("C2", "C", 2, 31.0),
        _result("C3", "C", 3, 32.0),
    ]
    replicate_stats = [
        ReplicateStats("A", 3, 11.0, 0.5, 0.9, 0.6, True, True),
        ReplicateStats("B", 3, 21.0, 0.5, 0.9, 0.7, True, True),
        ReplicateStats("C", 3, 31.0, 0.5, 0.9, 0.8, True, True),
    ]

    group_result = run_group_comparison(replicate_stats, results)

    assert group_result.test_name == "one-way ANOVA"
    assert group_result.p_value is not None


def test_run_group_comparison_selects_kruskal_for_small_n() -> None:
    results = [
        _result("A1", "A", 1, 10.0),
        _result("A2", "A", 2, 20.0),
        _result("B1", "B", 1, 30.0),
        _result("B2", "B", 2, 40.0),
    ]
    replicate_stats = [
        ReplicateStats("A", 2, 15.0, 5.0, None, None, False, False),
        ReplicateStats("B", 2, 35.0, 5.0, None, None, False, False),
    ]

    group_result = run_group_comparison(replicate_stats, results)

    assert group_result.test_name == "Kruskal-Wallis"
    assert "fewer than 3 technical replicates" in group_result.rationale
