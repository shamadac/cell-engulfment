from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from models import SampleResult


@dataclass(frozen=True)
class ReplicateStats:
    biological_replicate: str
    n: int
    mean_engulfment_rate: float
    sem_engulfment_rate: float
    shapiro_wilk_stat: float | None
    shapiro_wilk_p: float | None
    normality_test_applicable: bool
    normality_passed: bool


@dataclass(frozen=True)
class GroupComparisonResult:
    test_name: str
    statistic: float | None
    p_value: float | None
    rationale: str


def _group_rates(results: list["SampleResult"]) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for result in results:
        grouped[result.biological_replicate].append(float(result.engulfment_rate))
    return dict(sorted(grouped.items()))


def compute_replicate_stats(results: list["SampleResult"]) -> list[ReplicateStats]:
    replicate_stats: list[ReplicateStats] = []
    for biological_replicate, rates in _group_rates(results).items():
        n = len(rates)
        mean_rate = float(np.mean(rates))
        sem_rate = float(stats.sem(rates)) if n > 1 else 0.0

        if n >= 3:
            shapiro_stat, shapiro_p = stats.shapiro(rates)
            normality_test_applicable = True
            normality_passed = bool(shapiro_p > 0.05)
        else:
            shapiro_stat = None
            shapiro_p = None
            normality_test_applicable = False
            normality_passed = False

        replicate_stats.append(
            ReplicateStats(
                biological_replicate=biological_replicate,
                n=n,
                mean_engulfment_rate=mean_rate,
                sem_engulfment_rate=sem_rate,
                shapiro_wilk_stat=None if shapiro_stat is None else float(shapiro_stat),
                shapiro_wilk_p=None if shapiro_p is None else float(shapiro_p),
                normality_test_applicable=normality_test_applicable,
                normality_passed=normality_passed,
            )
        )

    return replicate_stats


def run_group_comparison(
    replicate_stats: list[ReplicateStats],
    results: list["SampleResult"],
) -> GroupComparisonResult:
    grouped_rates = _group_rates(results)
    groups = list(grouped_rates.values())

    if len(groups) < 2:
        return GroupComparisonResult(
            test_name="not_applicable",
            statistic=None,
            p_value=None,
            rationale="At least two biological replicates are required for a group comparison.",
        )

    all_normal = all(
        stat_item.normality_test_applicable and stat_item.normality_passed
        for stat_item in replicate_stats
    )

    if all_normal:
        statistic, p_value = stats.f_oneway(*groups)
        return GroupComparisonResult(
            test_name="one-way ANOVA",
            statistic=float(statistic),
            p_value=float(p_value),
            rationale="All biological replicates had valid Shapiro-Wilk results and passed at alpha=0.05.",
        )

    statistic, p_value = stats.kruskal(*groups)
    insufficient_n = any(not stat_item.normality_test_applicable for stat_item in replicate_stats)
    rationale = (
        "At least one biological replicate had fewer than 3 technical replicates."
        if insufficient_n
        else "At least one biological replicate failed Shapiro-Wilk at alpha=0.05."
    )
    return GroupComparisonResult(
        test_name="Kruskal-Wallis",
        statistic=float(statistic),
        p_value=float(p_value),
        rationale=rationale,
    )
