"""Shared helpers for turning object tables into per-sample results."""

from __future__ import annotations

import json
from collections import Counter

import pandas as pd

from data_loader import Sample
from engulfment import EngulfmentResult
from models import SampleResult


def volume_stats(df: pd.DataFrame) -> tuple[float, float, float]:
    """Return mean, median, and sample standard deviation for object volumes."""
    if df.empty:
        return (float("nan"), float("nan"), float("nan"))
    series = df["Volume (micron^3)"]
    return (float(series.mean()), float(series.median()), float(series.std()))


def summarize_reject_reasons(df: pd.DataFrame) -> str:
    """Compact pipe-delimited rejection reasons into a stable JSON count string."""
    if "reject_reason" not in df.columns:
        return ""

    counts: Counter[str] = Counter()
    for value in df["reject_reason"].fillna(""):
        if not value:
            continue
        for item in str(value).split("|"):
            if item:
                counts[item] += 1
    if not counts:
        return ""
    return json.dumps(dict(sorted(counts.items())), sort_keys=True)


def build_sample_result(
    *,
    sample: Sample,
    hflu_before: pd.DataFrame,
    scer_before: pd.DataFrame,
    hflu_after: pd.DataFrame,
    scer_after: pd.DataFrame,
    engulfment_result: EngulfmentResult,
    backend: str,
    runtime_seconds: float = 0.0,
    from_cache: bool = False,
    staged_local: bool = False,
    review_required: bool = False,
    ambiguous_bacteria_count: int = 0,
) -> SampleResult:
    """Build the canonical result row for one sample.

    The function keeps all downstream writers on the same schema whether a sample
    came from legacy object CSVs or from the Python-native ND2 backend.
    """
    hflu_before_stats = volume_stats(hflu_before)
    scer_before_stats = volume_stats(scer_before)
    hflu_after_stats = volume_stats(hflu_after)
    scer_after_stats = volume_stats(scer_after)

    engulfed_centroids = [
        [round(float(value), 6) for value in centroid]
        for centroid in engulfment_result.engulfed_yeast_centroids
    ]

    return SampleResult(
        sample_name=sample.prefix,
        session_label=sample.session_label,
        biological_replicate=sample.biological_replicate,
        technical_replicate=sample.technical_replicate,
        hflu_count_before=len(hflu_before),
        hflu_count_after=len(hflu_after),
        scer_count_before=len(scer_before),
        scer_count_after=len(scer_after),
        hflu_mean_vol_before=hflu_before_stats[0],
        hflu_median_vol_before=hflu_before_stats[1],
        hflu_std_vol_before=hflu_before_stats[2],
        hflu_mean_vol_after=hflu_after_stats[0],
        hflu_median_vol_after=hflu_after_stats[1],
        hflu_std_vol_after=hflu_after_stats[2],
        scer_mean_vol_before=scer_before_stats[0],
        scer_median_vol_before=scer_before_stats[1],
        scer_std_vol_before=scer_before_stats[2],
        scer_mean_vol_after=scer_after_stats[0],
        scer_median_vol_after=scer_after_stats[1],
        scer_std_vol_after=scer_after_stats[2],
        engulfing_yeast_count=engulfment_result.engulfing_yeast_count,
        engulfment_rate=engulfment_result.engulfment_rate,
        engulfed_yeast_centroids=engulfed_centroids,
        backend=backend,
        runtime_seconds=runtime_seconds,
        from_cache=from_cache,
        staged_local=staged_local,
        review_required=review_required,
        ambiguous_bacteria_count=ambiguous_bacteria_count,
        hflu_rejected_count=max(0, len(hflu_before) - len(hflu_after)),
        scer_rejected_count=max(0, len(scer_before) - len(scer_after)),
        hflu_reject_reasons=summarize_reject_reasons(hflu_before),
        scer_reject_reasons=summarize_reject_reasons(scer_before),
    )
