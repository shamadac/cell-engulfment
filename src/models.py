"""Dataclasses shared across analysis, reporting, and plotting modules."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SampleResult:
    """One row of sample-level analysis output."""

    sample_name: str
    session_label: str
    biological_replicate: str
    technical_replicate: int
    hflu_count_before: int
    hflu_count_after: int
    scer_count_before: int
    scer_count_after: int
    hflu_mean_vol_before: float
    hflu_median_vol_before: float
    hflu_std_vol_before: float
    hflu_mean_vol_after: float
    hflu_median_vol_after: float
    hflu_std_vol_after: float
    scer_mean_vol_before: float
    scer_median_vol_before: float
    scer_std_vol_before: float
    scer_mean_vol_after: float
    scer_median_vol_after: float
    scer_std_vol_after: float
    engulfing_yeast_count: int
    engulfment_rate: float
    engulfed_yeast_centroids: list[list[float]] = field(default_factory=list)
    backend: str = "legacy_csv"
    runtime_seconds: float = 0.0
    from_cache: bool = False
    staged_local: bool = False
    review_required: bool = False
    ambiguous_bacteria_count: int = 0
    hflu_rejected_count: int = 0
    scer_rejected_count: int = 0
    hflu_reject_reasons: str = ""
    scer_reject_reasons: str = ""
