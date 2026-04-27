from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd
from pandas.api.types import is_float_dtype


def write_per_sample_csv(result, output_dir: Path) -> None:
    per_sample_dir = output_dir / "per_sample"
    per_sample_dir.mkdir(parents=True, exist_ok=True)
    output_path = per_sample_dir / f"results_{result.sample_name}.csv"
    pd.DataFrame([asdict(result)]).to_csv(output_path, index=False)


def write_summary_csv(results, output_dir: Path) -> None:
    summary_path = output_dir / "summary.csv"
    rows = []
    for result in results:
        row = asdict(result)
        row.pop("runtime_seconds", None)
        row.pop("from_cache", None)
        row.pop("staged_local", None)
        rows.append(row)
    df = pd.DataFrame(rows)
    for column in df.columns:
        if is_float_dtype(df[column]):
            df[column] = df[column].round(6)
    df.to_csv(summary_path, index=False)


def write_qc_summary_csv(results, output_dir: Path) -> None:
    rows = [
        {
            "sample_name": result.sample_name,
            "session_label": result.session_label,
            "backend": result.backend,
            "review_required": result.review_required,
            "ambiguous_bacteria_count": result.ambiguous_bacteria_count,
            "hflu_rejected_count": result.hflu_rejected_count,
            "scer_rejected_count": result.scer_rejected_count,
            "hflu_reject_reasons": result.hflu_reject_reasons,
            "scer_reject_reasons": result.scer_reject_reasons,
        }
        for result in results
    ]
    output_path = output_dir / "qc_summary.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)


def write_performance_csv(results, output_dir: Path) -> None:
    rows = [
        {
            "sample_name": result.sample_name,
            "session_label": result.session_label,
            "backend": result.backend,
            "runtime_seconds": result.runtime_seconds,
            "from_cache": result.from_cache,
            "staged_local": result.staged_local,
        }
        for result in results
    ]
    output_path = output_dir / "performance.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)


def write_statistical_report(replicate_stats, group_result, output_dir: Path) -> None:
    rows: list[dict] = []
    for replicate_stat in replicate_stats:
        rows.append(
            {
                "row_type": "replicate",
                "biological_replicate": replicate_stat.biological_replicate,
                "n": replicate_stat.n,
                "mean_engulfment_rate": replicate_stat.mean_engulfment_rate,
                "sem_engulfment_rate": replicate_stat.sem_engulfment_rate,
                "normality_test": "shapiro_wilk" if replicate_stat.normality_test_applicable else "not_applicable",
                "normality_test_applicable": replicate_stat.normality_test_applicable,
                "normality_passed": replicate_stat.normality_passed,
                "shapiro_wilk_stat": replicate_stat.shapiro_wilk_stat,
                "shapiro_wilk_p": replicate_stat.shapiro_wilk_p,
                "group_test_name": None,
                "group_test_statistic": None,
                "group_test_p_value": None,
                "rationale": None,
            }
        )

    rows.append(
        {
            "row_type": "group_comparison",
            "biological_replicate": None,
            "n": None,
            "mean_engulfment_rate": None,
            "sem_engulfment_rate": None,
            "normality_test": None,
            "normality_test_applicable": None,
            "normality_passed": None,
            "shapiro_wilk_stat": None,
            "shapiro_wilk_p": None,
            "group_test_name": group_result.test_name,
            "group_test_statistic": group_result.statistic,
            "group_test_p_value": group_result.p_value,
            "rationale": group_result.rationale,
        }
    )

    output_path = output_dir / "statistical_report.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
