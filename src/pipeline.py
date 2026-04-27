from __future__ import annotations

from datetime import datetime
from pathlib import Path

from analysis_core import build_sample_result
from config_loader import PipelineConfig
from data_loader import Sample, discover_samples, load_sample_csvs
from diagnostics import plot_size_histogram
from engulfment import classify_sample
from figures import plot_bar_chart, plot_boxplot, plot_violin
from imagej_runner import run_imagej_macro
from logger import setup_logger
from models import SampleResult
from output_writer import (
    write_per_sample_csv,
    write_performance_csv,
    write_qc_summary_csv,
    write_statistical_report,
    write_summary_csv,
)
from python_backend import process_nd2_session
from size_filter import apply_size_filter
from statistics import compute_replicate_stats, run_group_comparison


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolved_backend(config: PipelineConfig, session) -> str:
    if config.pipeline.backend is not None:
        return config.pipeline.backend
    if session.nd2_dir is not None and config.imagej_executable is not None:
        return "legacy_fiji"
    return "legacy_csv"


def process_sample(
    sample: Sample,
    config: PipelineConfig,
    logger,
    *,
    backend_label: str = "legacy_csv",
) -> SampleResult:
    hflu_df, scer_df = load_sample_csvs(sample)
    output_dir = Path(logger.output_dir)  # type: ignore[attr-defined]

    plot_size_histogram(
        hflu_df,
        "hflu",
        sample.prefix,
        config.size_filters.hflu_min_um3,
        config.size_filters.hflu_max_um3,
        output_dir,
        config.figures,
    )
    plot_size_histogram(
        scer_df,
        "scer",
        sample.prefix,
        config.size_filters.scer_min_um3,
        config.size_filters.scer_max_um3,
        output_dir,
        config.figures,
    )

    hflu_filtered = apply_size_filter(
        hflu_df,
        config.size_filters.hflu_min_um3,
        config.size_filters.hflu_max_um3,
        "hflu",
        sample.prefix,
        logger,
    )
    scer_filtered = apply_size_filter(
        scer_df,
        config.size_filters.scer_min_um3,
        config.size_filters.scer_max_um3,
        "scer",
        sample.prefix,
        logger,
    )
    engulfment_result = classify_sample(hflu_filtered, scer_filtered)
    return build_sample_result(
        sample=sample,
        hflu_before=hflu_df,
        scer_before=scer_df,
        hflu_after=hflu_filtered,
        scer_after=scer_filtered,
        engulfment_result=engulfment_result,
        backend=backend_label,
    )


def _process_legacy_session(session, config: PipelineConfig, logger, output_dir: Path) -> list[SampleResult]:
    macro_path = PROJECT_ROOT / "macros" / "image_processing.ijm"
    backend = _resolved_backend(config, session)

    if backend == "legacy_fiji":
        if config.imagej_executable is None or session.nd2_dir is None:
            logger.error(
                "Skipping session '%s' because legacy_fiji requires imagej_executable and nd2_dir",
                session.label,
            )
            return []

        session.csv_dir.mkdir(parents=True, exist_ok=True)
        macro_success = run_imagej_macro(
            config.imagej_executable,
            session.nd2_dir,
            session.csv_dir,
            macro_path,
            logger,
        )
        if not macro_success:
            logger.error("Skipping session '%s' because ImageJ processing failed", session.label)
            return []

    samples = discover_samples(session.csv_dir, session.label, logger)
    if not samples:
        logger.warning("No valid paired samples discovered in %s", session.csv_dir)
        return []

    results: list[SampleResult] = []
    for sample in samples:
        try:
            result = process_sample(sample, config, logger, backend_label=backend)
        except Exception:
            logger.exception("Failed to process sample '%s' in session '%s'", sample.prefix, session.label)
            continue

        results.append(result)
        write_per_sample_csv(result, output_dir)
        logger.info(
            "Processed sample %s: engulfing_yeast_count=%s, engulfment_rate=%.3f",
            result.sample_name,
            result.engulfing_yeast_count,
            result.engulfment_rate,
        )

    return results


def run_pipeline(config: PipelineConfig) -> list[SampleResult]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.output_base_dir / timestamp
    logger = setup_logger(output_dir)
    logger.info("Starting cell-engulfment pipeline")

    results: list[SampleResult] = []
    for session in config.sessions:
        backend = _resolved_backend(config, session)
        logger.info("Processing session '%s' with backend '%s'", session.label, backend)

        if backend == "python_native_nd2":
            if session.nd2_dir is None:
                logger.error(
                    "Skipping session '%s' because python_native_nd2 requires nd2_dir",
                    session.label,
                )
                continue
            session_results = process_nd2_session(session, config, output_dir, logger)
            for result in session_results:
                write_per_sample_csv(result, output_dir)
            results.extend(session_results)
            continue

        results.extend(_process_legacy_session(session, config, logger, output_dir))

    write_summary_csv(results, output_dir)
    write_qc_summary_csv(results, output_dir)
    write_performance_csv(results, output_dir)

    if results:
        replicate_stats = compute_replicate_stats(results)
        group_result = run_group_comparison(replicate_stats, results)
        write_statistical_report(replicate_stats, group_result, output_dir)
        plot_boxplot(replicate_stats, results, group_result, output_dir, config.figures)
        plot_bar_chart(replicate_stats, group_result, output_dir, config.figures)
        if config.figures.violin_plot:
            plot_violin(replicate_stats, results, output_dir, config.figures)
    else:
        logger.warning("No results were produced; skipping statistics and figures.")

    logger.info("Pipeline finished with %s processed samples", len(results))
    return results
