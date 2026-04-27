"""Tests for run-level figure generation helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from figures import plot_bar_chart, plot_boxplot, plot_violin
from pipeline import SampleResult
from statistics import GroupComparisonResult, compute_replicate_stats


def _result(sample_name: str, replicate: str, technical_replicate: int, rate: float) -> SampleResult:
    return SampleResult(
        sample_name=sample_name,
        session_label="figure-session",
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


def _results() -> list[SampleResult]:
    return [
        _result("A1", "A", 1, 50.0),
        _result("A2", "A", 2, 100.0),
        _result("B1", "B", 1, 0.0),
        _result("B2", "B", 2, 50.0),
        _result("C1", "C", 1, 100.0),
        _result("C2", "C", 2, 50.0),
    ]


def test_plot_boxplot_creates_file_with_labels_and_annotation(tmp_path: Path, monkeypatch) -> None:
    saved: list[tuple[Figure, Path, dict]] = []
    original_savefig = Figure.savefig
    monkeypatch.setattr(plt, "close", lambda fig=None: None)

    def spy(self, fname, *args, **kwargs):
        saved.append((self, Path(fname), kwargs))
        return original_savefig(self, fname, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", spy)

    config = SimpleNamespace(dpi=300, format="png", violin_plot=True)
    stats = compute_replicate_stats(_results())
    group_result = GroupComparisonResult("Kruskal-Wallis", 5.0, 0.01, "test rationale")

    output_path = plot_boxplot(stats, _results(), group_result, tmp_path, config)

    fig, _, kwargs = saved[-1]
    ax = fig.axes[0]
    assert output_path.exists()
    assert kwargs["dpi"] == 300
    assert ax.get_ylabel() == "Engulfment Rate (%)"
    assert any("p =" in text.get_text() for text in ax.texts)


def test_plot_bar_chart_creates_file_with_labels_and_annotation(tmp_path: Path, monkeypatch) -> None:
    saved: list[tuple[Figure, Path, dict]] = []
    original_savefig = Figure.savefig
    monkeypatch.setattr(plt, "close", lambda fig=None: None)

    def spy(self, fname, *args, **kwargs):
        saved.append((self, Path(fname), kwargs))
        return original_savefig(self, fname, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", spy)

    config = SimpleNamespace(dpi=300, format="png", violin_plot=True)
    stats = compute_replicate_stats(_results())
    group_result = GroupComparisonResult("Kruskal-Wallis", 5.0, 0.01, "test rationale")

    output_path = plot_bar_chart(stats, group_result, tmp_path, config)

    fig, _, kwargs = saved[-1]
    ax = fig.axes[0]
    assert output_path.exists()
    assert kwargs["dpi"] == 300
    assert ax.get_ylabel() == "Engulfment Rate (%)"
    assert any("p =" in text.get_text() for text in ax.texts)


def test_plot_violin_creates_file_when_enabled(tmp_path: Path) -> None:
    config = SimpleNamespace(dpi=300, format="png", violin_plot=True)
    stats = compute_replicate_stats(_results())

    output_path = plot_violin(stats, _results(), tmp_path, config)

    assert output_path.exists()
