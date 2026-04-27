from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def apply_publication_style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")
    ax.tick_params(labelsize=10)
    ax.title.set_fontsize(12)
    ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_size(10)


def _output_path(output_dir: Path, filename: str, config) -> Path:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir / f"{filename}.{config.format}"


def _group_results(results) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for result in results:
        grouped[result.biological_replicate].append(float(result.engulfment_rate))
    return dict(sorted(grouped.items()))


def _replicate_colors(replicates: list[str]) -> dict[str, str]:
    palette = ["#355070", "#6d597a", "#b56576", "#e56b6f", "#eaac8b", "#4f772d"]
    return {replicate: palette[index % len(palette)] for index, replicate in enumerate(replicates)}


def _annotate_group_result(ax: plt.Axes, group_result) -> None:
    if group_result is None or group_result.test_name == "not_applicable" or group_result.p_value is None:
        return
    ax.text(
        0.99,
        0.98,
        f"{group_result.test_name}\np = {group_result.p_value:.4g}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "boxstyle": "round,pad=0.3"},
    )


def plot_boxplot(stats, results, group_result, output_dir: Path, config) -> Path:
    grouped = _group_results(results)
    replicates = list(grouped)
    colors = _replicate_colors(replicates)
    data = [grouped[replicate] for replicate in replicates]
    positions = np.arange(1, len(replicates) + 1)
    output_path = _output_path(output_dir, "boxplot_engulfment_rate", config)

    fig, ax = plt.subplots(figsize=(8, 5))
    boxplot = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True)
    for patch, replicate in zip(boxplot["boxes"], replicates, strict=False):
        patch.set_facecolor(colors[replicate])
        patch.set_alpha(0.6)

    rng = np.random.default_rng(42)
    for position, replicate in zip(positions, replicates, strict=False):
        values = np.array(grouped[replicate], dtype=float)
        jitter = rng.uniform(-0.08, 0.08, size=len(values))
        ax.scatter(
            np.full(len(values), position) + jitter,
            values,
            color=colors[replicate],
            edgecolors="black",
            linewidths=0.5,
            s=28,
            zorder=3,
        )

    ax.set_xticks(positions, replicates)
    ax.set_xlabel("Biological Replicate")
    ax.set_ylabel("Engulfment Rate (%)")
    ax.set_title("Engulfment Rate by Biological Replicate")
    apply_publication_style(ax)
    _annotate_group_result(ax, group_result)
    fig.tight_layout()
    fig.savefig(output_path, dpi=config.dpi, format=config.format)
    plt.close(fig)
    return output_path


def plot_bar_chart(stats, group_result, output_dir: Path, config) -> Path:
    replicates = [item.biological_replicate for item in stats]
    means = [item.mean_engulfment_rate for item in stats]
    sems = [item.sem_engulfment_rate for item in stats]
    positions = np.arange(len(replicates))
    colors = _replicate_colors(replicates)
    output_path = _output_path(output_dir, "barchart_engulfment_rate", config)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        positions,
        means,
        yerr=sems,
        capsize=4,
        color=[colors[replicate] for replicate in replicates],
        edgecolor="black",
        linewidth=0.7,
    )
    ax.set_xticks(positions, replicates)
    ax.set_xlabel("Biological Replicate")
    ax.set_ylabel("Engulfment Rate (%)")
    ax.set_title("Mean Engulfment Rate by Biological Replicate")
    apply_publication_style(ax)
    _annotate_group_result(ax, group_result)
    fig.tight_layout()
    fig.savefig(output_path, dpi=config.dpi, format=config.format)
    plt.close(fig)
    return output_path


def plot_violin(stats, results, output_dir: Path, config) -> Path:
    grouped = _group_results(results)
    replicates = list(grouped)
    positions = np.arange(1, len(replicates) + 1)
    output_path = _output_path(output_dir, "violin_engulfment_rate", config)

    fig, ax = plt.subplots(figsize=(8, 5))
    violins = ax.violinplot([grouped[replicate] for replicate in replicates], positions=positions, showmeans=True)
    colors = _replicate_colors(replicates)
    for body, replicate in zip(violins["bodies"], replicates, strict=False):
        body.set_facecolor(colors[replicate])
        body.set_edgecolor("black")
        body.set_alpha(0.6)

    ax.set_xticks(positions, replicates)
    ax.set_xlabel("Biological Replicate")
    ax.set_ylabel("Engulfment Rate (%)")
    ax.set_title("Engulfment Rate Distribution by Biological Replicate")
    apply_publication_style(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=config.dpi, format=config.format)
    plt.close(fig)
    return output_path
