"""Diagnostic plots for object-size distributions."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def plot_size_histogram(
    df: pd.DataFrame,
    cell_type: str,
    sample_name: str,
    min_vol: float,
    max_vol: float,
    output_dir: Path,
    config,
) -> Path:
    """Write a volume histogram with the configured size-filter bounds marked."""
    diagnostics_dir = output_dir / "cell_sizes"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    output_path = diagnostics_dir / f"{sample_name}_{cell_type}_size_histogram.{config.format}"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["Volume (micron^3)"], bins=50, color="#7aa6c2", edgecolor="white")
    ax.axvline(min_vol, color="#b22222", linestyle="--", linewidth=1.5)
    ax.axvline(max_vol, color="#b22222", linestyle="--", linewidth=1.5)
    ax.set_title(f"{sample_name} {cell_type} volume distribution")
    ax.set_xlabel("Volume (micron^3)")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(output_path, dpi=config.dpi, format=config.format)
    plt.close(fig)

    return output_path
