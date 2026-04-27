"""Volume-based object filtering for legacy measurement tables."""

from __future__ import annotations

import pandas as pd


def apply_size_filter(
    df: pd.DataFrame,
    min_vol: float,
    max_vol: float,
    cell_type: str,
    sample_name: str,
    logger,
) -> pd.DataFrame:
    """Keep objects whose measured volumes fall inside an inclusive range."""
    if min_vol > max_vol:
        raise ValueError("min_vol must be less than or equal to max_vol")

    initial_count = len(df)
    filtered_df = df[df["Volume (micron^3)"].between(min_vol, max_vol, inclusive="both")].copy()
    final_count = len(filtered_df)

    logger.info(
        "Applied %s size filter to %s: thresholds=[%s, %s], before=%s, after=%s",
        cell_type,
        sample_name,
        min_vol,
        max_vol,
        initial_count,
        final_count,
    )

    if initial_count > 0:
        retention_fraction = final_count / initial_count
        if retention_fraction < 0.10:
            logger.warning(
                "Low %s retention for %s after size filtering: %.3f",
                cell_type,
                sample_name,
                retention_fraction,
            )

    return filtered_df
