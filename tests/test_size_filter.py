"""Tests for volume-based object filtering."""

from __future__ import annotations

import logging

import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from size_filter import apply_size_filter


def _logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = True
    logger.setLevel(logging.INFO)
    return logger


def test_apply_size_filter_keeps_boundary_values(caplog: pytest.LogCaptureFixture) -> None:
    df = pd.DataFrame({"Volume (micron^3)": [0.1, 0.2, 0.3, 0.5, 0.6]})
    logger = _logger("size_filter_boundary")

    with caplog.at_level(logging.INFO, logger=logger.name):
        filtered = apply_size_filter(df, 0.2, 0.5, "hflu", "A1", logger)

    assert filtered["Volume (micron^3)"].tolist() == [0.2, 0.3, 0.5]
    assert "thresholds=[0.2, 0.5]" in caplog.text


def test_apply_size_filter_handles_empty_dataframe() -> None:
    df = pd.DataFrame({"Volume (micron^3)": []})
    filtered = apply_size_filter(df, 0.2, 0.5, "hflu", "A1", _logger("size_filter_empty"))
    assert filtered.empty


@settings(max_examples=50)
@given(
    volumes=st.lists(
        st.floats(min_value=0, max_value=500, allow_nan=False, allow_infinity=False, width=32),
        max_size=20,
    ),
    min_vol=st.floats(min_value=0, max_value=250, allow_nan=False, allow_infinity=False, width=32),
    max_vol=st.floats(min_value=250, max_value=500, allow_nan=False, allow_infinity=False, width=32),
)
def test_filtered_count_never_exceeds_input_count(volumes, min_vol, max_vol) -> None:
    df = pd.DataFrame({"Volume (micron^3)": volumes})
    filtered = apply_size_filter(df, min_vol, max_vol, "scer", "prop", _logger("size_filter_prop"))

    assert len(filtered) <= len(df)
    assert filtered["Volume (micron^3)"].between(min_vol, max_vol, inclusive="both").all()
