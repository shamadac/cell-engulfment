"""Tests for legacy and mask-based engulfment classification."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from engulfment import (
    classify_mask_containment,
    classify_sample,
    compute_hflu_radius,
    compute_scer_radius,
    is_engulfed,
    validate_shell_cavity_support,
)


COLUMNS = ["XM", "YM", "ZM", "BX", "BY", "BZ", "B-width", "B-height", "B-depth", "Volume (micron^3)"]


def _df(rows: list[list[float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=COLUMNS)


@st.composite
def object_dataframe(draw, max_rows: int = 5):
    row_count = draw(st.integers(min_value=0, max_value=max_rows))
    rows: list[list[float]] = []
    for _ in range(row_count):
        rows.append(
            [
                draw(st.floats(min_value=-20, max_value=20, allow_nan=False, allow_infinity=False, width=32)),
                draw(st.floats(min_value=-20, max_value=20, allow_nan=False, allow_infinity=False, width=32)),
                draw(st.floats(min_value=-20, max_value=20, allow_nan=False, allow_infinity=False, width=32)),
                0.0,
                0.0,
                0.0,
                draw(st.floats(min_value=0, max_value=10, allow_nan=False, allow_infinity=False, width=32)),
                draw(st.floats(min_value=0, max_value=10, allow_nan=False, allow_infinity=False, width=32)),
                draw(st.floats(min_value=0, max_value=10, allow_nan=False, allow_infinity=False, width=32)),
                draw(st.floats(min_value=0, max_value=200, allow_nan=False, allow_infinity=False, width=32)),
            ]
        )
    return _df(rows)


def test_is_engulfed_detects_inside_case() -> None:
    scer_row = _df([[0, 0, 0, 0, 0, 0, 8, 8, 8, 50]]).iloc[0]
    hflu_row = _df([[0, 0, 0, 0, 0, 0, 1, 1, 1, 0.3]]).iloc[0]
    assert is_engulfed(scer_row, hflu_row) is True


def test_is_engulfed_detects_outside_case() -> None:
    scer_row = _df([[0, 0, 0, 0, 0, 0, 8, 8, 8, 50]]).iloc[0]
    hflu_row = _df([[20, 20, 20, 0, 0, 0, 1, 1, 1, 0.3]]).iloc[0]
    assert is_engulfed(scer_row, hflu_row) is False


def test_is_engulfed_is_true_on_boundary() -> None:
    scer_row = _df([[0, 0, 0, 0, 0, 0, 4, 4, 4, 50]]).iloc[0]
    hflu_row = _df([[2, 0, 0, 0, 0, 0, 0, 0, 0, 0.3]]).iloc[0]
    assert is_engulfed(scer_row, hflu_row) is True


def test_classify_sample_returns_expected_indices() -> None:
    scer_df = _df(
        [
            [0, 0, 0, 0, 0, 0, 8, 8, 8, 50],
            [100, 100, 100, 0, 0, 0, 8, 8, 8, 50],
        ]
    )
    hflu_df = _df([[0, 0, 0, 0, 0, 0, 1, 1, 1, 0.3]])
    result = classify_sample(hflu_df, scer_df)

    assert result.engulfing_yeast_count == 1
    assert result.engulfed_yeast_indices == [0]
    assert result.engulfment_rate == pytest.approx(50.0)


def test_classify_sample_handles_empty_inputs() -> None:
    result = classify_sample(_df([]), _df([]))
    assert result.engulfing_yeast_count == 0
    assert result.engulfment_rate == 0.0
    assert result.engulfed_yeast_indices == []


@settings(max_examples=50)
@given(df=object_dataframe())
def test_computed_radii_are_non_negative(df: pd.DataFrame) -> None:
    for _, row in df.iterrows():
        assert compute_hflu_radius(row) >= 0
        assert compute_scer_radius(row) >= 0


@settings(max_examples=50)
@given(hflu_df=object_dataframe(max_rows=4), scer_df=object_dataframe(max_rows=4))
def test_engulfment_rate_is_bounded(hflu_df: pd.DataFrame, scer_df: pd.DataFrame) -> None:
    result = classify_sample(hflu_df, scer_df)
    assert 0.0 <= result.engulfment_rate <= 100.0


@settings(max_examples=50)
@given(hflu_df=object_dataframe(max_rows=4), scer_df=object_dataframe(max_rows=4))
def test_classification_is_reproducible(hflu_df: pd.DataFrame, scer_df: pd.DataFrame) -> None:
    first = classify_sample(hflu_df, scer_df)
    second = classify_sample(hflu_df, scer_df)
    assert first == second


def test_classify_mask_containment_requires_overlap_and_centroid_inside() -> None:
    hflu_df = pd.DataFrame(
        [
            {
                "label_id": 1,
                "XM": 1.0,
                "YM": 1.0,
                "ZM": 1.0,
                "BX": 0.0,
                "BY": 0.0,
                "BZ": 0.0,
                "B-width": 1.0,
                "B-height": 1.0,
                "B-depth": 1.0,
                "Volume (micron^3)": 1.0,
                "voxel_centroid_z": 1.0,
                "voxel_centroid_y": 1.0,
                "voxel_centroid_x": 1.0,
            },
            {
                "label_id": 2,
                "XM": 3.0,
                "YM": 3.0,
                "ZM": 3.0,
                "BX": 2.0,
                "BY": 2.0,
                "BZ": 2.0,
                "B-width": 1.0,
                "B-height": 1.0,
                "B-depth": 1.0,
                "Volume (micron^3)": 1.0,
                "voxel_centroid_z": 3.0,
                "voxel_centroid_y": 3.0,
                "voxel_centroid_x": 3.0,
            },
        ]
    )
    scer_df = pd.DataFrame(
        [
            {
                "label_id": 10,
                "XM": 1.5,
                "YM": 1.5,
                "ZM": 1.5,
                "BX": 0.0,
                "BY": 0.0,
                "BZ": 0.0,
                "B-width": 4.0,
                "B-height": 4.0,
                "B-depth": 4.0,
                "Volume (micron^3)": 64.0,
            }
        ]
    )

    hflu_labels = np.zeros((5, 5, 5), dtype=np.int32)
    hflu_labels[1, 1, 1] = 1
    hflu_labels[3, 3, 3] = 2
    scer_labels = np.zeros((5, 5, 5), dtype=np.int32)
    scer_labels[0:3, 0:3, 0:3] = 10

    result = classify_mask_containment(hflu_df, scer_df, hflu_labels, scer_labels)

    assert result.engulfing_yeast_count == 1
    assert result.engulfed_yeast_indices == [0]
    assert result.ambiguous_bacteria_count == 0


def test_classify_mask_containment_rejects_overlap_when_centroid_is_outside() -> None:
    hflu_df = pd.DataFrame(
        [
            {
                "label_id": 1,
                "XM": 4.0,
                "YM": 4.0,
                "ZM": 4.0,
                "BX": 0.0,
                "BY": 0.0,
                "BZ": 0.0,
                "B-width": 1.0,
                "B-height": 1.0,
                "B-depth": 1.0,
                "Volume (micron^3)": 1.0,
                "voxel_centroid_z": 4.0,
                "voxel_centroid_y": 4.0,
                "voxel_centroid_x": 4.0,
            }
        ]
    )
    scer_df = pd.DataFrame(
        [
            {
                "label_id": 10,
                "XM": 1.5,
                "YM": 1.5,
                "ZM": 1.5,
                "BX": 0.0,
                "BY": 0.0,
                "BZ": 0.0,
                "B-width": 5.0,
                "B-height": 5.0,
                "B-depth": 5.0,
                "Volume (micron^3)": 125.0,
            }
        ]
    )

    hflu_labels = np.zeros((6, 6, 6), dtype=np.int32)
    hflu_labels[1, 1, 1] = 1
    scer_labels = np.zeros((6, 6, 6), dtype=np.int32)
    scer_labels[0:3, 0:3, 0:3] = 10

    result = classify_mask_containment(hflu_df, scer_df, hflu_labels, scer_labels)

    assert result.engulfing_yeast_count == 0
    assert result.assignment_rows == []


def test_classify_mask_containment_deduplicates_shared_bacteria() -> None:
    hflu_df = pd.DataFrame(
        [
            {
                "label_id": 1,
                "XM": 2.0,
                "YM": 2.0,
                "ZM": 2.0,
                "BX": 1.0,
                "BY": 1.0,
                "BZ": 1.0,
                "B-width": 2.0,
                "B-height": 2.0,
                "B-depth": 2.0,
                "Volume (micron^3)": 8.0,
                "voxel_centroid_z": 2.0,
                "voxel_centroid_y": 2.0,
                "voxel_centroid_x": 2.0,
            }
        ]
    )
    scer_df = pd.DataFrame(
        [
            {
                "label_id": 10,
                "XM": 2.0,
                "YM": 2.0,
                "ZM": 2.0,
                "BX": 0.0,
                "BY": 0.0,
                "BZ": 0.0,
                "B-width": 4.0,
                "B-height": 4.0,
                "B-depth": 4.0,
                "Volume (micron^3)": 64.0,
            },
            {
                "label_id": 11,
                "XM": 2.0,
                "YM": 2.0,
                "ZM": 2.0,
                "BX": 1.0,
                "BY": 1.0,
                "BZ": 1.0,
                "B-width": 4.0,
                "B-height": 4.0,
                "B-depth": 4.0,
                "Volume (micron^3)": 64.0,
            },
        ]
    )

    hflu_labels = np.zeros((6, 6, 6), dtype=np.int32)
    hflu_labels[2:4, 2:4, 2:4] = 1
    scer_labels = np.zeros((6, 6, 6), dtype=np.int32)
    scer_labels[2:4, 2:4, 2:3] = 10
    scer_labels[2:4, 2:4, 3:4] = 11

    result = classify_mask_containment(
        hflu_df,
        scer_df,
        hflu_labels,
        scer_labels,
        min_inside_fraction=0.4,
    )

    assert result.engulfing_yeast_count == 1
    assert result.ambiguous_bacteria_count == 1


def test_classify_mask_containment_rejects_partial_overlap_below_threshold() -> None:
    hflu_df = pd.DataFrame(
        [
            {
                "label_id": 1,
                "XM": 1.5,
                "YM": 1.5,
                "ZM": 1.5,
                "BX": 1.0,
                "BY": 1.0,
                "BZ": 1.0,
                "B-width": 2.0,
                "B-height": 2.0,
                "B-depth": 2.0,
                "Volume (micron^3)": 8.0,
                "voxel_centroid_z": 1.0,
                "voxel_centroid_y": 1.0,
                "voxel_centroid_x": 1.0,
            }
        ]
    )
    scer_df = pd.DataFrame(
        [
            {
                "label_id": 10,
                "XM": 1.5,
                "YM": 1.5,
                "ZM": 1.5,
                "BX": 0.0,
                "BY": 0.0,
                "BZ": 0.0,
                "B-width": 3.0,
                "B-height": 3.0,
                "B-depth": 3.0,
                "Volume (micron^3)": 27.0,
            }
        ]
    )
    hflu_labels = np.zeros((4, 4, 4), dtype=np.int32)
    hflu_labels[1:3, 1:3, 1:3] = 1
    scer_labels = np.zeros((4, 4, 4), dtype=np.int32)
    scer_labels[1:2, 1:3, 1:3] = 10

    result = classify_mask_containment(
        hflu_df,
        scer_df,
        hflu_labels,
        scer_labels,
        min_inside_fraction=0.95,
    )

    assert result.engulfing_yeast_count == 0


def test_classify_mask_containment_interior_margin_rejects_boundary_hugging_object() -> None:
    hflu_df = pd.DataFrame(
        [
            {
                "label_id": 1,
                "XM": 2.0,
                "YM": 0.0,
                "ZM": 1.0,
                "BX": 0.0,
                "BY": 0.0,
                "BZ": 0.0,
                "B-width": 1.0,
                "B-height": 1.0,
                "B-depth": 1.0,
                "Volume (micron^3)": 1.0,
                "voxel_centroid_z": 1.0,
                "voxel_centroid_y": 0.0,
                "voxel_centroid_x": 2.0,
            }
        ]
    )
    scer_df = pd.DataFrame(
        [
            {
                "label_id": 10,
                "XM": 2.0,
                "YM": 2.0,
                "ZM": 1.0,
                "BX": 0.0,
                "BY": 0.0,
                "BZ": 0.0,
                "B-width": 5.0,
                "B-height": 5.0,
                "B-depth": 3.0,
                "Volume (micron^3)": 75.0,
            }
        ]
    )

    hflu_labels = np.zeros((3, 5, 5), dtype=np.int32)
    hflu_labels[1, 0, 2] = 1
    scer_labels = np.zeros((3, 5, 5), dtype=np.int32)
    scer_labels[:, :, :] = 10

    without_margin = classify_mask_containment(
        hflu_df,
        scer_df,
        hflu_labels,
        scer_labels,
        min_inside_fraction=0.95,
        interior_margin_um=0.0,
        spacing_zyx=(1.0, 1.0, 1.0),
    )
    with_margin = classify_mask_containment(
        hflu_df,
        scer_df,
        hflu_labels,
        scer_labels,
        min_inside_fraction=0.95,
        interior_margin_um=1.1,
        spacing_zyx=(1.0, 1.0, 1.0),
    )

    assert without_margin.engulfing_yeast_count == 1
    assert with_margin.engulfing_yeast_count == 0


def test_validate_shell_cavity_support_keeps_only_shell_enclosed_assignments() -> None:
    hflu_df = pd.DataFrame(
        [
            {
                "label_id": 1,
                "XM": 2.0,
                "YM": 2.0,
                "ZM": 1.0,
                "BX": 0.0,
                "BY": 0.0,
                "BZ": 0.0,
                "B-width": 1.0,
                "B-height": 1.0,
                "B-depth": 1.0,
                "Volume (micron^3)": 1.0,
                "voxel_centroid_z": 1.0,
                "voxel_centroid_y": 2.0,
                "voxel_centroid_x": 2.0,
            },
            {
                "label_id": 2,
                "XM": 2.0,
                "YM": 8.0,
                "ZM": 1.0,
                "BX": 0.0,
                "BY": 7.0,
                "BZ": 0.0,
                "B-width": 1.0,
                "B-height": 1.0,
                "B-depth": 1.0,
                "Volume (micron^3)": 1.0,
                "voxel_centroid_z": 1.0,
                "voxel_centroid_y": 8.0,
                "voxel_centroid_x": 2.0,
            },
        ]
    )
    scer_df = pd.DataFrame(
        [
            {
                "label_id": 10,
                "XM": 2.0,
                "YM": 2.0,
                "ZM": 1.0,
                "BX": 0.0,
                "BY": 0.0,
                "BZ": 0.0,
                "B-width": 5.0,
                "B-height": 5.0,
                "B-depth": 3.0,
                "Volume (micron^3)": 50.0,
            },
            {
                "label_id": 11,
                "XM": 2.0,
                "YM": 8.0,
                "ZM": 1.0,
                "BX": 0.0,
                "BY": 6.0,
                "BZ": 0.0,
                "B-width": 5.0,
                "B-height": 5.0,
                "B-depth": 3.0,
                "Volume (micron^3)": 50.0,
            },
        ]
    )

    hflu_labels = np.zeros((3, 12, 5), dtype=np.int32)
    hflu_labels[1, 2, 2] = 1
    hflu_labels[1, 8, 2] = 2
    scer_labels = np.zeros((3, 12, 5), dtype=np.int32)
    scer_labels[:, 0:5, 0:5] = 10
    scer_labels[:, 6:11, 0:5] = 11
    scer_support = np.zeros_like(scer_labels, dtype=bool)
    for z_index in range(3):
        scer_support[z_index, 0, 0:5] = True
        scer_support[z_index, 4, 0:5] = True
        scer_support[z_index, 0:5, 0] = True
        scer_support[z_index, 0:5, 4] = True
        scer_support[z_index, 6:11, 0:5] = True

    result = classify_mask_containment(
        hflu_df,
        scer_df,
        hflu_labels,
        scer_labels,
        min_inside_fraction=0.95,
    )

    validated = validate_shell_cavity_support(
        result,
        hflu_labels,
        scer_labels,
        scer_support,
        scer_df.reset_index(drop=True),
        min_overlap_fraction=0.5,
        min_centroid_slice_fraction=0.5,
    )

    assert result.engulfing_yeast_count == 2
    assert validated.engulfing_yeast_count == 1
    assert validated.engulfed_yeast_indices == [0]
    assert any(row["shell_cavity_validation_passed"] for row in validated.assignment_rows)
    assert any(not row["shell_cavity_validation_passed"] for row in validated.assignment_rows)
