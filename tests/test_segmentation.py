"""Tests for 3D segmentation, measurement, and object rejection rules."""

from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

pytest.importorskip("skimage")

from segmentation import _measure_labels, segment_hflu_stack, segment_scer_stack


def _draw_disk(image: np.ndarray, z_index: int, center_y: int, center_x: int, radius: int, value: float) -> None:
    yy, xx = np.ogrid[: image.shape[1], : image.shape[2]]
    mask = (yy - center_y) ** 2 + (xx - center_x) ** 2 <= radius ** 2
    image[z_index, mask] = value


def test_segment_scer_splits_touching_cells_and_marks_border_objects() -> None:
    stack = np.zeros((9, 64, 64), dtype=np.float32)
    for z_index in range(3, 6):
        _draw_disk(stack, z_index, 24, 22, 8, 1.0)
        _draw_disk(stack, z_index, 24, 40, 8, 1.0)
        _draw_disk(stack, z_index, 1, 1, 6, 1.0)

    config = SimpleNamespace(
        gaussian_sigma_xyz=(0.0, 1.0, 1.0),
        threshold_method="otsu",
        min_volume_um3=40.0,
        max_volume_um3=700.0,
        remove_border_objects=True,
        watershed=True,
        watershed_min_distance_um=4.0,
    )

    artifacts = segment_scer_stack(stack, (1.0, 1.0, 1.0), config)

    assert len(artifacts.measurements) >= 3
    retained = artifacts.measurements[artifacts.measurements["reject_reason"] == ""]
    assert len(retained) >= 2


def test_segment_hflu_rejects_speckle_like_objects() -> None:
    stack = np.zeros((6, 32, 32), dtype=np.float32)
    stack[2:4, 10:12, 10:16] = 10.0
    stack[3, 25, 25] = 3.0

    config = SimpleNamespace(
        gaussian_sigma_xyz=(0.5, 0.75, 0.75),
        threshold_method="triangle",
        min_volume_um3=1.0,
        max_volume_um3=100.0,
        remove_border_objects=True,
        watershed=True,
        watershed_min_distance_um=1.0,
    )

    artifacts = segment_hflu_stack(stack, (1.0, 1.0, 1.0), config)

    assert len(artifacts.measurements) >= 1
    assert artifacts.measurements["reject_reason"].str.contains("speckle_like").any()
    assert (artifacts.measurements["reject_reason"] == "").any()


def test_measure_labels_marks_border_touch_objects() -> None:
    labels = np.zeros((4, 8, 8), dtype=np.int32)
    labels[:, 0:2, 0:2] = 1
    labels[1:3, 3:5, 3:5] = 2
    measurements = _measure_labels(
        labels,
        np.zeros_like(labels, dtype=np.float32),
        (1.0, 1.0, 1.0),
        "scer",
        1.0,
        100.0,
        True,
    )

    border_rows = measurements[measurements["label_id"] == 1]
    interior_rows = measurements[measurements["label_id"] == 2]
    assert border_rows["reject_reason"].str.contains("border_touch").all()
    assert not interior_rows["reject_reason"].str.contains("border_touch").any()
