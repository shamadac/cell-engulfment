"""Tests for ND2 backend caching, staging, and sample processing."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from python_backend import _process_nd2_sample, _stage_nd2_file
from segmentation import ChannelArtifacts


def _measurement_rows() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    hflu_df = pd.DataFrame(
        [
            {
                "label_id": 1,
                "XM": 1.0,
                "YM": 1.0,
                "ZM": 1.0,
                "BX": 1.0,
                "BY": 1.0,
                "BZ": 1.0,
                "B-width": 1.0,
                "B-height": 1.0,
                "B-depth": 1.0,
                "Volume (micron^3)": 1.0,
                "voxel_centroid_z": 1.0,
                "voxel_centroid_y": 1.0,
                "voxel_centroid_x": 1.0,
                "bbox_volume_um3": 1.0,
                "occupancy": 1.0,
                "max_aspect_ratio": 1.0,
                "border_touch": False,
                "reject_reason": "",
            }
        ]
    )
    scer_df = pd.DataFrame(
        [
            {
                "label_id": 10,
                "XM": 1.0,
                "YM": 1.0,
                "ZM": 1.0,
                "BX": 0.0,
                "BY": 0.0,
                "BZ": 0.0,
                "B-width": 3.0,
                "B-height": 3.0,
                "B-depth": 3.0,
                "Volume (micron^3)": 27.0,
                "voxel_centroid_z": 1.0,
                "voxel_centroid_y": 1.0,
                "voxel_centroid_x": 1.0,
                "bbox_volume_um3": 27.0,
                "occupancy": 1.0,
                "max_aspect_ratio": 1.0,
                "border_touch": False,
                "reject_reason": "",
            }
        ]
    )
    hflu_labels = np.zeros((4, 4, 4), dtype=np.int32)
    hflu_labels[1, 1, 1] = 1
    scer_labels = np.zeros((4, 4, 4), dtype=np.int32)
    scer_labels[0:3, 0:3, 0:3] = 10
    return hflu_df, scer_df, hflu_labels, scer_labels


def test_process_nd2_sample_uses_cache_on_second_run(tmp_path: Path, monkeypatch) -> None:
    nd2_path = tmp_path / "A1.nd2"
    nd2_path.write_text("fake", encoding="utf-8")
    session_csv_dir = tmp_path / "exports"
    output_dir = tmp_path / "output"
    cache_dir = tmp_path / "cache"

    hflu_df, scer_df, hflu_labels, scer_labels = _measurement_rows()

    def fake_read(*args, **kwargs):
        zeros = np.zeros((4, 4, 4), dtype=np.float32)
        return zeros, zeros, (1.0, 1.0, 1.0)

    def fake_segment_hflu(*args, **kwargs):
        return ChannelArtifacts(
            raw_stack=np.zeros((4, 4, 4), dtype=np.float32),
            preprocessed_stack=np.zeros((4, 4, 4), dtype=np.float32),
            binary_mask=hflu_labels > 0,
            labels=hflu_labels,
            measurements=hflu_df.copy(),
        )

    def fake_segment_scer(*args, **kwargs):
        return ChannelArtifacts(
            raw_stack=np.zeros((4, 4, 4), dtype=np.float32),
            preprocessed_stack=np.zeros((4, 4, 4), dtype=np.float32),
            binary_mask=scer_labels > 0,
            labels=scer_labels,
            measurements=scer_df.copy(),
        )

    monkeypatch.setattr("python_backend._read_nd2_channels", fake_read)
    monkeypatch.setattr("python_backend.segment_hflu_stack", fake_segment_hflu)
    monkeypatch.setattr("python_backend.segment_scer_stack", fake_segment_scer)

    task = {
        "session_label": "Example Session",
        "sample_prefix": "A1",
        "nd2_path": str(nd2_path),
        "session_csv_dir": str(session_csv_dir),
        "cache_dir": str(cache_dir),
        "output_dir": str(output_dir),
        "config": {
            "pipeline": {
                "backend": "python_native_nd2",
                "workers": 1,
                "cache_dir": str(cache_dir),
                "save_qc_overlays": False,
                "save_label_stacks": False,
                "stage_nd2_to_local": "never",
            },
            "segmentation": {
                "scer": {
                    "channel": 0,
                    "gaussian_sigma_xyz": [1.0, 1.0, 1.0],
                    "threshold_method": "otsu",
                    "min_volume_um3": 10.0,
                    "max_volume_um3": 100.0,
                    "remove_border_objects": True,
                    "watershed": True,
                    "watershed_min_distance_um": 4.0,
                },
                "hflu": {
                    "channel": 1,
                    "gaussian_sigma_xyz": [0.5, 0.5, 0.5],
                    "threshold_method": "triangle",
                    "min_volume_um3": 0.2,
                    "max_volume_um3": 5.0,
                    "remove_border_objects": True,
                    "watershed": True,
                    "watershed_min_distance_um": 1.0,
                },
            },
            "engulfment": {
                "method": "mask_containment",
                "min_inside_fraction": 0.95,
                "allow_shared_bacteria": False,
                "save_assignment_details": False,
            },
        },
    }

    first = _process_nd2_sample(task)
    assert first.from_cache is False
    assert (session_csv_dir / "A1_hflu.csv").exists()
    assert (session_csv_dir / "A1_scer.csv").exists()

    def fail_read(*args, **kwargs):
        raise AssertionError("cache should have been used")

    monkeypatch.setattr("python_backend._read_nd2_channels", fail_read)
    second = _process_nd2_sample(task)

    assert second.from_cache is True
    assert second.engulfment_rate == first.engulfment_rate


def test_stage_nd2_file_copies_from_onedrive_paths(tmp_path: Path) -> None:
    onedrive_dir = tmp_path / "onedrive_test" / "microscope"
    onedrive_dir.mkdir(parents=True)
    source_path = onedrive_dir / "A1.nd2"
    source_path.write_text("fake", encoding="utf-8")
    staged_path, staged = _stage_nd2_file(source_path, tmp_path / "cache", "Session", "auto")

    assert staged is True
    assert staged_path.exists()
    assert staged_path.read_text(encoding="utf-8") == "fake"
