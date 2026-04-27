from __future__ import annotations

import logging
from pathlib import Path

import pytest

from data_loader import Sample, discover_samples, load_sample_csvs


HEADER = "XM,YM,ZM,BX,BY,BZ,B-width,B-height,B-depth,Volume (micron^3)\n"
ROW = "0,0,0,0,0,0,1,1,1,0.3\n"


def _logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = True
    logger.setLevel(logging.INFO)
    return logger


def test_discover_samples_finds_pairs(tmp_path: Path) -> None:
    (tmp_path / "A1_hflu.csv").write_text(HEADER + ROW, encoding="utf-8")
    (tmp_path / "A1_scer.csv").write_text(HEADER + ROW, encoding="utf-8")
    (tmp_path / "B2_hflu.csv").write_text(HEADER + ROW, encoding="utf-8")
    (tmp_path / "B2_scer.csv").write_text(HEADER + ROW, encoding="utf-8")

    samples = discover_samples(tmp_path, "Session 1", _logger("loader_pairs"))

    assert samples == [
        Sample("A1", "Session 1", "A", 1, tmp_path / "A1_hflu.csv", tmp_path / "A1_scer.csv"),
        Sample("B2", "Session 1", "B", 2, tmp_path / "B2_hflu.csv", tmp_path / "B2_scer.csv"),
    ]


def test_discover_samples_skips_unpaired_and_invalid(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    logger = _logger("loader_skip")
    (tmp_path / "A1_hflu.csv").write_text(HEADER + ROW, encoding="utf-8")
    (tmp_path / "bad_name_scer.csv").write_text(HEADER + ROW, encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger=logger.name):
        samples = discover_samples(tmp_path, "Session 2", logger)

    assert samples == []
    assert "invalid scer filename prefix" in caplog.text
    assert "Skipping unpaired sample 'A1'" in caplog.text


def test_load_sample_csvs_validates_required_columns(tmp_path: Path) -> None:
    (tmp_path / "A1_hflu.csv").write_text("XM,YM\n0,0\n", encoding="utf-8")
    (tmp_path / "A1_scer.csv").write_text(HEADER + ROW, encoding="utf-8")
    sample = Sample("A1", "Session 3", "A", 1, tmp_path / "A1_hflu.csv", tmp_path / "A1_scer.csv")

    with pytest.raises(ValueError):
        load_sample_csvs(sample)
