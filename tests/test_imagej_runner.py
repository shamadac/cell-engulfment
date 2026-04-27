from __future__ import annotations

import logging
from pathlib import Path

import pytest

from imagej_runner import MACRO_ARG_SEPARATOR, run_imagej_macro


class _CompletedProcess:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = True
    logger.setLevel(logging.INFO)
    return logger


def test_run_imagej_macro_invokes_subprocess(tmp_path: Path, monkeypatch) -> None:
    captured: list[list[str]] = []

    def fake_run(command, capture_output, text, check):
        captured.append(command)
        return _CompletedProcess(0, stdout="ok")

    monkeypatch.setattr("imagej_runner.subprocess.run", fake_run)

    nd2_dir = tmp_path / "nd2"
    csv_dir = tmp_path / "csv"
    macro_path = tmp_path / "macro.ijm"
    nd2_dir.mkdir()
    macro_path.write_text("", encoding="utf-8")

    success = run_imagej_macro("ImageJ.exe", nd2_dir, csv_dir, macro_path, _logger("imagej_success"))

    assert success is True
    assert captured
    assert captured[0][1] == "-batch"
    assert f"{nd2_dir}{MACRO_ARG_SEPARATOR}{csv_dir}" == captured[0][-1]
    assert (csv_dir / "processed_images").exists()


def test_run_imagej_macro_logs_failure(tmp_path: Path, monkeypatch, caplog: pytest.LogCaptureFixture) -> None:
    def fake_run(command, capture_output, text, check):
        return _CompletedProcess(1, stdout="out", stderr="err")

    monkeypatch.setattr("imagej_runner.subprocess.run", fake_run)
    logger = _logger("imagej_failure")

    with caplog.at_level(logging.ERROR, logger=logger.name):
        success = run_imagej_macro("ImageJ.exe", tmp_path / "nd2", tmp_path / "csv", tmp_path / "macro.ijm", logger)

    assert success is False
    assert "ImageJ macro failed" in caplog.text
