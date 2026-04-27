from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import yaml

from config_loader import load_config
from models import SampleResult
from data_loader import discover_samples
from logger import setup_logger
from pipeline import process_sample, run_pipeline


ROOT = Path(__file__).resolve().parents[1]
CSV_HEADER = "XM,YM,ZM,BX,BY,BZ,B-width,B-height,B-depth,Volume (micron^3)\n"
SCER_ROWS = "0,0,0,0,0,0,8,8,8,50\n100,100,100,0,0,0,8,8,8,50\n"
HFLU_ROWS = {
    0: "250,250,250,0,0,0,1,1,1,0.3\n300,300,300,0,0,0,1,1,1,0.3\n",
    50: "0,0,0,0,0,0,1,1,1,0.3\n250,250,250,0,0,0,1,1,1,0.3\n",
    100: "0,0,0,0,0,0,1,1,1,0.3\n100,100,100,0,0,0,1,1,1,0.3\n",
}


def _write_sample(session_dir: Path, prefix: str, rate: int) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / f"{prefix}_scer.csv").write_text(CSV_HEADER + SCER_ROWS, encoding="utf-8")
    (session_dir / f"{prefix}_hflu.csv").write_text(CSV_HEADER + HFLU_ROWS[rate], encoding="utf-8")


def test_process_sample_end_to_end(tmp_path: Path) -> None:
    config = load_config(ROOT / "tests" / "fixtures" / "test_config.yaml")
    logger = setup_logger(tmp_path / "run")
    session_dir = ROOT / "tests" / "fixtures" / "sample_session"
    sample = discover_samples(session_dir, "Fixture Session", logger)[0]

    result = process_sample(sample, config, logger)

    assert result.sample_name == "A1"
    assert result.hflu_count_after == 2
    assert result.scer_count_after == 2
    assert result.engulfment_rate == 50.0
    assert (tmp_path / "run" / "cell_sizes" / "A1_hflu_size_histogram.png").exists()


def test_run_pipeline_multiple_sessions(tmp_path: Path) -> None:
    session_one = tmp_path / "session_one"
    session_two = tmp_path / "session_two"
    _write_sample(session_one, "A1", 50)
    _write_sample(session_one, "B1", 0)
    _write_sample(session_two, "A2", 100)
    _write_sample(session_two, "B2", 50)

    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "size_filters": {
                    "hflu_min_um3": 0.2,
                    "hflu_max_um3": 0.5,
                    "scer_min_um3": 40.0,
                    "scer_max_um3": 100.0,
                },
                "sessions": [
                    {"label": "Session One", "csv_dir": str(session_one)},
                    {"label": "Session Two", "csv_dir": str(session_two)},
                ],
                "output_base_dir": str(tmp_path / "pipeline_output"),
                "imagej_executable": None,
                "figures": {"dpi": 300, "format": "png", "violin_plot": False},
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    results = run_pipeline(config)

    output_dirs = [path for path in (tmp_path / "pipeline_output").iterdir() if path.is_dir()]
    assert len(results) == 4
    assert len(output_dirs) == 1
    assert (output_dirs[0] / "summary.csv").exists()
    assert (output_dirs[0] / "qc_summary.csv").exists()
    assert (output_dirs[0] / "performance.csv").exists()
    assert (output_dirs[0] / "statistical_report.csv").exists()
    log_files = list(output_dirs[0].glob("process_log_*.txt"))
    assert len(log_files) == 1


def test_cli_smoke_test_uses_fixture_config() -> None:
    output_root = ROOT / "output"
    before = {path.name for path in output_root.iterdir() if path.is_dir()} if output_root.exists() else set()

    completed = subprocess.run(
        [sys.executable, "src/main.py", "--config", "tests/fixtures/test_config.yaml"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr

    after = {path.name for path in output_root.iterdir() if path.is_dir()}
    new_dirs = sorted(after - before)
    assert new_dirs
    latest_output = output_root / new_dirs[-1]
    assert (latest_output / "summary.csv").exists()
    assert (latest_output / "qc_summary.csv").exists()
    assert (latest_output / "performance.csv").exists()
    assert (latest_output / "statistical_report.csv").exists()


def test_run_pipeline_accepts_python_native_backend(tmp_path: Path, monkeypatch) -> None:
    session_dir = tmp_path / "session_csv"
    session_dir.mkdir()
    nd2_dir = tmp_path / "nd2"
    nd2_dir.mkdir()
    (nd2_dir / "A1.nd2").write_text("", encoding="utf-8")
    config_path = tmp_path / "v2.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "size_filters": {
                    "hflu_min_um3": 0.2,
                    "hflu_max_um3": 0.5,
                    "scer_min_um3": 40.0,
                    "scer_max_um3": 100.0,
                },
                "sessions": [
                    {
                        "label": "Session One",
                        "csv_dir": str(session_dir),
                        "nd2_dir": str(nd2_dir),
                    }
                ],
                "output_base_dir": str(tmp_path / "pipeline_output"),
                "pipeline": {
                    "backend": "python_native_nd2",
                    "workers": 1,
                },
                "figures": {"dpi": 300, "format": "png", "violin_plot": False},
            }
        ),
        encoding="utf-8",
    )

    fake_result = SampleResult(
        sample_name="A1",
        session_label="Session One",
        biological_replicate="A",
        technical_replicate=1,
        hflu_count_before=3,
        hflu_count_after=2,
        scer_count_before=2,
        scer_count_after=1,
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
        engulfment_rate=100.0,
        backend="python_native_nd2",
        runtime_seconds=1.0,
        from_cache=False,
        staged_local=False,
        review_required=False,
        ambiguous_bacteria_count=0,
        hflu_rejected_count=1,
        scer_rejected_count=1,
    )

    monkeypatch.setattr("pipeline.process_nd2_session", lambda session, config, output_dir, logger: [fake_result])

    config = load_config(config_path)
    results = run_pipeline(config)

    assert len(results) == 1
    output_dirs = [path for path in (tmp_path / "pipeline_output").iterdir() if path.is_dir()]
    assert len(output_dirs) == 1
    assert (output_dirs[0] / "summary.csv").exists()
    assert (output_dirs[0] / "qc_summary.csv").exists()
    assert (output_dirs[0] / "performance.csv").exists()
