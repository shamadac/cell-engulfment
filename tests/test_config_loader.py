from __future__ import annotations

from pathlib import Path

import pytest

from config_loader import ConfigError, load_config


ROOT = Path(__file__).resolve().parents[1]


def test_valid_config_loads_fixture() -> None:
    config = load_config(ROOT / "tests" / "fixtures" / "test_config.yaml")
    assert len(config.sessions) == 1
    assert config.sessions[0].label == "Fixture Session"
    assert config.sessions[0].csv_dir.is_absolute()
    assert config.imagej_executable is None
    assert config.figures.violin_plot is True
    assert config.pipeline.backend is None
    assert config.pipeline.workers == 2
    assert config.pipeline.cache_dir == config.output_base_dir / "cache"
    assert config.segmentation is not None
    assert config.segmentation.scer.min_volume_um3 == config.size_filters.scer_min_um3
    assert config.segmentation.scer.threshold_scale == 1.0
    assert config.segmentation.hflu.max_volume_um3 == config.size_filters.hflu_max_um3
    assert config.engulfment.interior_margin_um == 0.0
    assert config.engulfment.require_shell_cavity_support is False


def test_explicit_v2_config_hydrates_aliases(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    nd2_dir = tmp_path / "nd2"
    nd2_dir.mkdir()
    config_path = tmp_path / "v2.yaml"
    config_path.write_text(
        "size_filters:\n"
        "  hflu_min_um3: 0.2\n"
        "  hflu_max_um3: 0.5\n"
        "  scer_min_um3: 40.0\n"
        "  scer_max_um3: 100.0\n"
        "sessions:\n"
        "  - label: v2\n"
        f'    csv_dir: "{session_dir.as_posix()}"\n'
        f'    nd2_dir: "{nd2_dir.as_posix()}"\n'
        "output_base_dir: output\n"
        "pipeline:\n"
        '  backend: "python_native_nd2"\n'
        "  workers: 3\n"
        '  stage_nd2_to_local: "always"\n'
        "segmentation:\n"
        "  scer:\n"
        '    channel: "Widefield Green"\n'
        "    gaussian_sigma_xyz: [1.0, 1.0, 1.0]\n"
        '    threshold_method: "otsu"\n'
        "    threshold_scale: 1.15\n"
        "    remove_border_objects: true\n"
        "    watershed: true\n"
        "    watershed_min_distance_um: 4.0\n"
        "  hflu:\n"
        "    channel: 1\n"
        "    gaussian_sigma_xyz: [0.5, 0.5, 0.5]\n"
        '    threshold_method: "triangle"\n'
        "    threshold_scale: 1.0\n"
        "    min_volume_um3: 0.3\n"
        "    max_volume_um3: 0.8\n"
        "    remove_border_objects: true\n"
        "    watershed: false\n"
        "    watershed_min_distance_um: 1.0\n"
        "engulfment:\n"
        '  method: "mask_containment"\n'
        "  min_inside_fraction: 0.97\n"
        "  interior_margin_um: 0.33\n"
        "  require_shell_cavity_support: true\n"
        "  shell_cavity_min_overlap_fraction: 0.6\n"
        "  shell_cavity_min_centroid_slice_fraction: 0.5\n"
        "  allow_shared_bacteria: false\n"
        "  save_assignment_details: true\n"
        "figures:\n"
        "  dpi: 300\n"
        '  format: "png"\n'
        "  violin_plot: false\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.pipeline.backend == "python_native_nd2"
    assert config.pipeline.workers == 3
    assert config.pipeline.stage_nd2_to_local == "always"
    assert config.segmentation is not None
    assert config.segmentation.scer.channel == "Widefield Green"
    assert config.segmentation.scer.threshold_scale == 1.15
    assert config.segmentation.scer.min_volume_um3 == config.size_filters.scer_min_um3
    assert config.segmentation.hflu.min_volume_um3 == 0.3
    assert config.engulfment.interior_margin_um == 0.33
    assert config.engulfment.method == "mask_containment"
    assert config.engulfment.require_shell_cavity_support is True


def test_missing_required_field_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "missing.yaml"
    config_path.write_text(
        "sessions:\n"
        "  - label: test\n"
        f"    csv_dir: {tmp_path.as_posix()}\n"
        "output_base_dir: output\n"
        "figures:\n"
        "  dpi: 300\n"
        '  format: "png"\n'
        "  violin_plot: false\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError):
        load_config(config_path)


def test_malformed_yaml_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "broken.yaml"
    config_path.write_text("size_filters: [\n", encoding="utf-8")

    with pytest.raises(ConfigError):
        load_config(config_path)


def test_missing_csv_dir_for_non_nd2_session_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_paths.yaml"
    config_path.write_text(
        "size_filters:\n"
        "  hflu_min_um3: 0.2\n"
        "  hflu_max_um3: 0.5\n"
        "  scer_min_um3: 40.0\n"
        "  scer_max_um3: 100.0\n"
        "sessions:\n"
        "  - label: bad\n"
        '    csv_dir: "missing_session"\n'
        "output_base_dir: output\n"
        "imagej_executable: null\n"
        "figures:\n"
        "  dpi: 300\n"
        '  format: "png"\n'
        "  violin_plot: false\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError):
        load_config(config_path)


def test_nd2_session_with_missing_csv_parent_raises(tmp_path: Path) -> None:
    nd2_dir = tmp_path / "nd2"
    nd2_dir.mkdir()
    config_path = tmp_path / "bad_nd2.yaml"
    config_path.write_text(
        "size_filters:\n"
        "  hflu_min_um3: 0.2\n"
        "  hflu_max_um3: 0.5\n"
        "  scer_min_um3: 40.0\n"
        "  scer_max_um3: 100.0\n"
        "sessions:\n"
        "  - label: nd2\n"
        '    csv_dir: "missing_parent/out"\n'
        f'    nd2_dir: "{nd2_dir.as_posix()}"\n'
        "output_base_dir: output\n"
        "imagej_executable: null\n"
        "figures:\n"
        "  dpi: 300\n"
        '  format: "png"\n'
        "  violin_plot: false\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError):
        load_config(config_path)
