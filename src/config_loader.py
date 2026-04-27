"""Configuration models and validation for command-line pipeline runs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


class ConfigError(Exception):
    """Raised when the pipeline configuration is missing or invalid."""


class SizeFilterConfig(BaseModel):
    """Volume thresholds applied to legacy CSV object tables."""

    hflu_min_um3: float = Field(ge=0)
    hflu_max_um3: float = Field(ge=0)
    scer_min_um3: float = Field(ge=0)
    scer_max_um3: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_bounds(self) -> "SizeFilterConfig":
        if self.hflu_min_um3 > self.hflu_max_um3:
            raise ValueError("hflu_min_um3 must be less than or equal to hflu_max_um3")
        if self.scer_min_um3 > self.scer_max_um3:
            raise ValueError("scer_min_um3 must be less than or equal to scer_max_um3")
        return self


class SessionConfig(BaseModel):
    """Input and output locations for one microscopy acquisition session."""

    label: str
    csv_dir: Path
    nd2_dir: Path | None = None

    @field_validator("label")
    @classmethod
    def validate_label(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("session label must not be empty")
        return value


class FigureConfig(BaseModel):
    """Settings shared by all output plots."""

    dpi: int = Field(default=300, ge=300)
    format: str = "png"
    violin_plot: bool = False

    @field_validator("format")
    @classmethod
    def normalize_format(cls, value: str) -> str:
        value = value.strip().lower()
        if not value:
            raise ValueError("figure format must not be empty")
        return value


class PipelineRuntimeConfig(BaseModel):
    """Runtime switches for backend selection, caching, and parallelism."""

    backend: Literal["legacy_csv", "legacy_fiji", "python_native_nd2"] | None = None
    workers: int = Field(default=2, ge=1)
    cache_dir: Path | None = None
    save_qc_overlays: bool = True
    save_label_stacks: bool = False
    stage_nd2_to_local: Literal["auto", "never", "always"] = "auto"


class ChannelSegmentationConfig(BaseModel):
    """Segmentation parameters for one fluorescence channel."""

    channel: int | str
    gaussian_sigma_xyz: tuple[float, float, float] = (0.8, 1.0, 1.0)
    threshold_method: Literal["otsu", "triangle", "yen", "li", "mean"] = "otsu"
    threshold_scale: float = Field(default=1.0, gt=0)
    min_volume_um3: float | None = Field(default=None, ge=0)
    max_volume_um3: float | None = Field(default=None, ge=0)
    remove_border_objects: bool = True
    watershed: bool = True
    watershed_min_distance_um: float = Field(default=2.0, ge=0)

    @field_validator("channel")
    @classmethod
    def validate_channel(cls, value: int | str) -> int | str:
        if isinstance(value, str) and not value.strip():
            raise ValueError("channel name must not be empty")
        return value

    @field_validator("gaussian_sigma_xyz")
    @classmethod
    def validate_sigma(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        if len(value) != 3:
            raise ValueError("gaussian_sigma_xyz must contain exactly 3 values")
        if any(item < 0 for item in value):
            raise ValueError("gaussian_sigma_xyz values must be non-negative")
        return value

    @model_validator(mode="after")
    def validate_volume_bounds(self) -> "ChannelSegmentationConfig":
        if self.min_volume_um3 is not None and self.max_volume_um3 is not None:
            if self.min_volume_um3 > self.max_volume_um3:
                raise ValueError("min_volume_um3 must be less than or equal to max_volume_um3")
        return self


class SegmentationConfig(BaseModel):
    """Pair of channel-specific segmentation configurations."""

    scer: ChannelSegmentationConfig
    hflu: ChannelSegmentationConfig


class EngulfmentConfig(BaseModel):
    """Parameters controlling bacterial containment classification."""

    method: Literal["centroid_sphere", "mask_containment"] = "centroid_sphere"
    min_inside_fraction: float = Field(default=0.95, ge=0.0, le=1.0)
    interior_margin_um: float = Field(default=0.0, ge=0.0)
    require_shell_cavity_support: bool = False
    shell_cavity_min_overlap_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    shell_cavity_min_centroid_slice_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    allow_shared_bacteria: bool = False
    save_assignment_details: bool = True


class PipelineConfig(BaseModel):
    """Top-level validated configuration loaded from YAML."""

    size_filters: SizeFilterConfig
    sessions: list[SessionConfig]
    output_base_dir: Path
    imagej_executable: Path | None = None
    figures: FigureConfig = FigureConfig()
    pipeline: PipelineRuntimeConfig = PipelineRuntimeConfig()
    segmentation: SegmentationConfig | None = None
    engulfment: EngulfmentConfig = EngulfmentConfig()

    @field_validator("sessions")
    @classmethod
    def validate_sessions(cls, value: list[SessionConfig]) -> list[SessionConfig]:
        if not value:
            raise ValueError("at least one session must be configured")
        return value

    @model_validator(mode="after")
    def hydrate_v2_defaults(self) -> "PipelineConfig":
        """Fill optional Python-native defaults from legacy size-filter settings."""
        if self.pipeline.cache_dir is None:
            self.pipeline.cache_dir = self.output_base_dir / "cache"

        if self.segmentation is None:
            self.segmentation = SegmentationConfig(
                scer=ChannelSegmentationConfig(
                    channel=0,
                    gaussian_sigma_xyz=(0.8, 1.0, 1.0),
                    threshold_method="otsu",
                    threshold_scale=1.0,
                    min_volume_um3=self.size_filters.scer_min_um3,
                    max_volume_um3=self.size_filters.scer_max_um3,
                    remove_border_objects=True,
                    watershed=True,
                    watershed_min_distance_um=4.0,
                ),
                hflu=ChannelSegmentationConfig(
                    channel=1,
                    gaussian_sigma_xyz=(0.5, 0.75, 0.75),
                    threshold_method="triangle",
                    threshold_scale=1.0,
                    min_volume_um3=self.size_filters.hflu_min_um3,
                    max_volume_um3=self.size_filters.hflu_max_um3,
                    remove_border_objects=True,
                    watershed=True,
                    watershed_min_distance_um=1.2,
                ),
            )
        else:
            if self.segmentation.hflu.min_volume_um3 is None:
                self.segmentation.hflu.min_volume_um3 = self.size_filters.hflu_min_um3
            if self.segmentation.hflu.max_volume_um3 is None:
                self.segmentation.hflu.max_volume_um3 = self.size_filters.hflu_max_um3
            if self.segmentation.scer.min_volume_um3 is None:
                self.segmentation.scer.min_volume_um3 = self.size_filters.scer_min_um3
            if self.segmentation.scer.max_volume_um3 is None:
                self.segmentation.scer.max_volume_um3 = self.size_filters.scer_max_um3

        return self


def _resolve_path(value: str | None, base_dir: Path) -> str | None:
    """Resolve relative config paths against the directory containing the YAML file."""
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def _prepare_raw_config(raw_config: dict, base_dir: Path) -> dict:
    """Normalize path-like strings before pydantic turns them into Path objects."""
    prepared = deepcopy(raw_config)
    prepared["output_base_dir"] = _resolve_path(prepared.get("output_base_dir"), base_dir)
    prepared["imagej_executable"] = _resolve_path(prepared.get("imagej_executable"), base_dir)

    pipeline = prepared.get("pipeline")
    if isinstance(pipeline, dict):
        pipeline["cache_dir"] = _resolve_path(pipeline.get("cache_dir"), base_dir)

    sessions = prepared.get("sessions", [])
    for session in sessions:
        session["csv_dir"] = _resolve_path(session.get("csv_dir"), base_dir)
        session["nd2_dir"] = _resolve_path(session.get("nd2_dir"), base_dir)

    return prepared


def _validate_paths(config: PipelineConfig) -> None:
    """Fail early when configured inputs or output parents are unavailable."""
    if config.imagej_executable is not None and not config.imagej_executable.exists():
        raise ConfigError(f"Configured ImageJ executable does not exist: {config.imagej_executable}")

    if config.pipeline.cache_dir is None:
        raise ConfigError("pipeline.cache_dir could not be resolved")

    for session in config.sessions:
        if session.nd2_dir is None:
            if not session.csv_dir.exists():
                raise ConfigError(
                    f"Session '{session.label}' csv_dir does not exist: {session.csv_dir}"
                )
        else:
            if not session.nd2_dir.exists():
                raise ConfigError(
                    f"Session '{session.label}' nd2_dir does not exist: {session.nd2_dir}"
                )
            if not session.csv_dir.parent.exists():
                raise ConfigError(
                    f"Session '{session.label}' csv_dir parent does not exist: {session.csv_dir.parent}"
                )


def load_config(config_path: str | Path) -> PipelineConfig:
    """Load, normalize, and validate a YAML configuration file."""
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = path.resolve()

    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            raw_config = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Malformed YAML in config file {path}: {exc}") from exc

    if not isinstance(raw_config, dict):
        raise ConfigError(f"Config file {path} must contain a top-level mapping")

    prepared_config = _prepare_raw_config(raw_config, path.parent)

    try:
        config = PipelineConfig.model_validate(prepared_config)
    except ValidationError as exc:
        raise ConfigError(f"Invalid config file {path}: {exc}") from exc

    _validate_paths(config)
    return config
