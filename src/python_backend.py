"""Python-native ND2 ingestion, segmentation, caching, and sample execution."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_core import build_sample_result
from config_loader import PipelineConfig, SessionConfig
from data_loader import SAMPLE_PREFIX_RE, Sample
from engulfment import classify_mask_containment, validate_shell_cavity_support
from models import SampleResult
from segmentation import segment_hflu_stack, segment_scer_stack

try:
    import nd2
except Exception as exc:  # pragma: no cover - dependency exercised in integration
    nd2 = None
    _ND2_IMPORT_ERROR = exc
else:
    _ND2_IMPORT_ERROR = None


PYTHON_BACKEND_VERSION = "python_native_nd2_v2.6"


def require_nd2() -> None:
    """Raise a clear error if Nikon ND2 support is unavailable."""
    if _ND2_IMPORT_ERROR is not None:
        raise RuntimeError("The nd2 package is required for the python_native_nd2 backend") from _ND2_IMPORT_ERROR


def _safe_slug(value: str) -> str:
    """Convert a session label into a filesystem-safe cache directory name."""
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")


def _json_default(value: Any) -> Any:
    """Serialize Path objects when hashing or storing config dictionaries."""
    if isinstance(value, Path):
        return str(value)
    return value


def _file_signature(path: Path) -> dict[str, Any]:
    """Capture the file fields that invalidate per-sample caches."""
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _is_onedrive_path(path: Path) -> bool:
    """Detect cloud-synced paths that benefit from local staging before reading."""
    return any("onedrive" in part.lower() for part in path.parts)


def _stage_nd2_file(source_path: Path, cache_dir: Path, session_label: str, mode: str) -> tuple[Path, bool]:
    """Optionally copy an ND2 file into the cache before processing."""
    if mode == "never":
        return source_path, False
    if mode == "auto" and not _is_onedrive_path(source_path):
        return source_path, False

    staged_dir = cache_dir / "staged_nd2" / _safe_slug(session_label)
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged_path = staged_dir / source_path.name

    source_sig = _file_signature(source_path)
    if staged_path.exists():
        staged_sig = _file_signature(staged_path)
        if staged_sig["size"] == source_sig["size"] and staged_sig["mtime_ns"] == source_sig["mtime_ns"]:
            return staged_path, True

    shutil.copy2(source_path, staged_path)
    return staged_path, True


def _discover_nd2_files(session: SessionConfig) -> list[tuple[str, Path]]:
    """Return valid sample-prefix/ND2-path pairs for a raw-image session."""
    if session.nd2_dir is None:
        return []
    pairs: list[tuple[str, Path]] = []
    for path in sorted(session.nd2_dir.glob("*.nd2")):
        prefix = path.stem
        if SAMPLE_PREFIX_RE.match(prefix):
            pairs.append((prefix, path))
    return pairs


def _resolve_channel_index(ndfile: Any, channel_spec: int | str) -> int:
    """Resolve a channel by zero-based index or by ND2 channel name."""
    channel_count = int(getattr(ndfile, "sizes", {}).get("C", 1))
    if isinstance(channel_spec, int):
        if channel_spec < 0 or channel_spec >= channel_count:
            raise ValueError(f"Channel index {channel_spec} is out of range for {channel_count} channels")
        return channel_spec

    channel_name = channel_spec.strip().lower()
    metadata = getattr(ndfile, "metadata", None)
    channels = getattr(metadata, "channels", []) if metadata is not None else []
    discovered_names: list[str] = []
    for index, channel in enumerate(channels):
        meta = getattr(channel, "channel", channel)
        name = getattr(meta, "name", None)
        if name is None:
            discovered_names.append("")
            continue
        discovered_names.append(str(name))
        if str(name).strip().lower() == channel_name:
            return index

    if channel_spec.isdigit():
        return _resolve_channel_index(ndfile, int(channel_spec))

    raise ValueError(
        f"Could not resolve channel '{channel_spec}'. Available names: {', '.join(name or '<unnamed>' for name in discovered_names)}"
    )


def _extract_channel_stack(data: np.ndarray, axis_names: list[str], channel_index: int) -> np.ndarray:
    """Extract one channel and normalize it to a float32 Z/Y/X stack."""
    indexer: list[int | slice] = []
    kept_axes: list[str] = []
    for axis_name, axis_size in zip(axis_names, data.shape, strict=False):
        if axis_name == "C":
            indexer.append(channel_index)
        elif axis_name in {"Z", "Y", "X"}:
            indexer.append(slice(None))
            kept_axes.append(axis_name)
        elif axis_size == 1:
            indexer.append(0)
        else:
            indexer.append(0)

    selected = np.asarray(data[tuple(indexer)])
    if selected.ndim == 2:
        selected = selected[np.newaxis, :, :]
        kept_axes = ["Z", "Y", "X"]
    elif "Z" not in kept_axes:
        selected = selected[np.newaxis, ...]
        kept_axes = ["Z", *kept_axes]

    transpose_order = [kept_axes.index(axis_name) for axis_name in ("Z", "Y", "X")]
    return np.transpose(selected, axes=transpose_order).astype(np.float32, copy=False)


def _read_nd2_channels(
    nd2_path: Path,
    scer_channel: int | str,
    hflu_channel: int | str,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Read configured scer/hflu channels and physical voxel spacing from an ND2 file."""
    require_nd2()
    with nd2.ND2File(nd2_path) as ndfile:
        data = ndfile.asarray()
        axis_names = list(getattr(ndfile, "sizes", {}).keys())
        scer_index = _resolve_channel_index(ndfile, scer_channel)
        hflu_index = _resolve_channel_index(ndfile, hflu_channel)
        scer_stack = _extract_channel_stack(data, axis_names, scer_index)
        hflu_stack = _extract_channel_stack(data, axis_names, hflu_index)
        voxel_size = ndfile.voxel_size()
        spacing_zyx = (
            float(getattr(voxel_size, "z", 1.0) or 1.0),
            float(getattr(voxel_size, "y", 1.0) or 1.0),
            float(getattr(voxel_size, "x", 1.0) or 1.0),
        )
    return scer_stack, hflu_stack, spacing_zyx


def _cache_manifest_path(sample_cache_dir: Path) -> Path:
    """Return the cache manifest path for one sample."""
    return sample_cache_dir / "manifest.json"


def _measurement_cache_paths(sample_cache_dir: Path) -> tuple[Path, Path, Path]:
    """Return the cached measurement and label-stack paths for one sample."""
    return (
        sample_cache_dir / "hflu_measurements.csv",
        sample_cache_dir / "scer_measurements.csv",
        sample_cache_dir / "labels.npz",
    )


def _load_cache(sample_cache_dir: Path, source_sig: dict[str, Any], config_hash: str) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]] | None:
    """Load cached segmentation outputs when input, config, and backend match."""
    manifest_path = _cache_manifest_path(sample_cache_dir)
    hflu_path, scer_path, labels_path = _measurement_cache_paths(sample_cache_dir)
    if not manifest_path.exists() or not hflu_path.exists() or not scer_path.exists() or not labels_path.exists():
        return None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("backend_version") != PYTHON_BACKEND_VERSION:
        return None
    if manifest.get("config_hash") != config_hash:
        return None
    if manifest.get("source_signature") != source_sig:
        return None

    arrays = dict(np.load(labels_path, allow_pickle=False))
    return manifest, pd.read_csv(hflu_path), pd.read_csv(scer_path), arrays


def _write_cache(
    sample_cache_dir: Path,
    *,
    source_sig: dict[str, Any],
    config_hash: str,
    hflu_df: pd.DataFrame,
    scer_df: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    timings: dict[str, float],
) -> None:
    """Persist per-sample segmentation outputs for deterministic resumability."""
    sample_cache_dir.mkdir(parents=True, exist_ok=True)
    hflu_path, scer_path, labels_path = _measurement_cache_paths(sample_cache_dir)
    hflu_df.to_csv(hflu_path, index=False)
    scer_df.to_csv(scer_path, index=False)
    np.savez_compressed(labels_path, **arrays)
    manifest = {
        "backend_version": PYTHON_BACKEND_VERSION,
        "config_hash": config_hash,
        "source_signature": source_sig,
        "timings": timings,
    }
    _cache_manifest_path(sample_cache_dir).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _normalize_projection(image: np.ndarray) -> np.ndarray:
    """Scale a projection into 0..1 for diagnostic image output."""
    image = np.asarray(image, dtype=float)
    if image.size == 0:
        return image
    max_value = float(np.nanmax(image))
    if max_value <= 0:
        return np.zeros_like(image, dtype=float)
    return image / max_value


def _save_qc_panel(
    qc_dir: Path,
    sample_prefix: str,
    *,
    scer_raw_max: np.ndarray,
    hflu_raw_max: np.ndarray,
    scer_mask_max: np.ndarray,
    hflu_mask_max: np.ndarray,
    engulfed_mask_max: np.ndarray,
) -> None:
    """Write max projections and a compact overlay panel for visual QC."""
    qc_dir.mkdir(parents=True, exist_ok=True)

    scer_norm = _normalize_projection(scer_raw_max)
    hflu_norm = _normalize_projection(hflu_raw_max)
    scer_mask = scer_mask_max.astype(float)
    hflu_mask = hflu_mask_max.astype(float)
    engulfed_mask = engulfed_mask_max.astype(float)

    plt.imsave(qc_dir / f"{sample_prefix}_scer_max.png", scer_norm, cmap="Blues")
    plt.imsave(qc_dir / f"{sample_prefix}_hflu_max.png", hflu_norm, cmap="Reds")

    overlay_rgb = np.dstack(
        [
            np.clip(hflu_norm + (engulfed_mask * 0.5), 0.0, 1.0),
            np.clip(engulfed_mask, 0.0, 1.0),
            np.clip(scer_norm, 0.0, 1.0),
        ]
    )

    figure, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    axes[0].imshow(scer_norm, cmap="Blues")
    axes[0].set_title("scer raw max")
    axes[1].imshow(scer_mask, cmap="gray")
    axes[1].set_title("scer filled mask")
    axes[2].imshow(hflu_mask, cmap="gray")
    axes[2].set_title("hflu mask")
    axes[3].imshow(overlay_rgb)
    axes[3].set_title("engulfment overlay")
    for axis in axes:
        axis.axis("off")
    figure.savefig(qc_dir / f"{sample_prefix}_overlay.png", dpi=200)
    plt.close(figure)


def _assignment_details_path(output_dir: Path, sample_prefix: str) -> Path:
    """Return the per-sample containment assignment CSV path."""
    assignment_dir = output_dir / "per_sample"
    assignment_dir.mkdir(parents=True, exist_ok=True)
    return assignment_dir / f"assignment_{sample_prefix}.csv"


def _build_task_config(config: PipelineConfig) -> dict[str, Any]:
    """Extract the config fields that affect per-sample ND2 processing."""
    return {
        "pipeline": config.pipeline.model_dump(mode="json"),
        "segmentation": config.segmentation.model_dump(mode="json") if config.segmentation is not None else None,
        "engulfment": config.engulfment.model_dump(mode="json"),
    }


def _config_hash(task_config: dict[str, Any]) -> str:
    """Hash processing settings so caches invalidate when analysis parameters change."""
    payload = json.dumps(task_config, default=_json_default, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _review_required(scer_df: pd.DataFrame, hflu_df: pd.DataFrame, ambiguous_bacteria_count: int) -> bool:
    """Flag samples whose segmentation or assignment metrics deserve manual review."""
    scer_before = max(1, len(scer_df))
    hflu_before = max(1, len(hflu_df))
    scer_merged = int(scer_df["reject_reason"].fillna("").str.contains("merged_shape").sum()) if "reject_reason" in scer_df else 0
    border_rejects = 0
    if "reject_reason" in scer_df:
        border_rejects += int(scer_df["reject_reason"].fillna("").str.contains("border_touch").sum())
    if "reject_reason" in hflu_df:
        border_rejects += int(hflu_df["reject_reason"].fillna("").str.contains("border_touch").sum())
    return (
        ambiguous_bacteria_count > 0
        or (scer_merged / scer_before) > 0.15
        or (border_rejects / (scer_before + hflu_before)) > 0.2
    )


def _process_nd2_sample(task: dict[str, Any]) -> SampleResult:
    """Process one ND2 file from raw stacks through final sample-level metrics."""
    sample_start = time.perf_counter()
    session_label = str(task["session_label"])
    sample_prefix = str(task["sample_prefix"])
    nd2_path = Path(task["nd2_path"])
    session_csv_dir = Path(task["session_csv_dir"])
    cache_dir = Path(task["cache_dir"])
    output_dir = Path(task["output_dir"])
    task_config = dict(task["config"])

    source_sig = _file_signature(nd2_path)
    config_hash = _config_hash(task_config)
    sample_cache_dir = cache_dir / _safe_slug(session_label) / sample_prefix

    # Staging is tracked separately from cache hits: a sample may be staged even
    # when segmentation is recomputed because the config or source changed.
    staged_path, staged_local = _stage_nd2_file(
        nd2_path,
        cache_dir,
        session_label,
        str(task_config["pipeline"]["stage_nd2_to_local"]),
    )

    arrays: dict[str, np.ndarray]
    timings: dict[str, float] = {"stage_seconds": time.perf_counter() - sample_start}
    cached = _load_cache(sample_cache_dir, source_sig, config_hash)
    if cached is not None:
        _, hflu_df, scer_df, arrays = cached
        from_cache = True
    else:
        segmentation_start = time.perf_counter()
        # Raw channels are processed independently so channel-specific thresholds
        # and size filters can be tuned without changing the output schema.
        scer_stack, hflu_stack, spacing_zyx = _read_nd2_channels(
            staged_path,
            task_config["segmentation"]["scer"]["channel"],
            task_config["segmentation"]["hflu"]["channel"],
        )
        scer_artifacts = segment_scer_stack(
            scer_stack,
            spacing_zyx,
            SimpleNamespace(**task_config["segmentation"]["scer"]),
        )
        hflu_artifacts = segment_hflu_stack(
            hflu_stack,
            spacing_zyx,
            SimpleNamespace(**task_config["segmentation"]["hflu"]),
        )
        hflu_df = hflu_artifacts.measurements
        scer_df = scer_artifacts.measurements
        arrays = {
            "hflu_labels": hflu_artifacts.labels.astype(np.int32),
            "scer_labels": scer_artifacts.labels.astype(np.int32),
            "scer_support_mask": (
                scer_artifacts.support_mask.astype(np.uint8)
                if scer_artifacts.support_mask is not None
                else np.zeros_like(scer_artifacts.labels, dtype=np.uint8)
            ),
            "hflu_raw_max": hflu_artifacts.raw_stack.max(axis=0).astype(np.float32),
            "scer_raw_max": scer_artifacts.raw_stack.max(axis=0).astype(np.float32),
            "hflu_mask_max": hflu_artifacts.binary_mask.max(axis=0).astype(np.uint8),
            "scer_mask_max": scer_artifacts.binary_mask.max(axis=0).astype(np.uint8),
            "spacing_zyx": np.asarray(spacing_zyx, dtype=np.float32),
        }
        timings["segmentation_seconds"] = time.perf_counter() - segmentation_start
        _write_cache(
            sample_cache_dir,
            source_sig=source_sig,
            config_hash=config_hash,
            hflu_df=hflu_df,
            scer_df=scer_df,
            arrays=arrays,
            timings=timings,
        )
        from_cache = False

    session_csv_dir.mkdir(parents=True, exist_ok=True)
    hflu_csv_path = session_csv_dir / f"{sample_prefix}_hflu.csv"
    scer_csv_path = session_csv_dir / f"{sample_prefix}_scer.csv"
    # The ND2 backend emits ImageJ-compatible object CSVs so downstream analysis
    # can use the same summary writers as legacy CSV sessions.
    hflu_df.to_csv(hflu_csv_path, index=False)
    scer_df.to_csv(scer_csv_path, index=False)

    hflu_filtered = hflu_df[hflu_df["reject_reason"].fillna("") == ""].copy() if "reject_reason" in hflu_df else hflu_df.copy()
    scer_filtered = scer_df[scer_df["reject_reason"].fillna("") == ""].copy() if "reject_reason" in scer_df else scer_df.copy()
    spacing_zyx = tuple(float(value) for value in np.asarray(arrays.get("spacing_zyx", (1.0, 1.0, 1.0))).tolist())
    engulfment_result = classify_mask_containment(
        hflu_filtered,
        scer_filtered,
        arrays["hflu_labels"],
        arrays["scer_labels"],
        min_inside_fraction=float(task_config["engulfment"]["min_inside_fraction"]),
        interior_margin_um=float(task_config["engulfment"].get("interior_margin_um", 0.0)),
        spacing_zyx=spacing_zyx,
        allow_shared_bacteria=bool(task_config["engulfment"]["allow_shared_bacteria"]),
    )
    if bool(task_config["engulfment"].get("require_shell_cavity_support", False)):
        # Shell-cavity support is an optional second pass for channels where the
        # larger object is represented by a membrane or wall stain.
        engulfment_result = validate_shell_cavity_support(
            engulfment_result,
            arrays["hflu_labels"],
            arrays["scer_labels"],
            arrays.get("scer_support_mask"),
            scer_filtered.reset_index(drop=True),
            min_overlap_fraction=float(task_config["engulfment"].get("shell_cavity_min_overlap_fraction", 0.5)),
            min_centroid_slice_fraction=float(
                task_config["engulfment"].get("shell_cavity_min_centroid_slice_fraction", 0.5)
            ),
        )

    if bool(task_config["engulfment"]["save_assignment_details"]):
        assignment_df = pd.DataFrame(engulfment_result.assignment_rows)
        assignment_df.to_csv(_assignment_details_path(output_dir, sample_prefix), index=False)

    if bool(task_config["pipeline"]["save_qc_overlays"]):
        # The overlay marks yeast labels that passed the containment classifier,
        # not every detected object in the field of view.
        engulfed_mask = np.isin(
            arrays["scer_labels"],
            scer_filtered.iloc[engulfment_result.engulfed_yeast_indices]["label_id"].to_numpy(dtype=int)
            if engulfment_result.engulfed_yeast_indices
            else np.array([], dtype=int),
        )
        _save_qc_panel(
            output_dir / "qc",
            sample_prefix,
            scer_raw_max=arrays["scer_raw_max"],
            hflu_raw_max=arrays["hflu_raw_max"],
            scer_mask_max=arrays["scer_mask_max"],
            hflu_mask_max=arrays["hflu_mask_max"],
            engulfed_mask_max=engulfed_mask.max(axis=0).astype(np.uint8),
        )

    if bool(task_config["pipeline"]["save_label_stacks"]):
        label_dir = output_dir / "qc" / "labels"
        label_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            label_dir / f"{sample_prefix}_labels.npz",
            hflu_labels=arrays["hflu_labels"],
            scer_labels=arrays["scer_labels"],
        )

    sample = Sample(
        prefix=sample_prefix,
        session_label=session_label,
        biological_replicate=sample_prefix[0],
        technical_replicate=int(sample_prefix[1:]),
        hflu_path=hflu_csv_path,
        scer_path=scer_csv_path,
    )
    return build_sample_result(
        sample=sample,
        hflu_before=hflu_df,
        scer_before=scer_df,
        hflu_after=hflu_filtered,
        scer_after=scer_filtered,
        engulfment_result=engulfment_result,
        backend="python_native_nd2",
        runtime_seconds=time.perf_counter() - sample_start,
        from_cache=from_cache,
        staged_local=staged_local,
        review_required=_review_required(scer_df, hflu_df, engulfment_result.ambiguous_bacteria_count),
        ambiguous_bacteria_count=engulfment_result.ambiguous_bacteria_count,
    )


def process_nd2_session(
    session: SessionConfig,
    config: PipelineConfig,
    output_dir: Path,
    logger,
) -> list[SampleResult]:
    """Process all valid ND2 files in a session with sample-level parallelism."""
    nd2_files = _discover_nd2_files(session)
    if not nd2_files:
        logger.warning("No ND2 files discovered for session '%s' in %s", session.label, session.nd2_dir)
        return []

    cache_dir = Path(config.pipeline.cache_dir or (config.output_base_dir / "cache"))
    task_config = _build_task_config(config)
    tasks = [
        {
            "session_label": session.label,
            "sample_prefix": prefix,
            "nd2_path": path,
            "session_csv_dir": str(session.csv_dir),
            "cache_dir": str(cache_dir),
            "output_dir": str(output_dir),
            "config": task_config,
        }
        for prefix, path in nd2_files
    ]

    # Parallelism is intentionally at the sample level; individual stack
    # segmentation remains single-process to limit memory pressure per worker.
    max_workers = min(int(config.pipeline.workers), max(1, len(tasks)))
    results: list[SampleResult] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_process_nd2_sample, task): task for task in tasks}
        for future in as_completed(future_map):
            task = future_map[future]
            sample_prefix = str(task["sample_prefix"])
            try:
                result = future.result()
            except Exception:
                logger.exception(
                    "Failed to process ND2 sample '%s' for session '%s'",
                    sample_prefix,
                    session.label,
                )
                continue

            logger.info(
                "Processed ND2 sample %s: engulfing_yeast_count=%s, engulfment_rate=%.3f, from_cache=%s",
                result.sample_name,
                result.engulfing_yeast_count,
                result.engulfment_rate,
                result.from_cache,
            )
            results.append(result)

    return sorted(results, key=lambda item: (item.biological_replicate, item.technical_replicate))
