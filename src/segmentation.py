"""3D segmentation and measurement routines for fluorescence image stacks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

try:
    from skimage import feature, filters, measure
    from skimage.segmentation import watershed
except Exception as exc:  # pragma: no cover - dependency exercised in integration
    feature = None
    filters = None
    measure = None
    watershed = None
    _SKIMAGE_IMPORT_ERROR = exc
else:
    _SKIMAGE_IMPORT_ERROR = None


def require_skimage() -> None:
    """Raise a clear error if the optional image-processing dependency is missing."""
    if _SKIMAGE_IMPORT_ERROR is not None:
        raise RuntimeError(
            "scikit-image is required for the python_native_nd2 backend"
        ) from _SKIMAGE_IMPORT_ERROR


@dataclass(frozen=True)
class ChannelArtifacts:
    """Intermediate arrays and measurements produced for one fluorescence channel."""

    raw_stack: np.ndarray
    preprocessed_stack: np.ndarray
    binary_mask: np.ndarray
    labels: np.ndarray
    measurements: pd.DataFrame
    support_mask: np.ndarray | None = None


def _compute_threshold(values: np.ndarray, method: str) -> float:
    """Compute an intensity threshold using one of the supported global methods."""
    require_skimage()
    finite = np.asarray(values[np.isfinite(values)], dtype=float)
    positive = finite[finite > 0]
    # Ignore zero-valued background when possible so sparse channels do not bias
    # automatic thresholds toward the lower edge of the camera range.
    if positive.size > 0:
        finite = positive
    if finite.size == 0:
        return 0.0

    method = method.lower()
    if method == "otsu":
        return float(filters.threshold_otsu(finite))
    if method == "triangle":
        return float(filters.threshold_triangle(finite))
    if method == "yen":
        return float(filters.threshold_yen(finite))
    if method == "li":
        return float(filters.threshold_li(finite))
    if method == "mean":
        return float(finite.mean())
    raise ValueError(f"Unsupported threshold method: {method}")


def _scaled_threshold(values: np.ndarray, method: str, scale: float) -> float:
    """Apply a user-configurable multiplier to an automatic threshold."""
    return _compute_threshold(values, method) * float(scale)


def _resolve_spacing(spacing_zyx: tuple[float, float, float]) -> tuple[float, float, float]:
    """Replace missing or invalid voxel spacing with 1.0 um fallbacks."""
    z, y, x = spacing_zyx
    return (
        float(z if z and z > 0 else 1.0),
        float(y if y and y > 0 else 1.0),
        float(x if x and x > 0 else 1.0),
    )


def _peak_min_distance_px(spacing_zyx: tuple[float, float, float], min_distance_um: float) -> int:
    """Convert a physical watershed seed distance into a conservative pixel radius."""
    min_spacing = min(_resolve_spacing(spacing_zyx))
    return max(1, int(round(min_distance_um / min_spacing)))


def _needs_watershed(labels: np.ndarray, voxel_volume_um3: float, max_volume_um3: float | None) -> bool:
    """Skip watershed unless at least one component is large enough to be merged."""
    if max_volume_um3 is None:
        return True
    if labels.max() == 0:
        return False
    size_threshold = max(8, int(round((max_volume_um3 / voxel_volume_um3) * 1.5)))
    counts = np.bincount(labels.ravel())
    if len(counts) <= 1:
        return False
    return bool(np.any(counts[1:] > size_threshold))


def _split_components(
    binary_mask: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    *,
    min_distance_um: float,
    enable_watershed: bool,
    max_volume_um3: float | None = None,
) -> np.ndarray:
    """Label binary components and optionally split oversized components by watershed."""
    require_skimage()
    initial_labels = measure.label(binary_mask.astype(np.uint8), connectivity=1)
    if initial_labels.max() == 0:
        return initial_labels.astype(np.int32)

    voxel_volume_um3 = float(np.prod(_resolve_spacing(spacing_zyx)))
    if not enable_watershed or not _needs_watershed(initial_labels, voxel_volume_um3, max_volume_um3):
        return initial_labels.astype(np.int32)

    # Physical voxel spacing keeps watershed splitting comparable across image
    # stacks with different pixel sizes or z-step intervals.
    distance = ndi.distance_transform_edt(binary_mask, sampling=spacing_zyx)
    coords = feature.peak_local_max(
        distance,
        labels=binary_mask.astype(np.uint8),
        min_distance=_peak_min_distance_px(spacing_zyx, min_distance_um),
        exclude_border=False,
    )
    if len(coords) == 0:
        return initial_labels.astype(np.int32)

    markers = np.zeros(binary_mask.shape, dtype=np.int32)
    markers[tuple(coords.T)] = np.arange(1, len(coords) + 1, dtype=np.int32)
    return watershed(-distance, markers, mask=binary_mask).astype(np.int32)


def _measure_labels(
    labels: np.ndarray,
    intensity_stack: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    cell_type: str,
    min_volume_um3: float,
    max_volume_um3: float,
    remove_border_objects: bool,
) -> pd.DataFrame:
    """Measure labeled objects and attach rule-based quality-control flags."""
    require_skimage()
    spacing_zyx = _resolve_spacing(spacing_zyx)
    voxel_volume_um3 = float(np.prod(spacing_zyx))
    rows: list[dict[str, float | int | bool | str]] = []

    for region in measure.regionprops(labels, intensity_image=intensity_stack):
        min_z, min_y, min_x, max_z, max_y, max_x = region.bbox
        depth_um = (max_z - min_z) * spacing_zyx[0]
        height_um = (max_y - min_y) * spacing_zyx[1]
        width_um = (max_x - min_x) * spacing_zyx[2]
        bbox_volume_um3 = max(depth_um * height_um * width_um, voxel_volume_um3)
        aspect_values = [value for value in (width_um, height_um, depth_um) if value > 0]
        max_aspect_ratio = (max(aspect_values) / min(aspect_values)) if len(aspect_values) >= 2 else 1.0
        z_border_touch = min_z == 0 or max_z == labels.shape[0]
        border_touch = (
            min_y == 0
            or min_x == 0
            or max_y == labels.shape[1]
            or max_x == labels.shape[2]
        )

        # region.area is in voxels; converting with voxel volume keeps size
        # filters independent of microscope magnification or z-step.
        volume_um3 = float(region.area * voxel_volume_um3)
        occupancy = float(volume_um3 / bbox_volume_um3) if bbox_volume_um3 > 0 else 0.0
        reject_reasons: list[str] = []
        if volume_um3 < min_volume_um3 or volume_um3 > max_volume_um3:
            reject_reasons.append("volume_out_of_range")
        if remove_border_objects and border_touch:
            reject_reasons.append("border_touch")
        if cell_type == "scer":
            if occupancy < 0.18 or (max_aspect_ratio > 3.0 and occupancy < 0.55):
                reject_reasons.append("merged_shape")
        else:
            if volume_um3 <= min_volume_um3 * 1.25 and max_aspect_ratio < 1.5:
                reject_reasons.append("speckle_like")

        centroid_z, centroid_y, centroid_x = region.centroid
        rows.append(
            {
                "label_id": int(region.label),
                "XM": float(centroid_x * spacing_zyx[2]),
                "YM": float(centroid_y * spacing_zyx[1]),
                "ZM": float(centroid_z * spacing_zyx[0]),
                "BX": float(min_x * spacing_zyx[2]),
                "BY": float(min_y * spacing_zyx[1]),
                "BZ": float(min_z * spacing_zyx[0]),
                "B-width": float(width_um),
                "B-height": float(height_um),
                "B-depth": float(depth_um),
                "Volume (micron^3)": volume_um3,
                "voxel_centroid_z": float(centroid_z),
                "voxel_centroid_y": float(centroid_y),
                "voxel_centroid_x": float(centroid_x),
                "bbox_volume_um3": float(bbox_volume_um3),
                "occupancy": occupancy,
                "max_aspect_ratio": float(max_aspect_ratio),
                "border_touch": bool(border_touch),
                "z_border_touch": bool(z_border_touch),
                "reject_reason": "|".join(sorted(set(reject_reasons))),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "label_id",
                "XM",
                "YM",
                "ZM",
                "BX",
                "BY",
                "BZ",
                "B-width",
                "B-height",
                "B-depth",
                "Volume (micron^3)",
                "voxel_centroid_z",
                "voxel_centroid_y",
                "voxel_centroid_x",
                "bbox_volume_um3",
                "occupancy",
                "max_aspect_ratio",
                "border_touch",
                "z_border_touch",
                "reject_reason",
            ]
        )
    return df.sort_values(["ZM", "YM", "XM"]).reset_index(drop=True)


def segment_scer_stack(
    stack: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    config,
) -> ChannelArtifacts:
    """Segment the scer channel as a shell-like signal with filled interiors."""
    require_skimage()
    stack = stack.astype(np.float32)
    background = ndi.gaussian_filter(
        stack,
        sigma=tuple(max(value * 4, 2.0) for value in config.gaussian_sigma_xyz),
    )
    enhanced = np.clip(stack - background, a_min=0.0, a_max=None)
    smoothed = ndi.gaussian_filter(enhanced, sigma=config.gaussian_sigma_xyz)
    binary_slices = np.zeros_like(smoothed, dtype=bool)
    support_slices = np.zeros_like(smoothed, dtype=bool)
    threshold_scale = float(getattr(config, "threshold_scale", 1.0))
    for z_index in range(smoothed.shape[0]):
        slice_data = smoothed[z_index]
        threshold = _scaled_threshold(slice_data, config.threshold_method, threshold_scale)
        mask = slice_data > threshold
        mask = ndi.binary_closing(mask, structure=np.ones((3, 3), dtype=bool))
        support_slices[z_index] = mask
        # The support mask preserves the wall/shell signal; the filled mask is
        # used as the candidate yeast-associated volume for containment tests.
        binary_slices[z_index] = ndi.binary_fill_holes(mask)

    binary_slices = ndi.binary_opening(binary_slices, structure=np.ones((1, 3, 3), dtype=bool))
    labels = _split_components(
        binary_slices,
        spacing_zyx,
        min_distance_um=config.watershed_min_distance_um,
        enable_watershed=config.watershed,
        max_volume_um3=config.max_volume_um3,
    )
    measurements = _measure_labels(
        labels,
        smoothed,
        spacing_zyx,
        "scer",
        float(config.min_volume_um3),
        float(config.max_volume_um3),
        config.remove_border_objects,
    )
    return ChannelArtifacts(
        raw_stack=stack,
        preprocessed_stack=smoothed,
        binary_mask=binary_slices,
        labels=labels,
        measurements=measurements,
        support_mask=support_slices,
    )


def segment_hflu_stack(
    stack: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    config,
) -> ChannelArtifacts:
    """Segment the hflu channel as discrete fluorescent bacterial objects."""
    require_skimage()
    stack = stack.astype(np.float32)
    background = ndi.gaussian_filter(stack, sigma=tuple(max(value * 3, 1.0) for value in config.gaussian_sigma_xyz))
    enhanced = np.clip(stack - background, a_min=0.0, a_max=None)
    smoothed = ndi.gaussian_filter(enhanced, sigma=config.gaussian_sigma_xyz)
    threshold = _scaled_threshold(
        smoothed,
        config.threshold_method,
        float(getattr(config, "threshold_scale", 1.0)),
    )
    binary_mask = smoothed > threshold
    # A small in-plane opening/closing removes isolated specks while avoiding
    # aggressive z-axis morphology on anisotropic stacks.
    binary_mask = ndi.binary_opening(binary_mask, structure=np.ones((1, 2, 2), dtype=bool))
    binary_mask = ndi.binary_closing(binary_mask, structure=np.ones((1, 2, 2), dtype=bool))

    labels = _split_components(
        binary_mask,
        spacing_zyx,
        min_distance_um=config.watershed_min_distance_um,
        enable_watershed=config.watershed,
        max_volume_um3=config.max_volume_um3,
    )
    measurements = _measure_labels(
        labels,
        smoothed,
        spacing_zyx,
        "hflu",
        float(config.min_volume_um3),
        float(config.max_volume_um3),
        config.remove_border_objects,
    )
    return ChannelArtifacts(
        raw_stack=stack,
        preprocessed_stack=smoothed,
        binary_mask=binary_mask,
        labels=labels,
        measurements=measurements,
        support_mask=None,
    )
