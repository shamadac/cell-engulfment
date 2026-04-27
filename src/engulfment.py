from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class EngulfmentResult:
    engulfing_yeast_count: int
    engulfment_rate: float
    engulfed_yeast_indices: list[int]
    engulfed_yeast_centroids: list[list[float]]
    ambiguous_bacteria_count: int = 0
    assignment_rows: list[dict[str, Any]] = field(default_factory=list)


def _slice_centroid_inside(mask_2d: np.ndarray, points: np.ndarray) -> bool:
    if points.size == 0:
        return False
    centroid = np.rint(points.mean(axis=0)).astype(int)
    y = int(np.clip(centroid[0], 0, mask_2d.shape[0] - 1))
    x = int(np.clip(centroid[1], 0, mask_2d.shape[1] - 1))
    return bool(mask_2d[y, x])


def _label_shell_cavity_mask(
    scer_labels: np.ndarray,
    scer_support_mask: np.ndarray,
    label_id: int,
) -> np.ndarray:
    cavity = np.zeros_like(scer_labels, dtype=bool)
    for z_index in range(scer_labels.shape[0]):
        label_plane = scer_labels[z_index] == label_id
        if not np.any(label_plane):
            continue
        support_plane = scer_support_mask[z_index] & label_plane
        cavity[z_index] = ndi.binary_fill_holes(support_plane) & ~support_plane & label_plane
    return cavity


def compute_hflu_radius(row: pd.Series) -> float:
    return float(
        np.sqrt(
            (row["B-width"] / 2) ** 2
            + (row["B-height"] / 2) ** 2
            + (row["B-depth"] / 2) ** 2
        )
    )


def compute_scer_radius(row: pd.Series) -> float:
    return float(max(row["B-width"], row["B-height"], row["B-depth"]) / 2)


def is_engulfed(scer_row: pd.Series, hflu_row: pd.Series) -> bool:
    scer_centroid = np.array([scer_row["XM"], scer_row["YM"], scer_row["ZM"]], dtype=float)
    hflu_centroid = np.array([hflu_row["XM"], hflu_row["YM"], hflu_row["ZM"]], dtype=float)
    distance = float(np.linalg.norm(scer_centroid - hflu_centroid))
    return distance + compute_hflu_radius(hflu_row) <= compute_scer_radius(scer_row)


def classify_sample(hflu_df: pd.DataFrame, scer_df: pd.DataFrame) -> EngulfmentResult:
    if hflu_df.empty or scer_df.empty:
        return EngulfmentResult(
            engulfing_yeast_count=0,
            engulfment_rate=0.0,
            engulfed_yeast_indices=[],
            engulfed_yeast_centroids=[],
        )

    hflu_centroids = hflu_df[["XM", "YM", "ZM"]].to_numpy(dtype=float)
    hflu_radii = hflu_df.apply(compute_hflu_radius, axis=1).to_numpy(dtype=float)

    scer_centroids = scer_df[["XM", "YM", "ZM"]].to_numpy(dtype=float)
    scer_radii = scer_df.apply(compute_scer_radius, axis=1).to_numpy(dtype=float)

    engulfed_indices: list[int] = []
    engulfed_centroids: list[list[float]] = []

    for index, (scer_centroid, scer_radius) in enumerate(zip(scer_centroids, scer_radii, strict=False)):
        distances = np.linalg.norm(hflu_centroids - scer_centroid, axis=1)
        if np.any(distances + hflu_radii <= scer_radius):
            engulfed_indices.append(index)
            engulfed_centroids.append(scer_centroid.tolist())

    engulfing_yeast_count = len(engulfed_indices)
    engulfment_rate = (engulfing_yeast_count / len(scer_df) * 100.0) if len(scer_df) > 0 else 0.0

    return EngulfmentResult(
        engulfing_yeast_count=engulfing_yeast_count,
        engulfment_rate=float(engulfment_rate),
        engulfed_yeast_indices=engulfed_indices,
        engulfed_yeast_centroids=engulfed_centroids,
    )


def _centroid_inside_label(label_volume: np.ndarray, label_id: int, row: pd.Series) -> bool:
    z = int(np.clip(round(float(row["voxel_centroid_z"])), 0, label_volume.shape[0] - 1))
    y = int(np.clip(round(float(row["voxel_centroid_y"])), 0, label_volume.shape[1] - 1))
    x = int(np.clip(round(float(row["voxel_centroid_x"])), 0, label_volume.shape[2] - 1))
    return bool(label_volume[z, y, x] == label_id)


def _overlap_counts(hflu_labels: np.ndarray, scer_labels: np.ndarray) -> dict[tuple[int, int], int]:
    hflu_flat = hflu_labels.ravel().astype(np.int64, copy=False)
    scer_flat = scer_labels.ravel().astype(np.int64, copy=False)
    valid = (hflu_flat > 0) & (scer_flat > 0)
    if not np.any(valid):
        return {}

    scer_factor = int(scer_flat.max()) + 1
    pair_codes = hflu_flat[valid] * scer_factor + scer_flat[valid]
    unique_codes, counts = np.unique(pair_codes, return_counts=True)
    return {
        (int(code // scer_factor), int(code % scer_factor)): int(count)
        for code, count in zip(unique_codes, counts, strict=False)
    }


def _apply_interior_margin(
    scer_labels: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    interior_margin_um: float,
) -> np.ndarray:
    if interior_margin_um <= 0 or scer_labels.max() == 0:
        return scer_labels

    result = np.zeros_like(scer_labels, dtype=np.int32)
    spacing_yx = (float(spacing_zyx[1]), float(spacing_zyx[2]))
    for label_id, bbox in enumerate(ndi.find_objects(scer_labels), start=1):
        if bbox is None:
            continue

        sublabels = scer_labels[bbox]
        submask = sublabels == label_id
        if not np.any(submask):
            continue

        eroded = np.zeros_like(submask, dtype=bool)
        for z_index in range(submask.shape[0]):
            plane = submask[z_index]
            if not np.any(plane):
                continue
            padded_plane = np.pad(plane, 1, mode="constant", constant_values=False)
            distance = ndi.distance_transform_edt(padded_plane, sampling=spacing_yx)[1:-1, 1:-1]
            eroded[z_index] = distance >= interior_margin_um

        if np.any(eroded):
            subresult = result[bbox]
            subresult[eroded] = label_id

    return result


def classify_mask_containment(
    hflu_df: pd.DataFrame,
    scer_df: pd.DataFrame,
    hflu_labels: np.ndarray,
    scer_interior_labels: np.ndarray,
    *,
    min_inside_fraction: float = 0.95,
    interior_margin_um: float = 0.0,
    spacing_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    allow_shared_bacteria: bool = False,
) -> EngulfmentResult:
    if hflu_df.empty or scer_df.empty:
        return EngulfmentResult(
            engulfing_yeast_count=0,
            engulfment_rate=0.0,
            engulfed_yeast_indices=[],
            engulfed_yeast_centroids=[],
        )

    hflu_lookup = hflu_df.reset_index(drop=True).copy()
    scer_lookup = scer_df.reset_index(drop=True).copy()
    interior_labels = _apply_interior_margin(
        scer_interior_labels,
        spacing_zyx,
        interior_margin_um,
    )
    hflu_centroids = hflu_lookup[["XM", "YM", "ZM"]].to_numpy(dtype=float)
    hflu_tree = cKDTree(hflu_centroids)
    max_hflu_radius = float(hflu_lookup.apply(compute_hflu_radius, axis=1).max()) if len(hflu_lookup) else 0.0
    hflu_voxel_counts = np.bincount(hflu_labels.ravel())
    overlaps = _overlap_counts(hflu_labels, interior_labels)

    candidate_rows: list[dict[str, Any]] = []
    for scer_idx, scer_row in scer_lookup.iterrows():
        scer_centroid = np.array([scer_row["XM"], scer_row["YM"], scer_row["ZM"]], dtype=float)
        search_radius = float(
            np.linalg.norm(
                np.array(
                    [
                        scer_row["B-width"] / 2 + max_hflu_radius,
                        scer_row["B-height"] / 2 + max_hflu_radius,
                        scer_row["B-depth"] / 2 + max_hflu_radius,
                    ],
                    dtype=float,
                )
            )
        )
        candidate_indices = hflu_tree.query_ball_point(scer_centroid, search_radius)
        scer_label_id = int(scer_row["label_id"])

        for hflu_idx in candidate_indices:
            hflu_row = hflu_lookup.iloc[int(hflu_idx)]
            hflu_label_id = int(hflu_row["label_id"])
            hflu_voxels = int(hflu_voxel_counts[hflu_label_id]) if hflu_label_id < len(hflu_voxel_counts) else 0
            if hflu_voxels == 0:
                continue

            overlap_voxels = overlaps.get((hflu_label_id, scer_label_id), 0)
            inside_fraction = overlap_voxels / hflu_voxels
            centroid_inside = _centroid_inside_label(interior_labels, scer_label_id, hflu_row)
            if inside_fraction >= min_inside_fraction:
                candidate_rows.append(
                    {
                        "bacteria_index": int(hflu_idx),
                        "bacteria_label_id": hflu_label_id,
                        "yeast_index": int(scer_idx),
                        "yeast_label_id": scer_label_id,
                        "inside_fraction": float(inside_fraction),
                        "overlap_voxels": overlap_voxels,
                        "centroid_inside": bool(centroid_inside),
                        "interior_margin_um": float(interior_margin_um),
                    }
                )

    if not candidate_rows:
        return EngulfmentResult(
            engulfing_yeast_count=0,
            engulfment_rate=0.0,
            engulfed_yeast_indices=[],
            engulfed_yeast_centroids=[],
        )

    by_bacteria: dict[int, list[dict[str, Any]]] = {}
    for row in candidate_rows:
        by_bacteria.setdefault(int(row["bacteria_index"]), []).append(row)

    chosen_rows: list[dict[str, Any]] = []
    ambiguous_bacteria_count = 0
    for matches in by_bacteria.values():
        ordered = sorted(
            matches,
            key=lambda item: (
                bool(item["centroid_inside"]),
                float(item["inside_fraction"]),
                int(item["overlap_voxels"]),
                -int(item["yeast_label_id"]),
            ),
            reverse=True,
        )
        if len(ordered) > 1:
            ambiguous_bacteria_count += 1
        chosen = dict(ordered[0])
        if not bool(chosen["centroid_inside"]):
            continue
        chosen["ambiguous"] = len(ordered) > 1
        chosen_rows.append(chosen)
        if allow_shared_bacteria:
            for extra in ordered[1:]:
                if not bool(extra["centroid_inside"]):
                    continue
                extra_row = dict(extra)
                extra_row["ambiguous"] = True
                chosen_rows.append(extra_row)

    unique_yeast_indices = sorted({int(row["yeast_index"]) for row in chosen_rows})
    engulfed_centroids = [
        scer_lookup.iloc[index][["XM", "YM", "ZM"]].astype(float).tolist()
        for index in unique_yeast_indices
    ]
    engulfment_rate = (len(unique_yeast_indices) / len(scer_lookup) * 100.0) if len(scer_lookup) else 0.0

    return EngulfmentResult(
        engulfing_yeast_count=len(unique_yeast_indices),
        engulfment_rate=float(engulfment_rate),
        engulfed_yeast_indices=unique_yeast_indices,
        engulfed_yeast_centroids=engulfed_centroids,
        ambiguous_bacteria_count=ambiguous_bacteria_count,
        assignment_rows=chosen_rows,
    )


def validate_shell_cavity_support(
    engulfment_result: EngulfmentResult,
    hflu_labels: np.ndarray,
    scer_labels: np.ndarray,
    scer_support_mask: np.ndarray | None,
    scer_lookup: pd.DataFrame,
    *,
    min_overlap_fraction: float,
    min_centroid_slice_fraction: float,
) -> EngulfmentResult:
    if not engulfment_result.assignment_rows or scer_support_mask is None:
        return engulfment_result

    cavity_cache: dict[int, np.ndarray] = {}
    enriched_rows: list[dict[str, Any]] = []
    validated_rows: list[dict[str, Any]] = []

    for row in engulfment_result.assignment_rows:
        bacteria_label_id = int(row["bacteria_label_id"])
        yeast_label_id = int(row["yeast_label_id"])
        bacterium_mask = hflu_labels == bacteria_label_id
        total_bacterium_voxels = int(bacterium_mask.sum())
        if total_bacterium_voxels <= 0:
            continue

        cavity_mask = cavity_cache.get(yeast_label_id)
        if cavity_mask is None:
            cavity_mask = _label_shell_cavity_mask(scer_labels, scer_support_mask.astype(bool, copy=False), yeast_label_id)
            cavity_cache[yeast_label_id] = cavity_mask

        overlap_voxels = int((bacterium_mask & cavity_mask).sum())
        cavity_overlap_fraction = overlap_voxels / total_bacterium_voxels

        occupied_slices = np.where(bacterium_mask.any(axis=(1, 2)))[0]
        centroid_slice_hits = 0
        for z_index in occupied_slices:
            slice_points = np.argwhere(bacterium_mask[z_index])
            if _slice_centroid_inside(cavity_mask[z_index], slice_points):
                centroid_slice_hits += 1
        centroid_slice_fraction = centroid_slice_hits / max(1, len(occupied_slices))
        validation_passed = (
            cavity_overlap_fraction >= min_overlap_fraction
            or centroid_slice_fraction >= min_centroid_slice_fraction
        )

        enriched_row = dict(row)
        enriched_row["shell_cavity_overlap_fraction"] = float(cavity_overlap_fraction)
        enriched_row["shell_cavity_centroid_slice_fraction"] = float(centroid_slice_fraction)
        enriched_row["shell_cavity_validation_passed"] = bool(validation_passed)
        enriched_rows.append(enriched_row)
        if validation_passed:
            validated_rows.append(enriched_row)

    if not enriched_rows:
        return engulfment_result

    unique_yeast_indices = sorted({int(row["yeast_index"]) for row in validated_rows})
    engulfed_centroids = [
        scer_lookup.iloc[index][["XM", "YM", "ZM"]].astype(float).tolist()
        for index in unique_yeast_indices
    ]
    engulfment_rate = (len(unique_yeast_indices) / len(scer_lookup) * 100.0) if len(scer_lookup) else 0.0
    return EngulfmentResult(
        engulfing_yeast_count=len(unique_yeast_indices),
        engulfment_rate=float(engulfment_rate),
        engulfed_yeast_indices=unique_yeast_indices,
        engulfed_yeast_centroids=engulfed_centroids,
        ambiguous_bacteria_count=engulfment_result.ambiguous_bacteria_count,
        assignment_rows=enriched_rows,
    )
