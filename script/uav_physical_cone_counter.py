#!/usr/bin/env python3
"""Compare legacy image-space counting with UAV motion-compensated physical IDs.

This is intentionally an independent experiment. It does not run or modify the
FALLEN, SHIFT, or MISSING business rules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from cone_counter import ConeCounter
from traffic_rules import AnalysisResult, ConeDetection


Point = tuple[float, float]
VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
DEFAULT_MIN_CONE_LONG_SIDE_RATIO = 24.0 / 1080.0


def _distance(first: Point, second: Point) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _apply_homography(matrix: Any, point: Point) -> Point | None:
    import numpy as np

    vector = np.asarray([point[0], point[1], 1.0], dtype=np.float64)
    projected = np.asarray(matrix, dtype=np.float64) @ vector
    denominator = float(projected[2])
    if not math.isfinite(denominator) or abs(denominator) < 1e-9:
        return None
    result = (float(projected[0] / denominator), float(projected[1] / denominator))
    return result if all(math.isfinite(value) for value in result) else None


def _polygon_contact(polygon: Sequence[Sequence[float]]) -> Point:
    bottom = sorted(((float(point[0]), float(point[1])) for point in polygon), key=lambda point: point[1], reverse=True)[:2]
    return ((bottom[0][0] + bottom[1][0]) / 2.0, (bottom[0][1] + bottom[1][1]) / 2.0)


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class MotionEstimate:
    valid: bool
    method: str
    inliers: int
    points: int
    inlier_ratio: float
    reprojection_error: float | None
    reference_from_current: Any


class GroundMotionEstimator:
    """Estimate frame-to-frame ground/background motion with feature flow."""

    def __init__(
        self,
        cv2: Any,
        np: Any,
        *,
        feature_scale: float = 0.5,
        max_features: int = 800,
        min_inliers: int = 25,
        min_inlier_ratio: float = 0.35,
        ransac_threshold: float = 4.0,
    ) -> None:
        if not 0.25 <= feature_scale <= 1.0:
            raise ValueError("feature_scale must be in [0.25, 1]")
        self.cv2 = cv2
        self.np = np
        self.feature_scale = feature_scale
        self.max_features = max_features
        self.min_inliers = min_inliers
        self.min_inlier_ratio = min_inlier_ratio
        self.ransac_threshold = ransac_threshold
        self._previous_gray: Any | None = None
        self._previous_polygons: tuple[tuple[Point, ...], ...] = ()
        self._reference_from_previous: Any | None = None

    def _gray_small(self, frame: Any) -> Any:
        gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
        if self.feature_scale == 1.0:
            return gray
        return self.cv2.resize(gray, None, fx=self.feature_scale, fy=self.feature_scale, interpolation=self.cv2.INTER_AREA)

    def _mask(self, shape: tuple[int, int]) -> Any:
        mask = self.np.full(shape, 255, dtype=self.np.uint8)
        if self._previous_polygons:
            polygons = [
                self.np.asarray(
                    [[round(point[0] * self.feature_scale), round(point[1] * self.feature_scale)] for point in polygon],
                    dtype=self.np.int32,
                )
                for polygon in self._previous_polygons
            ]
            blocked = self.np.zeros(shape, dtype=self.np.uint8)
            self.cv2.fillPoly(blocked, polygons, 255)
            kernel = self.np.ones((9, 9), dtype=self.np.uint8)
            blocked = self.cv2.dilate(blocked, kernel)
            mask = self.cv2.bitwise_not(blocked)
        return mask

    def _identity(self) -> Any:
        return self.np.eye(3, dtype=self.np.float64)

    def update(self, frame: Any, polygons: Sequence[Sequence[Point]]) -> MotionEstimate:
        current_gray = self._gray_small(frame)
        current_polygons = tuple(tuple((float(x), float(y)) for x, y in polygon) for polygon in polygons)
        if self._previous_gray is None:
            self._previous_gray = current_gray
            self._previous_polygons = current_polygons
            self._reference_from_previous = self._identity()
            return MotionEstimate(False, "initial", 0, 0, 0.0, None, self._reference_from_previous.copy())

        previous_points = self.cv2.goodFeaturesToTrack(
            self._previous_gray,
            maxCorners=self.max_features,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7,
            mask=self._mask(self._previous_gray.shape[:2]),
        )
        method = "insufficient_features"
        valid = False
        inliers = 0
        point_count = 0
        inlier_ratio = 0.0
        reprojection_error: float | None = None
        current_from_previous_small = self._identity()
        if previous_points is not None and len(previous_points) >= 8:
            tracked_points, status, errors = self.cv2.calcOpticalFlowPyrLK(
                self._previous_gray,
                current_gray,
                previous_points,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(self.cv2.TERM_CRITERIA_EPS | self.cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
            if tracked_points is not None and status is not None:
                good = status.reshape(-1).astype(bool)
                previous = previous_points.reshape(-1, 2)[good]
                current = tracked_points.reshape(-1, 2)[good]
                finite = self.np.isfinite(previous).all(axis=1) & self.np.isfinite(current).all(axis=1)
                previous = previous[finite]
                current = current[finite]
                point_count = int(len(previous))
                if point_count >= 8:
                    matrix, inlier_mask = self.cv2.findHomography(
                        previous,
                        current,
                        self.cv2.RANSAC,
                        self.ransac_threshold * self.feature_scale,
                    )
                    if matrix is not None and inlier_mask is not None:
                        inlier_mask = inlier_mask.reshape(-1).astype(bool)
                        inliers = int(inlier_mask.sum())
                        inlier_ratio = inliers / point_count
                        if inliers:
                            projected = self.cv2.perspectiveTransform(previous.reshape(-1, 1, 2), matrix).reshape(-1, 2)
                            reprojection_error = float(self.np.mean(self.np.linalg.norm(projected[inlier_mask] - current[inlier_mask], axis=1)))
                        if inliers >= self.min_inliers and inlier_ratio >= self.min_inlier_ratio:
                            current_from_previous_small = self.np.asarray(matrix, dtype=self.np.float64)
                            method = "homography"
                            valid = True
                        else:
                            method = "homography_rejected"
                    else:
                        method = "homography_failed"

        scale = self.feature_scale
        scale_matrix = self.np.asarray([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]], dtype=self.np.float64)
        inverse_scale = self.np.asarray([[1.0 / scale, 0.0, 0.0], [0.0, 1.0 / scale, 0.0], [0.0, 0.0, 1.0]], dtype=self.np.float64)
        current_from_previous = inverse_scale @ current_from_previous_small @ scale_matrix
        previous_from_current = self.np.linalg.inv(current_from_previous) if valid else self._identity()
        reference_from_previous = self._reference_from_previous
        if reference_from_previous is None:
            reference_from_previous = self._identity()
        reference_from_current = reference_from_previous @ previous_from_current
        self._previous_gray = current_gray
        self._previous_polygons = current_polygons
        self._reference_from_previous = reference_from_current
        return MotionEstimate(valid, method, inliers, point_count, inlier_ratio, reprojection_error, reference_from_current.copy())


@dataclass
class _PhysicalTrack:
    physical_id: int
    map_point: Point
    image_point: Point
    votes: deque[bool]
    velocity: Point = (0.0, 0.0)
    state: str = "tentative"
    age: int = 1
    hits: int = 1
    missed_frames: int = 0
    counted: bool = False


@dataclass(frozen=True)
class PhysicalTrack:
    """Public snapshot of one motion-compensated physical cone track."""

    physical_id: int
    map_point: Point
    image_point: Point
    state: str
    age: int
    hits: int
    missed_frames: int
    counted: bool


@dataclass(frozen=True)
class PhysicalCountResult:
    stable_visible_count: int
    unique_seen_count: int
    new_physical_ids: tuple[int, ...]
    active_physical_ids: tuple[int, ...]
    held_physical_ids: tuple[int, ...]
    detection_to_physical_id: tuple[int | None, ...]
    tracking_degraded: bool
    policy: dict[str, int | float | bool]
    recovery_detection_indices: tuple[int, ...] = ()
    tracks: tuple[PhysicalTrack, ...] = ()


@dataclass(frozen=True)
class PhysicalRoi:
    """One stable ROI for a geometrically coherent physical cone route."""

    roi_id: int
    bounds: tuple[float, float, float, float]
    member_ids: tuple[int, ...]
    held_ids: tuple[int, ...]
    direction: Point
    state: str


@dataclass(frozen=True)
class PhysicalRoiResult:
    """Stable image-space ROIs derived from confirmed physical tracks."""

    rois: tuple[PhysicalRoi, ...]


@dataclass(frozen=True)
class _PhysicalRouteObservation:
    member_ids: tuple[int, ...]
    member_points: tuple[tuple[int, Point], ...]
    direction: Point
    map_center: Point
    spacing: float


@dataclass
class _PhysicalRoiState:
    roi_id: int
    member_ids: tuple[int, ...]
    member_points: tuple[tuple[int, Point], ...]
    direction: Point
    map_center: Point
    spacing: float
    bounds: tuple[float, float, float, float]
    state: str = "tentative"
    consecutive_hits: int = 1
    missed_frames: int = 0


class PhysicalRoiTracker:
    """Group physical tracks into stable route ROIs in the reference image."""

    def __init__(
        self,
        *,
        smoothing: float = 0.25,
        padding: float = 24.0,
        hold_frames: int = 8,
        confirm_frames: int = 3,
        min_cones: int = 2,
        lateral_distance_ratio: float = 0.6,
        longitudinal_link_ratio: float = 2.5,
        max_direction_difference: float = 20.0,
        identity_memory_frames: int = 30,
    ) -> None:
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing must be in (0, 1]")
        if padding < 0.0:
            raise ValueError("padding must not be negative")
        if hold_frames < 0:
            raise ValueError("hold_frames must not be negative")
        if confirm_frames < 1:
            raise ValueError("confirm_frames must be at least 1")
        if min_cones < 2:
            raise ValueError("min_cones must be at least 2")
        if lateral_distance_ratio <= 0.0:
            raise ValueError("lateral_distance_ratio must be positive")
        if longitudinal_link_ratio <= 1.0:
            raise ValueError("longitudinal_link_ratio must be greater than 1")
        if not 0.0 < max_direction_difference < 90.0:
            raise ValueError("max_direction_difference must be in (0, 90)")
        if identity_memory_frames < hold_frames:
            raise ValueError("identity_memory_frames must be at least hold_frames")
        self.smoothing = smoothing
        self.padding = padding
        self.hold_frames = hold_frames
        self.confirm_frames = confirm_frames
        self.min_cones = min_cones
        self.lateral_distance_ratio = lateral_distance_ratio
        self.longitudinal_link_ratio = longitudinal_link_ratio
        self.max_direction_difference = max_direction_difference
        self.identity_memory_frames = identity_memory_frames
        self._states: list[_PhysicalRoiState] = []
        self._next_roi_id = 1
        self._member_bounds: dict[int, tuple[float, float, float, float]] = {}

    @staticmethod
    def _canonical_direction(direction: Point) -> Point:
        length = math.hypot(direction[0], direction[1])
        if length <= 1e-9:
            return (1.0, 0.0)
        normalized = (direction[0] / length, direction[1] / length)
        if normalized[0] < 0.0 or (
            abs(normalized[0]) <= 1e-9 and normalized[1] < 0.0
        ):
            return (-normalized[0], -normalized[1])
        return normalized

    @classmethod
    def _principal_direction(cls, points: Sequence[Point]) -> Point:
        if len(points) < 2:
            return (1.0, 0.0)
        mean_x = statistics.fmean(point[0] for point in points)
        mean_y = statistics.fmean(point[1] for point in points)
        covariance_xx = statistics.fmean(
            (point[0] - mean_x) ** 2 for point in points
        )
        covariance_yy = statistics.fmean(
            (point[1] - mean_y) ** 2 for point in points
        )
        covariance_xy = statistics.fmean(
            (point[0] - mean_x) * (point[1] - mean_y) for point in points
        )
        angle = 0.5 * math.atan2(
            2.0 * covariance_xy, covariance_xx - covariance_yy
        )
        return cls._canonical_direction((math.cos(angle), math.sin(angle)))

    @staticmethod
    def _direction_difference(first: Point, second: Point) -> float:
        cosine = min(1.0, max(-1.0, abs(first[0] * second[0] + first[1] * second[1])))
        return math.degrees(math.acos(cosine))

    @staticmethod
    def _typical_spacing(points: Sequence[Point]) -> float:
        if len(points) < 2:
            return 55.0
        neighbor_rank = 1 if len(points) >= 3 else 0
        distances = []
        for index, point in enumerate(points):
            ordered = sorted(
                _distance(point, other)
                for other_index, other in enumerate(points)
                if other_index != index
            )
            distances.append(ordered[min(neighbor_rank, len(ordered) - 1)])
        positive = [distance for distance in distances if distance > 1e-6]
        return statistics.median(positive) if positive else 55.0

    @staticmethod
    def _local_spacing(points: Sequence[Point]) -> float:
        if len(points) < 2:
            return 55.0
        distances = []
        for index, point in enumerate(points):
            nearest = min(
                _distance(point, other)
                for other_index, other in enumerate(points)
                if other_index != index
            )
            if nearest > 1e-6:
                distances.append(nearest)
        return statistics.median(distances) if distances else 55.0

    @staticmethod
    def _line_distance(point: Point, origin: Point, direction: Point) -> float:
        normal = (-direction[1], direction[0])
        return abs(
            (point[0] - origin[0]) * normal[0]
            + (point[1] - origin[1]) * normal[1]
        )

    @staticmethod
    def _continuous_components(
        indices: Sequence[int],
        points: Sequence[Point],
        direction: Point,
        maximum_gap: float,
    ) -> list[tuple[int, ...]]:
        ordered = sorted(
            indices,
            key=lambda index: (
                points[index][0] * direction[0]
                + points[index][1] * direction[1]
            ),
        )
        if not ordered:
            return []
        components: list[list[int]] = [[ordered[0]]]
        previous_projection = (
            points[ordered[0]][0] * direction[0]
            + points[ordered[0]][1] * direction[1]
        )
        for index in ordered[1:]:
            projection = (
                points[index][0] * direction[0]
                + points[index][1] * direction[1]
            )
            if projection - previous_projection > maximum_gap:
                components.append([])
            components[-1].append(index)
            previous_projection = projection
        return [tuple(component) for component in components]

    def _route_observations(
        self, tracks: Sequence[PhysicalTrack]
    ) -> list[_PhysicalRouteObservation]:
        if len(tracks) < self.min_cones:
            return []
        points = [track.map_point for track in tracks]
        spacing = self._typical_spacing(points)
        lateral_limit = max(4.0, spacing * self.lateral_distance_ratio)
        longitudinal_limit = max(8.0, spacing * self.longitudinal_link_ratio)
        candidates: dict[tuple[int, ...], tuple[Point, float]] = {}

        for first_index in range(len(points)):
            for second_index in range(first_index + 1, len(points)):
                first = points[first_index]
                second = points[second_index]
                pair_distance = _distance(first, second)
                if pair_distance <= 1e-6 or pair_distance > longitudinal_limit:
                    continue
                pair_direction = self._canonical_direction(
                    (second[0] - first[0], second[1] - first[1])
                )
                supported = [
                    index
                    for index, point in enumerate(points)
                    if self._line_distance(point, first, pair_direction)
                    <= lateral_limit
                ]
                for component in self._continuous_components(
                    supported, points, pair_direction, longitudinal_limit
                ):
                    if (
                        first_index not in component
                        or second_index not in component
                        or len(component) < self.min_cones
                    ):
                        continue
                    component_points = [points[index] for index in component]
                    direction = self._principal_direction(component_points)
                    if (
                        self._direction_difference(pair_direction, direction)
                        > self.max_direction_difference
                    ):
                        continue
                    center = (
                        statistics.fmean(point[0] for point in component_points),
                        statistics.fmean(point[1] for point in component_points),
                    )
                    refined = tuple(
                        index
                        for index in component
                        if self._line_distance(points[index], center, direction)
                        <= lateral_limit
                    )
                    if len(refined) < self.min_cones:
                        continue
                    refined_points = [points[index] for index in refined]
                    local_limit = max(
                        8.0,
                        self._local_spacing(refined_points)
                        * self.longitudinal_link_ratio,
                    )
                    for final_component in self._continuous_components(
                        refined, points, direction, local_limit
                    ):
                        if len(final_component) < self.min_cones:
                            continue
                        final_points = [points[index] for index in final_component]
                        final_direction = self._principal_direction(final_points)
                        if (
                            self._direction_difference(
                                pair_direction, final_direction
                            )
                            > self.max_direction_difference
                        ):
                            continue
                        final_center = (
                            statistics.fmean(point[0] for point in final_points),
                            statistics.fmean(point[1] for point in final_points),
                        )
                        residual = statistics.fmean(
                            self._line_distance(
                                points[index], final_center, final_direction
                            )
                            for index in final_component
                        )
                        key = tuple(sorted(final_component))
                        previous = candidates.get(key)
                        if previous is None or residual < previous[1]:
                            candidates[key] = (final_direction, residual)

        remaining = set(range(len(tracks)))
        groups: list[tuple[tuple[int, ...], Point]] = []
        for indices, (direction, residual) in sorted(
            candidates.items(),
            key=lambda item: (-len(item[0]), item[1][1], item[0]),
        ):
            if len(indices) < self.min_cones or not set(indices).issubset(remaining):
                continue
            groups.append((indices, direction))
            remaining.difference_update(indices)

        observations = []
        for indices, direction in groups:
            group_tracks = [tracks[index] for index in indices]
            observations.append(
                _PhysicalRouteObservation(
                    member_ids=tuple(
                        sorted(track.physical_id for track in group_tracks)
                    ),
                    member_points=tuple(
                        sorted(
                            (track.physical_id, track.map_point)
                            for track in group_tracks
                        )
                    ),
                    direction=direction,
                    map_center=(
                        statistics.fmean(track.map_point[0] for track in group_tracks),
                        statistics.fmean(track.map_point[1] for track in group_tracks),
                    ),
                    spacing=self._local_spacing(
                        [track.map_point for track in group_tracks]
                    ),
                )
            )
        observations.sort(key=lambda item: (item.map_center[0], item.map_center[1]))
        return observations

    @staticmethod
    def _clip(
        bounds: tuple[float, float, float, float], image_size: tuple[int, int]
    ) -> tuple[float, float, float, float]:
        width, height = image_size
        x1 = min(float(width), max(0.0, bounds[0]))
        y1 = min(float(height), max(0.0, bounds[1]))
        x2 = min(float(width), max(x1, bounds[2]))
        y2 = min(float(height), max(y1, bounds[3]))
        return x1, y1, x2, y2

    def _raw_bounds(
        self,
        member_ids: Sequence[int],
        tracks_by_id: dict[int, PhysicalTrack],
        image_size: tuple[int, int],
    ) -> tuple[float, float, float, float]:
        bounds = []
        for physical_id in member_ids:
            observed = self._member_bounds.get(physical_id)
            if observed is None:
                point = tracks_by_id[physical_id].image_point
                observed = (point[0], point[1], point[0], point[1])
            bounds.append(observed)
        raw = (
            min(item[0] for item in bounds) - self.padding,
            min(item[1] for item in bounds) - self.padding,
            max(item[2] for item in bounds) + self.padding,
            max(item[3] for item in bounds) + self.padding,
        )
        return self._clip(raw, image_size)

    def _smooth_bounds(
        self,
        previous: tuple[float, float, float, float],
        current: tuple[float, float, float, float],
        image_size: tuple[int, int],
    ) -> tuple[float, float, float, float]:
        return self._clip(
            tuple(
                old * (1.0 - self.smoothing) + new * self.smoothing
                for old, new in zip(previous, current)
            ),  # type: ignore[arg-type]
            image_size,
        )

    def update(
        self,
        tracks: Sequence[PhysicalTrack],
        detection_bounds: dict[int, tuple[float, float, float, float]],
        image_size: tuple[int, int],
    ) -> PhysicalRoiResult:
        usable = [
            track
            for track in tracks
            if track.state == "confirmed"
            or (track.state == "held" and track.missed_frames <= self.hold_frames)
        ]
        usable_ids = {track.physical_id for track in usable}
        tracks_by_id = {track.physical_id: track for track in usable}
        for physical_id, bounds in detection_bounds.items():
            if physical_id in usable_ids:
                self._member_bounds[physical_id] = bounds
        self._member_bounds = {
            physical_id: bounds
            for physical_id, bounds in self._member_bounds.items()
            if physical_id in usable_ids
        }

        observations = self._route_observations(usable)
        assignments: dict[int, int] = {}
        used_states: set[int] = set()
        used_observations: set[int] = set()
        matches = []
        for state_index, state in enumerate(self._states):
            state_members = set(state.member_ids)
            state_points = dict(state.member_points)
            for observation_index, observation in enumerate(observations):
                shared_ids = state_members.intersection(observation.member_ids)
                overlap = len(shared_ids)
                center_distance = _distance(
                    state.map_center, observation.map_center
                )
                if (
                    self._direction_difference(
                        state.direction, observation.direction
                    )
                    > self.max_direction_difference
                ):
                    continue
                match_kind = 0
                scale = max(8.0, state.spacing)
                if overlap:
                    observation_points = dict(observation.member_points)
                    shared_deltas = [
                        (
                            observation_points[physical_id][0]
                            - state_points[physical_id][0],
                            observation_points[physical_id][1]
                            - state_points[physical_id][1],
                        )
                        for physical_id in shared_ids
                    ]
                    longitudinal = statistics.median(
                        abs(
                            delta[0] * state.direction[0]
                            + delta[1] * state.direction[1]
                        )
                        for delta in shared_deltas
                    )
                    lateral = statistics.median(
                        abs(
                            -delta[0] * state.direction[1]
                            + delta[1] * state.direction[0]
                        )
                        for delta in shared_deltas
                    )
                    if (
                        lateral > scale * self.lateral_distance_ratio
                        or longitudinal
                        > scale * self.longitudinal_link_ratio
                    ):
                        continue
                else:
                    scale = max(state.spacing, observation.spacing)
                    delta = (
                        observation.map_center[0] - state.map_center[0],
                        observation.map_center[1] - state.map_center[1],
                    )
                    longitudinal = abs(
                        delta[0] * state.direction[0]
                        + delta[1] * state.direction[1]
                    )
                    lateral = abs(
                        -delta[0] * state.direction[1]
                        + delta[1] * state.direction[0]
                    )
                    if (
                        lateral > scale * self.lateral_distance_ratio
                        or longitudinal > scale * self.longitudinal_link_ratio
                    ):
                        continue
                    match_kind = 1
                matches.append(
                    (
                        match_kind,
                        -overlap,
                        center_distance,
                        state_index,
                        observation_index,
                    )
                )
        for _, _, _, state_index, observation_index in sorted(matches):
            if state_index in used_states or observation_index in used_observations:
                continue
            assignments[state_index] = observation_index
            used_states.add(state_index)
            used_observations.add(observation_index)

        retained: list[_PhysicalRoiState] = []
        for state_index, state in enumerate(self._states):
            observation_index = assignments.get(state_index)
            if observation_index is None:
                if set(state.member_ids).intersection(usable_ids):
                    state.missed_frames += 1
                else:
                    state.missed_frames = max(
                        state.missed_frames + 1, self.hold_frames + 1
                    )
                state.consecutive_hits = 0
                if (
                    state.state == "tentative"
                    or state.missed_frames > self.identity_memory_frames
                ):
                    continue
                state.state = (
                    "held"
                    if state.missed_frames <= self.hold_frames
                    else "dormant"
                )
                retained.append(state)
                continue

            observation = observations[observation_index]
            raw = self._raw_bounds(
                observation.member_ids, tracks_by_id, image_size
            )
            state.bounds = self._smooth_bounds(state.bounds, raw, image_size)
            state.member_ids = observation.member_ids
            state.member_points = observation.member_points
            state.direction = observation.direction
            state.map_center = observation.map_center
            state.spacing = observation.spacing
            state.missed_frames = 0
            state.consecutive_hits += 1
            if state.consecutive_hits >= self.confirm_frames:
                state.state = "confirmed"
            retained.append(state)

        for observation_index, observation in enumerate(observations):
            if observation_index in used_observations:
                continue
            raw = self._raw_bounds(
                observation.member_ids, tracks_by_id, image_size
            )
            state = _PhysicalRoiState(
                roi_id=self._next_roi_id,
                member_ids=observation.member_ids,
                member_points=observation.member_points,
                direction=observation.direction,
                map_center=observation.map_center,
                spacing=observation.spacing,
                bounds=raw,
                state="confirmed" if self.confirm_frames == 1 else "tentative",
            )
            self._next_roi_id += 1
            retained.append(state)

        self._states = retained
        rois = []
        visible_states = [
            state
            for state in self._states
            if state.state not in {"tentative", "dormant"}
        ]
        selected_states = []
        selected_member_ids: set[int] = set()
        for state in sorted(
            visible_states,
            key=lambda item: (
                0 if item.state == "confirmed" else 1,
                item.missed_frames,
                item.roi_id,
            ),
        ):
            if selected_member_ids.intersection(state.member_ids):
                continue
            selected_states.append(state)
            selected_member_ids.update(state.member_ids)
        for state in sorted(selected_states, key=lambda item: item.roi_id):
            held_ids = tuple(
                physical_id
                for physical_id in state.member_ids
                if physical_id in tracks_by_id
                and tracks_by_id[physical_id].state == "held"
            )
            rois.append(
                PhysicalRoi(
                    roi_id=state.roi_id,
                    bounds=state.bounds,
                    member_ids=state.member_ids,
                    held_ids=held_ids,
                    direction=state.direction,
                    state=state.state,
                )
            )
        return PhysicalRoiResult(tuple(rois))


class UavPhysicalCounter:
    """Track cone contacts in a reference ground image using ego-motion compensation."""

    def __init__(
        self,
        *,
        confirm_window: int = 5,
        confirm_votes: int = 3,
        max_missed_frames: int = 12,
        match_distance_pixels: float = 55.0,
        new_id_on_degraded_motion: bool = False,
    ) -> None:
        if confirm_window < 1 or not 1 <= confirm_votes <= confirm_window:
            raise ValueError("confirm_votes must be within confirm_window")
        if max_missed_frames < 0 or match_distance_pixels <= 0.0:
            raise ValueError("invalid physical counter policy")
        self.confirm_window = confirm_window
        self.confirm_votes = confirm_votes
        self.max_missed_frames = max_missed_frames
        self.match_distance_pixels = match_distance_pixels
        self.new_id_on_degraded_motion = new_id_on_degraded_motion
        self._states: list[_PhysicalTrack] = []
        self._next_id = 1
        self._unique_seen_count = 0

    @staticmethod
    def _spacing(points: Sequence[Point]) -> float:
        if len(points) < 2:
            return 55.0
        nearest = []
        for index, point in enumerate(points):
            distances = [_distance(point, other) for other_index, other in enumerate(points) if other_index != index]
            if distances:
                nearest.append(min(distances))
        return statistics.median(nearest) if nearest else 55.0

    def update(
        self,
        contacts: Sequence[Point],
        images: Sequence[Point],
        motion: MotionEstimate,
        can_create_physical_id: Sequence[bool] | None = None,
    ) -> PhysicalCountResult:
        if len(contacts) != len(images):
            raise ValueError("contacts and images must have the same length")
        creation_eligible = (
            tuple(True for _ in images)
            if can_create_physical_id is None
            else tuple(bool(value) for value in can_create_physical_id)
        )
        if len(creation_eligible) != len(images):
            raise ValueError(
                "can_create_physical_id must have one value per detection"
            )
        map_points = [
            _apply_homography(motion.reference_from_current, contact)
            for contact in contacts
        ]
        usable = all(point is not None for point in map_points)
        if not usable:
            map_points = [None] * len(images)
        valid_motion = motion.valid or not self._states
        maximum_distance = max(self.match_distance_pixels, min(180.0, self._spacing([point for point in map_points if point is not None]) * 1.25))
        candidates = sorted(
            (
                0 if creation_eligible[detection_index] else 1,
                _distance(
                    (
                        state.map_point[0] + state.velocity[0] * (state.missed_frames + 1),
                        state.map_point[1] + state.velocity[1] * (state.missed_frames + 1),
                    ),
                    point,
                ),
                state_index,
                detection_index,
            )
            for state_index, state in enumerate(self._states)
            for detection_index, point in enumerate(map_points)
            if point is not None
            and (
                creation_eligible[detection_index]
                or state.state in {"confirmed", "held"}
            )
        )
        assignments: dict[int, int] = {}
        used_states: set[int] = set()
        used_detections: set[int] = set()
        for _, distance, state_index, detection_index in candidates:
            if distance > maximum_distance:
                continue
            if state_index in used_states or detection_index in used_detections:
                continue
            used_states.add(state_index)
            used_detections.add(detection_index)
            assignments[detection_index] = state_index

        matched_states = set(assignments.values())
        retained: list[_PhysicalTrack] = []
        for state_index, state in enumerate(self._states):
            state.age += 1
            if state_index in matched_states:
                retained.append(state)
                continue
            state.missed_frames += 1
            state.votes.append(False)
            if state.missed_frames <= self.max_missed_frames:
                if state.state == "confirmed":
                    state.state = "held"
                retained.append(state)

        detection_ids: list[int | None] = [None] * len(images)
        new_ids: list[int] = []
        recovery_detection_indices: list[int] = []
        for detection_index, image_point in enumerate(images):
            map_point = map_points[detection_index]
            state_index = assignments.get(detection_index)
            if state_index is None:
                if not creation_eligible[detection_index]:
                    continue
                if not valid_motion and not self.new_id_on_degraded_motion:
                    continue
                assert map_point is not None
                state = _PhysicalTrack(self._next_id, map_point, contacts[detection_index], deque(maxlen=self.confirm_window))
                self._next_id += 1
                retained.append(state)
            else:
                state = self._states[state_index]
                assert map_point is not None
                elapsed = state.missed_frames + 1
                measured_velocity = ((map_point[0] - state.map_point[0]) / elapsed, (map_point[1] - state.map_point[1]) / elapsed)
                state.velocity = (state.velocity[0] * 0.5 + measured_velocity[0] * 0.5, state.velocity[1] * 0.5 + measured_velocity[1] * 0.5)
                state.map_point = map_point
                state.image_point = contacts[detection_index]
                state.missed_frames = 0
                state.hits += 1
                if state.state == "held":
                    state.state = "confirmed"
            state.votes.append(True)
            if state.state == "tentative" and len(state.votes) == self.confirm_window and sum(state.votes) >= self.confirm_votes:
                state.state = "confirmed"
            if state.state == "confirmed" and not state.counted:
                state.counted = True
                self._unique_seen_count += 1
                new_ids.append(state.physical_id)
            detection_ids[detection_index] = state.physical_id
            if not creation_eligible[detection_index]:
                recovery_detection_indices.append(detection_index)

        self._states = retained
        active_ids = tuple(state.physical_id for state in self._states if state.state == "confirmed" and state.missed_frames == 0)
        held_ids = tuple(state.physical_id for state in self._states if state.state == "held")
        return PhysicalCountResult(
            stable_visible_count=len(active_ids),
            unique_seen_count=self._unique_seen_count,
            new_physical_ids=tuple(new_ids),
            active_physical_ids=active_ids,
            held_physical_ids=held_ids,
            detection_to_physical_id=tuple(detection_ids),
            tracking_degraded=not motion.valid,
            policy={
                "confirm_window": self.confirm_window,
                "confirm_votes": self.confirm_votes,
                "max_missed_frames": self.max_missed_frames,
                "match_distance_pixels": self.match_distance_pixels,
                "new_id_on_degraded_motion": self.new_id_on_degraded_motion,
                "low_confidence_existing_id_only": True,
            },
            recovery_detection_indices=tuple(recovery_detection_indices),
            tracks=tuple(
                PhysicalTrack(
                    physical_id=state.physical_id,
                    map_point=state.map_point,
                    image_point=state.image_point,
                    state=state.state,
                    age=state.age,
                    hits=state.hits,
                    missed_frames=state.missed_frames,
                    counted=state.counted,
                )
                for state in self._states
            ),
        )


def _extract_cones(result: Any, np: Any, *, cone_class: str, image_size: tuple[int, int], min_long_side_ratio: float) -> list[ConeDetection]:
    if result.obb is None or len(result.obb) == 0:
        return []
    minimum_long_side = min(image_size) * min_long_side_ratio
    polygons = result.obb.xyxyxyxy.detach().cpu().numpy()
    confidences = result.obb.conf.detach().cpu().numpy()
    class_ids = result.obb.cls.detach().cpu().numpy().astype(int)
    detections: list[ConeDetection] = []
    for polygon_array, confidence, class_id in zip(polygons, confidences, class_ids):
        if _class_name(result.names, int(class_id)) != cone_class:
            continue
        polygon = tuple((float(point[0]), float(point[1])) for point in polygon_array)
        detection = ConeDetection(polygon=polygon, confidence=float(confidence), class_id=int(class_id), label=cone_class)
        if detection.long_axis_length >= minimum_long_side:
            detections.append(detection)
    return detections


def _video_fps(cv2: Any, source: str) -> float:
    capture = cv2.VideoCapture(source)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    return fps if fps > 0.0 else 25.0


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--physical-match-conf",
        type=float,
        default=0.20,
        help="Low confidence floor used only to rematch confirmed physical IDs",
    )
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--cone-class", default="cone")
    parser.add_argument("--min-cone-long-side-ratio", type=float, default=DEFAULT_MIN_CONE_LONG_SIDE_RATIO)
    parser.add_argument("--confirm-window", type=int, default=5)
    parser.add_argument("--confirm-votes", type=int, default=3)
    parser.add_argument("--max-missed-frames", type=int, default=12)
    parser.add_argument("--match-distance-pixels", type=float, default=55.0)
    parser.add_argument("--feature-scale", type=float, default=0.5)
    parser.add_argument("--max-features", type=int, default=800)
    parser.add_argument("--min-motion-inliers", type=int, default=25)
    parser.add_argument("--min-motion-inlier-ratio", type=float, default=0.35)
    parser.add_argument("--motion-ransac-threshold", type=float, default=4.0)
    parser.add_argument("--new-id-on-degraded-motion", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.weights.is_file():
        raise SystemExit(f"Weights do not exist: {args.weights}")
    if not 0.0 <= args.min_cone_long_side_ratio < 1.0:
        raise SystemExit("--min-cone-long-side-ratio must be in [0, 1)")
    if not 0.0 <= args.physical_match_conf <= args.conf <= 1.0:
        raise SystemExit("--physical-match-conf must be in [0, --conf]")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        import cv2
        import numpy as np
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Missing inference dependency") from exc

    source_path = Path(args.source)
    is_video = source_path.suffix.lower() in VIDEO_SUFFIXES or args.source.isdigit()
    fps = _video_fps(cv2, args.source) if is_video else 0.0
    model = YOLO(str(args.weights.resolve()))
    estimator = GroundMotionEstimator(
        cv2,
        np,
        feature_scale=args.feature_scale,
        max_features=args.max_features,
        min_inliers=args.min_motion_inliers,
        min_inlier_ratio=args.min_motion_inlier_ratio,
        ransac_threshold=args.motion_ransac_threshold,
    )
    legacy_counter = ConeCounter(
        confirm_window=args.confirm_window if is_video else 1,
        confirm_votes=min(args.confirm_votes, args.confirm_window) if is_video else 1,
        max_missed_frames=args.max_missed_frames,
    )
    physical_counter = UavPhysicalCounter(
        confirm_window=args.confirm_window if is_video else 1,
        confirm_votes=min(args.confirm_votes, args.confirm_window) if is_video else 1,
        max_missed_frames=args.max_missed_frames,
        match_distance_pixels=args.match_distance_pixels,
        new_id_on_degraded_motion=args.new_id_on_degraded_motion,
    )
    event_path = args.output / "events.jsonl"
    config = {
        "command": [str(item) for item in sys.argv],
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "weights": {"path": str(args.weights.resolve()), "sha256": _sha256_file(args.weights.resolve())},
        "source": {"path": str(source_path.resolve()), "sha256": _sha256_file(source_path)},
        "method": "frame_feature_homography_to_reference_ground_image",
        "anomaly_detection": "not_run",
    }
    (args.output / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    writer = None
    if args.save_video:
        capture = cv2.VideoCapture(args.source)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        writer = cv2.VideoWriter(str(args.output / f"{source_path.stem}_physical_count.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps or 25.0, (width, height))

    frame_count = 0
    detected_frames = 0
    legacy_max_visible = 0
    physical_max_visible = 0
    legacy_new_id_events = 0
    physical_new_id_events = 0
    recovery_candidate_count = 0
    recovery_matched_count = 0
    degraded_frames = 0
    motion_methods: Counter[str] = Counter()
    with event_path.open("w", encoding="utf-8") as events_file:
        try:
            predictions = model.predict(
                source=int(args.source) if args.source.isdigit() else args.source,
                stream=True,
                imgsz=args.imgsz,
                conf=args.physical_match_conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
            )
            for frame_index, result in enumerate(predictions):
                frame = result.orig_img.copy()
                height, width = frame.shape[:2]
                physical_candidates = _extract_cones(
                    result,
                    np,
                    cone_class=args.cone_class,
                    image_size=(width, height),
                    min_long_side_ratio=args.min_cone_long_side_ratio,
                )
                primary_detections = [
                    detection
                    for detection in physical_candidates
                    if detection.confidence >= args.conf
                ]
                creation_eligible = [
                    detection.confidence >= args.conf
                    for detection in physical_candidates
                ]
                contacts = [
                    detection.ground_contact for detection in physical_candidates
                ]
                images = [detection.center for detection in physical_candidates]
                motion = estimator.update(
                    frame,
                    [detection.polygon for detection in physical_candidates],
                )
                legacy = legacy_counter.update(
                    AnalysisResult(ordered=primary_detections)
                )
                physical = physical_counter.update(
                    contacts,
                    images,
                    motion,
                    can_create_physical_id=creation_eligible,
                )
                recovery_indices = set(physical.recovery_detection_indices)
                frame_count += 1
                detected_frames += int(bool(primary_detections))
                legacy_max_visible = max(legacy_max_visible, legacy.stable_visible_count)
                physical_max_visible = max(physical_max_visible, physical.stable_visible_count)
                legacy_new_id_events += len(legacy.new_cone_ids)
                physical_new_id_events += len(physical.new_physical_ids)
                recovery_candidate_count += creation_eligible.count(False)
                recovery_matched_count += len(recovery_indices)
                motion_methods[motion.method] += 1
                degraded_frames += int(physical.tracking_degraded)
                event = {
                    "frame_index": frame_index,
                    "video_time_seconds": frame_index / fps if fps > 0.0 else 0.0,
                    "detected_cone_count": len(primary_detections),
                    "physical_candidate_count": len(physical_candidates),
                    "legacy": {
                        "stable_visible_count": legacy.stable_visible_count,
                        "unique_seen_count": legacy.unique_seen_count,
                        "new_ids": list(legacy.new_cone_ids),
                        "active_ids": list(legacy.active_cone_ids),
                        "held_ids": list(legacy.held_cone_ids),
                    },
                    "physical": {
                        "stable_visible_count": physical.stable_visible_count,
                        "unique_seen_count": physical.unique_seen_count,
                        "new_ids": list(physical.new_physical_ids),
                        "active_ids": list(physical.active_physical_ids),
                        "held_ids": list(physical.held_physical_ids),
                        "detection_to_physical_id": list(physical.detection_to_physical_id),
                        "recovery_detection_indices": list(
                            physical.recovery_detection_indices
                        ),
                        "tracking_degraded": physical.tracking_degraded,
                    },
                    "motion": {
                        "valid": motion.valid,
                        "method": motion.method,
                        "points": motion.points,
                        "inliers": motion.inliers,
                        "inlier_ratio": motion.inlier_ratio,
                        "reprojection_error": motion.reprojection_error,
                    },
                    "detections": [
                        {
                            "center": detection.center,
                            "ground_contact": detection.ground_contact,
                            "confidence": detection.confidence,
                            "long_side_pixels": detection.long_axis_length,
                            "physical_id": (
                                physical.detection_to_physical_id[
                                    detection_index
                                ]
                            ),
                            "match_mode": (
                                "recovery"
                                if detection_index in recovery_indices
                                else "primary"
                            ),
                        }
                        for detection_index, detection in enumerate(
                            physical_candidates
                        )
                        if creation_eligible[detection_index]
                        or physical.detection_to_physical_id[detection_index]
                        is not None
                    ],
                }
                events_file.write(json.dumps(event, ensure_ascii=False) + "\n")
                if writer is not None:
                    cv2.putText(frame, f"legacy visible={legacy.stable_visible_count} total={legacy.unique_seen_count}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"physical visible={physical.stable_visible_count} total={physical.unique_seen_count} {'DEGRADED' if physical.tracking_degraded else ''}", (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
                    for detection_index, detection in enumerate(
                        physical_candidates
                    ):
                        physical_id = physical.detection_to_physical_id[
                            detection_index
                        ]
                        recovered = detection_index in recovery_indices
                        if not creation_eligible[detection_index] and not recovered:
                            continue
                        polygon = np.asarray(detection.polygon, dtype=np.int32).reshape((-1, 1, 2))
                        color = (80, 220, 80) if recovered else (0, 220, 0)
                        cv2.polylines(frame, [polygon], True, color, 2, cv2.LINE_AA)
                        label = (
                            f"RECOVER P{physical_id}"
                            if recovered
                            else f"P{physical_id or '-'}"
                        )
                        cv2.putText(frame, label, (round(detection.center[0]) + 6, round(detection.center[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                    writer.write(frame)
        finally:
            if writer is not None:
                writer.release()

    final_event = event if frame_count else {}
    summary = {
        "frames": frame_count,
        "detected_frames": detected_frames,
        "duration_seconds": round((frame_count - 1) / fps, 3) if frame_count and fps > 0.0 else 0.0,
        "legacy": final_event.get("legacy", {}),
        "physical": final_event.get("physical", {}),
        "comparison": {
            "legacy_max_stable_visible_count": legacy_max_visible,
            "physical_max_stable_visible_count": physical_max_visible,
            "legacy_new_id_events": legacy_new_id_events,
            "physical_new_id_events": physical_new_id_events,
            "physical_recovery_candidates": recovery_candidate_count,
            "physical_recovery_matched": recovery_matched_count,
        },
        "motion": {
            "degraded_frames": degraded_frames,
            "degraded_ratio": round(degraded_frames / frame_count, 4) if frame_count else 0.0,
            "methods": dict(sorted(motion_methods.items())),
        },
        "anomaly_detection": "not_run",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
