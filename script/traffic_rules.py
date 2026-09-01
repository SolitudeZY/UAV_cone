"""Geometry-only traffic-cone rules, independent of the inference runtime."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Sequence

from ground_projection import GroundPlaneProjector


Point = tuple[float, float]


def _distance(first: Point, second: Point) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _axis_difference(first: float, second: float) -> float:
    difference = abs((first - second) % 180.0)
    return min(difference, 180.0 - difference)


@dataclass(frozen=True)
class ConeDetection:
    polygon: tuple[Point, Point, Point, Point]
    confidence: float
    class_id: int = 0
    label: str = "cone"
    classifier_fallen: bool = False
    classifier_fallen_probability: float | None = None

    @property
    def center(self) -> Point:
        return (
            sum(point[0] for point in self.polygon) / 4.0,
            sum(point[1] for point in self.polygon) / 4.0,
        )

    @property
    def ground_contact(self) -> Point:
        """Bottom-edge midpoint used as the cone's ground-plane contact."""
        bottom = sorted(self.polygon, key=lambda point: point[1], reverse=True)[:2]
        return (
            (bottom[0][0] + bottom[1][0]) / 2.0,
            (bottom[0][1] + bottom[1][1]) / 2.0,
        )

    @property
    def long_axis_angle(self) -> float:
        """Long-axis angle in image coordinates, degrees in [0, 180)."""
        edges = []
        for index, point in enumerate(self.polygon):
            other = self.polygon[(index + 1) % 4]
            edges.append((_distance(point, other), point, other))
        _, first, second = max(edges, key=lambda edge: edge[0])
        return math.degrees(math.atan2(second[1] - first[1], second[0] - first[0])) % 180.0

    @property
    def long_axis_length(self) -> float:
        """Length of the longest OBB edge in image pixels."""
        return max(
            _distance(point, self.polygon[(index + 1) % 4])
            for index, point in enumerate(self.polygon)
        )

    @property
    def aspect_ratio(self) -> float:
        lengths = [
            _distance(point, self.polygon[(index + 1) % 4])
            for index, point in enumerate(self.polygon)
        ]
        short = min(lengths)
        return max(lengths) / short if short > 1e-9 else float("inf")


@dataclass(frozen=True)
class MissingAlert:
    after_index: int
    gap_pixels: float
    reference_gap_pixels: float
    estimated_count: int
    positions: tuple[Point, ...]
    gap_corrected: float | None = None
    reference_gap_corrected: float | None = None
    distance_unit: str = "pixels"


@dataclass(frozen=True)
class DisplacementAlert:
    index: int
    distance_pixels: float
    distance_meters: float | None
    distance_corrected: float | None = None
    distance_unit: str = "pixels"


@dataclass(frozen=True)
class FallenAlert:
    index: int
    angle_degrees: float
    difference_degrees: float
    angle_rule_positive: bool = True
    classifier_positive: bool = False
    classifier_probability: float | None = None


@dataclass(frozen=True)
class ShiftCornerExemption:
    corner_index: int
    protected_indices: tuple[int, ...]
    before_arm_indices: tuple[int, ...]
    after_arm_indices: tuple[int, ...]
    arm_angle_degrees: float
    before_residual_ratio: float
    after_residual_ratio: float
    vertex_distance_ratio: float
    distance_unit: str = "pixels"


@dataclass
class AnalysisResult:
    ordered: list[ConeDetection] = field(default_factory=list)
    routes: list[list[int]] = field(default_factory=list)
    median_gap_pixels: float | None = None
    median_gap_distance: float | None = None
    distance_unit: str = "pixels"
    ground_centers: list[Point | None] = field(default_factory=list)
    perspective_calibration: dict[str, float | str | None] | None = None
    missing: list[MissingAlert] = field(default_factory=list)
    missing_edge_ignored: list[int] = field(default_factory=list)
    displaced: list[DisplacementAlert] = field(default_factory=list)
    shift_corner_exemptions: list[ShiftCornerExemption] = field(default_factory=list)
    isolated: list[int] = field(default_factory=list)
    fallen: list[FallenAlert] = field(default_factory=list)
    edge_ignored: list[int] = field(default_factory=list)


def _principal_direction(points: Sequence[Point]) -> Point:
    if len(points) < 2:
        return (1.0, 0.0)
    mean_x = statistics.fmean(point[0] for point in points)
    mean_y = statistics.fmean(point[1] for point in points)
    covariance_xx = statistics.fmean((point[0] - mean_x) ** 2 for point in points)
    covariance_yy = statistics.fmean((point[1] - mean_y) ** 2 for point in points)
    covariance_xy = statistics.fmean(
        (point[0] - mean_x) * (point[1] - mean_y) for point in points
    )
    angle = 0.5 * math.atan2(2.0 * covariance_xy, covariance_xx - covariance_yy)
    return (math.cos(angle), math.sin(angle))


def order_along_route(detections: Sequence[ConeDetection]) -> list[ConeDetection]:
    points = [detection.center for detection in detections]
    direction = _principal_direction(points)
    return sorted(
        detections,
        key=lambda detection: (
            detection.center[0] * direction[0] + detection.center[1] * direction[1]
        ),
    )


def _local_reference_gap(gaps: Sequence[float], index: int, window: int) -> float | None:
    start = max(0, index - window)
    stop = min(len(gaps), index + window + 1)
    neighbors = [gaps[position] for position in range(start, stop) if position != index]
    if not neighbors:
        return None
    return statistics.median(neighbors)


def _line_distance(point: Point, neighbors: Sequence[Point]) -> float:
    mean_x = statistics.fmean(item[0] for item in neighbors)
    mean_y = statistics.fmean(item[1] for item in neighbors)
    direction = _principal_direction(neighbors)
    normal = (-direction[1], direction[0])
    return abs((point[0] - mean_x) * normal[0] + (point[1] - mean_y) * normal[1])


def _solve_3x3(matrix: Sequence[Sequence[float]], values: Sequence[float]) -> tuple[float, float, float] | None:
    augmented = [list(row) + [value] for row, value in zip(matrix, values)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-9:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(3):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                current - factor * pivot_value
                for current, pivot_value in zip(augmented[row], augmented[column])
            ]
    return tuple(augmented[row][3] for row in range(3))


def _quadratic_curve_distance(point: Point, neighbors: Sequence[Point]) -> float:
    """Normal residual from a local quadratic route fitted without the point."""
    if len(neighbors) < 4:
        return _line_distance(point, neighbors)
    mean_x = statistics.fmean(item[0] for item in neighbors)
    mean_y = statistics.fmean(item[1] for item in neighbors)
    direction = _principal_direction(neighbors)
    normal = (-direction[1], direction[0])
    coordinates = [
        (
            (item[0] - mean_x) * direction[0] + (item[1] - mean_y) * direction[1],
            (item[0] - mean_x) * normal[0] + (item[1] - mean_y) * normal[1],
        )
        for item in neighbors
    ]
    sums = {
        power: sum(longitudinal**power for longitudinal, _ in coordinates)
        for power in range(5)
    }
    rhs = [
        sum((longitudinal**power) * lateral for longitudinal, lateral in coordinates)
        for power in (2, 1, 0)
    ]
    coefficients = _solve_3x3(
        (
            (sums[4], sums[3], sums[2]),
            (sums[3], sums[2], sums[1]),
            (sums[2], sums[1], sums[0]),
        ),
        rhs,
    )
    if coefficients is None:
        return _line_distance(point, neighbors)
    longitudinal = (
        (point[0] - mean_x) * direction[0]
        + (point[1] - mean_y) * direction[1]
    )
    lateral = (
        (point[0] - mean_x) * normal[0]
        + (point[1] - mean_y) * normal[1]
    )
    quadratic, linear, intercept = coefficients
    expected = quadratic * longitudinal**2 + linear * longitudinal + intercept
    return abs(lateral - expected)


def _undirected_angle(first: Point, second: Point) -> float:
    cosine = abs(first[0] * second[0] + first[1] * second[1])
    return math.degrees(math.acos(min(1.0, max(-1.0, cosine))))


def _max_line_residual(points: Sequence[Point]) -> float:
    return max((_line_distance(point, points) for point in points), default=0.0)


def _supported_shift_corners(
    points: Sequence[Point],
    route_indices: Sequence[int],
    median_gap: float,
    *,
    neighbor_count: int,
    min_arm_cones: int,
    min_angle_degrees: float,
    max_arm_residual_ratio: float,
    max_vertex_distance_ratio: float,
    protect_neighbors: int,
    distance_unit: str,
) -> tuple[list[ShiftCornerExemption], set[int], tuple[int, ...]]:
    """Find well-supported route bends that should not be treated as SHIFT."""
    exemptions: list[ShiftCornerExemption] = []
    protected_local_indices: set[int] = set()
    for index in range(min_arm_cones, len(points) - min_arm_cones):
        before = points[max(0, index - neighbor_count) : index]
        after = points[index + 1 : index + neighbor_count + 1]
        if len(before) < min_arm_cones or len(after) < min_arm_cones:
            continue

        before_direction = _principal_direction(before)
        after_direction = _principal_direction(after)
        arm_angle = _undirected_angle(before_direction, after_direction)
        if arm_angle < min_angle_degrees:
            continue

        before_residual_ratio = _max_line_residual(before) / median_gap
        after_residual_ratio = _max_line_residual(after) / median_gap
        if (
            before_residual_ratio > max_arm_residual_ratio
            or after_residual_ratio > max_arm_residual_ratio
        ):
            continue

        vertex_distance_ratio = max(
            _line_distance(points[index], before),
            _line_distance(points[index], after),
        ) / median_gap
        if vertex_distance_ratio > max_vertex_distance_ratio:
            continue

        local_start = max(0, index - protect_neighbors)
        local_stop = min(len(points), index + protect_neighbors + 1)
        protected = tuple(route_indices[item] for item in range(local_start, local_stop))
        protected_local_indices.update(range(local_start, local_stop))
        exemptions.append(
            ShiftCornerExemption(
                corner_index=route_indices[index],
                protected_indices=protected,
                before_arm_indices=tuple(
                    route_indices[item]
                    for item in range(max(0, index - neighbor_count), index)
                ),
                after_arm_indices=tuple(
                    route_indices[item]
                    for item in range(
                        index + 1, min(len(points), index + neighbor_count + 1)
                    )
                ),
                arm_angle_degrees=arm_angle,
                before_residual_ratio=before_residual_ratio,
                after_residual_ratio=after_residual_ratio,
                vertex_distance_ratio=vertex_distance_ratio,
                distance_unit=distance_unit,
            )
        )
    return (
        exemptions,
        protected_local_indices,
        tuple(
            route_indices.index(exemption.corner_index) for exemption in exemptions
        ),
    )


def _same_arm_neighbor_indices(
    point_count: int,
    index: int,
    neighbor_count: int,
    corner_indices: Sequence[int],
) -> list[int]:
    """Return local fitting neighbors without crossing a supported corner."""
    left_boundary = 0
    right_boundary = point_count
    for corner_index in corner_indices:
        if corner_index < index:
            left_boundary = max(left_boundary, corner_index)
        elif corner_index > index:
            right_boundary = min(right_boundary, corner_index + 1)
    before = range(max(left_boundary, index - neighbor_count), index)
    after = range(index + 1, min(right_boundary, index + neighbor_count + 1))
    return [*before, *after]


def _adaptive_route_groups(
    detections: Sequence[ConeDetection],
    link_ratio: float,
    route_hints: Sequence[Sequence[Point]] | None = None,
    route_hint_ratio: float = 2.0,
) -> list[list[ConeDetection]]:
    """Group spatially connected cones using the frame local nearest-neighbor scale."""
    if not detections:
        return []
    if link_ratio <= 1.0:
        raise ValueError("route_link_ratio must be greater than 1")
    if len(detections) == 1:
        return [[detections[0]]]

    centers = [detection.center for detection in detections]
    nearest = [
        min(_distance(center, other) for other_index, other in enumerate(centers) if other_index != index)
        for index, center in enumerate(centers)
    ]
    positive_nearest = [distance for distance in nearest if distance > 1e-6]
    if not positive_nearest:
        return [order_along_route(detections)]
    typical_spacing = statistics.median(positive_nearest)

    adjacency: list[set[int]] = [set() for _ in detections]
    for first_index in range(len(detections)):
        for second_index in range(first_index + 1, len(detections)):
            local_spacing = min(
                max(nearest[first_index], nearest[second_index]),
                typical_spacing,
            )
            if _distance(centers[first_index], centers[second_index]) <= link_ratio * local_spacing:
                adjacency[first_index].add(second_index)
                adjacency[second_index].add(first_index)

    groups: list[list[ConeDetection]] = []
    visited: set[int] = set()
    for start in range(len(detections)):
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        indices: list[int] = []
        while stack:
            current = stack.pop()
            indices.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        groups.append(order_along_route([detections[index] for index in indices]))

    if route_hints:
        stabilized: list[list[ConeDetection]] = []
        maximum_hint_distance = typical_spacing * route_hint_ratio
        usable_hints = [hint for hint in route_hints if hint]
        for group in groups:
            hinted: dict[int, list[ConeDetection]] = {}
            group_residual: list[ConeDetection] = []
            for detection in group:
                distances = [
                    min(_distance(detection.center, point) for point in hint)
                    for hint in usable_hints
                ]
                if distances and min(distances) <= maximum_hint_distance:
                    hint_index = min(range(len(distances)), key=distances.__getitem__)
                    hinted.setdefault(hint_index, []).append(detection)
                else:
                    group_residual.append(detection)
            stabilized.extend(
                order_along_route(hinted[index]) for index in sorted(hinted)
            )
            if group_residual:
                stabilized.extend(
                    _adaptive_route_groups(group_residual, link_ratio)
                )
        groups = stabilized

    groups.sort(
        key=lambda group: (
            statistics.fmean(detection.center[0] for detection in group),
            statistics.fmean(detection.center[1] for detection in group),
        )
    )
    return groups


def _terminal_isolated_indices(
    detections: Sequence[ConeDetection], maximum_gap_ratio: float
) -> set[int]:
    """Find route endpoints both far from and off the supported route core."""
    if maximum_gap_ratio <= 1.0 or len(detections) < 4:
        return set()
    gaps = [
        _distance(first.center, second.center)
        for first, second in zip(detections, detections[1:])
    ]
    isolated: set[int] = set()
    for endpoint, gap_index in ((0, 0), (len(detections) - 1, len(gaps) - 1)):
        reference_gaps = [
            gap for index, gap in enumerate(gaps) if index != gap_index and gap > 1e-6
        ]
        if (
            reference_gaps
            and gaps[gap_index] > statistics.median(reference_gaps) * maximum_gap_ratio
            and _line_distance(
                detections[endpoint].center,
                [
                    detection.center
                    for detection in (
                        detections[1:] if endpoint == 0 else detections[:-1]
                    )
                ],
            )
            > statistics.median(reference_gaps) * 0.5
        ):
            isolated.add(endpoint)
    return isolated


def _touches_image_edge(
    detection: ConeDetection,
    image_size: tuple[int, int] | None,
    margin_ratio: float,
) -> bool:
    if image_size is None or margin_ratio <= 0:
        return False
    width, height = image_size
    margin = min(width, height) * margin_ratio
    return any(
        point[0] <= margin
        or point[1] <= margin
        or point[0] >= width - margin
        or point[1] >= height - margin
        for point in detection.polygon
    )


def analyze_cones(
    detections: Sequence[ConeDetection],
    *,
    gap_ratio: float = 1.8,
    gap_window: int = 5,
    offset_ratio: float = 0.25,
    offset_threshold_meters: float = 0.5,
    meters_per_pixel: float | None = None,
    neighbor_count: int = 5,
    shift_neighbor_count: int = 2,
    upright_angle: float = 90.0,
    fallen_angle: float = 30.0,
    min_aspect_ratio: float = 1.2,
    route_link_ratio: float = 3.0,
    route_hints: Sequence[Sequence[Point]] | None = None,
    route_hint_ratio: float = 2.0,
    min_route_cones: int = 2,
    image_size: tuple[int, int] | None = None,
    fallen_edge_margin_ratio: float = 0.05,
    missing_edge_margin_ratio: float = 0.02,
    shift_corner_exemption: bool = True,
    shift_corner_min_arm_cones: int = 2,
    shift_corner_min_angle_degrees: float = 45.0,
    shift_corner_max_arm_residual_ratio: float = 0.25,
    shift_corner_max_vertex_distance_ratio: float = 0.45,
    shift_corner_protect_neighbors: int = 1,
    isolated_endpoint_gap_ratio: float = 2.5,
    ground_projector: GroundPlaneProjector | None = None,
) -> AnalysisResult:
    if gap_ratio <= 1.0:
        raise ValueError("gap_ratio must be greater than 1")
    if meters_per_pixel is not None and meters_per_pixel <= 0:
        raise ValueError("meters_per_pixel must be positive")
    if ground_projector is not None and image_size is None:
        raise ValueError("image_size is required for ground projection")
    if ground_projector is not None and meters_per_pixel is not None:
        raise ValueError(
            "ground_projector and meters_per_pixel are mutually exclusive"
        )
    if route_link_ratio <= 1.0:
        raise ValueError("route_link_ratio must be greater than 1")
    if route_hint_ratio <= 0:
        raise ValueError("route_hint_ratio must be positive")
    if min_route_cones < 1:
        raise ValueError("min_route_cones must be at least 1")
    if shift_neighbor_count < 2:
        raise ValueError("shift_neighbor_count must be at least 2")
    if not 0.0 <= fallen_edge_margin_ratio < 0.5:
        raise ValueError("fallen_edge_margin_ratio must be in [0, 0.5)")
    if not 0.0 <= missing_edge_margin_ratio < 0.5:
        raise ValueError("missing_edge_margin_ratio must be in [0, 0.5)")
    if shift_corner_min_arm_cones < 2:
        raise ValueError("shift_corner_min_arm_cones must be at least 2")
    if not 0.0 < shift_corner_min_angle_degrees <= 90.0:
        raise ValueError("shift_corner_min_angle_degrees must be in (0, 90]")
    if shift_corner_max_arm_residual_ratio < 0.0:
        raise ValueError("shift_corner_max_arm_residual_ratio must be non-negative")
    if shift_corner_max_vertex_distance_ratio < 0.0:
        raise ValueError("shift_corner_max_vertex_distance_ratio must be non-negative")
    if shift_corner_protect_neighbors < 0:
        raise ValueError("shift_corner_protect_neighbors must be non-negative")
    if isolated_endpoint_gap_ratio <= 1.0:
        raise ValueError("isolated_endpoint_gap_ratio must be greater than 1")

    upright: list[ConeDetection] = []
    fallen: list[tuple[ConeDetection, float, bool]] = []
    edge_ignored: list[ConeDetection] = []
    for detection in detections:
        difference = _axis_difference(detection.long_axis_angle, upright_angle)
        angle_is_fallen = (
            difference >= fallen_angle
            and detection.aspect_ratio >= min_aspect_ratio
        )
        fused_is_fallen = angle_is_fallen or detection.classifier_fallen
        if fused_is_fallen and _touches_image_edge(
            detection, image_size, fallen_edge_margin_ratio
        ):
            edge_ignored.append(detection)
        elif fused_is_fallen:
            fallen.append((detection, difference, angle_is_fallen))
        else:
            upright.append(detection)

    result = AnalysisResult()
    for group in _adaptive_route_groups(
        upright, route_link_ratio, route_hints, route_hint_ratio
    ):
        start = len(result.ordered)
        result.ordered.extend(group)
        isolated_local_indices = _terminal_isolated_indices(
            group, isolated_endpoint_gap_ratio
        )
        route = [
            start + index
            for index in range(len(group))
            if index not in isolated_local_indices
        ]
        result.isolated.extend(start + index for index in isolated_local_indices)
        if len(route) >= min_route_cones:
            result.routes.append(route)
        else:
            result.isolated.extend(route)
    for detection, difference, angle_is_fallen in fallen:
        index = len(result.ordered)
        result.ordered.append(detection)
        result.fallen.append(
            FallenAlert(
                index,
                detection.long_axis_angle,
                difference,
                angle_rule_positive=angle_is_fallen,
                classifier_positive=detection.classifier_fallen,
                classifier_probability=detection.classifier_fallen_probability,
            )
        )
    for detection in edge_ignored:
        index = len(result.ordered)
        result.ordered.append(detection)
        result.edge_ignored.append(index)

    if ground_projector is not None:
        assert image_size is not None
        result.distance_unit = ground_projector.distance_unit
        result.perspective_calibration = ground_projector.metadata(image_size)
        result.ground_centers = [
            ground_projector.image_to_ground(detection.ground_contact, image_size)
            for detection in result.ordered
        ]
    else:
        result.ground_centers = [None for _ in result.ordered]

    all_positive_pixel_gaps: list[float] = []
    all_positive_distances: list[float] = []
    for route_indices in result.routes:
        route = [result.ordered[index] for index in route_indices]
        if len(route) < 2:
            continue
        pixel_gaps = [
            _distance(first.center, second.center)
            for first, second in zip(route, route[1:])
        ]
        positive_pixel_gaps = [gap for gap in pixel_gaps if gap > 1e-6]
        all_positive_pixel_gaps.extend(positive_pixel_gaps)
        if ground_projector is not None:
            projected = [result.ground_centers[index] for index in route_indices]
            if any(point is None for point in projected):
                continue
            measurement_centers = [point for point in projected if point is not None]
            gaps = [
                _distance(first, second)
                for first, second in zip(
                    measurement_centers, measurement_centers[1:]
                )
            ]
        else:
            measurement_centers = [detection.center for detection in route]
            gaps = pixel_gaps
        positive_gaps = [gap for gap in gaps if gap > 1e-6]
        if not positive_gaps:
            continue
        all_positive_distances.extend(positive_gaps)
        route_median_gap = statistics.median(positive_gaps)

        for gap_index, gap in enumerate(gaps):
            terminal_detections: list[ConeDetection] = []
            if gap_index == 0:
                terminal_detections.append(route[0])
            if gap_index == len(gaps) - 1:
                terminal_detections.append(route[-1])
            if any(
                _touches_image_edge(
                    detection, image_size, missing_edge_margin_ratio
                )
                for detection in terminal_detections
            ):
                result.missing_edge_ignored.append(route_indices[gap_index])
                continue

            reference = _local_reference_gap(gaps, gap_index, gap_window)
            if reference is None or reference <= 1e-6 or gap <= reference * gap_ratio:
                continue
            pixel_reference = _local_reference_gap(
                pixel_gaps, gap_index, gap_window
            )
            if pixel_reference is None:
                continue
            estimated_count = max(1, round(gap / reference) - 1)
            first = route[gap_index].center
            second = route[gap_index + 1].center
            if ground_projector is not None:
                assert image_size is not None
                first_ground = measurement_centers[gap_index]
                second_ground = measurement_centers[gap_index + 1]
                positions = tuple(
                    ground_projector.ground_to_image(
                        (
                            first_ground[0]
                            + (second_ground[0] - first_ground[0])
                            * step
                            / (estimated_count + 1),
                            first_ground[1]
                            + (second_ground[1] - first_ground[1])
                            * step
                            / (estimated_count + 1),
                        ),
                        image_size,
                    )
                    for step in range(1, estimated_count + 1)
                )
            else:
                positions = tuple(
                    (
                        first[0]
                        + (second[0] - first[0])
                        * step
                        / (estimated_count + 1),
                        first[1]
                        + (second[1] - first[1])
                        * step
                        / (estimated_count + 1),
                    )
                    for step in range(1, estimated_count + 1)
                )
            result.missing.append(
                MissingAlert(
                    route_indices[gap_index],
                    pixel_gaps[gap_index],
                    pixel_reference,
                    estimated_count,
                    positions,
                    gap_corrected=(gap if ground_projector is not None else None),
                    reference_gap_corrected=(
                        reference if ground_projector is not None else None
                    ),
                    distance_unit=result.distance_unit,
                )
            )

        if len(route) < 5:
            continue
        corner_protected_indices: set[int] = set()
        corner_local_indices: tuple[int, ...] = ()
        if shift_corner_exemption:
            (
                exemptions,
                corner_protected_indices,
                corner_local_indices,
            ) = _supported_shift_corners(
                measurement_centers,
                route_indices,
                route_median_gap,
                neighbor_count=neighbor_count,
                min_arm_cones=shift_corner_min_arm_cones,
                min_angle_degrees=shift_corner_min_angle_degrees,
                max_arm_residual_ratio=shift_corner_max_arm_residual_ratio,
                max_vertex_distance_ratio=shift_corner_max_vertex_distance_ratio,
                protect_neighbors=shift_corner_protect_neighbors,
                distance_unit=result.distance_unit,
            )
            result.shift_corner_exemptions.extend(exemptions)
        if ground_projector is not None:
            threshold_distance = (
                offset_threshold_meters
                if ground_projector.altitude_meters is not None
                else route_median_gap * offset_ratio
            )
        else:
            threshold_distance = (
                offset_threshold_meters / meters_per_pixel
                if meters_per_pixel is not None
                else route_median_gap * offset_ratio
            )
        for index in range(2, len(route) - 2):
            if index in corner_protected_indices:
                continue
            neighbor_indices = _same_arm_neighbor_indices(
                len(route), index, shift_neighbor_count, corner_local_indices
            )
            if len(neighbor_indices) < 4:
                continue
            neighbors = [route[item].center for item in neighbor_indices]
            distance_pixels = min(
                _line_distance(route[index].center, neighbors),
                _quadratic_curve_distance(route[index].center, neighbors),
            )
            corrected_neighbors = [measurement_centers[item] for item in neighbor_indices]
            distance_corrected = min(
                _line_distance(measurement_centers[index], corrected_neighbors),
                _quadratic_curve_distance(measurement_centers[index], corrected_neighbors),
            )
            # Leave a small geometric margin for fitting noise on smooth
            # curves; the temporal tracker still provides the business gate.
            effective_threshold = threshold_distance * 1.10
            if distance_corrected <= effective_threshold:
                continue
            distance_meters = (
                distance_corrected
                if ground_projector is not None
                and ground_projector.altitude_meters is not None
                else distance_pixels * meters_per_pixel
                if meters_per_pixel is not None
                else None
            )
            result.displaced.append(
                DisplacementAlert(
                    route_indices[index],
                    distance_pixels,
                    distance_meters,
                    distance_corrected=(
                        distance_corrected
                        if ground_projector is not None
                        else None
                    ),
                    distance_unit=result.distance_unit,
                )
            )

    if all_positive_pixel_gaps:
        result.median_gap_pixels = statistics.median(all_positive_pixel_gaps)
    if all_positive_distances:
        result.median_gap_distance = statistics.median(all_positive_distances)
    return result
