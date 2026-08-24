#!/usr/bin/env python3
"""Run OBB inference and traffic-cone spatial rules on images or video."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

from alert_logger import AlertJournal, format_review_alert
from cone_counter import ConeCounter, ConeCountingResult
from fallen_tracker import ConeTrackRegistry, FallenTrackingResult
from gap_tracker import GapThresholdTracker, GapTrackingResult
from ground_projection import GroundPlaneProjector
from route_tracker import (
    RouteTracker,
    RouteTrackingResult,
    observations_from_analysis,
)
from shift_tracker import ShiftTracker
from traffic_rules import AnalysisResult, ConeDetection, MissingAlert, analyze_cones


VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
DEFAULT_MIN_CONE_LONG_SIDE_RATIO = 24.0 / 1080.0


def _parse_roi(value: str) -> tuple[float, float, float, float]:
    try:
        coordinates = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI must contain four numbers: x1,y1,x2,y2") from exc
    if len(coordinates) != 4 or coordinates[0] >= coordinates[2] or coordinates[1] >= coordinates[3]:
        raise argparse.ArgumentTypeError("ROI must satisfy x1 < x2 and y1 < y2")
    return coordinates


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="Path to best.pt or last.pt")
    parser.add_argument("--source", required=True, help="Image, directory, video, stream URL, or camera index")
    parser.add_argument("--output", type=Path, default=project_root / "script" / "runs" / "infer")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default=0, help="For example 0, cpu, or mps")
    parser.add_argument("--cone-class", default="cone")
    parser.add_argument(
        "--min-cone-long-side-ratio",
        type=float,
        default=DEFAULT_MIN_CONE_LONG_SIDE_RATIO,
        help=(
            "Ignore cone OBBs whose long side is below this fraction of the "
            "frame's shorter side; 0 disables the filter"
        ),
    )
    parser.add_argument(
        "--classifier-weights",
        type=Path,
        help="Optional MobileNetV3 candidate-review checkpoint; omitted keeps legacy behavior",
    )
    parser.add_argument(
        "--classifier-thresholds",
        type=Path,
        help="Calibrated thresholds JSON; defaults to thresholds.json beside classifier weights",
    )
    parser.add_argument(
        "--classifier-mode",
        choices=("audit", "fallen", "full"),
        default="full",
        help=(
            "audit records decisions only; fallen adds posture votes; full also "
            "removes calibrated not_cone candidates before all traffic rules"
        ),
    )
    parser.add_argument("--classifier-device", help="Defaults to --device")
    parser.add_argument("--classifier-crop-context", type=float, default=1.6)
    parser.add_argument("--classifier-batch", type=int, default=64)
    parser.add_argument(
        "--roi",
        type=_parse_roi,
        default=None,
        help="Analyze one cone sequence inside x1,y1,x2,y2; detections outside remain visible",
    )
    parser.add_argument("--gap-ratio", type=float, default=1.8)
    parser.add_argument("--gap-window", type=int, default=5)
    parser.add_argument("--gap-history-frames", type=int, default=30)
    parser.add_argument("--gap-history-min-frames", type=int, default=3)
    parser.add_argument(
        "--gap-close-ratio",
        type=float,
        default=0.55,
        help="Exclude gaps below this ratio of the per-route historical nominal gap",
    )
    parser.add_argument(
        "--gap-max-drop-ratio",
        type=float,
        default=0.05,
        help="Maximum per-frame decrease of a route nominal gap",
    )
    parser.add_argument(
        "--gap-min-pixels",
        type=float,
        default=0.0,
        help="Legacy pixel-mode gap floor; ignored by perspective-corrected distances",
    )
    parser.add_argument("--gap-vote-window", type=int, default=5)
    parser.add_argument("--gap-confirm-votes", type=int, default=3)
    parser.add_argument(
        "--gap-alert-ratio",
        type=float,
        default=2.2,
        help="Strong MISSING ratio used to promote a yellow suspect to red",
    )
    parser.add_argument(
        "--gap-suspect-seconds",
        type=float,
        default=0.8,
        help="Credible yellow MISSING duration required before red promotion",
    )
    parser.add_argument("--gap-alert-confirm-frames", type=int, default=5)
    parser.add_argument(
        "--gap-max-ratio-disagreement",
        type=float,
        default=0.6,
        help="Maximum relative disagreement between pixel and corrected gap ratios",
    )
    parser.add_argument("--gap-clear-frames", type=int, default=3)
    parser.add_argument(
        "--gap-membership-cooldown",
        type=int,
        default=5,
        help="Frames (3-5) to suppress gaps adjacent to route member changes",
    )
    parser.add_argument(
        "--route-link-ratio",
        type=float,
        default=3.0,
        help="Adaptive ROI link distance as a multiple of local nearest-neighbor spacing",
    )
    parser.add_argument("--route-hint-ratio", type=float, default=2.0)
    parser.add_argument("--min-route-cones", type=int, default=2)
    parser.add_argument(
        "--min-alert-cones",
        type=int,
        default=3,
        help="Minimum current cones in a route before business alerts are allowed",
    )
    parser.add_argument(
        "--gap-history-min-cones",
        type=int,
        default=5,
        help="Minimum cones in a stable route before its nominal gap history may update",
    )
    parser.add_argument(
        "--roi-memory-frames",
        type=int,
        default=5,
        help="Frames to retain each independently missing confirmed route",
    )
    parser.add_argument("--route-confirm-frames", type=int, default=3)
    parser.add_argument("--route-recovery-frames", type=int, default=2)
    parser.add_argument("--route-match-ratio", type=float, default=2.5)
    parser.add_argument("--route-max-angle", type=float, default=45.0)
    parser.add_argument("--roi-smoothing", type=float, default=0.35)
    parser.add_argument("--roi-max-smoothing", type=float, default=0.75)
    parser.add_argument("--no-temporal-roi", action="store_true")
    parser.add_argument(
        "--camera-pitch-deg",
        type=float,
        default=None,
        help="Camera optical-axis depression below horizontal; requires --camera-hfov-deg",
    )
    parser.add_argument(
        "--camera-hfov-deg",
        type=float,
        default=None,
        help="Camera horizontal field of view; requires --camera-pitch-deg",
    )
    parser.add_argument(
        "--drone-altitude-m",
        type=float,
        default=None,
        help="Camera height above the cone ground plane; enables metric distances",
    )
    parser.add_argument(
        "--no-perspective-correction",
        action="store_true",
        help="Explicitly keep pixel distances; perspective correction is off by default",
    )
    parser.add_argument(
        "--meters-per-pixel",
        type=float,
        default=None,
        help="Ground calibration at this view; enables metric displacement threshold",
    )
    parser.add_argument("--offset-threshold-m", type=float, default=0.5)
    parser.add_argument(
        "--offset-ratio",
        type=float,
        default=0.25,
        help="Uncalibrated fallback: displacement / median cone spacing",
    )
    parser.add_argument(
        "--upright-angle",
        type=float,
        default=90.0,
        help="Expected upright cone long-axis angle in image coordinates",
    )
    parser.add_argument("--fallen-angle", type=float, default=30.0)
    parser.add_argument(
        "--fallen-reactivation-cooldown-frames",
        type=int,
        default=30,
        help="Frames to suppress reactivation of the same cleared FALLEN cone",
    )
    parser.add_argument(
        "--fallen-vote-window",
        type=int,
        default=7,
        help="Recent observations used to confirm one stable FALLEN cone",
    )
    parser.add_argument(
        "--fallen-confirm-votes",
        type=int,
        default=5,
        help="FALLEN votes required after the complete vote window is observed",
    )
    parser.add_argument(
        "--fallen-match-distance-ratio",
        type=float,
        default=0.6,
        help="Maximum center motion as a multiple of local cone spacing",
    )
    parser.add_argument(
        "--fallen-max-angle-jump",
        type=float,
        default=20.0,
        help="Maximum long-axis angle change between accepted FALLEN votes",
    )
    parser.add_argument("--fallen-edge-margin", type=float, default=0.05)
    parser.add_argument(
        "--cone-count-confirm-window",
        type=int,
        default=5,
        help="Recent observations used to confirm one cone before counting it",
    )
    parser.add_argument(
        "--cone-count-confirm-votes",
        type=int,
        default=3,
        help="Positive observations required within the cone count window",
    )
    parser.add_argument(
        "--cone-count-max-missed-frames",
        type=int,
        default=5,
        help="Keep a confirmed cone track through this many missed frames",
    )
    parser.add_argument(
        "--cone-count-match-distance-ratio",
        type=float,
        default=0.8,
        help="Matching distance as a multiple of local cone spacing",
    )
    parser.add_argument(
        "--missing-edge-margin",
        type=float,
        default=0.02,
        help="Suppress terminal missing gaps when the outer cone is this close to an image edge",
    )
    parser.add_argument(
        "--no-shift-corner-exemption",
        action="store_true",
        help="Disable supported route-corner exemption for SHIFT diagnostics",
    )
    parser.add_argument("--shift-corner-min-arm-cones", type=int, default=2)
    parser.add_argument("--shift-corner-min-angle", type=float, default=45.0)
    parser.add_argument(
        "--shift-corner-max-arm-residual-ratio", type=float, default=0.25
    )
    parser.add_argument(
        "--shift-corner-max-vertex-distance-ratio", type=float, default=0.45
    )
    parser.add_argument("--shift-corner-protect-neighbors", type=int, default=1)
    parser.add_argument(
        "--shift-neighbor-count",
        type=int,
        default=2,
        help="Nearest cones used on each side of a SHIFT candidate",
    )
    parser.add_argument(
        "--isolated-endpoint-gap-ratio",
        type=float,
        default=2.5,
        help="Exclude a route endpoint whose gap exceeds this multiple of the route core",
    )
    parser.add_argument(
        "--shift-confirm-frames",
        type=int,
        default=3,
        help="Consecutive route-level SHIFT frames required for a business alert",
    )
    parser.add_argument(
        "--shift-membership-stable-frames",
        type=int,
        default=10,
        help="Frames route membership must remain unchanged before SHIFT confirmation",
    )
    parser.add_argument(
        "--exclude-classes",
        default="person",
        help="Comma-separated classes from the same OBB model whose boxes suppress cone centers",
    )
    parser.add_argument(
        "--person-weights",
        default=None,
        help="Optional COCO person detector weights used to suppress cones inside person boxes",
    )
    parser.add_argument("--person-conf", type=float, default=0.25)
    parser.add_argument(
        "--person-classes",
        default="0",
        help="Comma-separated exclusion class IDs; use 0,2 for the project PPE model",
    )
    parser.add_argument("--person-imgsz", type=int, default=640)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    return parser


def require_runtime():
    try:
        import cv2
        import numpy as np
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing inference dependency. Run: pip install -r script/requirements.txt"
        ) from exc
    return cv2, np, YOLO


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _inside_any_box(
    point: tuple[float, float], boxes: Sequence[Sequence[float]]
) -> bool:
    return any(
        box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]
        for box in boxes
    )


def _same_model_exclusion_boxes(
    result: Any, excluded_classes: set[str]
) -> list[tuple[float, float, float, float]]:
    if not excluded_classes or result.obb is None or len(result.obb) == 0:
        return []
    polygons = result.obb.xyxyxyxy.detach().cpu().numpy()
    class_ids = result.obb.cls.detach().cpu().numpy().astype(int)
    boxes: list[tuple[float, float, float, float]] = []
    for polygon, class_id in zip(polygons, class_ids):
        if _class_name(result.names, int(class_id)) not in excluded_classes:
            continue
        boxes.append(
            (
                float(polygon[:, 0].min()),
                float(polygon[:, 1].min()),
                float(polygon[:, 0].max()),
                float(polygon[:, 1].max()),
            )
        )
    return boxes


def _extract_detections(
    result: Any,
    cone_class: str,
    roi: tuple[float, float, float, float] | None,
    exclusion_boxes: Sequence[Sequence[float]] = (),
    image_size: tuple[int, int] | None = None,
    min_cone_long_side_ratio: float = 0.0,
    candidate_classifier: Any | None = None,
    frame: Any | None = None,
    classifier_mode: str = "full",
) -> tuple[list[ConeDetection], list[dict[str, Any]], int, int]:
    if not 0.0 <= min_cone_long_side_ratio < 1.0:
        raise ValueError("min_cone_long_side_ratio must be in [0, 1)")
    if classifier_mode not in {"audit", "fallen", "full"}:
        raise ValueError("classifier_mode must be audit, fallen, or full")
    if candidate_classifier is not None and frame is None:
        raise ValueError("frame is required when candidate classification is enabled")
    cones: list[ConeDetection] = []
    all_detections: list[dict[str, Any]] = []
    person_suppressed = 0
    small_cone_suppressed = 0
    obb = result.obb
    if obb is None or len(obb) == 0:
        return cones, all_detections, person_suppressed, small_cone_suppressed
    minimum_long_side = (
        min(image_size) * min_cone_long_side_ratio
        if image_size is not None
        else 0.0
    )
    polygons = obb.xyxyxyxy.detach().cpu().numpy()
    confidences = obb.conf.detach().cpu().numpy()
    class_ids = obb.cls.detach().cpu().numpy().astype(int)
    prepared: list[tuple[dict[str, Any], ConeDetection | None]] = []
    review_record_indices: list[int] = []
    review_polygons: list[Any] = []
    for polygon_array, confidence, class_id in zip(polygons, confidences, class_ids):
        polygon = tuple((float(point[0]), float(point[1])) for point in polygon_array)
        if len(polygon) != 4:
            continue
        label = _class_name(result.names, int(class_id))
        item = {
            "polygon": polygon,
            "confidence": float(confidence),
            "class_id": int(class_id),
            "label": label,
        }
        if label == cone_class:
            cone = ConeDetection(**item)
            center_x, center_y = cone.center
            if _inside_any_box((center_x, center_y), exclusion_boxes):
                person_suppressed += 1
                continue
            item["long_side_pixels"] = cone.long_axis_length
            item["minimum_long_side_pixels"] = minimum_long_side
            item["postprocess_status"] = "accepted"
            if cone.long_axis_length < minimum_long_side:
                item["postprocess_status"] = "too_small"
                small_cone_suppressed += 1
                prepared.append((item, None))
                continue
            prepared.append((item, cone))
            if candidate_classifier is not None:
                review_record_indices.append(len(prepared) - 1)
                review_polygons.append(polygon_array)
        else:
            prepared.append((item, None))

    reviews_by_record: dict[int, dict[str, Any]] = {}
    if candidate_classifier is not None and review_polygons:
        reviews = candidate_classifier.classify(frame, review_polygons)
        if len(reviews) != len(review_record_indices):
            raise RuntimeError("Classifier returned a different number of reviews")
        reviews_by_record = dict(zip(review_record_indices, reviews))
    elif candidate_classifier is not None:
        candidate_classifier.last_timing_ms = {
            "crop": 0.0,
            "inference": 0.0,
            "total": 0.0,
        }

    for record_index, (item, cone) in enumerate(prepared):
        review = reviews_by_record.get(record_index)
        if review is not None:
            review = dict(review)
            item["classifier"] = review
            rejected = classifier_mode == "full" and review["not_cone_rejected"]
            fallen_applied = (
                classifier_mode in {"fallen", "full"}
                and review["fallen_vote"]
                and not rejected
            )
            review["fallen_vote_applied"] = fallen_applied
            if rejected:
                review["action"] = "reject_not_cone"
                item["postprocess_status"] = "classifier_not_cone"
            elif fallen_applied:
                review["action"] = "fallen_vote"
            elif classifier_mode == "audit":
                review["action"] = "audit_only"
            else:
                review["action"] = "accept"
            if rejected:
                all_detections.append(item)
                continue
            assert cone is not None
            cone = ConeDetection(
                polygon=cone.polygon,
                confidence=cone.confidence,
                class_id=cone.class_id,
                label=cone.label,
                classifier_fallen=fallen_applied,
                classifier_fallen_probability=review["probabilities"]["fallen_cone"],
            )
        all_detections.append(item)
        if cone is not None:
            center_x, center_y = cone.center
            if roi is None or (
                roi[0] <= center_x <= roi[2] and roi[1] <= center_y <= roi[3]
            ):
                cones.append(cone)
    return cones, all_detections, person_suppressed, small_cone_suppressed


def _point(point: tuple[float, float]) -> tuple[int, int]:
    return (round(point[0]), round(point[1]))


def _route_bounds(analysis: AnalysisResult, route: list[int]) -> tuple[float, float, float, float]:
    points = [
        point
        for index in route
        for point in analysis.ordered[index].polygon
    ]
    return (
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    )


def _route_id(analysis: AnalysisResult, detection_index: int) -> int | None:
    return next(
        (route_id for route_id, route in enumerate(analysis.routes) if detection_index in route),
        None,
    )


def _stable_route_id(
    analysis: AnalysisResult,
    detection_index: int,
    tracking: RouteTrackingResult | None,
) -> int | None:
    if tracking is None:
        return None
    route_id = _route_id(analysis, detection_index)
    if route_id is not None:
        track = next(
            (
                item
                for item in tracking.tracks
                if item.current_route_index == route_id
            ),
            None,
        )
        return track.business_stable_id if track is not None else None
    if not 0 <= detection_index < len(analysis.ordered):
        return None
    point = analysis.ordered[detection_index].center
    candidates: list[tuple[float, int]] = []
    for track in tracking.tracks:
        if track.state == "tentative" or not track.centers:
            continue
        distance = min(
            ((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2) ** 0.5
            for center in track.centers
        )
        if distance <= max(20.0, track.spacing * 2.5):
            candidates.append((distance, track.business_stable_id))
    return min(candidates)[1] if candidates else None


def _alert_allowed(
    analysis: AnalysisResult,
    detection_index: int,
    tracking: RouteTrackingResult | None,
    min_alert_cones: int = 3,
) -> bool:
    if tracking is None:
        route_id = _route_id(analysis, detection_index)
        return route_id is None or len(analysis.routes[route_id]) >= min_alert_cones
    route_id = _route_id(analysis, detection_index)
    if route_id is None:
        return False
    track = next(
        (
            item
            for item in tracking.tracks
            if item.current_route_index == route_id and item.alert_ready
        ),
        None,
    )
    if track is None or track.current_route_index is None:
        return False
    return len(analysis.routes[track.current_route_index]) >= min_alert_cones


def _draw_detections(
    cv2: Any,
    np: Any,
    frame: Any,
    detections: Sequence[dict[str, Any]],
) -> Any:
    for item in detections:
        polygon = np.asarray(item["polygon"], dtype=np.int32).reshape((-1, 1, 2))
        if item.get("postprocess_status") == "too_small":
            color = (160, 160, 160)
            caption = "SMALL"
        elif item.get("postprocess_status") == "classifier_not_cone":
            color = (0, 165, 255)
            probability = item["classifier"]["probabilities"]["not_cone"]
            caption = f"REJECT N={probability:.2f}"
        else:
            color = (255, 180, 0) if item["label"] != "cone" else (50, 210, 50)
            caption = f"{item['label']} {item['confidence']:.2f}"
            if item.get("classifier") is not None:
                review = item["classifier"]
                probabilities = review["probabilities"]
                class_code = {
                    "fallen_cone": "F",
                    "not_cone": "N",
                    "upright_cone": "U",
                }[review["argmax_class"]]
                caption = (
                    f"M:{class_code} "
                    f"{probabilities[review['argmax_class']]:.2f}"
                )
                if review["fallen_vote_applied"]:
                    color = (255, 0, 255)
        cv2.polylines(frame, [polygon], True, color, 2, cv2.LINE_AA)
        first = tuple(int(value) for value in polygon[0, 0])
        cv2.putText(
            frame,
            caption,
            (first[0], max(18, first[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return frame


def _draw_frame(
    cv2: Any,
    np: Any,
    frame: Any,
    detections: list[dict[str, Any]],
    analysis: AnalysisResult,
    roi: tuple[float, float, float, float] | None,
    tracking: RouteTrackingResult | None = None,
    gap_tracking: GapTrackingResult | None = None,
    min_alert_cones: int = 3,
    confirmed_shift_route_ids: frozenset[int] | None = None,
    fallen_tracking: FallenTrackingResult | None = None,
    cone_counting: ConeCountingResult | None = None,
) -> Any:
    fallen_indices = {
        observation.index
        for observation in (
            fallen_tracking.active if fallen_tracking is not None else ()
        )
    }
    displaced_indices = {
        alert.index
        for alert in analysis.displaced
        if _alert_allowed(analysis, alert.index, tracking, min_alert_cones)
        and (
            confirmed_shift_route_ids is None
            or _stable_route_id(analysis, alert.index, tracking)
            in confirmed_shift_route_ids
        )
    }
    edge_ignored_indices = set(analysis.edge_ignored)
    isolated_indices = set(analysis.isolated)

    _draw_detections(cv2, np, frame, detections)

    route_colors = ((0, 220, 255), (255, 80, 80), (80, 255, 120), (255, 120, 255))
    if tracking is None:
        drawable_routes = [
            (route_id + 1, "confirmed", _route_bounds(analysis, route), route)
            for route_id, route in enumerate(analysis.routes)
        ]
    else:
        drawable_routes = [
            (
                track.business_stable_id,
                track.state,
                track.roi,
                (
                    analysis.routes[track.current_route_index]
                    if track.matched
                    and track.current_route_index is not None
                    and track.current_route_index < len(analysis.routes)
                    else []
                ),
            )
            for track in tracking.tracks
            if track.state != "tentative"
        ]
    for stable_id, state, bounds, route in drawable_routes:
        color = route_colors[(stable_id - 1) % len(route_colors)]
        centers = [analysis.ordered[index].center for index in route]
        for first, second in zip(centers, centers[1:]):
            cv2.line(frame, _point(first), _point(second), color, 2, cv2.LINE_AA)
        if route or tracking is not None:
            x1, y1, x2, y2 = bounds
            padding = 8
            top_left = _point((max(0, x1 - padding), max(0, y1 - padding)))
            bottom_right = _point((x2 + padding, y2 + padding))
            cv2.rectangle(frame, top_left, bottom_right, color, 1, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"AUTO ROI {stable_id}{' HOLD' if state == 'held' else ''}",
                (top_left[0], max(16, top_left[1] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    for index, detection in enumerate(analysis.ordered):
        center = _point(detection.center)
        color = (
            (160, 160, 160)
            if index in edge_ignored_indices or index in isolated_indices
            else (0, 0, 255)
            if index in fallen_indices or index in displaced_indices
            else (0, 255, 255)
        )
        cv2.circle(frame, center, 5, color, -1, cv2.LINE_AA)
        if index in edge_ignored_indices:
            cv2.putText(frame, "EDGE SKIP", (center[0] + 7, center[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        elif index in isolated_indices:
            cv2.putText(frame, "ISOLATED", (center[0] + 7, center[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        elif index in fallen_indices:
            cv2.putText(frame, "FALLEN", (center[0] + 7, center[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        elif index in displaced_indices:
            cv2.putText(frame, "SHIFT", (center[0] + 7, center[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        if cone_counting is not None:
            stable_id = cone_counting.index_to_stable_id.get(index)
            if stable_id is not None:
                cv2.putText(
                    frame,
                    f"ID {stable_id}",
                    (center[0] + 7, center[1] + 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    color,
                    1,
                    cv2.LINE_AA,
                )

    suspect_alerts = gap_tracking.suspect if gap_tracking is not None else ()
    for alert in suspect_alerts:
        for position in alert.positions:
            center = _point(position)
            cv2.circle(frame, center, 9, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "MISSING?", (center[0] + 10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

    missing_alerts = (
        gap_tracking.missing if gap_tracking is not None else analysis.missing
    )
    for alert in missing_alerts:
        if gap_tracking is None and not _alert_allowed(
            analysis, alert.after_index, tracking, min_alert_cones
        ):
            continue
        for position in alert.positions:
            center = _point(position)
            cv2.circle(frame, center, 9, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "MISSING", (center[0] + 10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
    if roi is not None:
        cv2.rectangle(frame, _point((roi[0], roi[1])), _point((roi[2], roi[3])), (255, 255, 0), 2)
    if cone_counting is not None:
        cv2.putText(
            frame,
            (
                f"CONE visible={cone_counting.stable_visible_count} "
                f"total={cone_counting.unique_seen_count}"
            ),
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return frame


def _events(
    frame_index: int,
    analysis: AnalysisResult,
    detections: list[dict[str, Any]],
    roi: tuple[float, float, float, float] | None,
    person_suppressed: int,
    tracking: RouteTrackingResult | None = None,
    gap_tracking: GapTrackingResult | None = None,
    video_time_seconds: float | None = None,
    min_alert_cones: int = 3,
    confirmed_shift_route_ids: frozenset[int] | None = None,
    shift_confirm_frames: int = 1,
    shift_membership_stable_frames: int = 1,
    fallen_tracking: FallenTrackingResult | None = None,
    cone_counting: ConeCountingResult | None = None,
    small_cone_suppressed: int = 0,
    classifier_policy: dict[str, Any] | None = None,
    classifier_timing_ms: dict[str, float] | None = None,
) -> dict[str, Any]:
    def missing_event(
        alert: MissingAlert, stable_gap_id: int | None = None
    ) -> dict[str, Any]:
        event = {
            "after_index": alert.after_index,
            "route_id": _route_id(analysis, alert.after_index),
            "stable_route_id": _stable_route_id(
                analysis, alert.after_index, tracking
            ),
            "gap_pixels": alert.gap_pixels,
            "reference_gap_pixels": alert.reference_gap_pixels,
            "estimated_count": alert.estimated_count,
            "positions": alert.positions,
            "gap_corrected": alert.gap_corrected,
            "reference_gap_corrected": alert.reference_gap_corrected,
            "distance_unit": alert.distance_unit,
        }
        if stable_gap_id is not None:
            event["stable_gap_id"] = stable_gap_id
        return event

    missing_events = [missing_event(alert) for alert in analysis.missing]
    stable_missing_events = (
        [
            missing_event(alert, stable_gap_id)
            for alert, stable_gap_id in zip(
                gap_tracking.missing, gap_tracking.missing_gap_ids
            )
        ]
        if gap_tracking is not None
        else [
            event
            for alert, event in zip(analysis.missing, missing_events)
            if _alert_allowed(
                analysis, alert.after_index, tracking, min_alert_cones
            )
        ]
    )
    suspect_missing_events = (
        [
            missing_event(alert, stable_gap_id)
            for alert, stable_gap_id in zip(
                gap_tracking.suspect, gap_tracking.suspect_gap_ids
            )
        ]
        if gap_tracking is not None
        else []
    )
    displaced_events = [
        {
            "index": alert.index,
            "route_id": _route_id(analysis, alert.index),
            "stable_route_id": _stable_route_id(analysis, alert.index, tracking),
            "distance_pixels": alert.distance_pixels,
            "distance_meters": alert.distance_meters,
            "distance_corrected": alert.distance_corrected,
            "distance_unit": alert.distance_unit,
        }
        for alert in analysis.displaced
    ]
    fallen_events = [
        {
            "index": alert.index,
            "stable_cone_id": None,
            "stable_route_id": _stable_route_id(analysis, alert.index, tracking),
            "obb_axis_angle_degrees": alert.angle_degrees,
            "upright_difference_degrees": alert.difference_degrees,
            "angle_rule_positive": alert.angle_rule_positive,
            "classifier_positive": alert.classifier_positive,
            "classifier_fallen_probability": alert.classifier_probability,
        }
        for alert in analysis.fallen
    ]
    fallen_by_index = {alert.index: alert for alert in analysis.fallen}
    stable_fallen_events = [
        {
            "index": observation.index,
            "stable_cone_id": observation.stable_cone_id,
            "stable_route_id": _stable_route_id(
                analysis, observation.index, tracking
            ),
            "obb_axis_angle_degrees": observation.angle_degrees,
            "upright_difference_degrees": observation.difference_degrees,
            "angle_rule_positive": (
                fallen_by_index[observation.index].angle_rule_positive
                if observation.index in fallen_by_index
                else None
            ),
            "classifier_positive": (
                fallen_by_index[observation.index].classifier_positive
                if observation.index in fallen_by_index
                else None
            ),
            "classifier_fallen_probability": (
                fallen_by_index[observation.index].classifier_probability
                if observation.index in fallen_by_index
                else None
            ),
        }
        for observation in (fallen_tracking.active if fallen_tracking is not None else ())
        if observation.index < 0 or 0 <= observation.index < len(analysis.ordered)
    ]
    return {
        "frame_index": frame_index,
        "timestamp_seconds": time.time(),
        "video_time_seconds": video_time_seconds,
        "distance_unit": analysis.distance_unit,
        "perspective_calibration": analysis.perspective_calibration,
        "cone_count": len(analysis.ordered),
        "stable_visible_count": (
            cone_counting.stable_visible_count if cone_counting is not None else None
        ),
        "unique_seen_count": (
            cone_counting.unique_seen_count if cone_counting is not None else None
        ),
        "new_cone_ids": (
            list(cone_counting.new_cone_ids) if cone_counting is not None else []
        ),
        "active_cone_ids": (
            list(cone_counting.active_cone_ids) if cone_counting is not None else []
        ),
        "held_cone_ids": (
            list(cone_counting.held_cone_ids) if cone_counting is not None else []
        ),
        "stable_cone_ids": (
            list(cone_counting.stable_cone_ids) if cone_counting is not None else []
        ),
        "tracked_cones": (
            list(cone_counting.tracked_cones) if cone_counting is not None else []
        ),
        "cone_counting_policy": (
            cone_counting.policy if cone_counting is not None else None
        ),
        "roi": roi,
        "person_suppressed": person_suppressed,
        "small_cone_suppressed": small_cone_suppressed,
        "classifier_policy": classifier_policy,
        "classifier_timing_ms": classifier_timing_ms,
        "classifier_counts": {
            "reviewed": sum("classifier" in item for item in detections),
            "not_cone_votes": sum(
                bool(item.get("classifier", {}).get("not_cone_rejected"))
                for item in detections
            ),
            "not_cone_suppressed": sum(
                item.get("postprocess_status") == "classifier_not_cone"
                for item in detections
            ),
            "fallen_votes": sum(
                bool(item.get("classifier", {}).get("fallen_vote"))
                for item in detections
            ),
            "fallen_votes_applied": sum(
                bool(item.get("classifier", {}).get("fallen_vote_applied"))
                for item in detections
            ),
        },
        "detections": [
            {
                **item,
                "polygon": [list(point) for point in item["polygon"]],
                "center": [
                    sum(point[0] for point in item["polygon"]) / 4.0,
                    sum(point[1] for point in item["polygon"]) / 4.0,
                ],
            }
            for item in detections
        ],
        "ordered_cone_centers": [detection.center for detection in analysis.ordered],
        "ordered_cone_ground_contacts": [
            detection.ground_contact for detection in analysis.ordered
        ],
        "ordered_cone_ground_centers": analysis.ground_centers,
        "adaptive_routes": [
            {
                "route_id": route_id,
                "stable_route_id": (
                    tracking.route_to_stable_id.get(route_id)
                    if tracking is not None
                    else None
                ),
                "indices": route,
                "roi": _route_bounds(analysis, route),
                "centers": [analysis.ordered[index].center for index in route],
            }
            for route_id, route in enumerate(analysis.routes)
        ],
        "tracked_routes": (
            [
                {
                    "stable_route_id": track.business_stable_id,
                    "parent_stable_id": track.parent_stable_id,
                    "state": track.state,
                    "matched": track.matched,
                    "alert_ready": track.alert_ready,
                    "current_route_id": track.current_route_index,
                    "roi": track.roi,
                    "raw_roi": track.raw_roi,
                    "centers": track.centers,
                    "hits": track.hits,
                    "hit_streak": track.hit_streak,
                    "age": track.age,
                    "missed_frames": track.missed_frames,
                    "velocity": track.velocity,
                }
                for track in tracking.tracks
            ]
            if tracking is not None
            else []
        ),
        "median_gap_pixels": analysis.median_gap_pixels,
        "median_gap_distance": analysis.median_gap_distance,
        "gap_alert_policy": (
            {
                "suspect_ratio": gap_tracking.policy.suspect_ratio,
                "alert_ratio": gap_tracking.policy.alert_ratio,
                "vote_window": gap_tracking.policy.vote_window,
                "confirm_votes": gap_tracking.policy.confirm_votes,
                "suspect_frames": gap_tracking.policy.suspect_frames,
                "alert_confirm_frames": gap_tracking.policy.alert_confirm_frames,
                "clear_frames": gap_tracking.policy.clear_frames,
                "max_ratio_disagreement": (
                    gap_tracking.policy.max_ratio_disagreement
                ),
            }
            if gap_tracking is not None and gap_tracking.policy is not None
            else None
        ),
        "fallen_alert_policy": (
            fallen_tracking.policy if fallen_tracking is not None else None
        ),
        "gap_thresholds": (
            [
                {
                    "stable_route_id": route.stable_id,
                    "current_route_id": route.current_route_index,
                    "nominal_gap_pixels": (
                        route.nominal_gap_pixels
                        if route.distance_unit == "pixels"
                        else None
                    ),
                    "frame_average_pixels": (
                        route.frame_average_pixels
                        if route.distance_unit == "pixels"
                        else None
                    ),
                    "average_of_averages_pixels": (
                        route.average_of_averages_pixels
                        if route.distance_unit == "pixels"
                        else None
                    ),
                    "history_size": route.history_size,
                    "confidence": route.confidence,
                    "excluded_close_gaps": route.excluded_close_gaps,
                    "threshold_source": route.threshold_source,
                    "distance_unit": route.distance_unit,
                    "nominal_gap_corrected": (
                        route.nominal_gap_pixels
                        if route.distance_unit != "pixels"
                        else None
                    ),
                    "frame_average_corrected": (
                        route.frame_average_pixels
                        if route.distance_unit != "pixels"
                        else None
                    ),
                    "average_of_averages_corrected": (
                        route.average_of_averages_pixels
                        if route.distance_unit != "pixels"
                        else None
                    ),
                }
                for route in gap_tracking.routes
            ]
            if gap_tracking is not None
            else []
        ),
        "tracked_gaps": (
            [
                {
                    "stable_gap_id": gap.stable_id,
                    "stable_route_id": gap.stable_route_id,
                    "after_index": gap.after_index,
                    "endpoint_ids": gap.endpoint_ids,
                    "candidate_missing": gap.candidate_missing,
                    "active_missing": gap.active_missing,
                    "positive_votes": gap.positive_votes,
                    "vote_count": gap.vote_count,
                    "normal_streak": gap.normal_streak,
                    "cooldown_remaining": gap.cooldown_remaining,
                    "business_eligible": gap.business_eligible,
                    "alert_level": gap.alert_level,
                    "suspect_frames": gap.suspect_frames,
                    "strong_alert_streak": gap.strong_alert_streak,
                    "distance_ratio": gap.distance_ratio,
                    "pixel_ratio": gap.pixel_ratio,
                    "measurement_conflict": gap.measurement_conflict,
                    "position_conflict": gap.position_conflict,
                }
                for gap in gap_tracking.gaps
            ]
            if gap_tracking is not None
            else []
        ),
        "missing": missing_events,
        "suspect_missing": suspect_missing_events,
        "stable_missing": stable_missing_events,
        "displaced": displaced_events,
        "stable_displaced": [
            event
            for alert, event in zip(analysis.displaced, displaced_events)
            if _alert_allowed(analysis, alert.index, tracking, min_alert_cones)
            and (
                confirmed_shift_route_ids is None
                or event["stable_route_id"] in confirmed_shift_route_ids
            )
        ],
        "shift_alert_policy": {
            "confirm_frames": shift_confirm_frames,
            "membership_stable_frames": shift_membership_stable_frames,
        },
        "shift_corner_exemptions": [
            {
                "corner_index": item.corner_index,
                "protected_indices": item.protected_indices,
                "before_arm_indices": item.before_arm_indices,
                "after_arm_indices": item.after_arm_indices,
                "arm_angle_degrees": item.arm_angle_degrees,
                "before_residual_ratio": item.before_residual_ratio,
                "after_residual_ratio": item.after_residual_ratio,
                "vertex_distance_ratio": item.vertex_distance_ratio,
                "distance_unit": item.distance_unit,
            }
            for item in analysis.shift_corner_exemptions
        ],
        "edge_ignored": analysis.edge_ignored,
        "isolated": analysis.isolated,
        "missing_edge_ignored": analysis.missing_edge_ignored,
        "fallen": fallen_events,
        "stable_fallen": stable_fallen_events,
        "suppressed_missing_by_fallen": [
            {
                "stable_route_id": item.stable_route_id,
                "fallen_stable_ids": item.fallen_stable_ids,
                "reason": item.reason,
                **missing_event(item.alert),
            }
            for item in (
                gap_tracking.suppressed_by_fallen if gap_tracking is not None else ()
            )
        ],
    }


def _video_fps(cv2: Any, source: str) -> float:
    if source.isdigit():
        return 25.0
    capture = cv2.VideoCapture(source)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    return fps if fps > 0 else 25.0


def _ground_projector_from_args(args: argparse.Namespace) -> GroundPlaneProjector | None:
    pitch_provided = args.camera_pitch_deg is not None
    hfov_provided = args.camera_hfov_deg is not None
    if pitch_provided != hfov_provided:
        raise SystemExit(
            "--camera-pitch-deg and --camera-hfov-deg must be provided together"
        )
    if args.no_perspective_correction and pitch_provided:
        raise SystemExit(
            "--no-perspective-correction cannot be combined with camera parameters"
        )
    if args.drone_altitude_m is not None and not pitch_provided:
        raise SystemExit(
            "--drone-altitude-m requires --camera-pitch-deg and --camera-hfov-deg"
        )
    if args.drone_altitude_m is not None and args.meters_per_pixel is not None:
        raise SystemExit(
            "--drone-altitude-m and --meters-per-pixel are mutually exclusive"
        )
    if pitch_provided and args.meters_per_pixel is not None:
        raise SystemExit(
            "camera perspective parameters and --meters-per-pixel are mutually exclusive"
        )
    if args.no_perspective_correction or not pitch_provided:
        return None
    return GroundPlaneProjector(
        pitch_degrees=args.camera_pitch_deg,
        horizontal_fov_degrees=args.camera_hfov_deg,
        altitude_meters=args.drone_altitude_m,
    )


def main() -> int:
    args = build_parser().parse_args()
    if not args.weights.is_file():
        raise SystemExit(f"Weights do not exist: {args.weights}")
    if not 0.0 <= args.min_cone_long_side_ratio < 1.0:
        raise SystemExit("--min-cone-long-side-ratio must be in [0, 1)")
    if args.cone_count_confirm_window < 1 or not (
        1 <= args.cone_count_confirm_votes <= args.cone_count_confirm_window
    ):
        raise SystemExit(
            "--cone-count-confirm-votes must be within "
            "--cone-count-confirm-window"
        )
    if args.cone_count_max_missed_frames < 0:
        raise SystemExit("--cone-count-max-missed-frames must not be negative")
    if args.cone_count_match_distance_ratio <= 0.0:
        raise SystemExit("--cone-count-match-distance-ratio must be positive")
    if args.classifier_thresholds is not None and args.classifier_weights is None:
        raise SystemExit("--classifier-thresholds requires --classifier-weights")
    ground_projector = _ground_projector_from_args(args)
    cv2, np, YOLO = require_runtime()
    model = YOLO(str(args.weights.resolve()))
    candidate_classifier = None
    classifier_policy = None
    if args.classifier_weights is not None:
        from cone_candidate_classifier import ConeCandidateClassifier

        thresholds = args.classifier_thresholds or (
            args.classifier_weights.parent / "thresholds.json"
        )
        classifier_device = (
            args.classifier_device
            if args.classifier_device is not None
            else args.device
        )
        try:
            candidate_classifier = ConeCandidateClassifier(
                args.classifier_weights,
                thresholds,
                device=classifier_device,
                crop_context=args.classifier_crop_context,
                batch_size=args.classifier_batch,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise SystemExit(f"Could not initialize candidate classifier: {exc}") from exc
        classifier_policy = {
            **candidate_classifier.policy,
            "mode": args.classifier_mode,
        }
        print(
            "Candidate classifier enabled: "
            f"mode={args.classifier_mode} "
            f"crop_context={args.classifier_crop_context} "
            f"device={classifier_device}"
        )
    person_model = YOLO(args.person_weights) if args.person_weights else None
    excluded_classes = {
        value.strip() for value in args.exclude_classes.split(",") if value.strip()
    }
    try:
        person_classes = [int(value) for value in args.person_classes.split(",")]
    except ValueError as exc:
        raise SystemExit("--person-classes must contain comma-separated integers") from exc
    source: str | int = int(args.source) if args.source.isdigit() else args.source
    predict_args = {
        "source": source,
        "stream": True,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "verbose": False,
    }
    if args.device is not None:
        predict_args["device"] = args.device

    args.output.mkdir(parents=True, exist_ok=True)
    event_path = args.output / "events.jsonl"
    cone_count_path = args.output / "cone_count.json"
    structured_alert_path = args.output / "alert.jsonl"
    review_alert_path = args.output / "alert.log"
    source_path = Path(args.source)
    is_video = (
        args.source.isdigit()
        or source_path.suffix.lower() in VIDEO_SUFFIXES
        or args.source.startswith(("rtsp://", "rtmp://", "http://", "https://"))
    )
    video_fps = _video_fps(cv2, args.source) if is_video else 0.0
    if args.gap_suspect_seconds <= 0.0:
        raise SystemExit("--gap-suspect-seconds must be positive")
    writer = None
    route_tracker = (
        RouteTracker(
            confirm_frames=args.route_confirm_frames,
            recovery_frames=args.route_recovery_frames,
            max_missed_frames=args.roi_memory_frames,
            match_distance_ratio=args.route_match_ratio,
            max_direction_difference=args.route_max_angle,
            base_smoothing=args.roi_smoothing,
            max_smoothing=args.roi_max_smoothing,
        )
        if is_video and not args.no_temporal_roi
        else None
    )
    gap_tracker = (
        GapThresholdTracker(
            history_frames=args.gap_history_frames,
            min_history_frames=args.gap_history_min_frames,
            close_ratio=args.gap_close_ratio,
            max_drop_ratio=args.gap_max_drop_ratio,
            minimum_gap_pixels=args.gap_min_pixels,
            vote_window=args.gap_vote_window,
            confirm_votes=args.gap_confirm_votes,
            alert_ratio=args.gap_alert_ratio,
            suspect_frames=max(
                1, math.ceil(args.gap_suspect_seconds * video_fps)
            ),
            alert_confirm_frames=args.gap_alert_confirm_frames,
            max_ratio_disagreement=args.gap_max_ratio_disagreement,
            clear_frames=args.gap_clear_frames,
            membership_cooldown_frames=args.gap_membership_cooldown,
            min_alert_cones=args.min_alert_cones,
            history_min_cones=args.gap_history_min_cones,
        )
        if route_tracker is not None
        else None
    )
    cone_counter = ConeCounter(
        confirm_window=(args.cone_count_confirm_window if is_video else 1),
        confirm_votes=(
            min(args.cone_count_confirm_votes, args.cone_count_confirm_window)
            if is_video
            else 1
        ),
        max_missed_frames=args.cone_count_max_missed_frames,
        match_distance_ratio=args.cone_count_match_distance_ratio,
    )
    fallen_tracker = ConeTrackRegistry(
        vote_window=args.fallen_vote_window,
        confirm_votes=args.fallen_confirm_votes,
        reactivation_cooldown_frames=args.fallen_reactivation_cooldown_frames,
        match_distance_ratio=args.fallen_match_distance_ratio,
        max_angle_jump_degrees=args.fallen_max_angle_jump,
    )
    shift_tracker = (
        ShiftTracker(
            confirm_frames=args.shift_confirm_frames,
            membership_stable_frames=args.shift_membership_stable_frames,
        )
        if route_tracker is not None
        else None
    )
    events_file = event_path.open("w", encoding="utf-8")
    structured_alert_file = structured_alert_path.open("w", encoding="utf-8")
    review_alert_file = review_alert_path.open("w", encoding="utf-8")
    alert_journal = AlertJournal()
    last_cone_counting: ConeCountingResult | None = None
    try:
        for frame_index, result in enumerate(model.predict(**predict_args)):
            video_time_seconds = (
                frame_index / video_fps if video_fps > 0.0 else 0.0
            )
            frame = result.orig_img.copy()
            height, width = frame.shape[:2]
            person_boxes = _same_model_exclusion_boxes(result, excluded_classes)
            if person_model is not None:
                person_result = person_model.predict(
                    frame,
                    classes=person_classes,
                    conf=args.person_conf,
                    imgsz=args.person_imgsz,
                    device=args.device,
                    verbose=False,
                )[0]
                if person_result.boxes is not None:
                    person_boxes.extend(person_result.boxes.xyxy.detach().cpu().tolist())
            (
                cones,
                detections,
                person_suppressed,
                small_cone_suppressed,
            ) = _extract_detections(
                result,
                args.cone_class,
                args.roi,
                person_boxes,
                image_size=(width, height),
                min_cone_long_side_ratio=args.min_cone_long_side_ratio,
                candidate_classifier=candidate_classifier,
                frame=frame,
                classifier_mode=args.classifier_mode,
            )
            analysis = analyze_cones(
                cones,
                gap_ratio=args.gap_ratio,
                gap_window=args.gap_window,
                route_link_ratio=args.route_link_ratio,
                route_hints=route_tracker.hints() if route_tracker is not None else None,
                route_hint_ratio=args.route_hint_ratio,
                min_route_cones=args.min_route_cones,
                image_size=(width, height),
                fallen_edge_margin_ratio=args.fallen_edge_margin,
                missing_edge_margin_ratio=args.missing_edge_margin,
                shift_corner_exemption=not args.no_shift_corner_exemption,
                shift_neighbor_count=args.shift_neighbor_count,
                shift_corner_min_arm_cones=args.shift_corner_min_arm_cones,
                shift_corner_min_angle_degrees=args.shift_corner_min_angle,
                shift_corner_max_arm_residual_ratio=(
                    args.shift_corner_max_arm_residual_ratio
                ),
                shift_corner_max_vertex_distance_ratio=(
                    args.shift_corner_max_vertex_distance_ratio
                ),
                shift_corner_protect_neighbors=args.shift_corner_protect_neighbors,
                isolated_endpoint_gap_ratio=args.isolated_endpoint_gap_ratio,
                offset_ratio=args.offset_ratio,
                offset_threshold_meters=args.offset_threshold_m,
                meters_per_pixel=args.meters_per_pixel,
                ground_projector=ground_projector,
                upright_angle=args.upright_angle,
                fallen_angle=args.fallen_angle,
            )
            tracking = (
                route_tracker.update(
                    observations_from_analysis(analysis), (width, height)
                )
                if route_tracker is not None
                else None
            )
            cone_counting = cone_counter.update(analysis)
            last_cone_counting = cone_counting
            fallen_tracking = fallen_tracker.update(analysis, tracking)
            active_fallen_indices = tuple(
                observation.index for observation in fallen_tracking.active
            )
            active_fallen_ids = {
                observation.index: observation.stable_cone_id
                for observation in fallen_tracking.active
            }
            gap_tracking = (
                gap_tracker.update(
                    analysis,
                    tracking,
                    gap_ratio=args.gap_ratio,
                    fallen_indices=active_fallen_indices,
                    fallen_stable_ids=active_fallen_ids,
                )
                if gap_tracker is not None and tracking is not None
                else None
            )
            confirmed_shift_route_ids = None
            if shift_tracker is not None and tracking is not None:
                candidate_shift_route_ids = {
                    stable_id
                    for alert in analysis.displaced
                    if _alert_allowed(
                        analysis, alert.index, tracking, args.min_alert_cones
                    )
                    and (
                        stable_id := _stable_route_id(
                            analysis, alert.index, tracking
                        )
                    )
                    is not None
                }
                route_member_counts: dict[int, int] = {}
                for track in tracking.tracks:
                    if track.current_route_index is None:
                        continue
                    stable_id = track.business_stable_id
                    route_member_counts[stable_id] = max(
                        route_member_counts.get(stable_id, 0),
                        len(analysis.routes[track.current_route_index]),
                    )
                confirmed_shift_route_ids = shift_tracker.update(
                    candidate_shift_route_ids, route_member_counts
                )
            frame = _draw_frame(
                cv2,
                np,
                frame,
                detections,
                analysis,
                args.roi,
                tracking,
                gap_tracking,
                args.min_alert_cones,
                confirmed_shift_route_ids,
                fallen_tracking,
                cone_counting,
            )
            event = _events(
                frame_index,
                analysis,
                detections,
                args.roi,
                person_suppressed,
                tracking,
                gap_tracking,
                video_time_seconds,
                args.min_alert_cones,
                confirmed_shift_route_ids,
                args.shift_confirm_frames,
                args.shift_membership_stable_frames,
                fallen_tracking,
                cone_counting=cone_counting,
                small_cone_suppressed=small_cone_suppressed,
                classifier_policy=classifier_policy,
                classifier_timing_ms=(
                    dict(candidate_classifier.last_timing_ms)
                    if candidate_classifier is not None
                    else None
                ),
            )
            events_file.write(
                json.dumps(
                    event,
                    ensure_ascii=False,
                )
                + "\n"
            )
            events_file.flush()
            for alert_record in alert_journal.update(
                event, video_time_seconds
            ):
                structured_alert_file.write(
                    json.dumps(alert_record, ensure_ascii=False) + "\n"
                )
                review_alert_file.write(
                    format_review_alert(alert_record) + "\n"
                )
            structured_alert_file.flush()
            review_alert_file.flush()

            if not args.no_save:
                if is_video:
                    if writer is None:
                        height, width = frame.shape[:2]
                        output_video = args.output / f"{source_path.stem or 'camera'}_result.mp4"
                        writer = cv2.VideoWriter(
                            str(output_video),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            video_fps,
                            (width, height),
                        )
                        if not writer.isOpened():
                            raise RuntimeError(f"Could not create output video: {output_video}")
                    writer.write(frame)
                else:
                    image_name = f"{Path(result.path).stem}_result.jpg"
                    cv2.imwrite(str(args.output / image_name), frame)
            if args.display:
                cv2.imshow("Traffic cone OBB", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    finally:
        events_file.close()
        structured_alert_file.close()
        review_alert_file.close()
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()
    print(f"Inference events: {event_path}")
    print(f"Structured alerts: {structured_alert_path}")
    print(f"Review alerts: {review_alert_path}")
    if last_cone_counting is not None:
        cone_count_path.write_text(
            json.dumps(
                {
                    "unique_seen_count": last_cone_counting.unique_seen_count,
                    "last_stable_visible_count": (
                        last_cone_counting.stable_visible_count
                    ),
                    "last_active_cone_ids": list(
                        last_cone_counting.active_cone_ids
                    ),
                    "last_held_cone_ids": list(last_cone_counting.held_cone_ids),
                    "counting_policy": last_cone_counting.policy,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            "Cone counting: "
            f"stable_visible={last_cone_counting.stable_visible_count} "
            f"unique_seen={last_cone_counting.unique_seen_count}"
        )
        print(f"Cone count summary: {cone_count_path}")
    if not args.no_save:
        print(f"Annotated output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
