#!/usr/bin/env python3
"""Run auditable YOLO OBB plus MobileNet candidate-classifier video inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from mine_classifier_candidates import crop_square, orientation_metadata
from mobilenet_classifier import build_transform, load_checkpoint


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _class_id(names: dict[int, str] | list[str], label: str) -> int:
    items = names.items() if isinstance(names, dict) else enumerate(names)
    for class_id, class_name in items:
        if str(class_name) == label:
            return int(class_id)
    raise ValueError(f"Model does not contain class {label!r}: {names}")


def _parse_point(value: str | None) -> tuple[float, float] | None:
    if value is None:
        return None
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("point must be x,y")
    return float(parts[0]), float(parts[1])


def _parse_range(value: str | None) -> tuple[float, float] | None:
    parsed = _parse_point(value)
    if parsed is not None and parsed[0] > parsed[1]:
        raise argparse.ArgumentTypeError("range minimum must not exceed maximum")
    return parsed


def _load_thresholds(path: Path) -> tuple[float, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["not_cone"]["threshold"]), float(payload["fallen"]["threshold"])


def _draw_label(
    frame: np.ndarray,
    polygon: np.ndarray,
    text: str,
    color: tuple[int, int, int],
    *,
    target: bool,
) -> None:
    points = polygon.astype(np.int32)
    cv2.polylines(frame, [points], True, color, 3 if target else 2, cv2.LINE_AA)
    x = max(0, int(points[:, 0].min()))
    y = max(18, int(points[:, 1].min()) - 5)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1, cv2.LINE_AA)


def _rate(count: int, total: int) -> float | None:
    return count / total if total else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolo-weights", type=Path, required=True)
    parser.add_argument("--classifier-weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--cone-class", default="cone")
    parser.add_argument("--crop-context", type=float, default=2.0)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--expected-scene", choices=("unlabeled", "glare-negative", "fallen-positive"), default="unlabeled")
    parser.add_argument("--target-start-center", type=str)
    parser.add_argument("--target-end-center", type=str)
    parser.add_argument("--target-max-distance", type=float, default=100.0)
    parser.add_argument(
        "--target-y-range",
        type=str,
        help="Optional manual truth constraint min_y,max_y used only for target matching",
    )
    parser.add_argument(
        "--negative-max-x",
        type=float,
        help="Manual negative ROI: candidate centers with x below this value contain no true cones",
    )
    args = parser.parse_args()

    start_center = _parse_point(args.target_start_center)
    end_center = _parse_point(args.target_end_center)
    target_y_range = _parse_range(args.target_y_range)
    if (start_center is None) != (end_center is None):
        raise SystemExit("--target-start-center and --target-end-center must be provided together")
    if args.expected_scene == "fallen-positive" and start_center is None:
        raise SystemExit("fallen-positive evaluation requires target start/end centers")
    if args.expected_scene == "glare-negative" and args.negative_max_x is None:
        raise SystemExit("glare-negative evaluation requires --negative-max-x")
    not_cone_threshold, fallen_threshold = _load_thresholds(args.thresholds)
    torch_device = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device}")
    yolo = YOLO(str(args.yolo_weights.resolve()), task="obb")
    cone_class_id = _class_id(yolo.names, args.cone_class)
    classifier, checkpoint = load_checkpoint(args.classifier_weights.resolve(), torch_device)
    transform = build_transform(False)
    class_to_idx = checkpoint["class_to_idx"]
    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]

    capture = cv2.VideoCapture(str(args.source.resolve()))
    if not capture.isOpened():
        raise SystemExit(f"Cannot open source: {args.source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count_hint = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        raise SystemExit("Source has invalid video metadata")

    args.output.mkdir(parents=True, exist_ok=True)
    output_video = args.output / f"{args.source.stem}_dual_result.mp4"
    events_path = args.output / "events.jsonl"
    summary_path = args.output / "summary.json"
    writer = cv2.VideoWriter(
        str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise SystemExit(f"Cannot create output video: {output_video}")

    counts: Counter[str] = Counter()
    timing = Counter()
    target_probability_sums: Counter[str] = Counter()
    processed_frames = 0
    try:
        with events_path.open("w", encoding="utf-8") as events:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_started = time.perf_counter()
                yolo_started = time.perf_counter()
                result = yolo.predict(
                    frame,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    device=args.device,
                    verbose=False,
                )[0]
                timing["yolo_seconds"] += time.perf_counter() - yolo_started
                polygons: list[np.ndarray] = []
                confidences: list[float] = []
                if result.obb is not None:
                    result_polygons = result.obb.xyxyxyxy.detach().cpu().numpy()
                    result_classes = result.obb.cls.detach().cpu().numpy().astype(int)
                    result_confidences = result.obb.conf.detach().cpu().numpy()
                    for polygon, class_id, confidence in zip(
                        result_polygons, result_classes, result_confidences
                    ):
                        if int(class_id) == cone_class_id:
                            polygons.append(polygon.astype(np.float32))
                            confidences.append(float(confidence))

                crop_started = time.perf_counter()
                crops = [
                    crop_square(
                        frame,
                        polygon,
                        context=args.crop_context,
                        output_size=args.crop_size,
                    )[0]
                    for polygon in polygons
                ]
                timing["crop_seconds"] += time.perf_counter() - crop_started
                probability_rows: list[list[float]] = []
                if crops:
                    tensors = [
                        transform(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                        for crop in crops
                    ]
                    classifier_started = time.perf_counter()
                    with torch.inference_mode():
                        probability_rows = (
                            classifier(torch.stack(tensors).to(torch_device))
                            .softmax(dim=1)
                            .cpu()
                            .tolist()
                        )
                    timing["classifier_seconds"] += time.perf_counter() - classifier_started

                target_index: int | None = None
                expected_center: tuple[float, float] | None = None
                if start_center is not None and polygons:
                    denominator = max(1, frame_count_hint - 1)
                    progress = min(1.0, processed_frames / denominator)
                    expected_center = (
                        start_center[0] + (end_center[0] - start_center[0]) * progress,
                        start_center[1] + (end_center[1] - start_center[1]) * progress,
                    )
                    eligible = [
                        index
                        for index, polygon in enumerate(polygons)
                        if target_y_range is None
                        or target_y_range[0] <= float(polygon.mean(axis=0)[1]) <= target_y_range[1]
                    ]
                    distances = {
                        index: float(
                            np.linalg.norm(
                                polygons[index].mean(axis=0) - np.asarray(expected_center)
                            )
                        )
                        for index in eligible
                    }
                    if distances:
                        closest = min(distances, key=distances.__getitem__)
                        if distances[closest] <= args.target_max_distance:
                            target_index = closest

                frame_event: dict[str, object] = {
                    "frame_index": processed_frames,
                    "time_seconds": processed_frames / fps,
                    "expected_target_center": expected_center,
                    "target_candidate_index": target_index,
                    "candidates": [],
                }
                frame_has_accepted = False
                frame_has_classifier_fallen = False
                frame_has_angle_fallen = False
                frame_has_fused_fallen = False
                for index, (polygon, confidence, values) in enumerate(
                    zip(polygons, confidences, probability_rows)
                ):
                    probabilities = {name: float(values[class_to_idx[name]]) for name in class_names}
                    orientation = orientation_metadata(
                        polygon,
                        (width, height),
                        upright_angle=90.0,
                        fallen_angle=30.0,
                        min_aspect_ratio=1.2,
                        edge_margin_ratio=0.05,
                    )
                    rejected = probabilities["not_cone"] >= not_cone_threshold
                    classifier_fallen = (not rejected) and probabilities["fallen_cone"] >= fallen_threshold
                    angle_fallen = bool(orientation["raw_angle_fallen"])
                    fused_fallen = (not rejected) and (classifier_fallen or angle_fallen)
                    predicted_class = max(class_names, key=probabilities.__getitem__)
                    is_target = index == target_index
                    is_negative_roi = (
                        args.negative_max_x is not None
                        and float(polygon.mean(axis=0)[0]) < args.negative_max_x
                    )
                    counts["yolo_candidates"] += 1
                    counts[f"argmax_{predicted_class}"] += 1
                    counts["rejected_not_cone"] += int(rejected)
                    counts["classifier_fallen"] += int(classifier_fallen)
                    counts["angle_fallen"] += int(angle_fallen)
                    counts["fused_fallen"] += int(fused_fallen)
                    if is_negative_roi:
                        counts["negative_roi_candidates"] += 1
                        counts["negative_roi_rejected"] += int(rejected)
                        counts["negative_roi_angle_fallen"] += int(angle_fallen)
                        counts["negative_roi_fused_fallen"] += int(fused_fallen)
                    frame_has_accepted |= not rejected
                    frame_has_classifier_fallen |= classifier_fallen
                    frame_has_angle_fallen |= angle_fallen
                    frame_has_fused_fallen |= fused_fallen
                    if is_target:
                        counts["target_yolo_detected_frames"] += 1
                        counts["target_rejected_frames"] += int(rejected)
                        counts["target_classifier_fallen_frames"] += int(classifier_fallen)
                        counts["target_angle_fallen_frames"] += int(angle_fallen)
                        counts["target_fused_fallen_frames"] += int(fused_fallen)
                        for name, value in probabilities.items():
                            target_probability_sums[name] += value
                    color = (0, 165, 255) if rejected else ((0, 0, 255) if fused_fallen else (0, 200, 0))
                    label = (
                        f"{'TARGET ' if is_target else ''}{predicted_class} "
                        f"N={probabilities['not_cone']:.2f} F={probabilities['fallen_cone']:.2f}"
                    )
                    _draw_label(frame, polygon, label, color, target=is_target)
                    frame_event["candidates"].append(
                        {
                            "index": index,
                            "polygon": polygon.tolist(),
                            "center": polygon.mean(axis=0).tolist(),
                            "yolo_confidence": confidence,
                            "probabilities": probabilities,
                            "argmax_class": predicted_class,
                            "not_cone_rejected": rejected,
                            "classifier_fallen": classifier_fallen,
                            "angle_fallen": angle_fallen,
                            "fused_fallen": fused_fallen,
                            "target_match": is_target,
                            "negative_roi": is_negative_roi,
                            "orientation": orientation,
                        }
                    )
                counts["frames_with_yolo_candidates"] += int(bool(polygons))
                counts["frames_with_accepted_candidates"] += int(frame_has_accepted)
                counts["frames_with_classifier_fallen"] += int(frame_has_classifier_fallen)
                counts["frames_with_angle_fallen"] += int(frame_has_angle_fallen)
                counts["frames_with_fused_fallen"] += int(frame_has_fused_fallen)
                negative_roi_candidates = [
                    item for item in frame_event["candidates"] if item["negative_roi"]
                ]
                counts["negative_roi_candidate_frames"] += int(bool(negative_roi_candidates))
                counts["negative_roi_accepted_frames"] += int(
                    any(not item["not_cone_rejected"] for item in negative_roi_candidates)
                )
                cv2.putText(
                    frame,
                    f"t={processed_frames / fps:.2f}s YOLO={len(polygons)} accepted={sum(1 for item in frame_event['candidates'] if not item['not_cone_rejected'])}",
                    (16, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(frame)
                events.write(json.dumps(frame_event, ensure_ascii=False) + "\n")
                processed_frames += 1
                timing["total_seconds"] += time.perf_counter() - frame_started
                if processed_frames % 30 == 0:
                    print(f"processed={processed_frames}/{frame_count_hint}", flush=True)
    finally:
        capture.release()
        writer.release()

    summary: dict[str, object] = {
        "source": str(args.source.resolve()),
        "expected_scene": args.expected_scene,
        "frames": processed_frames,
        "fps": fps,
        "settings": {
            "imgsz": args.imgsz,
            "confidence": args.conf,
            "iou": args.iou,
            "device": args.device,
            "crop_context": args.crop_context,
            "crop_size": args.crop_size,
            "not_cone_threshold": not_cone_threshold,
            "fallen_threshold": fallen_threshold,
        },
        "models": {
            "yolo": str(args.yolo_weights.resolve()),
            "yolo_sha256": _sha256(args.yolo_weights),
            "classifier": str(args.classifier_weights.resolve()),
            "classifier_sha256": _sha256(args.classifier_weights),
        },
        "counts": dict(counts),
        "rates": {
            "candidate_not_cone_rejection": _rate(counts["rejected_not_cone"], counts["yolo_candidates"]),
            "frame_all_candidate_suppression": _rate(
                counts["frames_with_yolo_candidates"] - counts["frames_with_accepted_candidates"],
                counts["frames_with_yolo_candidates"],
            ),
        },
        "timing_ms_per_frame": {
            name.removesuffix("_seconds"): value * 1000.0 / processed_frames
            for name, value in timing.items()
        },
        "artifacts": {"video": output_video.name, "events": events_path.name},
    }
    if args.expected_scene == "glare-negative":
        summary["glare_false_positive_evaluation"] = {
            "manual_negative_roi": {
                "rule": f"candidate_center_x < {args.negative_max_x}",
                "ground_truth_cones": 0,
                "scope": "conservative water-only region; excludes ambiguous bridge-edge candidates",
            },
            "baseline_false_positive_boxes": counts["negative_roi_candidates"],
            "postfilter_false_positive_boxes": counts["negative_roi_candidates"] - counts["negative_roi_rejected"],
            "box_false_positive_reduction": _rate(
                counts["negative_roi_rejected"], counts["negative_roi_candidates"]
            ),
            "baseline_false_positive_frames": counts["negative_roi_candidate_frames"],
            "postfilter_false_positive_frames": counts["negative_roi_accepted_frames"],
            "global_fallen_vote_comparison": {
                "baseline_raw_angle_fallen_boxes": counts["angle_fallen"],
                "postfilter_fused_fallen_boxes": counts["fused_fallen"],
                "box_reduction": _rate(
                    counts["angle_fallen"] - counts["fused_fallen"],
                    counts["angle_fallen"],
                ),
                "baseline_raw_angle_fallen_frames": counts["frames_with_angle_fallen"],
                "postfilter_fused_fallen_frames": counts["frames_with_fused_fallen"],
                "frame_reduction": _rate(
                    counts["frames_with_angle_fallen"] - counts["frames_with_fused_fallen"],
                    counts["frames_with_angle_fallen"],
                ),
            },
        }
    if args.expected_scene == "fallen-positive":
        detected = counts["target_yolo_detected_frames"]
        summary["fallen_target_evaluation"] = {
            "expected_positive_frames": processed_frames,
            "target_yolo_detected_frames": detected,
            "target_yolo_frame_recall": _rate(detected, processed_frames),
            "target_retained_frames": detected - counts["target_rejected_frames"],
            "target_retained_frame_recall": _rate(detected - counts["target_rejected_frames"], processed_frames),
            "target_classifier_fallen_frames": counts["target_classifier_fallen_frames"],
            "target_classifier_fallen_frame_recall": _rate(counts["target_classifier_fallen_frames"], processed_frames),
            "target_angle_fallen_frames": counts["target_angle_fallen_frames"],
            "target_angle_fallen_frame_recall": _rate(counts["target_angle_fallen_frames"], processed_frames),
            "target_fused_fallen_frames": counts["target_fused_fallen_frames"],
            "target_fused_fallen_frame_recall": _rate(counts["target_fused_fallen_frames"], processed_frames),
            "mean_target_probabilities": {
                name: _rate(target_probability_sums[name], detected) for name in class_names
            },
            "manual_target_path": {
                "start_center": start_center,
                "end_center": end_center,
                "max_distance": args.target_max_distance,
                "y_range": target_y_range,
            },
        }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
