#!/usr/bin/env python3
"""Mine reviewable cone crops for the candidate posture classifier."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from traffic_rules import ConeDetection


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")
CSV_FIELDS = (
    "candidate_id",
    "source_split",
    "source_image",
    "source_label",
    "candidate_kind",
    "review_pool",
    "candidate_index",
    "crop_path",
    "preview_path",
    "yolo_confidence",
    "raw_angle_fallen",
    "angle_rule_positive",
    "edge_ignored",
    "long_axis_angle",
    "upright_difference",
    "aspect_ratio",
    "match_status",
    "matched_gt_index",
    "match_iou",
    "polygon_json",
    "source_width",
    "source_height",
    "crop_center_x",
    "crop_center_y",
    "crop_side",
)


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--cone-class", default="cone")
    parser.add_argument("--crop-context", type=float, default=2.0)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--upright-angle", type=float, default=90.0)
    parser.add_argument("--fallen-angle", type=float, default=30.0)
    parser.add_argument("--min-aspect-ratio", type=float, default=1.2)
    parser.add_argument("--fallen-edge-margin", type=float, default=0.05)
    parser.add_argument("--project-root", type=Path, default=project_root)
    return parser


def polygon_iou(first: np.ndarray, second: np.ndarray) -> float:
    first_hull = cv2.convexHull(first.astype(np.float32))
    second_hull = cv2.convexHull(second.astype(np.float32))
    area_first = abs(float(cv2.contourArea(first_hull)))
    area_second = abs(float(cv2.contourArea(second_hull)))
    intersection, _ = cv2.intersectConvexConvex(first_hull, second_hull)
    union = area_first + area_second - float(intersection)
    return float(intersection) / union if union > 0.0 else 0.0


def match_polygons(
    predictions: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    threshold: float,
) -> tuple[dict[int, tuple[int, float]], set[int]]:
    candidates = sorted(
        (
            polygon_iou(prediction, target),
            prediction_index,
            target_index,
        )
        for prediction_index, prediction in enumerate(predictions)
        for target_index, target in enumerate(targets)
    )
    matches: dict[int, tuple[int, float]] = {}
    matched_targets: set[int] = set()
    for overlap, prediction_index, target_index in reversed(candidates):
        if overlap < threshold:
            break
        if prediction_index in matches or target_index in matched_targets:
            continue
        matches[prediction_index] = (target_index, overlap)
        matched_targets.add(target_index)
    return matches, matched_targets


def _axis_difference(first: float, second: float) -> float:
    difference = abs((first - second) % 180.0)
    return min(difference, 180.0 - difference)


def orientation_metadata(
    polygon: np.ndarray,
    image_size: tuple[int, int],
    *,
    upright_angle: float,
    fallen_angle: float,
    min_aspect_ratio: float,
    edge_margin_ratio: float,
) -> dict[str, float | bool]:
    detection = ConeDetection(
        polygon=tuple((float(point[0]), float(point[1])) for point in polygon),
        confidence=1.0,
    )
    difference = _axis_difference(detection.long_axis_angle, upright_angle)
    angle_rule_positive = (
        difference >= fallen_angle and detection.aspect_ratio >= min_aspect_ratio
    )
    width, height = image_size
    margin = min(width, height) * edge_margin_ratio
    edge_ignored = angle_rule_positive and any(
        point[0] <= margin
        or point[1] <= margin
        or point[0] >= width - margin
        or point[1] >= height - margin
        for point in detection.polygon
    )
    return {
        "raw_angle_fallen": angle_rule_positive and not edge_ignored,
        "angle_rule_positive": angle_rule_positive,
        "edge_ignored": edge_ignored,
        "long_axis_angle": detection.long_axis_angle,
        "upright_difference": difference,
        "aspect_ratio": detection.aspect_ratio,
    }


def crop_square(
    image: np.ndarray,
    polygon: np.ndarray,
    *,
    context: float,
    output_size: int,
    fill_value: int = 114,
) -> tuple[np.ndarray, dict[str, float]]:
    if context <= 0.0:
        raise ValueError("context must be positive")
    if output_size <= 0:
        raise ValueError("output_size must be positive")
    center = polygon.mean(axis=0)
    edge_lengths = [
        float(np.linalg.norm(polygon[(index + 1) % 4] - polygon[index]))
        for index in range(4)
    ]
    side = max(2.0, max(edge_lengths) * context)
    crop_pixels = max(2, int(math.ceil(side)))
    x1 = int(math.floor(float(center[0]) - crop_pixels / 2.0))
    y1 = int(math.floor(float(center[1]) - crop_pixels / 2.0))
    x2 = x1 + crop_pixels
    y2 = y1 + crop_pixels

    source_height, source_width = image.shape[:2]
    source_x1 = max(0, x1)
    source_y1 = max(0, y1)
    source_x2 = min(source_width, x2)
    source_y2 = min(source_height, y2)
    crop = np.full((crop_pixels, crop_pixels, 3), fill_value, dtype=image.dtype)
    if source_x1 < source_x2 and source_y1 < source_y2:
        destination_x1 = source_x1 - x1
        destination_y1 = source_y1 - y1
        crop[
            destination_y1 : destination_y1 + (source_y2 - source_y1),
            destination_x1 : destination_x1 + (source_x2 - source_x1),
        ] = image[source_y1:source_y2, source_x1:source_x2]
    interpolation = cv2.INTER_AREA if crop_pixels > output_size else cv2.INTER_CUBIC
    resized = cv2.resize(crop, (output_size, output_size), interpolation=interpolation)
    return resized, {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "side": float(crop_pixels),
        "x1": float(x1),
        "y1": float(y1),
    }


def draw_preview(
    crop: np.ndarray,
    polygon: np.ndarray,
    crop_geometry: dict[str, float],
    *,
    color: tuple[int, int, int],
) -> np.ndarray:
    scale = crop.shape[1] / crop_geometry["side"]
    transformed = np.asarray(
        [
            [
                (float(point[0]) - crop_geometry["x1"]) * scale,
                (float(point[1]) - crop_geometry["y1"]) * scale,
            ]
            for point in polygon
        ],
        dtype=np.int32,
    )
    preview = crop.copy()
    cv2.polylines(preview, [transformed], True, color, 2, cv2.LINE_AA)
    return preview


def read_targets(
    label_path: Path,
    image_size: tuple[int, int],
    cone_class_id: int,
) -> list[np.ndarray]:
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing label file: {label_path}")
    width, height = image_size
    targets: list[np.ndarray] = []
    for line_number, row in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not row.strip():
            continue
        values = row.split()
        if len(values) != 9:
            raise ValueError(f"Expected OBB class plus 8 coordinates: {label_path}:{line_number}")
        if int(values[0]) != cone_class_id:
            continue
        coordinates = [float(value) for value in values[1:]]
        targets.append(
            np.asarray(
                [
                    [coordinates[index] * width, coordinates[index + 1] * height]
                    for index in range(0, 8, 2)
                ],
                dtype=np.float32,
            )
        )
    return targets


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _class_id(names: Any, label: str) -> int:
    items = names.items() if isinstance(names, dict) else enumerate(names)
    for class_id, class_name in items:
        if str(class_name) == label:
            return int(class_id)
    raise ValueError(f"Model does not contain class {label!r}: {names}")


def _image_paths(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )


def _candidate_id(
    split: str, source_relative: str, candidate_kind: str, candidate_index: int
) -> str:
    value = f"{split}|{source_relative}|{candidate_kind}|{candidate_index}"
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:20]


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise OSError(f"Failed to write image: {path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(project_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _record(
    *,
    candidate_id: str,
    source_split: str,
    source_image: str,
    source_label: str,
    candidate_kind: str,
    review_pool: str,
    candidate_index: int,
    crop_path: str,
    preview_path: str,
    confidence: float | None,
    orientation: dict[str, float | bool],
    match_status: str,
    matched_gt_index: int | None,
    match_iou_value: float | None,
    polygon: np.ndarray,
    image_size: tuple[int, int],
    crop_geometry: dict[str, float],
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "source_split": source_split,
        "source_image": source_image,
        "source_label": source_label,
        "candidate_kind": candidate_kind,
        "review_pool": review_pool,
        "candidate_index": candidate_index,
        "crop_path": crop_path,
        "preview_path": preview_path,
        "yolo_confidence": "" if confidence is None else f"{confidence:.8f}",
        "raw_angle_fallen": str(bool(orientation["raw_angle_fallen"])).lower(),
        "angle_rule_positive": str(bool(orientation["angle_rule_positive"])).lower(),
        "edge_ignored": str(bool(orientation["edge_ignored"])).lower(),
        "long_axis_angle": f"{float(orientation['long_axis_angle']):.6f}",
        "upright_difference": f"{float(orientation['upright_difference']):.6f}",
        "aspect_ratio": f"{float(orientation['aspect_ratio']):.6f}",
        "match_status": match_status,
        "matched_gt_index": "" if matched_gt_index is None else matched_gt_index,
        "match_iou": "" if match_iou_value is None else f"{match_iou_value:.8f}",
        "polygon_json": json.dumps(polygon.tolist(), separators=(",", ":")),
        "source_width": image_size[0],
        "source_height": image_size[1],
        "crop_center_x": f"{crop_geometry['center_x']:.4f}",
        "crop_center_y": f"{crop_geometry['center_y']:.4f}",
        "crop_side": f"{crop_geometry['side']:.4f}",
    }


def _candidate_images(
    output: Path,
    split: str,
    review_pool: str,
    candidate_id: str,
    image: np.ndarray,
    polygon: np.ndarray,
    *,
    context: float,
    crop_size: int,
    preview_color: tuple[int, int, int],
) -> tuple[str, str, dict[str, float]]:
    crop, geometry = crop_square(
        image, polygon, context=context, output_size=crop_size
    )
    preview = draw_preview(crop, polygon, geometry, color=preview_color)
    crop_relative = Path("crops") / split / review_pool / f"{candidate_id}.jpg"
    preview_relative = Path("previews") / split / review_pool / f"{candidate_id}.jpg"
    _write_image(output / crop_relative, crop)
    _write_image(output / preview_relative, preview)
    return crop_relative.as_posix(), preview_relative.as_posix(), geometry


def mine_candidates(args: argparse.Namespace) -> dict[str, Any]:
    from ultralytics import YOLO

    if args.imgsz <= 0 or args.batch <= 0 or args.crop_size <= 0:
        raise ValueError("imgsz, batch, and crop-size must be positive")
    if not 0.0 <= args.conf <= 1.0 or not 0.0 <= args.iou <= 1.0:
        raise ValueError("conf and iou must be within [0, 1]")
    if not 0.0 <= args.match_iou <= 1.0:
        raise ValueError("match-iou must be within [0, 1]")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    all_images = {
        split: _image_paths(args.images / split)
        for split in SPLITS
    }
    if not all(all_images.values()):
        raise ValueError(f"Each split must contain images: {args.images}")

    model = YOLO(str(args.weights), task="obb")
    cone_class_id = _class_id(model.names, args.cone_class)
    started_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "status": "running",
        "started_at": started_at,
        "weights": str(args.weights.resolve()),
        "weights_sha256": _sha256_file(args.weights),
        "images": str(args.images.resolve()),
        "labels": str(args.labels.resolve()),
        "git_commit": _git_commit(args.project_root),
        "parameters": {
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "match_iou": args.match_iou,
            "device": str(args.device),
            "batch": args.batch,
            "cone_class": args.cone_class,
            "crop_context": args.crop_context,
            "crop_size": args.crop_size,
            "upright_angle": args.upright_angle,
            "fallen_angle": args.fallen_angle,
            "min_aspect_ratio": args.min_aspect_ratio,
            "fallen_edge_margin": args.fallen_edge_margin,
        },
        "source_images": {split: len(paths) for split, paths in all_images.items()},
    }
    _atomic_json(args.output / "run_manifest.json", manifest)

    partial_csv = args.output / "candidates.partial.csv"
    counts: Counter[str] = Counter()
    processed_images: Counter[str] = Counter()
    seen_sources: set[Path] = set()
    with partial_csv.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for split in SPLITS:
            expected = {path.resolve(): path for path in all_images[split]}
            results: Iterable[Any] = model.predict(
                source=str(args.images / split),
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                batch=args.batch,
                stream=True,
                verbose=False,
            )
            for result in results:
                source_path = Path(result.path).resolve()
                if source_path not in expected:
                    raise RuntimeError(f"Unexpected predictor source: {source_path}")
                if source_path in seen_sources:
                    raise RuntimeError(f"Predictor returned duplicate source: {source_path}")
                seen_sources.add(source_path)
                source_original = expected[source_path]
                image = result.orig_img
                if image is None:
                    image = cv2.imread(str(source_original))
                if image is None:
                    raise ValueError(f"Unreadable image: {source_original}")
                height, width = image.shape[:2]
                label_path = args.labels / split / f"{source_original.stem}.txt"
                targets = read_targets(label_path, (width, height), cone_class_id)

                predictions: list[np.ndarray] = []
                prediction_confidences: list[float] = []
                if result.obb is not None and len(result.obb) > 0:
                    polygons = result.obb.xyxyxyxy.detach().cpu().numpy()
                    confidences = result.obb.conf.detach().cpu().numpy()
                    class_ids = result.obb.cls.detach().cpu().numpy().astype(int)
                    for polygon, confidence, class_id in zip(
                        polygons, confidences, class_ids
                    ):
                        if _class_name(result.names, int(class_id)) != args.cone_class:
                            continue
                        predictions.append(np.asarray(polygon, dtype=np.float32))
                        prediction_confidences.append(float(confidence))

                matches, matched_targets = match_polygons(
                    predictions, targets, args.match_iou
                )
                source_relative = source_original.relative_to(args.images).as_posix()
                label_relative = label_path.relative_to(args.labels).as_posix()
                for prediction_index, (polygon, confidence) in enumerate(
                    zip(predictions, prediction_confidences)
                ):
                    orientation = orientation_metadata(
                        polygon,
                        (width, height),
                        upright_angle=args.upright_angle,
                        fallen_angle=args.fallen_angle,
                        min_aspect_ratio=args.min_aspect_ratio,
                        edge_margin_ratio=args.fallen_edge_margin,
                    )
                    match = matches.get(prediction_index)
                    if bool(orientation["raw_angle_fallen"]):
                        review_pool = "raw_angle_fallen"
                    elif match is None:
                        review_pool = "prediction_without_gt"
                    else:
                        review_pool = "angle_normal_cone"
                    candidate_id = _candidate_id(
                        split, source_relative, "prediction", prediction_index
                    )
                    crop_path, preview_path, geometry = _candidate_images(
                        args.output,
                        split,
                        review_pool,
                        candidate_id,
                        image,
                        polygon,
                        context=args.crop_context,
                        crop_size=args.crop_size,
                        preview_color=(255, 210, 0),
                    )
                    if match is None:
                        match_status = (
                            "no_cone_gt_in_image"
                            if not targets
                            else "prediction_without_gt"
                        )
                        matched_gt_index = None
                        match_iou_value = None
                    else:
                        matched_gt_index, match_iou_value = match
                        match_status = "matched_gt"
                    writer.writerow(
                        _record(
                            candidate_id=candidate_id,
                            source_split=split,
                            source_image=source_relative,
                            source_label=label_relative,
                            candidate_kind="prediction",
                            review_pool=review_pool,
                            candidate_index=prediction_index,
                            crop_path=crop_path,
                            preview_path=preview_path,
                            confidence=confidence,
                            orientation=orientation,
                            match_status=match_status,
                            matched_gt_index=matched_gt_index,
                            match_iou_value=match_iou_value,
                            polygon=polygon,
                            image_size=(width, height),
                            crop_geometry=geometry,
                        )
                    )
                    counts[review_pool] += 1
                    counts["prediction"] += 1

                for target_index, polygon in enumerate(targets):
                    if target_index in matched_targets:
                        continue
                    review_pool = "gt_without_prediction"
                    orientation = orientation_metadata(
                        polygon,
                        (width, height),
                        upright_angle=args.upright_angle,
                        fallen_angle=args.fallen_angle,
                        min_aspect_ratio=args.min_aspect_ratio,
                        edge_margin_ratio=args.fallen_edge_margin,
                    )
                    candidate_id = _candidate_id(
                        split, source_relative, "ground_truth", target_index
                    )
                    crop_path, preview_path, geometry = _candidate_images(
                        args.output,
                        split,
                        review_pool,
                        candidate_id,
                        image,
                        polygon,
                        context=args.crop_context,
                        crop_size=args.crop_size,
                        preview_color=(0, 210, 255),
                    )
                    writer.writerow(
                        _record(
                            candidate_id=candidate_id,
                            source_split=split,
                            source_image=source_relative,
                            source_label=label_relative,
                            candidate_kind="ground_truth",
                            review_pool=review_pool,
                            candidate_index=target_index,
                            crop_path=crop_path,
                            preview_path=preview_path,
                            confidence=None,
                            orientation=orientation,
                            match_status="gt_without_prediction",
                            matched_gt_index=target_index,
                            match_iou_value=None,
                            polygon=polygon,
                            image_size=(width, height),
                            crop_geometry=geometry,
                        )
                    )
                    counts[review_pool] += 1
                    counts["ground_truth"] += 1
                processed_images[split] += 1
                destination.flush()
            if processed_images[split] != len(all_images[split]):
                raise RuntimeError(
                    f"Predictor processed {processed_images[split]} of "
                    f"{len(all_images[split])} {split} images"
                )

    final_csv = args.output / "candidates.csv"
    os.replace(partial_csv, final_csv)
    summary = {
        "source_images": dict(processed_images),
        "candidates": sum(counts[pool] for pool in (
            "raw_angle_fallen",
            "angle_normal_cone",
            "prediction_without_gt",
            "gt_without_prediction",
        )),
        "by_pool": {
            pool: counts[pool]
            for pool in (
                "raw_angle_fallen",
                "angle_normal_cone",
                "prediction_without_gt",
                "gt_without_prediction",
            )
        },
        "prediction_candidates": counts["prediction"],
        "ground_truth_misses": counts["ground_truth"],
    }
    _atomic_json(args.output / "summary.json", summary)
    manifest.update(
        {
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
        }
    )
    _atomic_json(args.output / "run_manifest.json", manifest)
    return summary


def main() -> int:
    args = build_parser().parse_args()
    summary = mine_candidates(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
