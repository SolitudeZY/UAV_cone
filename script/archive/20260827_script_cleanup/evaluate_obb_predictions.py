#!/usr/bin/env python3
"""Compare infer_obb.py JSONL detections with a YOLO OBB label directory."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    first = cv2.convexHull(first.astype(np.float32))
    second = cv2.convexHull(second.astype(np.float32))
    area_first = abs(float(cv2.contourArea(first)))
    area_second = abs(float(cv2.contourArea(second)))
    intersection, _ = cv2.intersectConvexConvex(first, second)
    union = area_first + area_second - intersection
    return intersection / union if union > 0.0 else 0.0


def evaluate(image_dir: Path, label_dir: Path, events_path: Path, threshold: float) -> dict[str, object]:
    images = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line]
    if len(images) != len(events):
        raise ValueError(f"Image/event count mismatch: {len(images)} images, {len(events)} events")
    true_positive: Counter[int] = Counter()
    ground_truth: Counter[int] = Counter()
    predictions: Counter[int] = Counter()
    per_image: list[dict[str, object]] = []
    for image_path, event in zip(images, events):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unreadable image: {image_path}")
        height, width = image.shape[:2]
        targets: list[tuple[int, np.ndarray]] = []
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.is_file():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                values = [float(value) for value in line.split()]
                targets.append((int(values[0]), np.array([
                    [values[index] * width, values[index + 1] * height]
                    for index in range(1, 9, 2)
                ], dtype=np.float32)))
        predicted = [
            (int(item["class_id"]), np.asarray(item["polygon"], dtype=np.float32))
            for item in event.get("detections", [])
        ]
        ground_truth.update(class_id for class_id, _ in targets)
        predictions.update(class_id for class_id, _ in predicted)
        candidates = sorted([
            (
                _iou(target_polygon, prediction_polygon), target_index, prediction_index, target_class
            )
            for target_index, (target_class, target_polygon) in enumerate(targets)
            for prediction_index, (prediction_class, prediction_polygon) in enumerate(predicted)
            if target_class == prediction_class
        ], reverse=True)
        matched_targets: set[int] = set()
        matched_predictions: set[int] = set()
        for iou, target_index, prediction_index, class_id in candidates:
            if iou < threshold:
                break
            if target_index in matched_targets or prediction_index in matched_predictions:
                continue
            matched_targets.add(target_index)
            matched_predictions.add(prediction_index)
            true_positive[class_id] += 1
        per_image.append({
            "image": image_path.name, "ground_truth": len(targets),
            "predictions": len(predicted), "matched": len(matched_targets),
        })
    classes = sorted(set(ground_truth) | set(predictions))
    return {
        "iou_threshold": threshold,
        "images": len(images),
        "classes": {
            str(class_id): {
                "ground_truth": ground_truth[class_id],
                "predictions": predictions[class_id],
                "true_positive": true_positive[class_id],
                "recall": round(true_positive[class_id] / ground_truth[class_id], 4) if ground_truth[class_id] else None,
                "precision": round(true_positive[class_id] / predictions[class_id], 4) if predictions[class_id] else None,
            }
            for class_id in classes
        },
        "lowest_match_images": sorted(
            per_image, key=lambda item: (item["matched"] / item["ground_truth"] if item["ground_truth"] else 1.0, item["image"])
        )[:10],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.images, args.labels, args.events, args.iou), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
