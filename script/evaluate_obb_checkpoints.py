#!/usr/bin/env python3
"""Evaluate YOLO OBB checkpoints at a fixed deployment threshold."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def polygon_iou(first: np.ndarray, second: np.ndarray) -> float:
    first = cv2.convexHull(first.astype(np.float32))
    second = cv2.convexHull(second.astype(np.float32))
    area_first = abs(float(cv2.contourArea(first)))
    area_second = abs(float(cv2.contourArea(second)))
    intersection, _ = cv2.intersectConvexConvex(first, second)
    union = area_first + area_second - intersection
    return intersection / union if union > 0.0 else 0.0


def read_targets(image_path: Path, label_dir: Path) -> list[tuple[int, np.ndarray]]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unreadable image: {image_path}")
    height, width = image.shape[:2]
    label_path = label_dir / f"{image_path.stem}.txt"
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing label: {label_path}")
    targets = []
    for row in label_path.read_text(encoding="utf-8").splitlines():
        if not row.strip():
            continue
        values = [float(value) for value in row.split()]
        targets.append(
            (
                int(values[0]),
                np.asarray(
                    [
                        [values[index] * width, values[index + 1] * height]
                        for index in range(1, 9, 2)
                    ],
                    dtype=np.float32,
                ),
            )
        )
    return targets


def evaluate_checkpoint(
    weights: Path,
    image_paths: list[Path],
    label_dir: Path,
    imgsz: int,
    confidence: float,
    iou_threshold: float,
    device: str,
    batch: int,
    source_directory: Path,
) -> dict[str, object]:
    from ultralytics import YOLO

    ground_truth: Counter[int] = Counter()
    predictions: Counter[int] = Counter()
    true_positive: Counter[int] = Counter()
    empty_images = 0
    false_positive_empty_images = 0
    false_positive_empty_boxes = 0
    false_positive_empty_images_by_class: Counter[int] = Counter()
    false_positive_empty_boxes_by_class: Counter[int] = Counter()
    model = YOLO(str(weights), task="obb")
    image_by_name = {path.name: path for path in image_paths}
    if len(image_by_name) != len(image_paths):
        raise ValueError("Image filenames must be unique")
    results = model.predict(
        source=str(source_directory),
        imgsz=imgsz,
        conf=confidence,
        iou=0.5,
        device=device,
        batch=batch,
        stream=True,
        verbose=False,
    )
    try:
        processed_names: set[str] = set()
        for result in results:
            image_path = image_by_name[Path(result.path).name]
            processed_names.add(image_path.name)
            targets = read_targets(image_path, label_dir)
            predicted = []
            if result.obb is not None:
                polygons = result.obb.xyxyxyxy.cpu().numpy()
                classes = result.obb.cls.cpu().numpy().astype(int)
                predicted = list(zip(classes.tolist(), polygons))
            ground_truth.update(class_id for class_id, _ in targets)
            predictions.update(class_id for class_id, _ in predicted)
            if not targets:
                empty_images += 1
                if predicted:
                    false_positive_empty_images += 1
                    false_positive_empty_boxes += len(predicted)
                    predicted_classes = Counter(class_id for class_id, _ in predicted)
                    false_positive_empty_images_by_class.update(predicted_classes.keys())
                    false_positive_empty_boxes_by_class.update(predicted_classes)

            candidates = sorted(
                (
                    polygon_iou(target_polygon, predicted_polygon),
                    target_index,
                    prediction_index,
                    target_class,
                )
                for target_index, (target_class, target_polygon) in enumerate(targets)
                for prediction_index, (prediction_class, predicted_polygon) in enumerate(predicted)
                if target_class == prediction_class
            )
            matched_targets: set[int] = set()
            matched_predictions: set[int] = set()
            for overlap, target_index, prediction_index, class_id in reversed(candidates):
                if overlap < iou_threshold:
                    break
                if target_index in matched_targets or prediction_index in matched_predictions:
                    continue
                matched_targets.add(target_index)
                matched_predictions.add(prediction_index)
                true_positive[class_id] += 1
        if processed_names != image_by_name.keys():
            raise RuntimeError("Predictor did not return exactly one result per input image")
    finally:
        del results
        del model
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    classes = sorted(set(ground_truth) | set(predictions))
    return {
        "weights": str(weights.resolve()),
        "images": len(image_paths),
        "empty_images": empty_images,
        "false_positive_empty_images": false_positive_empty_images,
        "false_positive_empty_boxes": false_positive_empty_boxes,
        "false_positive_empty_images_by_class": {
            str(class_id): false_positive_empty_images_by_class[class_id]
            for class_id in classes
        },
        "false_positive_empty_boxes_by_class": {
            str(class_id): false_positive_empty_boxes_by_class[class_id]
            for class_id in classes
        },
        "classes": {
            str(class_id): {
                "ground_truth": ground_truth[class_id],
                "predictions": predictions[class_id],
                "true_positive": true_positive[class_id],
                "false_positive": predictions[class_id] - true_positive[class_id],
                "false_negative": ground_truth[class_id] - true_positive[class_id],
                "precision": round(true_positive[class_id] / predictions[class_id], 6)
                if predictions[class_id]
                else None,
                "recall": round(true_positive[class_id] / ground_truth[class_id], 6)
                if ground_truth[class_id]
                else None,
            }
            for class_id in classes
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", nargs="+", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument(
        "--image-prefix",
        default="",
        help="Evaluate only image filenames beginning with this prefix",
    )
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    image_paths = sorted(
        path
        for path in args.images.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and path.name.startswith(args.image_prefix)
    )
    if not image_paths:
        raise SystemExit("No matching images found")
    with tempfile.TemporaryDirectory(prefix="obb-eval-") as temporary:
        source_directory = Path(temporary)
        for image_path in image_paths:
            os.symlink(image_path.resolve(), source_directory / image_path.name)
        payload = {
            "settings": {
                "images": str(args.images.resolve()),
                "labels": str(args.labels.resolve()),
                "image_prefix": args.image_prefix,
                "imgsz": args.imgsz,
                "confidence": args.conf,
                "iou_threshold": args.iou,
            },
            "checkpoints": [
                evaluate_checkpoint(
                    weights,
                    image_paths,
                    args.labels,
                    args.imgsz,
                    args.conf,
                    args.iou,
                    args.device,
                    args.batch,
                    source_directory,
                )
                for weights in args.weights
            ],
        }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
