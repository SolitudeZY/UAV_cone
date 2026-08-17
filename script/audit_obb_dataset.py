#!/usr/bin/env python3
"""Audit one or more YOLO OBB datasets without modifying source files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _find_split(root: Path) -> tuple[Path, Path]:
    for name in ("train", "Train"):
        image_dir = root / "images" / name
        label_dir = root / "labels" / name
        if image_dir.is_dir() and label_dir.is_dir():
            return image_dir, label_dir
    raise FileNotFoundError(f"No matching images/train or images/Train under {root}")


def _polygon_area(points: list[tuple[float, float]]) -> float:
    return abs(sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
    )) / 2.0


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(root: Path) -> dict[str, object]:
    root = root.resolve()
    image_dir, label_dir = _find_split(root)
    images = {
        path.stem: path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    labels = {path.stem: path for path in label_dir.glob("*.txt")}
    issues: list[dict[str, object]] = []
    class_counts: Counter[int] = Counter()
    dimensions: Counter[str] = Counter()
    object_counts: list[int] = []
    hashes: defaultdict[str, list[str]] = defaultdict(list)

    for stem in sorted(images.keys() - labels.keys()):
        issues.append({"severity": "error", "type": "missing_label", "file": str(images[stem])})
    for stem in sorted(labels.keys() - images.keys()):
        issues.append({"severity": "error", "type": "orphan_label", "file": str(labels[stem])})

    for stem, image_path in sorted(images.items()):
        image = cv2.imread(str(image_path))
        if image is None:
            issues.append({"severity": "error", "type": "unreadable_image", "file": str(image_path)})
            continue
        height, width = image.shape[:2]
        dimensions[f"{width}x{height}"] += 1
        hashes[_file_hash(image_path)].append(str(image_path))
        label_path = labels.get(stem)
        if label_path is None:
            continue
        rows = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        object_counts.append(len(rows))
        if not rows:
            issues.append({"severity": "warning", "type": "empty_label", "file": str(label_path)})
        seen: set[tuple[float, ...]] = set()
        for line_number, row in enumerate(rows, 1):
            fields = row.split()
            if len(fields) != 9:
                issues.append({"severity": "error", "type": "field_count", "file": str(label_path), "line": line_number})
                continue
            try:
                values = tuple(float(field) for field in fields)
            except ValueError:
                issues.append({"severity": "error", "type": "non_numeric", "file": str(label_path), "line": line_number})
                continue
            class_id = int(values[0])
            if values[0] != class_id or class_id < 0:
                issues.append({"severity": "error", "type": "invalid_class", "file": str(label_path), "line": line_number})
                continue
            class_counts[class_id] += 1
            coordinates = values[1:]
            if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in coordinates):
                issues.append({"severity": "error", "type": "coordinate_range", "file": str(label_path), "line": line_number, "values": coordinates})
            points = list(zip(coordinates[::2], coordinates[1::2]))
            area = _polygon_area(points)
            if area <= 1e-8:
                issues.append({"severity": "error", "type": "degenerate_polygon", "file": str(label_path), "line": line_number})
            elif area * width * height < 4.0:
                issues.append({"severity": "warning", "type": "tiny_polygon", "file": str(label_path), "line": line_number, "area_pixels": area * width * height})
            rounded = tuple(round(value, 8) for value in values)
            if rounded in seen:
                issues.append({"severity": "error", "type": "duplicate_box", "file": str(label_path), "line": line_number})
            seen.add(rounded)

    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    for paths in duplicate_groups:
        issues.append({"severity": "warning", "type": "exact_duplicate_images", "files": paths})
    errors = sum(issue["severity"] == "error" for issue in issues)
    warnings = len(issues) - errors
    return {
        "root": str(root),
        "images": len(images),
        "labels": len(labels),
        "labeled_images": len(images.keys() & labels.keys()),
        "objects": sum(class_counts.values()),
        "objects_per_class": dict(sorted(class_counts.items())),
        "objects_per_labeled_image": {
            "min": min(object_counts, default=0),
            "max": max(object_counts, default=0),
            "mean": round(sum(object_counts) / len(object_counts), 2) if object_counts else 0,
        },
        "dimensions": dict(dimensions.most_common()),
        "exact_duplicate_groups": len(duplicate_groups),
        "errors": errors,
        "warnings": warnings,
        "issues": issues,
    }


def write_preview(root: Path, output: Path, columns: int = 4) -> None:
    image_dir, label_dir = _find_split(root.resolve())
    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    tile_width, tile_height = 480, 300
    rows = math.ceil(len(image_paths) / columns)
    canvas = np.full((rows * tile_height, columns * tile_width, 3), 245, dtype=np.uint8)
    colors = ((0, 220, 0), (0, 180, 255), (255, 80, 0), (255, 0, 255))
    for index, image_path in enumerate(image_paths):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        source_height, source_width = image.shape[:2]
        scale = min(tile_width / source_width, (tile_height - 28) / source_height)
        resized = cv2.resize(image, (round(source_width * scale), round(source_height * scale)))
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.is_file():
            for row in label_path.read_text(encoding="utf-8").splitlines():
                fields = row.split()
                if len(fields) != 9:
                    continue
                values = [float(field) for field in fields]
                class_id = int(values[0])
                points = np.array([
                    [round(values[i] * source_width * scale), round(values[i + 1] * source_height * scale)]
                    for i in range(1, 9, 2)
                ], dtype=np.int32)
                cv2.polylines(resized, [points], True, colors[class_id % len(colors)], 2, cv2.LINE_AA)
            status = ""
            text_color = (30, 30, 30)
        else:
            status = "  MISSING LABEL"
            text_color = (0, 0, 255)
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        cv2.putText(canvas, image_path.name + status, (x + 5, y + tile_height - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, text_color, 1, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise OSError(f"Failed to write preview: {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    reports = [audit(root) for root in args.roots]
    if args.preview_dir:
        for root in args.roots:
            write_preview(root, args.preview_dir / f"{root.name}_preview.jpg")
    payload = {"datasets": reports}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 1 if any(report["errors"] for report in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
