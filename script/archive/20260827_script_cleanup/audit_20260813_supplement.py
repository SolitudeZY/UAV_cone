#!/usr/bin/env python3
"""Audit and inventory the 2026-08-13 mixed CVAT/negative supplement."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import cv2


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
TIMESTAMP_RE = re.compile(r"_(\d{6})\.(\d{3})$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timestamp_seconds(path: Path) -> float:
    match = TIMESTAMP_RE.search(path.stem)
    if match is None:
        raise ValueError(f"Cannot parse timestamp from {path.name}")
    return int(match.group(1)) + int(match.group(2)) / 1000.0


def validated_rows(label: Path) -> tuple[list[str], int, int]:
    rows: list[str] = []
    clipped = 0
    duplicates = 0
    seen: set[tuple[float, ...]] = set()
    for line_number, raw in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        fields = raw.split()
        if len(fields) != 9:
            raise ValueError(f"{label}:{line_number}: expected 9 fields")
        values = tuple(float(value) for value in fields)
        class_id = int(values[0])
        if values[0] != class_id or class_id not in (0, 1, 2):
            raise ValueError(f"{label}:{line_number}: invalid class {values[0]}")
        if values in seen:
            duplicates += 1
            continue
        seen.add(values)
        coordinates = list(values[1:])
        bounded = [min(1.0, max(0.0, value)) for value in coordinates]
        clipped += sum(before != after for before, after in zip(coordinates, bounded))
        points = list(zip(bounded[::2], bounded[1::2]))
        area = abs(sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
        )) / 2.0
        if area <= 1e-8:
            raise ValueError(f"{label}:{line_number}: degenerate polygon after clipping")
        rows.append(" ".join([str(class_id), *(f"{value:.6f}" for value in bounded)]))
    return rows, clipped, duplicates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    image_dir = source / "images" / "Train"
    label_dir = source / "labels" / "Train"
    manual_negative_dir = source / "纯负样本-人工"
    images = {
        path.stem: path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    labels = {path.stem: path for path in label_dir.glob("*.txt")}
    manual_negatives = sorted(
        path for path in manual_negative_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )

    class_counts: Counter[int] = Counter()
    labeled: list[dict[str, object]] = []
    clipped_coordinates = 0
    duplicate_rows = 0
    for stem in sorted(images.keys() & labels.keys()):
        rows, clipped, duplicates = validated_rows(labels[stem])
        classes = Counter(int(row.split()[0]) for row in rows)
        class_counts.update(classes)
        clipped_coordinates += clipped
        duplicate_rows += duplicates
        labeled.append({
            "image": images[stem].name,
            "timestamp": timestamp_seconds(images[stem]),
            "sha256": sha256(images[stem]),
            "objects": {str(key): value for key, value in sorted(classes.items())},
            "validated_rows": rows,
        })

    missing_labels = [
        {
            "image": images[stem].name,
            "timestamp": timestamp_seconds(images[stem]),
            "sha256": sha256(images[stem]),
        }
        for stem in sorted(images.keys() - labels.keys())
    ]
    negative_records = [
        {
            "image": path.name,
            "timestamp": timestamp_seconds(path),
            "sha256": sha256(path),
        }
        for path in manual_negatives
    ]
    hashes = [record["sha256"] for record in labeled + missing_labels + negative_records]
    payload = {
        "source": str(source),
        "cvat_images": len(images),
        "cvat_labels": len(labels),
        "labeled_images": len(labeled),
        "missing_label_images": len(missing_labels),
        "manual_negative_images": len(negative_records),
        "objects_per_class": {str(key): value for key, value in sorted(class_counts.items())},
        "clipped_coordinates": clipped_coordinates,
        "duplicate_label_rows_removed": duplicate_rows,
        "orphan_labels": sorted(labels.keys() - images.keys()),
        "duplicate_image_hashes": len(hashes) - len(set(hashes)),
        "labeled": labeled,
        "missing_labels": missing_labels,
        "manual_negatives": negative_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        key: payload[key] for key in (
            "cvat_images", "cvat_labels", "labeled_images", "missing_label_images",
            "manual_negative_images", "objects_per_class", "clipped_coordinates",
            "duplicate_label_rows_removed", "orphan_labels", "duplicate_image_hashes",
        )
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
