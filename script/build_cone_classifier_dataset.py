#!/usr/bin/env python3
"""Build an audited ImageFolder dataset from reviewed cone candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from mine_classifier_candidates import crop_square


CLASSES = ("fallen_cone", "not_cone", "upright_cone")
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates-root",
        type=Path,
        default=root / "dataset" / "cone_candidate_classifier_20260814" / "candidates",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "dataset" / "cone_posture_classifier_20260814",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory after all input checks pass.",
    )
    parser.add_argument(
        "--crop-context",
        type=float,
        help="Regenerate unboxed crops from source images with this context multiplier.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(path: Path) -> str:
    with Image.open(path) as image:
        pixels = list(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR).getdata())
    bits = 0
    for row in range(8):
        for column in range(8):
            bits = (bits << 1) | int(pixels[row * 9 + column] > pixels[row * 9 + column + 1])
    return f"{bits:016x}"


def difference_hash_bytes(payload: bytes) -> str:
    with Image.open(io.BytesIO(payload)) as image:
        pixels = list(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR).getdata())
    bits = 0
    for row in range(8):
        for column in range(8):
            bits = (bits << 1) | int(pixels[row * 9 + column] > pixels[row * 9 + column + 1])
    return f"{bits:016x}"


@lru_cache(maxsize=16)
def read_source_image(source_path: str) -> np.ndarray:
    image = cv2.imread(source_path)
    if image is None:
        raise SystemExit(f"Unreadable source image: {source_path}")
    return image


def regenerate_crop(source_path: Path, polygon_json: str, context: float) -> bytes:
    image = read_source_image(str(source_path))
    polygon = np.asarray(json.loads(polygon_json), dtype=np.float32)
    crop, _ = crop_square(image, polygon, context=context, output_size=224)
    encoded, buffer = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not encoded:
        raise SystemExit(f"Failed to encode regenerated crop: {source_path}")
    return buffer.tobytes()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"Missing required CSV: {path}")
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load_inputs(root: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    candidates = read_csv(root / "candidates.csv")
    annotation_rows = read_csv(root / "review" / "annotations.csv")
    annotations: dict[str, dict[str, str]] = {}
    for row in annotation_rows:
        candidate_id = row["candidate_id"]
        if candidate_id in annotations:
            raise SystemExit(f"Duplicate annotation for candidate_id={candidate_id}")
        annotations[candidate_id] = row

    candidate_ids = {row["candidate_id"] for row in candidates}
    missing = candidate_ids - annotations.keys()
    extra = annotations.keys() - candidate_ids
    if missing or extra:
        raise SystemExit(
            f"Review is incomplete or inconsistent: missing={len(missing)}, extra={len(extra)}"
        )
    invalid = sorted({row["label"] for row in annotation_rows} - set(CLASSES) - {"ignore"})
    if invalid:
        raise SystemExit(f"Unsupported review labels: {invalid}")
    return candidates, annotations


def source_images_root(candidates_root: Path) -> Path:
    manifest_path = candidates_root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = Path(manifest["images"]).expanduser().resolve()
    if not path.is_dir():
        raise SystemExit(f"Source image root from run_manifest.json does not exist: {path}")
    return path


def main() -> int:
    args = parse_args()
    candidates_root = args.candidates_root.resolve()
    output = args.output.resolve()
    if args.crop_context is not None and (
        not math.isfinite(args.crop_context) or args.crop_context < 1.0
    ):
        raise SystemExit("--crop-context must be a finite value >= 1.0")
    candidates, annotations = load_inputs(candidates_root)
    images_root = source_images_root(candidates_root)

    reviewed_counts = Counter(annotations[row["candidate_id"]]["label"] for row in candidates)
    usable: list[dict[str, str]] = []
    for row in candidates:
        label = annotations[row["candidate_id"]]["label"]
        if label == "ignore":
            continue
        split = row["source_split"]
        if split not in SPLITS:
            raise SystemExit(f"Unsupported source_split={split!r} for {row['candidate_id']}")
        crop_path = candidates_root / row["crop_path"]
        source_path = images_root / row["source_image"]
        if not source_path.is_file() or (args.crop_context is None and not crop_path.is_file()):
            raise SystemExit(f"Missing crop/source for {row['candidate_id']}: {crop_path}, {source_path}")
        item = dict(row)
        item["label"] = label
        if args.crop_context is None:
            item["crop_sha256"] = sha256_file(crop_path)
            item["crop_dhash"] = difference_hash(crop_path)
            item["crop_absolute"] = str(crop_path)
        else:
            crop_bytes = regenerate_crop(source_path, row["polygon_json"], args.crop_context)
            item["crop_sha256"] = hashlib.sha256(crop_bytes).hexdigest()
            item["crop_dhash"] = difference_hash_bytes(crop_bytes)
            item["crop_bytes"] = crop_bytes
        item["source_absolute"] = str(source_path)
        usable.append(item)

    exact_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in usable:
        exact_groups[item["crop_sha256"]].append(item)

    conflicts = []
    cross_split_exact = []
    for digest, group in exact_groups.items():
        labels = sorted({item["label"] for item in group})
        splits = sorted({item["source_split"] for item in group})
        if len(labels) > 1:
            conflicts.append({"sha256": digest, "labels": labels, "ids": [x["candidate_id"] for x in group]})
        if len(splits) > 1:
            cross_split_exact.append({"sha256": digest, "splits": splits, "ids": [x["candidate_id"] for x in group]})
    if conflicts or cross_split_exact:
        raise SystemExit(
            f"Unsafe crop duplicates: label_conflicts={len(conflicts)}, "
            f"cross_split={len(cross_split_exact)}"
        )

    deduplicated = [
        sorted(group, key=lambda item: (item["source_image"], item["candidate_id"]))[0]
        for group in exact_groups.values()
    ]
    deduplicated.sort(key=lambda item: (item["source_split"], item["label"], item["candidate_id"]))

    source_hash_cache: dict[str, str] = {}
    source_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in deduplicated:
        source_path = item["source_absolute"]
        if source_path not in source_hash_cache:
            source_hash_cache[source_path] = sha256_file(Path(source_path))
        item["source_sha256"] = source_hash_cache[source_path]
        source_groups[item["source_sha256"]].append(item)
    source_cross_split = [
        {
            "source_sha256": digest,
            "splits": sorted({item["source_split"] for item in group}),
            "sources": sorted({item["source_image"] for item in group}),
        }
        for digest, group in source_groups.items()
        if len({item["source_split"] for item in group}) > 1
    ]
    if source_cross_split:
        raise SystemExit(f"Found {len(source_cross_split)} exact source images across splits")

    perceptual_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in deduplicated:
        perceptual_groups[item["crop_dhash"]].append(item)
    perceptual_cross_split = [
        {
            "dhash": digest,
            "splits": sorted({item["source_split"] for item in group}),
            "ids": [item["candidate_id"] for item in group],
        }
        for digest, group in perceptual_groups.items()
        if len(group) > 1 and len({item["source_split"] for item in group}) > 1
    ]

    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"Output already exists (use --overwrite): {output}")
        shutil.rmtree(output)

    for split in SPLITS:
        for label in CLASSES:
            (output / split / label).mkdir(parents=True, exist_ok=True)
    metadata = output / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)

    sample_fields = (
        "candidate_id", "split", "class", "image_path", "crop_sha256", "crop_dhash",
        "source_image", "source_sha256", "candidate_kind", "review_pool", "yolo_confidence",
        "raw_angle_fallen", "matched_gt_index", "polygon_json", "crop_context",
    )
    sample_rows = []
    for item in deduplicated:
        filename = f"{item['candidate_id']}__{item['crop_sha256'][:12]}.jpg"
        relative = Path(item["source_split"]) / item["label"] / filename
        if args.crop_context is None:
            shutil.copy2(item["crop_absolute"], output / relative)
        else:
            (output / relative).write_bytes(item["crop_bytes"])
        sample_rows.append(
            {
                "candidate_id": item["candidate_id"],
                "split": item["source_split"],
                "class": item["label"],
                "image_path": relative.as_posix(),
                "crop_sha256": item["crop_sha256"],
                "crop_dhash": item["crop_dhash"],
                "source_image": item["source_image"],
                "source_sha256": item["source_sha256"],
                "candidate_kind": item["candidate_kind"],
                "review_pool": item["review_pool"],
                "yolo_confidence": item["yolo_confidence"],
                "raw_angle_fallen": item["raw_angle_fallen"],
                "matched_gt_index": item["matched_gt_index"],
                "polygon_json": item["polygon_json"],
                "crop_context": args.crop_context or 2.0,
            }
        )
    with (metadata / "samples.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sample_fields)
        writer.writeheader()
        writer.writerows(sample_rows)

    split_counts = Counter((item["source_split"], item["label"]) for item in deduplicated)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidates_root": str(candidates_root),
        "source_images_root": str(images_root),
        "output": str(output),
        "policy": {
            "classes": list(CLASSES),
            "ignored_label": "ignore",
            "split_source": "source_split from candidates.csv; never reassigned",
            "exact_deduplication": "SHA-256 of unboxed 224x224 crop",
            "perceptual_audit": "64-bit difference hash; audit only, not auto-removed",
            "crop_source": "reviewed crop" if args.crop_context is None else "regenerated from source image",
            "crop_context": args.crop_context or 2.0,
        },
        "input": {
            "candidates": len(candidates),
            "reviewed_counts": dict(sorted(reviewed_counts.items())),
            "usable_before_deduplication": len(usable),
        },
        "deduplication": {
            "unique_crops": len(deduplicated),
            "removed_exact_duplicates": len(usable) - len(deduplicated),
            "duplicate_groups": sum(len(group) > 1 for group in exact_groups.values()),
            "label_conflicts": 0,
            "cross_split_exact_crops": 0,
            "cross_split_exact_sources": 0,
            "cross_split_same_dhash_groups": len(perceptual_cross_split),
        },
        "split_class_counts": {
            split: {label: split_counts[(split, label)] for label in CLASSES}
            for split in SPLITS
        },
        "input_sha256": {
            "candidates.csv": sha256_file(candidates_root / "candidates.csv"),
            "annotations.csv": sha256_file(candidates_root / "review" / "annotations.csv"),
        },
    }
    (metadata / "manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (metadata / "perceptual_cross_split.json").write_text(
        json.dumps(perceptual_cross_split, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
