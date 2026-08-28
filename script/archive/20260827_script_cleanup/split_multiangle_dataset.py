#!/usr/bin/env python3
"""Build a fused OBB dataset with time-blocked holdouts from one video source."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

from extend_fused_dataset import IMAGE_SUFFIXES, SPLITS, _validated_label_lines


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def parse_video_timestamp(image_name: str) -> tuple[str, float]:
    stem = Path(image_name).stem
    try:
        video, timestamp = stem.rsplit("_", 1)
        return video, float(timestamp)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Expected filename ending in _<seconds>: {image_name}"
        ) from exc


def _load_plan(path: Path) -> tuple[float, dict[str, list[tuple[str, float, float]]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    guard_seconds = float(payload.get("guard_seconds", 0))
    if guard_seconds < 0:
        raise ValueError("guard_seconds cannot be negative")
    ranges: dict[str, list[tuple[str, float, float]]] = {"val": [], "test": []}
    for split in ("val", "test"):
        for item in payload.get("holdouts", {}).get(split, []):
            video = str(item["video"])
            start, end = float(item["start"]), float(item["end"])
            if start >= end:
                raise ValueError(f"Invalid {split} range for {video}: {start} >= {end}")
            ranges[split].append((video, start, end))
    if not ranges["val"] or not ranges["test"]:
        raise ValueError("Plan must define at least one val and one test range")
    for val_video, val_start, val_end in ranges["val"]:
        for test_video, test_start, test_end in ranges["test"]:
            if val_video == test_video and max(val_start, test_start) < min(val_end, test_end):
                raise ValueError(f"Overlapping val/test ranges for {val_video}")
    return guard_seconds, ranges


def assign_split(
    video: str,
    timestamp: float,
    guard_seconds: float,
    ranges: dict[str, list[tuple[str, float, float]]],
) -> str:
    matches = [
        split
        for split, split_ranges in ranges.items()
        if any(
            range_video == video and start <= timestamp < end
            for range_video, start, end in split_ranges
        )
    ]
    if len(matches) > 1:
        raise ValueError(f"Timestamp belongs to multiple holdouts: {video} {timestamp}")
    if matches:
        return matches[0]
    distance = min(
        (
            min(abs(timestamp - start), abs(timestamp - end))
            for split_ranges in ranges.values()
            for range_video, start, end in split_ranges
            if range_video == video
        ),
        default=float("inf"),
    )
    return "guard" if distance < guard_seconds else "train"


def build_dataset(base: Path, source: Path, audit_path: Path, plan_path: Path, output: Path) -> dict[str, object]:
    base, source, audit_path, plan_path, output = (
        base.resolve(), source.resolve(), audit_path.resolve(), plan_path.resolve(), output.resolve()
    )
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if Path(str(audit.get("source", ""))).resolve() != source:
        raise ValueError("Audit source does not match --source")
    if audit.get("selection") != "labeled_plus_approved_missing":
        raise ValueError("This builder requires a labeled_plus_approved_missing audit")
    approved_missing = set(audit.get("approved_missing_label_images", []))
    if approved_missing:
        raise ValueError("Positive-only split does not accept approved missing labels")
    excluded = set(audit.get("excluded_images", []))
    guard_seconds, ranges = _load_plan(plan_path)

    config = yaml.safe_load((base / "data.yaml").read_text(encoding="utf-8")) or {}
    names = list(config.get("names", []))
    class_count = int(config.get("nc", len(names)))
    if len(names) != class_count:
        raise ValueError(f"Base data.yaml has nc={class_count}, names={len(names)}")
    source_images = source / "images" / "Train"
    source_labels = source / "labels" / "Train"
    if not source_images.is_dir() or not source_labels.is_dir():
        raise FileNotFoundError("Expected source images/Train and labels/Train directories")

    temporary = output.parent / f".{output.name}.building"
    if temporary.exists():
        raise FileExistsError(f"Inspect interrupted build directory first: {temporary}")
    image_counts: Counter[str] = Counter()
    object_counts: Counter[tuple[str, int]] = Counter()
    added_counts: Counter[str] = Counter()
    added_objects: Counter[tuple[str, int]] = Counter()
    assignments: dict[str, list[dict[str, object]]] = {
        "train": [], "val": [], "test": [], "guard": []
    }
    removed_base_cross_split_duplicates: list[dict[str, str]] = []
    clipped_rows = 0
    image_hashes: dict[str, tuple[str, str]] = {}
    try:
        base_inventory: dict[str, list[tuple[str, Path]]] = {}
        for split in SPLITS:
            for image in sorted((base / "images" / split).iterdir()):
                if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES:
                    base_inventory.setdefault(_sha256(image), []).append((split, image))
        split_priority = {"train": 0, "val": 1, "test": 2}
        preferred_split = {
            digest: max({split for split, _ in entries}, key=split_priority.__getitem__)
            for digest, entries in base_inventory.items()
        }
        for split in SPLITS:
            (temporary / "images" / split).mkdir(parents=True)
            (temporary / "labels" / split).mkdir(parents=True)
            for image in sorted((base / "images" / split).iterdir()):
                if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                label = base / "labels" / split / f"{image.stem}.txt"
                if not label.is_file():
                    raise FileNotFoundError(f"Base image has no label: {image}")
                digest = _sha256(image)
                if split != preferred_split[digest]:
                    removed_base_cross_split_duplicates.append({
                        "removed_split": split,
                        "removed_image": str(image),
                        "kept_split": preferred_split[digest],
                        "sha256": digest,
                    })
                    continue
                image_hashes[digest] = (split, str(image))
                _link_or_copy(image, temporary / "images" / split / image.name)
                shutil.copy2(label, temporary / "labels" / split / label.name)
                image_counts[split] += 1
                for row in label.read_text(encoding="utf-8").splitlines():
                    if row.strip():
                        object_counts[(split, int(float(row.split()[0])))] += 1

        for image in sorted(source_images.iterdir()):
            if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES or image.name in excluded:
                continue
            label = source_labels / f"{image.stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(f"Audited selected image has no label: {image}")
            video, timestamp = parse_video_timestamp(image.name)
            split = assign_split(video, timestamp, guard_seconds, ranges)
            lines, changed = _validated_label_lines(label, class_count)
            clipped_rows += changed
            classes = [int(line.split()[0]) for line in lines]
            assignments[split].append({
                "image": image.name,
                "video": video,
                "timestamp": timestamp,
                "objects": dict(sorted(Counter(classes).items())),
            })
            added_counts[split] += 1
            for class_id in classes:
                added_objects[(split, class_id)] += 1
            if split == "guard":
                continue
            digest = _sha256(image)
            if digest in image_hashes:
                raise ValueError(f"Source duplicates existing {image_hashes[digest][1]}: {image}")
            image_hashes[digest] = (split, str(image))
            destination_stem = f"new_multiangle__{image.stem}"
            _link_or_copy(image, temporary / "images" / split / f"{destination_stem}{image.suffix.lower()}")
            (temporary / "labels" / split / f"{destination_stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            image_counts[split] += 1
            for class_id in classes:
                object_counts[(split, class_id)] += 1

        selected_count = sum(added_counts.values())
        expected_count = int(audit.get("approved_image_count", selected_count))
        if selected_count != expected_count:
            raise ValueError(f"Audit expected {expected_count} selected images, found {selected_count}")
        manifest: dict[str, object] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "base_dataset": str(base),
            "source": str(source),
            "audit_file": str(audit_path),
            "audit_sha256": _sha256(audit_path),
            "split_plan": str(plan_path),
            "split_plan_sha256": _sha256(plan_path),
            "split_policy": "explicit video time blocks; remaining frames train-only if outside guard",
            "base_cross_split_duplicate_policy": "keep test, then val, then train",
            "removed_base_cross_split_duplicates": removed_base_cross_split_duplicates,
            "guard_seconds": guard_seconds,
            "classes": {str(index): name for index, name in enumerate(names)},
            "clipped_label_rows": clipped_rows,
            "added_images": {split: added_counts[split] for split in assignments},
            "added_objects": {
                split: {str(class_id): added_objects[(split, class_id)] for class_id in range(class_count)}
                for split in assignments
            },
            "images": {split: image_counts[split] for split in SPLITS},
            "objects": {
                split: {str(class_id): object_counts[(split, class_id)] for class_id in range(class_count)}
                for split in SPLITS
            },
            "assignments": assignments,
        }
        (temporary / "data.yaml").write_text(
            yaml.safe_dump({
                "path": ".", "train": "images/train", "val": "images/val",
                "test": "images/test", "nc": class_count, "names": names,
            }, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if output.exists():
            output.rmdir()
        temporary.replace(output)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_dataset(args.base, args.source, args.audit, args.plan, args.output)
    print(json.dumps({key: manifest[key] for key in ("added_images", "added_objects", "images", "objects")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
