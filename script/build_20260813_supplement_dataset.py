#!/usr/bin/env python3
"""Merge the audited 2026-08-13 supplement into all dataset splits."""

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


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def assign_split(timestamp: float, plan: dict[str, object]) -> str:
    for split in ("guard", "test", "val"):
        for start, end in plan["ranges"].get(split, []):
            if float(start) <= timestamp <= float(end):
                return split
    return str(plan.get("default_split", "train"))


def add_counts(label: Path, split: str, objects: Counter[tuple[str, int]]) -> None:
    for row in label.read_text(encoding="utf-8").splitlines():
        if row.strip():
            objects[(split, int(row.split()[0]))] += 1


def build(base: Path, source: Path, inventory_path: Path, plan_path: Path, output: Path) -> dict[str, object]:
    base = base.resolve()
    source = source.resolve()
    inventory_path = inventory_path.resolve()
    plan_path = plan_path.resolve()
    output = output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")

    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if Path(inventory["source"]).resolve() != source or Path(plan["source"]).resolve() != source:
        raise ValueError("Inventory/plan source does not match --source")
    if Path(plan["base_dataset"]).resolve() != base:
        raise ValueError("Plan base_dataset does not match --base")

    config = yaml.safe_load((base / "data.yaml").read_text(encoding="utf-8")) or {}
    names = list(config.get("names", []))
    class_count = int(config.get("nc", len(names)))
    if names != ["cone", "traffic_sign", "person"] or class_count != 3:
        raise ValueError(f"Expected three-class base, got nc={class_count}, names={names}")

    excluded_manual = set(plan.get("excluded_manual_negatives", {}))
    missing_labels = {record["image"] for record in inventory["missing_labels"]}
    source_images = source / "images" / "Train"
    source_labels = source / "labels" / "Train"
    manual_negative_dir = source / "纯负样本-人工"
    additions: list[dict[str, object]] = []
    for record in inventory["labeled"]:
        image = source_images / record["image"]
        label = source_labels / f"{image.stem}.txt"
        if sha256(image) != record["sha256"]:
            raise ValueError(f"Source image changed since audit: {image}")
        additions.append({**record, "kind": "cvat_labeled", "image_path": image, "label_path": label})
    for record in inventory["manual_negatives"]:
        if record["image"] in excluded_manual:
            continue
        image = manual_negative_dir / record["image"]
        if sha256(image) != record["sha256"]:
            raise ValueError(f"Manual negative changed since audit: {image}")
        additions.append({**record, "kind": "manual_negative", "image_path": image, "label_path": None})

    source_inventory = {
        path.name for path in source_images.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    if source_inventory != {
        record["image"] for record in inventory["labeled"] + inventory["missing_labels"]
    }:
        raise ValueError("CVAT image inventory changed since audit")
    if missing_labels & {record["image"] for record in additions}:
        raise ValueError("A CVAT missing-label image was selected")

    temporary = output.parent / f".{output.name}.building"
    if temporary.exists():
        raise FileExistsError(f"Inspect interrupted build directory first: {temporary}")
    image_counts: Counter[str] = Counter()
    object_counts: Counter[tuple[str, int]] = Counter()
    added_images: Counter[str] = Counter()
    added_objects: Counter[tuple[str, int]] = Counter()
    negative_images: Counter[str] = Counter()
    assignments: list[dict[str, object]] = []
    hashes: dict[str, tuple[str, str]] = {}
    try:
        for split in SPLITS:
            (temporary / "images" / split).mkdir(parents=True)
            (temporary / "labels" / split).mkdir(parents=True)
            for image in sorted((base / "images" / split).iterdir()):
                if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                label = base / "labels" / split / f"{image.stem}.txt"
                if not label.is_file():
                    raise FileNotFoundError(f"Base image has no label: {image}")
                digest = sha256(image)
                if digest in hashes and hashes[digest][0] != split:
                    raise ValueError(
                        f"Base contains a cross-split duplicate of {hashes[digest][1]}: {image}"
                    )
                hashes.setdefault(digest, (split, str(image)))
                link_or_copy(image, temporary / "images" / split / image.name)
                shutil.copy2(label, temporary / "labels" / split / label.name)
                image_counts[split] += 1
                add_counts(label, split, object_counts)

        for record in sorted(additions, key=lambda item: (float(item["timestamp"]), str(item["kind"]))):
            split = assign_split(float(record["timestamp"]), plan)
            rows = list(record.get("validated_rows", [])) if record["kind"] == "cvat_labeled" else []
            assignment = {
                "image": record["image"],
                "timestamp": record["timestamp"],
                "kind": record["kind"],
                "split": split,
                "sha256": record["sha256"],
                "objects": record.get("objects", {}),
            }
            assignments.append(assignment)
            if split == "guard":
                continue
            image = Path(record["image_path"])
            digest = str(record["sha256"])
            if digest in hashes:
                raise ValueError(f"New image duplicates {hashes[digest][1]}: {image}")
            hashes[digest] = (split, str(image))
            stem = f"new_20260813_{record['kind']}__{image.stem}"
            destination_image = temporary / "images" / split / f"{stem}{image.suffix.lower()}"
            destination_label = temporary / "labels" / split / f"{stem}.txt"
            link_or_copy(image, destination_image)
            destination_label.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
            image_counts[split] += 1
            added_images[split] += 1
            if not rows:
                negative_images[split] += 1
            for row in rows:
                class_id = int(row.split()[0])
                object_counts[(split, class_id)] += 1
                added_objects[(split, class_id)] += 1

        manifest: dict[str, object] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "base_dataset": str(base),
            "source": str(source),
            "inventory": str(inventory_path),
            "inventory_sha256": sha256(inventory_path),
            "split_plan": str(plan_path),
            "split_plan_sha256": sha256(plan_path),
            "split_policy": plan["policy"],
            "classes": {str(index): name for index, name in enumerate(names)},
            "source_audit": {
                "cvat_images": inventory["cvat_images"],
                "cvat_labeled_selected": inventory["labeled_images"],
                "cvat_missing_label_excluded": inventory["missing_label_images"],
                "manual_negative_inventory": inventory["manual_negative_images"],
                "manual_negative_excluded": plan["excluded_manual_negatives"],
                "label_coordinates_clipped": inventory["clipped_coordinates"],
                "duplicate_label_rows_removed": inventory["duplicate_label_rows_removed"],
            },
            "added_images": {split: added_images[split] for split in SPLITS},
            "added_negative_images": {split: negative_images[split] for split in SPLITS},
            "added_objects": {
                split: {str(class_id): added_objects[(split, class_id)] for class_id in range(class_count)}
                for split in SPLITS
            },
            "guard_images": sum(item["split"] == "guard" for item in assignments),
            "images": {split: image_counts[split] for split in SPLITS},
            "objects": {
                split: {str(class_id): object_counts[(split, class_id)] for class_id in range(class_count)}
                for split in SPLITS
            },
            "assignments": assignments,
        }
        (temporary / "data.yaml").write_text(yaml.safe_dump({
            "path": ".",
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": class_count,
            "names": names,
        }, allow_unicode=True, sort_keys=False), encoding="utf-8")
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.base, args.source, args.inventory, args.plan, args.output)
    print(json.dumps({
        key: manifest[key]
        for key in ("added_images", "added_negative_images", "added_objects", "guard_images", "images", "objects")
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
