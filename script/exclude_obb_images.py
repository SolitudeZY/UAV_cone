#!/usr/bin/env python3
"""Build an OBB dataset while excluding audited whole image/label pairs."""

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


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def class_names(config: dict[str, object]) -> list[str]:
    value = config.get("names", [])
    if isinstance(value, dict):
        names = [str(value[key]) for key in sorted(value, key=int)]
    elif isinstance(value, list):
        names = [str(name) for name in value]
    else:
        raise ValueError("data.yaml 'names' must be a list or mapping")
    class_count = int(config.get("nc", len(names)))
    if class_count <= 0 or class_count != len(names):
        raise ValueError(f"data.yaml has nc={class_count}, but {len(names)} names")
    return names


def label_counts(path: Path, class_count: int) -> Counter[int]:
    counts: Counter[int] = Counter()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        fields = raw_line.split()
        if len(fields) != 9:
            raise ValueError(
                f"{path}:{line_number}: expected class plus 8 OBB coordinates"
            )
        try:
            values = [float(field) for field in fields]
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: non-numeric OBB row") from exc
        class_id = int(values[0])
        if values[0] != class_id or not 0 <= class_id < class_count:
            raise ValueError(f"{path}:{line_number}: invalid class {values[0]:g}")
        if any(not 0.0 <= coordinate <= 1.0 for coordinate in values[1:]):
            raise ValueError(f"{path}:{line_number}: coordinate outside [0, 1]")
        counts[class_id] += 1
    return counts


def load_exclusions(plan_path: Path) -> tuple[dict[str, object], set[Path]]:
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    values = payload.get("excluded_images", [])
    if not isinstance(values, list) or not values:
        raise ValueError("Exclusion plan must contain a non-empty excluded_images list")
    exclusions: set[Path] = set()
    for value in values:
        relative = Path(str(value))
        parts = relative.parts
        if (
            relative.is_absolute()
            or len(parts) != 3
            or parts[0] != "images"
            or parts[1] not in SPLITS
            or relative.suffix.lower() not in IMAGE_SUFFIXES
            or ".." in parts
        ):
            raise ValueError(f"Invalid excluded image path: {value}")
        if relative in exclusions:
            raise ValueError(f"Duplicate excluded image path: {value}")
        exclusions.add(relative)
    return payload, exclusions


def build_filtered_dataset(
    source: Path, plan_path: Path, output: Path
) -> dict[str, object]:
    source = source.resolve()
    plan_path = plan_path.resolve()
    output = output.resolve()
    if source == output:
        raise ValueError("Source and output must be different directories")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")

    config_path = source / "data.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing source config: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    names = class_names(config)
    class_count = len(names)
    plan, exclusions = load_exclusions(plan_path)

    missing_exclusions = {
        relative for relative in exclusions if not (source / relative).is_file()
    }
    if missing_exclusions:
        raise FileNotFoundError(
            f"Excluded images missing from source: {sorted(map(str, missing_exclusions))}"
        )

    temporary = output.parent / f".{output.name}.building"
    if temporary.exists():
        raise FileExistsError(f"Inspect interrupted build directory first: {temporary}")

    image_counts: Counter[str] = Counter()
    object_counts: Counter[tuple[str, int]] = Counter()
    removed_objects: Counter[tuple[str, int]] = Counter()
    removed: list[dict[str, object]] = []
    try:
        for split in SPLITS:
            source_images = source / "images" / split
            source_labels = source / "labels" / split
            if not source_images.is_dir() or not source_labels.is_dir():
                raise FileNotFoundError(
                    f"Missing images/{split} or labels/{split} in {source}"
                )
            target_images = temporary / "images" / split
            target_labels = temporary / "labels" / split
            target_images.mkdir(parents=True)
            target_labels.mkdir(parents=True)

            images = {
                path.stem: path
                for path in source_images.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            }
            labels = {path.stem: path for path in source_labels.glob("*.txt")}
            missing_labels = sorted(images.keys() - labels.keys())
            orphan_labels = sorted(labels.keys() - images.keys())
            if missing_labels or orphan_labels:
                raise ValueError(
                    f"Split {split} has {len(missing_labels)} missing and "
                    f"{len(orphan_labels)} orphan labels"
                )

            for stem, image in sorted(images.items()):
                label = labels[stem]
                counts = label_counts(label, class_count)
                relative_image = image.relative_to(source)
                if relative_image in exclusions:
                    removed.append(
                        {
                            "image": relative_image.as_posix(),
                            "label": label.relative_to(source).as_posix(),
                            "image_sha256": file_hash(image),
                            "label_sha256": file_hash(label),
                            "objects": {
                                str(class_id): counts[class_id]
                                for class_id in range(class_count)
                            },
                        }
                    )
                    for class_id, count in counts.items():
                        removed_objects[(split, class_id)] += count
                    continue

                link_or_copy(image, target_images / image.name)
                shutil.copy2(label, target_labels / label.name)
                image_counts[split] += 1
                for class_id, count in counts.items():
                    object_counts[(split, class_id)] += count

        if len(removed) != len(exclusions):
            raise RuntimeError(
                f"Removed {len(removed)} images, expected {len(exclusions)}"
            )

        resolved_config = {
            "path": ".",
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": class_count,
            "names": names,
        }
        (temporary / "data.yaml").write_text(
            yaml.safe_dump(resolved_config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        manifest: dict[str, object] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_dataset": str(source),
            "exclusion_plan": str(plan_path),
            "exclusion_plan_sha256": file_hash(plan_path),
            "policy": "exclude complete image/label pairs; never retain images with removed true-object labels",
            "reason": plan.get("reason", "audited whole-frame exclusion"),
            "classes": {str(index): name for index, name in enumerate(names)},
            "image_storage": "hardlink with copy fallback; labels copied",
            "removed_images": removed,
            "removed_image_counts": {
                split: sum(item["image"].startswith(f"images/{split}/") for item in removed)
                for split in SPLITS
            },
            "removed_objects": {
                split: {
                    str(class_id): removed_objects[(split, class_id)]
                    for class_id in range(class_count)
                }
                for split in SPLITS
            },
            "images": {split: image_counts[split] for split in SPLITS},
            "objects": {
                split: {
                    str(class_id): object_counts[(split, class_id)]
                    for class_id in range(class_count)
                }
                for split in SPLITS
            },
            "provenance": {
                "source_data_yaml_sha256": file_hash(config_path),
                "source_manifest_sha256": (
                    file_hash(source / "manifest.json")
                    if (source / "manifest.json").is_file()
                    else None
                ),
            },
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if output.exists():
            output.rmdir()
        temporary.replace(output)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = build_filtered_dataset(args.source, args.plan, args.output)
    summary = {
        key: manifest[key]
        for key in ("removed_image_counts", "removed_objects", "images", "objects")
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
