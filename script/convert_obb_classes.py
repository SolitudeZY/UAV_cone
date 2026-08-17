#!/usr/bin/env python3
"""Create a three-class traffic-cone dataset from the legacy four-class set."""

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
SOURCE_NAMES = ("cone", "traffic_sign", "person", "vest")
TARGET_NAMES = SOURCE_NAMES[:3]
REMOVED_CLASS_ID = 3


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_inventory_hash(digest, relative_path: Path, size: int, content: bytes | None = None) -> None:
    digest.update(relative_path.as_posix().encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(size).encode("ascii"))
    digest.update(b"\0")
    if content is not None:
        digest.update(content)
    digest.update(b"\n")


def _class_names(config: dict[str, object]) -> tuple[str, ...]:
    names_value = config.get("names", [])
    if isinstance(names_value, dict):
        names = tuple(str(names_value[key]) for key in sorted(names_value, key=int))
    elif isinstance(names_value, list):
        names = tuple(str(name) for name in names_value)
    else:
        raise ValueError("data.yaml 'names' must be a list or mapping")
    class_count = int(config.get("nc", len(names)))
    if class_count != len(names):
        raise ValueError(f"data.yaml has nc={class_count}, but {len(names)} names")
    return names


def _filter_label(path: Path) -> tuple[bytes, Counter[int], Counter[int]]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Label is not UTF-8: {path}") from exc

    source_counts: Counter[int] = Counter()
    target_counts: Counter[int] = Counter()
    retained: list[str] = []
    for line_number, line_with_ending in enumerate(text.splitlines(keepends=True), 1):
        line = line_with_ending.strip()
        if not line:
            retained.append(line_with_ending)
            continue
        fields = line.split()
        if len(fields) != 9:
            raise ValueError(
                f"{path}:{line_number}: expected class + 8 OBB coordinates, got {len(fields)}"
            )
        try:
            values = [float(field) for field in fields]
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: non-numeric OBB row") from exc
        class_id = int(values[0])
        if values[0] != class_id or not 0 <= class_id < len(SOURCE_NAMES):
            raise ValueError(f"{path}:{line_number}: invalid class {values[0]:g}")
        if any(not 0.0 <= value <= 1.0 for value in values[1:]):
            raise ValueError(f"{path}:{line_number}: coordinate outside [0, 1]")
        source_counts[class_id] += 1
        if class_id == REMOVED_CLASS_ID:
            continue
        retained.append(line_with_ending)
        target_counts[class_id] += 1
    return "".join(retained).encode("utf-8"), source_counts, target_counts


def convert_dataset(source: Path, output: Path) -> dict[str, object]:
    source = source.resolve()
    output = output.resolve()
    if source == output:
        raise ValueError("Source and output must be different directories")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")

    config_path = source / "data.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing source config: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    names = _class_names(config)
    if names != SOURCE_NAMES:
        raise ValueError(
            "This conversion requires source classes exactly "
            f"{list(SOURCE_NAMES)}, got {list(names)}"
        )

    temporary = output.parent / f".{output.name}.building"
    if temporary.exists():
        raise FileExistsError(f"Inspect interrupted build directory first: {temporary}")

    image_counts: Counter[str] = Counter()
    source_objects: Counter[tuple[str, int]] = Counter()
    target_objects: Counter[tuple[str, int]] = Counter()
    source_image_inventory = hashlib.sha256()
    source_label_inventory = hashlib.sha256()
    target_label_inventory = hashlib.sha256()

    try:
        for split in SPLITS:
            source_image_dir = source / "images" / split
            source_label_dir = source / "labels" / split
            if not source_image_dir.is_dir() or not source_label_dir.is_dir():
                raise FileNotFoundError(f"Missing images/{split} or labels/{split} in {source}")
            target_image_dir = temporary / "images" / split
            target_label_dir = temporary / "labels" / split
            target_image_dir.mkdir(parents=True)
            target_label_dir.mkdir(parents=True)

            images = [
                path for path in sorted(source_image_dir.iterdir())
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ]
            image_stems = [path.stem for path in images]
            if len(image_stems) != len(set(image_stems)):
                raise ValueError(f"Duplicate image stems in {source_image_dir}")
            labels = {path.stem: path for path in source_label_dir.glob("*.txt")}
            missing = sorted(set(image_stems) - labels.keys())
            orphaned = sorted(labels.keys() - set(image_stems))
            if missing or orphaned:
                raise ValueError(
                    f"Split {split} has {len(missing)} missing and {len(orphaned)} orphan labels"
                )

            for image in images:
                relative_image = image.relative_to(source)
                _link_or_copy(image, target_image_dir / image.name)
                _update_inventory_hash(
                    source_image_inventory, relative_image, image.stat().st_size
                )

                label = labels[image.stem]
                filtered, before, after = _filter_label(label)
                target_label = target_label_dir / label.name
                target_label.write_bytes(filtered)
                source_objects.update({(split, key): count for key, count in before.items()})
                target_objects.update({(split, key): count for key, count in after.items()})
                source_bytes = label.read_bytes()
                _update_inventory_hash(
                    source_label_inventory,
                    label.relative_to(source),
                    len(source_bytes),
                    source_bytes,
                )
                _update_inventory_hash(
                    target_label_inventory,
                    target_label.relative_to(temporary),
                    len(filtered),
                    filtered,
                )
                image_counts[split] += 1

        resolved_config = {
            "path": ".",
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(TARGET_NAMES),
            "names": list(TARGET_NAMES),
        }
        (temporary / "data.yaml").write_text(
            yaml.safe_dump(resolved_config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        manifest: dict[str, object] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_dataset": str(source),
            "conversion": "drop vest class; preserve cone/traffic_sign/person IDs and splits",
            "source_classes": {str(index): name for index, name in enumerate(SOURCE_NAMES)},
            "target_classes": {str(index): name for index, name in enumerate(TARGET_NAMES)},
            "removed_class": {"id": REMOVED_CLASS_ID, "name": SOURCE_NAMES[REMOVED_CLASS_ID]},
            "image_storage": "hardlink with copy fallback",
            "images": {split: image_counts[split] for split in SPLITS},
            "source_objects": {
                split: {
                    str(class_id): source_objects[(split, class_id)]
                    for class_id in range(len(SOURCE_NAMES))
                }
                for split in SPLITS
            },
            "target_objects": {
                split: {
                    str(class_id): target_objects[(split, class_id)]
                    for class_id in range(len(TARGET_NAMES))
                }
                for split in SPLITS
            },
            "removed_objects": {
                split: source_objects[(split, REMOVED_CLASS_ID)] for split in SPLITS
            },
            "provenance": {
                "source_data_yaml_sha256": _sha256(config_path),
                "source_image_inventory_sha256": source_image_inventory.hexdigest(),
                "source_label_inventory_sha256": source_label_inventory.hexdigest(),
                "target_label_inventory_sha256": target_label_inventory.hexdigest(),
            },
        }
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


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=project_root / "dataset" / "2026-08-04-long-neg47",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "dataset" / "2026-08-04-long-neg47-3class",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = convert_dataset(args.source, args.output)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
