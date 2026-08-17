#!/usr/bin/env python3
"""Move audited historical train-only additions into grouped regression holdouts."""

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

from extend_fused_dataset import IMAGE_SUFFIXES, SPLITS


def file_hash(path: Path) -> str:
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


def find_source_image(source: Path, filename: str) -> Path:
    matches = [
        path for path in source.rglob(filename)
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {filename} under {source}, found {len(matches)}")
    return matches[0]


def load_plan(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    additions = payload.get("additions", [])
    if not additions:
        raise ValueError("Plan must contain additions")
    seen: set[tuple[str, str]] = set()
    for addition in additions:
        source = str(Path(str(addition["source"])).resolve())
        for split in (*SPLITS, "guard"):
            filenames = addition.get(split, [])
            if len(filenames) != len(set(filenames)):
                raise ValueError(f"Duplicate filenames in {source} {split}")
            for filename in filenames:
                key = (source, str(filename))
                if key in seen:
                    raise ValueError(f"Image assigned more than once: {key}")
                seen.add(key)
    return additions


def build_dataset(base: Path, plan_path: Path, output: Path) -> dict[str, object]:
    base, plan_path, output = base.resolve(), plan_path.resolve(), output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    additions = load_plan(plan_path)
    config = yaml.safe_load((base / "data.yaml").read_text(encoding="utf-8")) or {}
    names = list(config.get("names", []))
    class_count = int(config.get("nc", len(names)))
    if len(names) != class_count:
        raise ValueError(f"Base data.yaml has nc={class_count}, names={len(names)}")

    base_train_by_hash: dict[str, list[tuple[Path, Path]]] = {}
    for image in sorted((base / "images" / "train").iterdir()):
        if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = base / "labels" / "train" / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"Base train image has no label: {image}")
        base_train_by_hash.setdefault(file_hash(image), []).append((image, label))

    assignments: list[dict[str, object]] = []
    moved_hashes: dict[str, str] = {}
    for addition in additions:
        source = Path(str(addition["source"])).resolve()
        if not source.exists():
            raise FileNotFoundError(f"Historical source does not exist: {source}")
        for split in (*SPLITS, "guard"):
            if split == "train":
                continue
            for filename in addition.get(split, []):
                source_image = find_source_image(source, str(filename))
                digest = file_hash(source_image)
                matches = base_train_by_hash.get(digest, [])
                if not matches:
                    raise FileNotFoundError(f"No base train copy matches {source_image}")
                if digest in moved_hashes:
                    raise ValueError(f"Duplicate source content in plan: {source_image}")
                moved_hashes[digest] = split
                assignments.append({
                    "source": str(source),
                    "image": source_image.name,
                    "sha256": digest,
                    "split": split,
                    "removed_train_copies": len(matches),
                    "label_source": str(matches[0][1]),
                })

    temporary = output.parent / f".{output.name}.building"
    if temporary.exists():
        raise FileExistsError(f"Inspect interrupted build directory first: {temporary}")
    image_counts: Counter[str] = Counter()
    object_counts: Counter[tuple[str, int]] = Counter()
    try:
        for split in SPLITS:
            (temporary / "images" / split).mkdir(parents=True)
            (temporary / "labels" / split).mkdir(parents=True)
            for image in sorted((base / "images" / split).iterdir()):
                if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                if split == "train" and file_hash(image) in moved_hashes:
                    continue
                label = base / "labels" / split / f"{image.stem}.txt"
                if not label.is_file():
                    raise FileNotFoundError(f"Base image has no label: {image}")
                link_or_copy(image, temporary / "images" / split / image.name)
                shutil.copy2(label, temporary / "labels" / split / label.name)
                image_counts[split] += 1
                for row in label.read_text(encoding="utf-8").splitlines():
                    if row.strip():
                        object_counts[(split, int(float(row.split()[0])))] += 1

        for index, assignment in enumerate(assignments):
            split = str(assignment["split"])
            if split == "guard":
                continue
            source_image = find_source_image(Path(str(assignment["source"])), str(assignment["image"]))
            label_source = Path(str(assignment["label_source"]))
            stem = f"historical_holdout_{index:03d}__{source_image.stem}"
            link_or_copy(source_image, temporary / "images" / split / f"{stem}{source_image.suffix.lower()}")
            shutil.copy2(label_source, temporary / "labels" / split / f"{stem}.txt")
            image_counts[split] += 1
            for row in label_source.read_text(encoding="utf-8").splitlines():
                if row.strip():
                    object_counts[(split, int(float(row.split()[0])))] += 1

        manifest: dict[str, object] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "base_dataset": str(base),
            "split_plan": str(plan_path),
            "split_plan_sha256": file_hash(plan_path),
            "split_policy": "historical train-only additions grouped by source; all repeated train copies removed; one holdout copy kept",
            "holdout_interpretation": "historical regression only because the initialization checkpoint previously saw these images",
            "classes": {str(index): name for index, name in enumerate(names)},
            "assignments": assignments,
            "images": {split: image_counts[split] for split in SPLITS},
            "objects": {
                split: {str(class_id): object_counts[(split, class_id)] for class_id in range(class_count)}
                for split in SPLITS
            },
        }
        (temporary / "data.yaml").write_text(
            yaml.safe_dump({
                "path": ".", "train": "images/train", "val": "images/val",
                "test": "images/test", "nc": class_count, "names": names,
            }, allow_unicode=True, sort_keys=False), encoding="utf-8"
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_dataset(args.base, args.plan, args.output)
    print(json.dumps({key: manifest[key] for key in ("images", "objects")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
