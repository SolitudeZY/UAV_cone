#!/usr/bin/env python3
"""Extend an existing fused OBB dataset with audited train-only sources."""

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


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _validated_label_lines(path: Path | None, class_count: int) -> tuple[list[str], int]:
    if path is None:
        return [], 0
    output: list[str] = []
    clipped_rows = 0
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        fields = raw_line.split()
        if len(fields) != 9:
            raise ValueError(f"{path}:{line_number}: expected class + 8 OBB coordinates")
        try:
            values = [float(field) for field in fields]
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: non-numeric OBB row") from exc
        class_id = int(values[0])
        if values[0] != class_id or not 0 <= class_id < class_count:
            raise ValueError(f"{path}:{line_number}: invalid class {values[0]:g}")
        coordinates = [min(1.0, max(0.0, value)) for value in values[1:]]
        if coordinates != values[1:]:
            clipped_rows += 1
        output.append(" ".join([str(class_id), *(f"{value:.10f}" for value in coordinates)]))
    return output, clipped_rows


def _source_directories(source: Path) -> tuple[Path, Path]:
    for split in ("Train", "Validation", "train", "val"):
        image_dir = source / "images" / split
        if image_dir.is_dir():
            return image_dir, source / "labels" / split
    if any(
        path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        for path in source.iterdir()
    ):
        return source, source / "labels"
    raise FileNotFoundError(
        f"No flat images or images/Train|Validation directory found: {source}"
    )


def _load_audit(path: Path, source: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Audit file does not exist: {path}")
    audit = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(audit, dict):
        raise ValueError(f"Audit file must contain a JSON object: {path}")
    recorded_source = audit.get("source")
    if recorded_source and Path(str(recorded_source)).resolve() != source:
        raise ValueError(f"Audit source does not match --add: {path}")
    if audit.get("selection") not in {
        "all_except_excluded",
        "labeled_plus_approved_missing",
    }:
        raise ValueError(f"Unsupported audit selection in {path}")
    return audit


def extend_dataset(
    base: Path,
    additions: list[Path],
    output: Path,
    repeat: int,
    allow_missing_as_negative: bool,
    image_list: Path | None = None,
    audit_files: list[Path] | None = None,
) -> dict[str, object]:
    base, output = base.resolve(), output.resolve()
    additions = [path.resolve() for path in additions]
    image_list = image_list.resolve() if image_list is not None else None
    audit_files = [path.resolve() for path in (audit_files or [])]
    if repeat < 1:
        raise ValueError("--repeat must be at least 1")
    if image_list is not None and len(additions) != 1:
        raise ValueError("--image-list requires exactly one --add directory")
    if image_list is not None and audit_files:
        raise ValueError("--image-list and --audit are mutually exclusive")
    if audit_files and len(audit_files) != len(additions):
        raise ValueError("--audit requires exactly one JSON file per --add directory")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    config = yaml.safe_load((base / "data.yaml").read_text(encoding="utf-8")) or {}
    names = config.get("names", [])
    class_count = int(config.get("nc", len(names)))
    if len(names) != class_count:
        raise ValueError(f"Base data.yaml has nc={class_count}, names={len(names)}")

    temporary = output.parent / f".{output.name}.building"
    if temporary.exists():
        raise FileExistsError(f"Inspect interrupted build directory first: {temporary}")
    image_counts: Counter[str] = Counter()
    object_counts: Counter[tuple[str, int]] = Counter()
    added_sources: list[dict[str, object]] = []
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
                _link_or_copy(image, temporary / "images" / split / image.name)
                shutil.copy2(label, temporary / "labels" / split / label.name)
                image_counts[split] += 1
                for row in label.read_text(encoding="utf-8").splitlines():
                    if row.strip():
                        object_counts[(split, int(float(row.split()[0])))] += 1

        for source_index, source in enumerate(additions):
            image_dir, label_dir = _source_directories(source)
            selected_names: set[str] | None = None
            audit = (
                _load_audit(audit_files[source_index], source)
                if audit_files
                else None
            )
            if image_list is not None:
                if not image_list.is_file():
                    raise FileNotFoundError(f"Image list does not exist: {image_list}")
                listed_names = [
                    line.strip()
                    for line in image_list.read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.lstrip().startswith("#")
                ]
                if not listed_names or len(set(listed_names)) != len(listed_names):
                    raise ValueError("--image-list must contain unique image filenames")
                if any(Path(name).name != name for name in listed_names):
                    raise ValueError("--image-list entries must be filenames, not paths")
                selected_names = set(listed_names)
                image_dir = source
                label_dir = source / "labels"
            all_source_images = sorted(
                path for path in image_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() in IMAGE_SUFFIXES
                and (selected_names is None or path.name in selected_names)
            )
            excluded_names: set[str] = set()
            approved_missing_names: set[str] = set()
            if audit is not None:
                excluded_names = set(audit.get("excluded_images", []))
                approved_missing_names = set(
                    audit.get("approved_missing_label_images", [])
                )
                available_names = {path.name for path in all_source_images}
                unknown = sorted(
                    (excluded_names | approved_missing_names) - available_names
                )
                if unknown:
                    raise FileNotFoundError(
                        f"Audit entries not found in {source}: {unknown[:10]}"
                    )
                if excluded_names & approved_missing_names:
                    raise ValueError(
                        f"Audit cannot both exclude and approve an image: {source}"
                    )
                if audit["selection"] == "labeled_plus_approved_missing":
                    source_images = [
                        path
                        for path in all_source_images
                        if path.name not in excluded_names
                        and (
                            (label_dir / f"{path.stem}.txt").is_file()
                            or path.name in approved_missing_names
                        )
                    ]
                else:
                    source_images = [
                        path
                        for path in all_source_images
                        if path.name not in excluded_names
                    ]
            else:
                source_images = all_source_images
            if selected_names is not None:
                missing_images = sorted(selected_names - {path.name for path in source_images})
                if missing_images:
                    raise FileNotFoundError(
                        f"Listed images not found in {source}: {missing_images[:10]}"
                    )
            source_counts: Counter[int] = Counter()
            negative_images = 0
            clipped_rows = 0
            prefix = source.name.replace("-", "")
            for image in source_images:
                label_path = label_dir / f"{image.stem}.txt"
                if not label_path.is_file() and not allow_missing_as_negative:
                    raise FileNotFoundError(
                        f"Missing label for {image}. Audit it first, then pass "
                        "--allow-missing-as-negative only for confirmed negative images."
                    )
                if (
                    audit is not None
                    and audit["selection"] == "labeled_plus_approved_missing"
                    and not label_path.is_file()
                    and image.name not in approved_missing_names
                ):
                    raise ValueError(f"Missing label was not approved by audit: {image}")
                lines, changed = _validated_label_lines(label_path if label_path.is_file() else None, class_count)
                clipped_rows += changed
                if not lines:
                    negative_images += 1
                classes = [int(line.split()[0]) for line in lines]
                source_counts.update(classes)
                for repeat_index in range(repeat):
                    stem = f"new_{prefix}_r{repeat_index}__{image.stem}"
                    destination_image = temporary / "images" / "train" / f"{stem}{image.suffix.lower()}"
                    destination_label = temporary / "labels" / "train" / f"{stem}.txt"
                    _link_or_copy(image, destination_image)
                    destination_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                    image_counts["train"] += 1
                    for class_id in classes:
                        object_counts[("train", class_id)] += 1
            added_sources.append({
                "path": str(source),
                "image_list": str(image_list) if image_list is not None else None,
                "audit_file": (
                    str(audit_files[source_index]) if audit_files else None
                ),
                "audit_sha256": (
                    hashlib.sha256(audit_files[source_index].read_bytes()).hexdigest()
                    if audit_files
                    else None
                ),
                "audit_selection": audit.get("selection") if audit else None,
                "audit_excluded_images": len(excluded_names),
                "audit_approved_missing_labels": len(approved_missing_names),
                "source_inventory_images": len(all_source_images),
                "source_images": len(source_images),
                "train_copies": len(source_images) * repeat,
                "negative_images": negative_images,
                "clipped_label_rows": clipped_rows,
                "source_objects": {str(key): value for key, value in sorted(source_counts.items())},
            })

        resolved_config = {
            "path": ".", "train": "images/train", "val": "images/val",
            "test": "images/test", "nc": class_count, "names": names,
        }
        (temporary / "data.yaml").write_text(
            yaml.safe_dump(resolved_config, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        manifest: dict[str, object] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "base_dataset": str(base),
            "additions": added_sources,
            "classes": {str(index): name for index, name in enumerate(names)},
            "new_train_repeat": repeat,
            "split_policy": "new video frames are train-only; base val/test are unchanged",
            "image_storage": "hardlink with copy fallback",
            "images": {split: image_counts[split] for split in SPLITS},
            "objects": {
                split: {str(class_id): object_counts[(split, class_id)] for class_id in range(class_count)}
                for split in SPLITS
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


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=project_root / "dataset" / "2026-07-15")
    parser.add_argument("--add", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=project_root / "dataset" / "2026-08-04")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--image-list",
        type=Path,
        default=None,
        help="For one flat --add directory, include only these audited image filenames",
    )
    parser.add_argument(
        "--allow-missing-as-negative",
        action="store_true",
        help="Create empty labels for source images confirmed by audit to contain no target",
    )
    parser.add_argument(
        "--audit",
        nargs="+",
        type=Path,
        default=None,
        help=(
            "One audit JSON per --add; supports all_except_excluded or "
            "labeled_plus_approved_missing selection"
        ),
    )
    args = parser.parse_args()
    manifest = extend_dataset(
        args.base,
        args.add,
        args.output,
        args.repeat,
        args.allow_missing_as_negative,
        args.image_list,
        args.audit,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
