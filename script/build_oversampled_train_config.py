#!/usr/bin/env python3
"""Build a YOLO data config with selected training filename prefixes repeated."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_repeat(value: str) -> tuple[str, int]:
    try:
        prefix, count_text = value.rsplit("=", 1)
        count = int(count_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("repeat must be PREFIX=COUNT") from exc
    if not prefix or count < 1:
        raise argparse.ArgumentTypeError("repeat prefix must be non-empty and count >= 1")
    return prefix, count


def resolve_dataset_root(config_path: Path, config: dict[str, object]) -> Path:
    root = Path(str(config.get("path", "."))).expanduser()
    return root.resolve() if root.is_absolute() else (config_path.parent / root).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeat", type=parse_repeat, action="append", default=[])
    args = parser.parse_args()

    data_path = args.data.resolve()
    config = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
    root = resolve_dataset_root(data_path, config)
    train_dir = root / str(config["train"])
    images = sorted(
        path.resolve()
        for path in train_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise SystemExit(f"No training images found under {train_dir}")

    repeats = dict(args.repeat)
    unknown = [prefix for prefix in repeats if not any(path.name.startswith(prefix) for path in images)]
    if unknown:
        raise SystemExit(f"No training images match prefixes: {unknown}")

    sampled: list[Path] = []
    matched: Counter[str] = Counter()
    for image in images:
        count = 1
        for prefix, repeat_count in repeats.items():
            if image.name.startswith(prefix):
                count = repeat_count
                matched[prefix] += 1
                break
        sampled.extend([image] * count)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_list = output_dir / "train_oversampled.txt"
    train_list.write_text("\n".join(str(path) for path in sampled) + "\n", encoding="utf-8")

    output_config = dict(config)
    output_config["path"] = str(root)
    output_config["train"] = str(train_list)
    config_path = output_dir / "data_oversampled.yaml"
    config_path.write_text(
        yaml.safe_dump(output_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    manifest = {
        "source_data": str(data_path),
        "source_train_images": len(images),
        "sampled_train_entries": len(sampled),
        "repeat_policy": repeats,
        "matched_images": dict(matched),
        "validation_and_test_unchanged": True,
        "train_list": str(train_list),
        "data_config": str(config_path),
    }
    (output_dir / "oversampling_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
