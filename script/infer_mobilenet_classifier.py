#!/usr/bin/env python3
"""Classify one image or a directory of unboxed 224x224 cone candidate crops."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image

from mobilenet_classifier import build_transform, load_checkpoint


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    source = args.source.resolve()
    paths = [source] if source.is_file() else sorted(
        path for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise SystemExit(f"No input images found: {source}")
    device = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device}")
    model, checkpoint = load_checkpoint(args.weights.resolve(), device)
    transform = build_transform(False)
    class_names = [name for name, _ in sorted(checkpoint["class_to_idx"].items(), key=lambda item: item[1])]
    rows = []
    for start in range(0, len(paths), args.batch):
        batch_paths = paths[start : start + args.batch]
        tensors = []
        for path in batch_paths:
            with Image.open(path) as image:
                tensors.append(transform(image.convert("RGB")))
        with torch.inference_mode():
            probabilities = model(torch.stack(tensors).to(device)).softmax(dim=1).cpu()
        for path, values in zip(batch_paths, probabilities.tolist()):
            predicted_index = max(range(len(values)), key=values.__getitem__)
            row = {
                "image": str(path),
                "prediction": class_names[predicted_index],
                "confidence": f"{values[predicted_index]:.8f}",
            }
            row.update({f"probability_{name}": f"{values[index]:.8f}" for index, name in enumerate(class_names)})
            rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} predictions to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
