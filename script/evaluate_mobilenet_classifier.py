#!/usr/bin/env python3
"""Evaluate a trained cone classifier on an untouched ImageFolder split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from mobilenet_classifier import build_transform, evaluate_loader, load_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="ImageFolder split directory")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device}")
    model, checkpoint = load_checkpoint(args.weights.resolve(), device)
    dataset = ImageFolder(args.data.resolve(), transform=build_transform(False))
    if dataset.class_to_idx != checkpoint["class_to_idx"]:
        raise SystemExit(
            f"Class mapping mismatch: data={dataset.class_to_idx}, checkpoint={checkpoint['class_to_idx']}"
        )
    class_names = [name for name, _ in sorted(dataset.class_to_idx.items(), key=lambda item: item[1])]
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    metrics = evaluate_loader(model, loader, nn.CrossEntropyLoss(), device, class_names)
    result = {"weights": str(args.weights.resolve()), "data": str(args.data.resolve()), **metrics}
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
