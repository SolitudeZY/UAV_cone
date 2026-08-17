#!/usr/bin/env python3
"""Calibrate independent not-cone and fallen thresholds on an ImageFolder split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from mobilenet_classifier import build_transform, load_checkpoint


def collect_probabilities(
    weights: Path,
    data: Path,
    *,
    device_name: str,
    batch_size: int,
    workers: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    device = torch.device("cpu" if device_name == "cpu" else f"cuda:{device_name}")
    model, checkpoint = load_checkpoint(weights.resolve(), device)
    dataset = ImageFolder(data.resolve(), transform=build_transform(False))
    if dataset.class_to_idx != checkpoint["class_to_idx"]:
        raise ValueError(
            f"Class mapping mismatch: data={dataset.class_to_idx}, "
            f"checkpoint={checkpoint['class_to_idx']}"
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    probabilities: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.inference_mode():
        for images, batch_targets in loader:
            probabilities.append(model(images.to(device)).softmax(dim=1).cpu())
            targets.append(batch_targets)
    return torch.cat(probabilities), torch.cat(targets), dataset.class_to_idx


def _candidate_thresholds(values: torch.Tensor) -> list[float]:
    unique = sorted({float(value) for value in values.tolist()})
    return [0.0, *unique, 1.0000001]


def select_not_cone_threshold(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    class_to_idx: dict[str, int],
    max_true_cone_rejection: float,
) -> dict[str, float | int]:
    index = class_to_idx["not_cone"]
    scores = probabilities[:, index]
    negatives = targets == index
    true_cones = ~negatives
    best: dict[str, float | int] | None = None
    for threshold in _candidate_thresholds(scores):
        rejected = scores >= threshold
        true_cone_rejection = float(rejected[true_cones].float().mean())
        if true_cone_rejection > max_true_cone_rejection + 1e-12:
            continue
        negative_recall = float(rejected[negatives].float().mean())
        candidate = {
            "threshold": threshold,
            "not_cone_recall": negative_recall,
            "true_cone_rejection_rate": true_cone_rejection,
            "not_cone_rejected": int(rejected[negatives].sum()),
            "not_cone_total": int(negatives.sum()),
            "true_cone_rejected": int(rejected[true_cones].sum()),
            "true_cone_total": int(true_cones.sum()),
        }
        if best is None or (
            candidate["not_cone_recall"], candidate["threshold"]
        ) > (best["not_cone_recall"], best["threshold"]):
            best = candidate
    if best is None:
        raise RuntimeError("No not-cone threshold satisfies the rejection constraint")
    return best


def select_fallen_threshold(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    class_to_idx: dict[str, int],
    minimum_recall: float,
) -> dict[str, float | int]:
    index = class_to_idx["fallen_cone"]
    scores = probabilities[:, index]
    positives = targets == index
    best: dict[str, float | int] | None = None
    for threshold in _candidate_thresholds(scores):
        predicted = scores >= threshold
        true_positive = int((predicted & positives).sum())
        predicted_count = int(predicted.sum())
        recall = true_positive / int(positives.sum())
        if recall + 1e-12 < minimum_recall:
            continue
        precision = true_positive / predicted_count if predicted_count else 0.0
        candidate = {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "true_positive": true_positive,
            "predicted": predicted_count,
            "fallen_total": int(positives.sum()),
        }
        if best is None or (
            candidate["precision"], candidate["threshold"]
        ) > (best["precision"], best["threshold"]):
            best = candidate
    if best is None:
        raise RuntimeError("No fallen threshold satisfies the recall constraint")
    return best


def evaluate_thresholds(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    class_to_idx: dict[str, int],
    not_cone_threshold: float,
    fallen_threshold: float,
) -> dict[str, object]:
    not_index = class_to_idx["not_cone"]
    fallen_index = class_to_idx["fallen_cone"]
    rejected = probabilities[:, not_index] >= not_cone_threshold
    predicted_fallen = (~rejected) & (probabilities[:, fallen_index] >= fallen_threshold)
    true_cones = targets != not_index
    true_fallen = targets == fallen_index
    true_not_cone = targets == not_index
    return {
        "examples": int(targets.numel()),
        "true_cone_rejection": {
            "count": int((rejected & true_cones).sum()),
            "total": int(true_cones.sum()),
            "rate": float((rejected & true_cones).sum() / true_cones.sum()),
        },
        "not_cone_rejection": {
            "count": int((rejected & true_not_cone).sum()),
            "total": int(true_not_cone.sum()),
            "rate": float((rejected & true_not_cone).sum() / true_not_cone.sum()),
        },
        "fallen_detection": {
            "count": int((predicted_fallen & true_fallen).sum()),
            "total": int(true_fallen.sum()),
            "recall": float((predicted_fallen & true_fallen).sum() / true_fallen.sum()),
            "predicted": int(predicted_fallen.sum()),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-true-cone-rejection", type=float, default=0.01)
    parser.add_argument("--minimum-fallen-recall", type=float, default=0.95)
    args = parser.parse_args()

    val_probabilities, val_targets, class_to_idx = collect_probabilities(
        args.weights,
        args.val,
        device_name=args.device,
        batch_size=args.batch,
        workers=args.workers,
    )
    not_cone = select_not_cone_threshold(
        val_probabilities,
        val_targets,
        class_to_idx,
        args.max_true_cone_rejection,
    )
    fallen = select_fallen_threshold(
        val_probabilities,
        val_targets,
        class_to_idx,
        args.minimum_fallen_recall,
    )
    payload: dict[str, object] = {
        "weights": str(args.weights.resolve()),
        "calibration_data": str(args.val.resolve()),
        "class_to_idx": class_to_idx,
        "constraints": {
            "max_true_cone_rejection": args.max_true_cone_rejection,
            "minimum_fallen_recall": args.minimum_fallen_recall,
        },
        "not_cone": not_cone,
        "fallen": fallen,
        "validation": evaluate_thresholds(
            val_probabilities,
            val_targets,
            class_to_idx,
            float(not_cone["threshold"]),
            float(fallen["threshold"]),
        ),
    }
    if args.test:
        test_probabilities, test_targets, test_mapping = collect_probabilities(
            args.weights,
            args.test,
            device_name=args.device,
            batch_size=args.batch,
            workers=args.workers,
        )
        if test_mapping != class_to_idx:
            raise ValueError(f"Test class mapping mismatch: {test_mapping}")
        payload["test_data"] = str(args.test.resolve())
        payload["test"] = evaluate_thresholds(
            test_probabilities,
            test_targets,
            class_to_idx,
            float(not_cone["threshold"]),
            float(fallen["threshold"]),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
