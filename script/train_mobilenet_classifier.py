#!/usr/bin/env python3
"""Train and independently evaluate the reviewed three-class cone classifier."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder

from mobilenet_classifier import (
    EXPECTED_CLASSES,
    build_model,
    build_transform,
    evaluate_loader,
    set_trainable_stage,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "dataset" / "cone_posture_classifier_20260814")
    parser.add_argument("--output", type=Path, default=root / "script" / "runs" / "mobilenetv3_small_cone_20260814")
    parser.add_argument("--head-epochs", type=int, default=4)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--finetune-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--unfreeze-last-blocks", type=int, default=4)
    parser.add_argument("--balance-power", type=float, default=0.5)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0", help="CUDA index such as 0, or cpu")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value.lower() == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(f"cuda:{value}")


def validate_datasets(datasets: dict[str, ImageFolder]) -> tuple[list[str], dict[str, int]]:
    class_to_idx = datasets["train"].class_to_idx
    if set(class_to_idx) != EXPECTED_CLASSES:
        raise SystemExit(f"Expected classes {sorted(EXPECTED_CLASSES)}, got {sorted(class_to_idx)}")
    for split, dataset in datasets.items():
        if dataset.class_to_idx != class_to_idx:
            raise SystemExit(f"Class mapping differs in split {split}: {dataset.class_to_idx}")
    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
    return class_names, class_to_idx


def make_loaders(args: argparse.Namespace, device: torch.device):
    datasets = {
        "train": ImageFolder(args.data / "train", transform=build_transform(True)),
        "val": ImageFolder(args.data / "val", transform=build_transform(False)),
        "test": ImageFolder(args.data / "test", transform=build_transform(False)),
    }
    class_names, class_to_idx = validate_datasets(datasets)
    counts = Counter(datasets["train"].targets)
    if any(counts[index] == 0 for index in range(len(class_names))):
        raise SystemExit(f"Every class requires train samples: {counts}")
    if not 0.0 <= args.balance_power <= 1.0:
        raise SystemExit("--balance-power must be between 0 and 1")
    largest = max(counts.values())
    sample_weights = [
        (largest / counts[target]) ** args.balance_power for target in datasets["train"].targets
    ]
    class_sampling_mass = {
        index: counts[index] * (largest / counts[index]) ** args.balance_power
        for index in range(len(class_names))
    }
    total_sampling_mass = sum(class_sampling_mass.values())
    expected_samples_per_epoch = {
        class_names[index]: len(datasets["train"])
        * class_sampling_mass[index]
        / total_sampling_mass
        for index in range(len(class_names))
    }
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True, generator=generator)
    common = {
        "batch_size": args.batch,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    loaders = {
        "train": DataLoader(datasets["train"], sampler=sampler, **common),
        "val": DataLoader(datasets["val"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }
    return datasets, loaders, class_names, class_to_idx, counts, expected_samples_per_epoch


def save_checkpoint(path: Path, model: nn.Module, args: argparse.Namespace, class_to_idx: dict, epoch: int, metrics: dict) -> None:
    torch.save(
        {
            "architecture": "mobilenet_v3_small",
            "input_size": 224,
            "class_to_idx": class_to_idx,
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "validation_metrics": metrics,
            "training_args": vars(args),
        },
        path,
    )


def selection_key(metrics: dict) -> tuple[float, float, float, float]:
    return (
        metrics["macro_f1"],
        metrics["per_class"]["fallen_cone"]["recall"],
        metrics["per_class"]["not_cone"]["recall"],
        metrics["accuracy"],
    )


def train_epoch(model, loader, criterion, optimizer, scaler, device) -> float:
    model.train()
    loss_sum = 0.0
    examples = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item() * targets.numel()
        examples += targets.numel()
    return loss_sum / examples


def write_confusion(path: Path, metrics: dict, class_names: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["actual\\predicted", *class_names])
        for name, row in zip(class_names, metrics["confusion_matrix"]):
            writer.writerow([name, *row])


def main() -> int:
    args = parse_args()
    args.data = args.data.resolve()
    args.output = args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"Output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    (
        datasets,
        loaders,
        class_names,
        class_to_idx,
        train_counts,
        expected_samples_per_epoch,
    ) = make_loaders(args, device)

    model = build_model(len(class_names), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = GradScaler(device.type, enabled=device.type == "cuda")
    phases = (
        ("head", args.head_epochs, args.head_lr, True),
        ("finetune", args.finetune_epochs, args.finetune_lr, False),
    )
    history = []
    best_key = (-1.0, -1.0, -1.0, -1.0)
    best_epoch = 0
    epochs_without_improvement = 0
    global_epoch = 0
    stop = False

    print(
        f"device={device} classes={class_to_idx} train_counts={dict(train_counts)} "
        f"balance_power={args.balance_power} "
        f"expected_samples_per_epoch={expected_samples_per_epoch}"
    )
    for phase, phase_epochs, learning_rate, head_only in phases:
        if phase_epochs <= 0 or stop:
            continue
        set_trainable_stage(
            model,
            head_only=head_only,
            unfreeze_last_blocks=args.unfreeze_last_blocks,
        )
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = AdamW(parameters, lr=learning_rate, weight_decay=args.weight_decay)
        for _ in range(phase_epochs):
            global_epoch += 1
            train_loss = train_epoch(model, loaders["train"], criterion, optimizer, scaler, device)
            val_metrics = evaluate_loader(model, loaders["val"], criterion, device, class_names)
            row = {
                "epoch": global_epoch,
                "phase": phase,
                "learning_rate": learning_rate,
                "train_loss": train_loss,
                "val": val_metrics,
            }
            history.append(row)
            current_key = selection_key(val_metrics)
            improved = current_key > best_key
            if improved:
                best_key = current_key
                best_epoch = global_epoch
                epochs_without_improvement = 0
                save_checkpoint(args.output / "best.pt", model, args, class_to_idx, global_epoch, val_metrics)
            else:
                epochs_without_improvement += 1
            save_checkpoint(args.output / "last.pt", model, args, class_to_idx, global_epoch, val_metrics)
            with (args.output / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            fallen_recall = val_metrics["per_class"]["fallen_cone"]["recall"]
            negative_recall = val_metrics["per_class"]["not_cone"]["recall"]
            print(
                f"epoch={global_epoch:02d} phase={phase} train_loss={train_loss:.4f} "
                f"val_loss={val_metrics['loss']:.4f} macro_f1={val_metrics['macro_f1']:.4f} "
                f"fallen_recall={fallen_recall:.4f} not_cone_recall={negative_recall:.4f} "
                f"best={improved}",
                flush=True,
            )
            if phase == "finetune" and epochs_without_improvement >= args.patience:
                stop = True
                break

    best = torch.load(args.output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    val_metrics = evaluate_loader(model, loaders["val"], criterion, device, class_names)
    test_metrics = evaluate_loader(model, loaders["test"], criterion, device, class_names)
    write_confusion(args.output / "confusion_matrix_val.csv", val_metrics, class_names)
    write_confusion(args.output / "confusion_matrix_test.csv", test_metrics, class_names)
    summary = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "data": str(args.data),
        "class_to_idx": class_to_idx,
        "split_sizes": {split: len(dataset) for split, dataset in datasets.items()},
        "train_class_counts": {class_names[index]: train_counts[index] for index in range(len(class_names))},
        "sampling": {
            "balance_power": args.balance_power,
            "expected_samples_per_epoch": expected_samples_per_epoch,
        },
        "best_epoch": best_epoch,
        "selection_order": ["val_macro_f1", "val_fallen_recall", "val_not_cone_recall", "val_accuracy"],
        "validation": val_metrics,
        "test": test_metrics,
        "artifacts": {"best": "best.pt", "last": "last.pt"},
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
