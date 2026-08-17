"""Shared MobileNetV3-small classifier construction, transforms, and metrics."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.transforms import InterpolationMode


IMAGE_SIZE = 224
EXPECTED_CLASSES = {"fallen_cone", "not_cone", "upright_cone"}


def build_transform(training: bool):
    normalize = transforms.Normalize(
        mean=MobileNet_V3_Small_Weights.DEFAULT.transforms().mean,
        std=MobileNet_V3_Small_Weights.DEFAULT.transforms().std,
    )
    if training:
        return transforms.Compose(
            [
                transforms.Resize(
                    (IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BILINEAR
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            normalize,
        ]
    )


def build_model(num_classes: int, *, pretrained: bool) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    input_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(input_features, num_classes)
    return model


def set_trainable_stage(model: nn.Module, *, head_only: bool, unfreeze_last_blocks: int) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    if not head_only:
        blocks = list(model.features.children())
        count = max(1, min(unfreeze_last_blocks, len(blocks)))
        for block in blocks[-count:]:
            for parameter in block.parameters():
                parameter.requires_grad = True


def confusion_metrics(confusion: torch.Tensor, class_names: list[str]) -> dict:
    matrix = confusion.to(torch.float64)
    total = matrix.sum().item()
    per_class = {}
    f1_values = []
    for index, name in enumerate(class_names):
        true_positive = matrix[index, index].item()
        false_negative = matrix[index, :].sum().item() - true_positive
        false_positive = matrix[:, index].sum().item() - true_positive
        support = int(matrix[index, :].sum().item())
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return {
        "accuracy": matrix.diag().sum().item() / total if total else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_axes": {"rows": "actual", "columns": "predicted"},
    }


@torch.inference_mode()
def evaluate_loader(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
) -> dict:
    model.eval()
    confusion = torch.zeros((len(class_names), len(class_names)), dtype=torch.int64)
    loss_sum = 0.0
    examples = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss_sum += criterion(logits, targets).item() * targets.numel()
        examples += targets.numel()
        predictions = logits.argmax(dim=1)
        indices = targets.detach().cpu() * len(class_names) + predictions.detach().cpu()
        confusion += torch.bincount(indices, minlength=len(class_names) ** 2).reshape(
            len(class_names), len(class_names)
        )
    metrics = confusion_metrics(confusion, class_names)
    metrics["loss"] = loss_sum / examples if examples else 0.0
    return metrics


def load_checkpoint(path, device: torch.device) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    class_to_idx = checkpoint["class_to_idx"]
    model = build_model(len(class_to_idx), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint
