"""Runtime wrapper for MobileNet traffic-cone candidate review."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from PIL import Image

from mine_classifier_candidates import crop_square
from mobilenet_classifier import EXPECTED_CLASSES, build_transform, load_checkpoint


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_device(value: str | int) -> torch.device:
    text = str(value).strip().lower()
    if text in {"cpu", "mps"} or text.startswith("cuda:"):
        return torch.device(text)
    try:
        return torch.device(f"cuda:{int(text)}")
    except ValueError as exc:
        raise ValueError(
            "classifier device must be cpu, mps, cuda:N, or a CUDA index"
        ) from exc


def load_thresholds(path: Path) -> tuple[float, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        not_cone = float(payload["not_cone"]["threshold"])
        fallen = float(payload["fallen"]["threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid classifier thresholds file: {path}") from exc
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in (not_cone, fallen)):
        raise ValueError("classifier thresholds must be finite values in [0, 1]")
    return not_cone, fallen


class ConeCandidateClassifier:
    """Batch candidate classifier with calibrated rejection and posture votes."""

    def __init__(
        self,
        weights: Path,
        thresholds: Path,
        *,
        device: str | int,
        crop_context: float = 1.6,
        batch_size: int = 64,
    ) -> None:
        if not weights.is_file():
            raise ValueError(f"Classifier weights do not exist: {weights}")
        if not thresholds.is_file():
            raise ValueError(f"Classifier thresholds do not exist: {thresholds}")
        if not math.isfinite(crop_context) or crop_context < 1.0:
            raise ValueError("classifier crop context must be finite and >= 1.0")
        if batch_size < 1:
            raise ValueError("classifier batch size must be at least 1")

        self.weights = weights.resolve()
        self.thresholds_path = thresholds.resolve()
        self.device = _torch_device(device)
        self.crop_context = crop_context
        self.batch_size = batch_size
        self.not_cone_threshold, self.fallen_threshold = load_thresholds(
            self.thresholds_path
        )
        self.model, checkpoint = load_checkpoint(self.weights, self.device)
        class_to_idx = checkpoint.get("class_to_idx", {})
        if set(class_to_idx) != EXPECTED_CLASSES:
            raise ValueError(
                f"Classifier must contain {sorted(EXPECTED_CLASSES)}, got {sorted(class_to_idx)}"
            )
        self.class_names = [
            name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])
        ]
        self.transform = build_transform(False)
        self.last_timing_ms = {"crop": 0.0, "inference": 0.0, "total": 0.0}
        self.policy = {
            "weights": str(self.weights),
            "weights_sha256": _sha256(self.weights),
            "thresholds": str(self.thresholds_path),
            "thresholds_sha256": _sha256(self.thresholds_path),
            "device": str(self.device),
            "crop_context": self.crop_context,
            "crop_size": 224,
            "batch_size": self.batch_size,
            "not_cone_threshold": self.not_cone_threshold,
            "fallen_threshold": self.fallen_threshold,
        }

    def classify(
        self, frame: np.ndarray, polygons: Sequence[Sequence[Sequence[float]]]
    ) -> list[dict[str, Any]]:
        total_started = time.perf_counter()
        crop_started = time.perf_counter()
        crops = [
            crop_square(
                frame,
                np.asarray(polygon, dtype=np.float32),
                context=self.crop_context,
                output_size=224,
            )[0]
            for polygon in polygons
        ]
        crop_seconds = time.perf_counter() - crop_started

        probability_rows: list[list[float]] = []
        inference_seconds = 0.0
        for start in range(0, len(crops), self.batch_size):
            tensors = [
                self.transform(
                    Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                )
                for crop in crops[start : start + self.batch_size]
            ]
            inference_started = time.perf_counter()
            with torch.inference_mode():
                probability_rows.extend(
                    self.model(torch.stack(tensors).to(self.device))
                    .softmax(dim=1)
                    .cpu()
                    .tolist()
                )
            inference_seconds += time.perf_counter() - inference_started

        reviews = []
        for values in probability_rows:
            probabilities = dict(zip(self.class_names, (float(value) for value in values)))
            argmax_class = max(probabilities, key=probabilities.__getitem__)
            reviews.append(
                {
                    "probabilities": probabilities,
                    "argmax_class": argmax_class,
                    "not_cone_rejected": (
                        probabilities["not_cone"] >= self.not_cone_threshold
                    ),
                    "fallen_vote": (
                        probabilities["fallen_cone"] >= self.fallen_threshold
                    ),
                }
            )

        self.last_timing_ms = {
            "crop": crop_seconds * 1000.0,
            "inference": inference_seconds * 1000.0,
            "total": (time.perf_counter() - total_started) * 1000.0,
        }
        return reviews
