#!/usr/bin/env python3
"""Serve a small review UI for mined cone candidates."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, abort, jsonify, render_template, request, send_file


LABELS = ("upright_cone", "fallen_cone", "not_cone", "ignore")
REVIEW_FILTERS = ("all", "unreviewed", *LABELS)
ANNOTATION_FIELDS = ("candidate_id", "label", "reviewed_at", "sequence")
POOL_ORDER = {
    "raw_angle_fallen": 0,
    "prediction_without_gt": 1,
    "gt_without_prediction": 2,
    "angle_normal_cone": 3,
}
POOL_NAMES = {
    "raw_angle_fallen": "角度 FALLEN",
    "prediction_without_gt": "预测无 GT",
    "gt_without_prediction": "GT 无预测",
    "angle_normal_cone": "角度正常",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--debug", action="store_true")
    return parser


class CandidateReviewStore:
    def __init__(self, data_root: Path, annotations_path: Path | None = None) -> None:
        self.data_root = data_root.resolve()
        self.candidates_path = self.data_root / "candidates.csv"
        self.annotations_path = (
            annotations_path.resolve()
            if annotations_path is not None
            else self.data_root / "review" / "annotations.csv"
        )
        self._lock = threading.RLock()
        self.candidates = self._load_candidates()
        self.candidate_by_id = {
            candidate["candidate_id"]: candidate for candidate in self.candidates
        }
        if len(self.candidate_by_id) != len(self.candidates):
            raise ValueError("candidates.csv contains duplicate candidate_id values")
        self.annotations = self._load_annotations()
        self.sequence = max(
            (int(item["sequence"]) for item in self.annotations.values()), default=0
        )

    def _load_candidates(self) -> list[dict[str, str]]:
        if not self.candidates_path.is_file():
            raise FileNotFoundError(f"Missing candidates CSV: {self.candidates_path}")
        with self.candidates_path.open("r", encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))
        required = {
            "candidate_id",
            "source_split",
            "source_image",
            "review_pool",
            "crop_path",
            "preview_path",
        }
        if not rows:
            raise ValueError("candidates.csv is empty")
        missing = required - rows[0].keys()
        if missing:
            raise ValueError(f"candidates.csv missing fields: {sorted(missing)}")

        def sort_key(row: dict[str, str]) -> tuple[int, int, float, str]:
            raw_fallen = row.get("raw_angle_fallen", "false").lower() == "true"
            try:
                confidence = float(row.get("yolo_confidence") or -1.0)
            except ValueError:
                confidence = -1.0
            return (
                0 if raw_fallen else 1,
                POOL_ORDER.get(row["review_pool"], 99),
                -confidence,
                row["candidate_id"],
            )

        return sorted(rows, key=sort_key)

    def _load_annotations(self) -> dict[str, dict[str, str]]:
        if not self.annotations_path.is_file():
            return {}
        with self.annotations_path.open("r", encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))
        annotations: dict[str, dict[str, str]] = {}
        for row in rows:
            candidate_id = row.get("candidate_id", "")
            label = row.get("label", "")
            if candidate_id not in self.candidate_by_id:
                raise ValueError(f"Annotation references unknown candidate: {candidate_id}")
            if label not in LABELS:
                raise ValueError(f"Annotation contains unknown label: {label}")
            annotations[candidate_id] = row
        return annotations

    def _write_annotations(self) -> None:
        self.annotations_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=self.annotations_path.parent,
            prefix="annotations.",
            suffix=".tmp",
            delete=False,
        ) as destination:
            temporary = Path(destination.name)
            writer = csv.DictWriter(destination, fieldnames=ANNOTATION_FIELDS)
            writer.writeheader()
            writer.writerows(
                sorted(
                    self.annotations.values(),
                    key=lambda item: int(item["sequence"]),
                )
            )
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, self.annotations_path)

    def _matches(self, candidate: dict[str, str], pool: str, split: str) -> bool:
        return (
            (pool == "all" or candidate["review_pool"] == pool)
            and (split == "all" or candidate["source_split"] == split)
        )

    def _stats(self, pool: str = "all", split: str = "all") -> dict[str, Any]:
        selected = [
            item for item in self.candidates if self._matches(item, pool, split)
        ]
        reviewed = sum(
            item["candidate_id"] in self.annotations for item in selected
        )
        selected_ids = {item["candidate_id"] for item in selected}
        label_counts = {label: 0 for label in LABELS}
        for candidate_id, annotation in self.annotations.items():
            if candidate_id in selected_ids:
                label_counts[annotation["label"]] += 1
        return {
            "total": len(selected),
            "reviewed": reviewed,
            "remaining": len(selected) - reviewed,
            "all_total": len(self.candidates),
            "all_reviewed": len(self.annotations),
            "label_counts": label_counts,
        }

    def _selected(
        self,
        pool: str,
        split: str,
        review: str,
    ) -> list[dict[str, str]]:
        if review not in REVIEW_FILTERS:
            raise ValueError(f"Unknown review filter: {review}")

        def matches_review(candidate: dict[str, str]) -> bool:
            annotation = self.annotations.get(candidate["candidate_id"])
            if review == "all":
                return True
            if review == "unreviewed":
                return annotation is None
            return annotation is not None and annotation["label"] == review

        return [
            candidate
            for candidate in self.candidates
            if self._matches(candidate, pool, split) and matches_review(candidate)
        ]

    def _payload(
        self,
        candidate: dict[str, str] | None,
        pool: str,
        split: str,
        review: str,
    ) -> dict[str, Any]:
        selected = self._selected(pool, split, review)
        previous_id: str | None = None
        next_id: str | None = None
        position: int | None = None
        if candidate is not None:
            candidate_id = candidate["candidate_id"]
            try:
                index = next(
                    index
                    for index, item in enumerate(selected)
                    if item["candidate_id"] == candidate_id
                )
            except StopIteration as exc:
                raise ValueError("Candidate does not match the active filters") from exc
            position = index + 1
            if index > 0:
                previous_id = selected[index - 1]["candidate_id"]
            if index + 1 < len(selected):
                next_id = selected[index + 1]["candidate_id"]

        return {
            "candidate": self._public_candidate(candidate),
            "navigation": {
                "previous_candidate_id": previous_id,
                "next_candidate_id": next_id,
                "position": position,
                "total": len(selected),
            },
            "stats": self._stats(pool, split),
            "pools": [
                {"value": pool_id, "name": POOL_NAMES.get(pool_id, pool_id)}
                for pool_id in sorted(
                    {item["review_pool"] for item in self.candidates},
                    key=lambda item: POOL_ORDER.get(item, 99),
                )
            ],
            "splits": sorted({item["source_split"] for item in self.candidates}),
        }

    def current(
        self,
        pool: str = "all",
        split: str = "all",
        review: str = "unreviewed",
    ) -> dict[str, Any]:
        with self._lock:
            selected = self._selected(pool, split, review)
            candidate = selected[0] if selected else None
            return self._payload(candidate, pool, split, review)

    def candidate(
        self,
        candidate_id: str,
        pool: str = "all",
        split: str = "all",
        review: str = "all",
    ) -> dict[str, Any]:
        with self._lock:
            candidate = self.candidate_by_id.get(candidate_id)
            if candidate is None:
                raise KeyError(candidate_id)
            return self._payload(candidate, pool, split, review)

    def _public_candidate(
        self, candidate: dict[str, str] | None
    ) -> dict[str, Any] | None:
        if candidate is None:
            return None
        candidate_id = candidate["candidate_id"]
        annotation = self.annotations.get(candidate_id)
        return {
            "candidate_id": candidate_id,
            "source_split": candidate["source_split"],
            "source_image": candidate["source_image"],
            "review_pool": candidate["review_pool"],
            "review_pool_name": POOL_NAMES.get(
                candidate["review_pool"], candidate["review_pool"]
            ),
            "candidate_kind": candidate.get("candidate_kind", ""),
            "yolo_confidence": candidate.get("yolo_confidence", ""),
            "raw_angle_fallen": candidate.get("raw_angle_fallen", "false") == "true",
            "long_axis_angle": candidate.get("long_axis_angle", ""),
            "match_status": candidate.get("match_status", ""),
            "review_label": annotation["label"] if annotation else None,
            "reviewed_at": annotation["reviewed_at"] if annotation else None,
            "preview_url": f"/asset/{candidate_id}/preview",
            "crop_url": f"/asset/{candidate_id}/crop",
        }

    def label(self, candidate_id: str, label: str) -> dict[str, Any]:
        if label not in LABELS:
            raise ValueError(f"Unknown label: {label}")
        with self._lock:
            if candidate_id not in self.candidate_by_id:
                raise KeyError(candidate_id)
            self.sequence += 1
            self.annotations[candidate_id] = {
                "candidate_id": candidate_id,
                "label": label,
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
                "sequence": str(self.sequence),
            }
            self._write_annotations()
            return self._stats()

    def undo(self) -> dict[str, Any] | None:
        with self._lock:
            if not self.annotations:
                return None
            candidate_id, annotation = max(
                self.annotations.items(), key=lambda item: int(item[1]["sequence"])
            )
            del self.annotations[candidate_id]
            self._write_annotations()
            return {
                "candidate": self._public_candidate(self.candidate_by_id[candidate_id]),
                "removed_label": annotation["label"],
                "stats": self._stats(),
            }

    def asset_path(self, candidate_id: str, kind: str) -> Path:
        if candidate_id not in self.candidate_by_id:
            raise KeyError(candidate_id)
        if kind not in {"preview", "crop"}:
            raise ValueError(kind)
        field = "preview_path" if kind == "preview" else "crop_path"
        path = (self.data_root / self.candidate_by_id[candidate_id][field]).resolve()
        if not path.is_relative_to(self.data_root):
            raise ValueError("Candidate asset escapes data root")
        if not path.is_file():
            raise FileNotFoundError(path)
        return path


def create_app(store: CandidateReviewStore) -> Flask:
    application = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "candidate_review_web" / "templates"),
        static_folder=str(Path(__file__).parent / "candidate_review_web" / "static"),
    )
    application.config["JSON_AS_ASCII"] = False

    @application.get("/")
    def index() -> str:
        return render_template("index.html")

    @application.get("/api/current")
    def current() -> Any:
        try:
            payload = store.current(
                pool=request.args.get("pool", "all"),
                split=request.args.get("split", "all"),
                review=request.args.get("review", "unreviewed"),
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(payload)

    @application.get("/api/candidate/<candidate_id>")
    def candidate(candidate_id: str) -> Any:
        try:
            payload = store.candidate(
                candidate_id,
                pool=request.args.get("pool", "all"),
                split=request.args.get("split", "all"),
                review=request.args.get("review", "all"),
            )
        except KeyError:
            return jsonify({"error": "unknown candidate"}), 404
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(payload)

    @application.post("/api/label")
    def label() -> Any:
        payload = request.get_json(silent=True) or {}
        try:
            stats = store.label(str(payload.get("candidate_id", "")), str(payload.get("label", "")))
        except KeyError:
            return jsonify({"error": "unknown candidate"}), 404
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"ok": True, "stats": stats})

    @application.post("/api/undo")
    def undo() -> Any:
        return jsonify({"ok": True, "result": store.undo()})

    @application.get("/asset/<candidate_id>/<kind>")
    def asset(candidate_id: str, kind: str) -> Any:
        try:
            path = store.asset_path(candidate_id, kind)
        except (KeyError, ValueError, FileNotFoundError):
            abort(404)
        return send_file(path, mimetype="image/jpeg", max_age=3600)

    return application


def main() -> int:
    args = build_parser().parse_args()
    store = CandidateReviewStore(args.data_root, args.annotations)
    application = create_app(store)
    application.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        use_reloader=False,
        threaded=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
