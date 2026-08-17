#!/usr/bin/env python3
"""Summarize infer_obb.py events and activation-only alert logs."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


ALERT_FIELDS = (
    "missing",
    "suspect_missing",
    "displaced",
    "fallen",
    "stable_missing",
    "stable_displaced",
    "stable_fallen",
)


def summarize(output: Path) -> dict[str, object]:
    events = [
        json.loads(line) for line in (output / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    detections = [item for event in events for item in event.get("detections", [])]
    class_counts = Counter(str(item.get("label")) for item in detections)
    postprocess_counts = Counter(
        str(item.get("postprocess_status", "unmarked")) for item in detections
    )
    cone_confidences = [
        float(item["confidence"]) for item in detections if item.get("label") == "cone"
    ]
    cone_frames = [event for event in events if event.get("cone_count", 0) > 0]
    alert_counts = {
        field: sum(len(event.get(field, [])) for event in events)
        for field in ALERT_FIELDS
    }
    activation_path = output / "alert.jsonl"
    activations = [
        json.loads(line) for line in activation_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if activation_path.is_file() else []
    classifier_events = [event for event in events if event.get("classifier_policy")]
    classifier_count_fields = (
        "reviewed",
        "not_cone_votes",
        "not_cone_suppressed",
        "fallen_votes",
        "fallen_votes_applied",
    )
    classifier_timing_fields = ("crop", "inference", "total")
    top_cone_frames = sorted(
        (
            {
                "frame_index": int(event["frame_index"]),
                "video_time_seconds": float(event.get("video_time_seconds", 0.0)),
                "cone_count": int(event.get("cone_count", 0)),
                "max_cone_confidence": max(
                    (float(item["confidence"]) for item in event.get("detections", []) if item.get("label") == "cone"),
                    default=0.0,
                ),
            }
            for event in cone_frames
        ),
        key=lambda item: (item["max_cone_confidence"], item["cone_count"]),
        reverse=True,
    )[:20]
    return {
        "output": str(output.resolve()),
        "frames": len(events),
        "duration_seconds": round(float(events[-1].get("video_time_seconds", 0.0)), 3) if events else 0.0,
        "detections_by_class": dict(sorted(class_counts.items())),
        "detections_by_postprocess_status": dict(sorted(postprocess_counts.items())),
        "classifier": (
            {
                "policy": classifier_events[0]["classifier_policy"],
                "counts": {
                    field: sum(
                        int(event.get("classifier_counts", {}).get(field, 0))
                        for event in classifier_events
                    )
                    for field in classifier_count_fields
                },
                "average_timing_ms": {
                    field: round(
                        statistics.fmean(
                            float(event.get("classifier_timing_ms", {}).get(field, 0.0))
                            for event in classifier_events
                        ),
                        3,
                    )
                    for field in classifier_timing_fields
                },
            }
            if classifier_events
            else None
        ),
        "cone": {
            "detections": len(cone_confidences),
            "frames_with_cone": len(cone_frames),
            "frame_ratio": round(len(cone_frames) / len(events), 4) if events else 0.0,
            "mean_confidence": round(statistics.fmean(cone_confidences), 4) if cone_confidences else None,
            "max_confidence": round(max(cone_confidences), 4) if cone_confidences else None,
            "confidence_ge_0_5": sum(value >= 0.5 for value in cone_confidences),
            "confidence_ge_0_75": sum(value >= 0.75 for value in cone_confidences),
            "max_per_frame": max((int(event.get("cone_count", 0)) for event in events), default=0),
        },
        "raw_and_stable_alert_events": alert_counts,
        "activation_alerts": {
            "count": len(activations),
            "by_type": dict(sorted(Counter(str(item.get("alert_type")) for item in activations).items())),
        },
        "top_cone_frames": top_cone_frames,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outputs", nargs="+", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    payload = {"runs": [summarize(path) for path in args.outputs]}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
