#!/usr/bin/env python3
"""Find stable-cone ID fragmentation from infer_obb.py event logs."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _records(output: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (output / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _runs(values: list[bool]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(values) - 1))
    return runs


def analyze(output: Path, *, proximity_pixels: float = 180.0) -> dict[str, Any]:
    events = _records(output)
    cone_counts = [int(event.get("cone_count", 0)) for event in events]
    nonzero_indices = [index for index, count in enumerate(cone_counts) if count > 0]
    first_nonzero = nonzero_indices[0] if nonzero_indices else None
    last_nonzero = nonzero_indices[-1] if nonzero_indices else None
    zero_runs = _runs([count == 0 for count in cone_counts])
    interior_zero_runs = [
        {
            "first_frame": events[start]["frame_index"],
            "last_frame": events[end]["frame_index"],
            "frames": end - start + 1,
            "start_seconds": round(float(events[start].get("video_time_seconds", 0.0)), 3),
            "end_seconds": round(float(events[end].get("video_time_seconds", 0.0)), 3),
        }
        for start, end in zero_runs
        if first_nonzero is not None and last_nonzero is not None
        and start > first_nonzero and end < last_nonzero
    ]
    appearances: dict[int, list[dict[str, Any]]] = defaultdict(list)
    first_seen: dict[int, dict[str, Any]] = {}
    previous_tracks: dict[int, dict[str, Any]] = {}
    possible_reassociations: list[dict[str, Any]] = []

    for event in events:
        tracks = {
            int(track["stable_id"]): track
            for track in event.get("tracked_cones", [])
        }
        for stable_id, track in tracks.items():
            appearance = {
                "frame_index": int(event["frame_index"]),
                "time_seconds": float(event.get("video_time_seconds", 0.0)),
                "state": str(track["state"]),
                "matched": bool(track["matched"]),
                "missed_frames": int(track["missed_frames"]),
                "center": (float(track["center_x"]), float(track["center_y"])),
            }
            appearances[stable_id].append(appearance)
            if stable_id not in first_seen:
                first_seen[stable_id] = appearance
                nearby_previous = [
                    (other_id, other_track)
                    for other_id, other_track in previous_tracks.items()
                    if other_id != stable_id
                    and not bool(other_track["matched"])
                    and _distance(
                        appearance["center"],
                        (
                            float(other_track["center_x"]),
                            float(other_track["center_y"]),
                        ),
                    ) <= proximity_pixels
                ]
                if nearby_previous:
                    closest_id, closest = min(
                        nearby_previous,
                        key=lambda item: _distance(
                            appearance["center"],
                            (
                                float(item[1]["center_x"]),
                                float(item[1]["center_y"]),
                            ),
                        ),
                    )
                    possible_reassociations.append(
                        {
                            "new_stable_id": stable_id,
                            "frame_index": appearance["frame_index"],
                            "time_seconds": round(appearance["time_seconds"], 3),
                            "new_center": [
                                round(appearance["center"][0], 1),
                                round(appearance["center"][1], 1),
                            ],
                            "unmatched_prior_id": closest_id,
                            "prior_center": [
                                round(float(closest["center_x"]), 1),
                                round(float(closest["center_y"]), 1),
                            ],
                            "distance_pixels": round(
                                _distance(
                                    appearance["center"],
                                    (
                                        float(closest["center_x"]),
                                        float(closest["center_y"]),
                                    ),
                                ),
                                1,
                            ),
                            "prior_missed_frames": int(closest["missed_frames"]),
                        }
                    )
        previous_tracks = tracks

    confirmed_ids = {
        stable_id
        for stable_id, history in appearances.items()
        if any(item["state"] in {"confirmed", "held"} for item in history)
    }
    confirmed_lifetimes = []
    for stable_id in sorted(confirmed_ids):
        history = appearances[stable_id]
        matched_frames = sum(item["matched"] for item in history)
        confirmed_lifetimes.append(
            {
                "stable_id": stable_id,
                "first_frame": history[0]["frame_index"],
                "last_frame": history[-1]["frame_index"],
                "span_frames": history[-1]["frame_index"] - history[0]["frame_index"] + 1,
                "matched_frames": matched_frames,
                "first_time_seconds": round(history[0]["time_seconds"], 3),
                "last_time_seconds": round(history[-1]["time_seconds"], 3),
            }
        )
    short_lived = [item for item in confirmed_lifetimes if item["span_frames"] <= 15]
    frame_counts = Counter(
        int(event.get("stable_visible_count", 0)) for event in events
    )
    return {
        "output": str(output.resolve()),
        "frames": len(events),
        "accepted_detector": {
            "frames_with_cone": len(nonzero_indices),
            "interior_zero_runs": interior_zero_runs,
            "small_cone_suppressed_total": sum(
                int(event.get("small_cone_suppressed", 0)) for event in events
            ),
            "frames_with_small_cone_suppression": sum(
                int(event.get("small_cone_suppressed", 0)) > 0 for event in events
            ),
        },
        "unique_seen_count": int(events[-1].get("unique_seen_count", 0)) if events else 0,
        "confirmed_track_count": len(confirmed_ids),
        "short_lived_confirmed_tracks": short_lived,
        "stable_visible_count_distribution": dict(sorted(frame_counts.items())),
        "possible_reassociations": possible_reassociations,
        "confirmed_track_lifetimes": confirmed_lifetimes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outputs", nargs="+", type=Path)
    parser.add_argument("--proximity-pixels", type=float, default=180.0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    invalid_outputs = [
        output for output in args.outputs if not (output / "events.jsonl").is_file()
    ]
    if invalid_outputs:
        parser.error(
            "each output must be an infer_obb directory containing events.jsonl: "
            + ", ".join(str(output) for output in invalid_outputs)
        )
    payload = {
        "runs": [
            analyze(output, proximity_pixels=args.proximity_pixels)
            for output in args.outputs
        ]
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
