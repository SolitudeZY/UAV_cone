"""Stable per-cone tracking and video-level counting."""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass
from typing import Sequence

from traffic_rules import AnalysisResult


Point = tuple[float, float]


def _distance(first: Point, second: Point) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


@dataclass
class _ConeState:
    stable_id: int
    center: Point
    votes: deque[bool]
    velocity: Point = (0.0, 0.0)
    state: str = "tentative"
    hits: int = 1
    hit_streak: int = 1
    age: int = 1
    missed_frames: int = 0
    counted: bool = False


@dataclass(frozen=True)
class ConeCountingResult:
    """Per-frame stable count and auditable track state."""

    stable_visible_count: int
    unique_seen_count: int
    new_cone_ids: tuple[int, ...]
    active_cone_ids: tuple[int, ...]
    held_cone_ids: tuple[int, ...]
    index_to_stable_id: dict[int, int]
    stable_cone_ids: tuple[int | None, ...]
    tracked_cones: tuple[dict[str, int | float | bool | str], ...]
    policy: dict[str, int | float]


class ConeCounter:
    """Confirm individual cone tracks before counting them once."""

    def __init__(
        self,
        *,
        confirm_window: int = 5,
        confirm_votes: int = 3,
        max_missed_frames: int = 5,
        match_distance_ratio: float = 0.8,
        min_match_distance: float = 15.0,
        max_match_distance: float = 120.0,
    ) -> None:
        if confirm_window < 1 or not 1 <= confirm_votes <= confirm_window:
            raise ValueError("confirm_votes must be within confirm_window")
        if max_missed_frames < 0:
            raise ValueError("max_missed_frames must not be negative")
        if match_distance_ratio <= 0.0:
            raise ValueError("match_distance_ratio must be positive")
        if min_match_distance <= 0.0 or max_match_distance < min_match_distance:
            raise ValueError("match distance bounds are invalid")
        self.confirm_window = confirm_window
        self.confirm_votes = confirm_votes
        self.max_missed_frames = max_missed_frames
        self.match_distance_ratio = match_distance_ratio
        self.min_match_distance = min_match_distance
        self.max_match_distance = max_match_distance
        self._states: list[_ConeState] = []
        self._next_id = 1
        self._unique_seen_count = 0

    @staticmethod
    def _spacing(centers: Sequence[Point]) -> float:
        if len(centers) < 2:
            return 30.0
        nearest = []
        for index, center in enumerate(centers):
            distances = [
                _distance(center, other)
                for other_index, other in enumerate(centers)
                if other_index != index
            ]
            if distances:
                nearest.append(min(distances))
        return statistics.median(nearest) if nearest else 30.0

    def _assign(
        self, centers: Sequence[Point], maximum_distance: float
    ) -> dict[int, int]:
        candidates = sorted(
            (
                _distance(
                    (
                        state.center[0]
                        + state.velocity[0] * (state.missed_frames + 1),
                        state.center[1]
                        + state.velocity[1] * (state.missed_frames + 1),
                    ),
                    center,
                ),
                state_index,
                center_index,
            )
            for state_index, state in enumerate(self._states)
            for center_index, center in enumerate(centers)
        )
        assignments: dict[int, int] = {}
        used_states: set[int] = set()
        used_centers: set[int] = set()
        for distance, state_index, center_index in candidates:
            if distance > maximum_distance:
                break
            if state_index in used_states or center_index in used_centers:
                continue
            used_states.add(state_index)
            used_centers.add(center_index)
            assignments[center_index] = state_index
        return assignments

    def update(self, analysis: AnalysisResult) -> ConeCountingResult:
        centers = [detection.center for detection in analysis.ordered]
        spacing = self._spacing(centers)
        maximum_distance = max(
            self.min_match_distance,
            min(self.max_match_distance, spacing * self.match_distance_ratio),
        )
        assignments = self._assign(centers, maximum_distance)
        matched_states = set(assignments.values())
        new_ids: list[int] = []

        retained: list[_ConeState] = []
        for state_index, state in enumerate(self._states):
            state.age += 1
            if state_index in matched_states:
                retained.append(state)
                continue
            state.missed_frames += 1
            state.hit_streak = 0
            state.votes.append(False)
            if state.missed_frames <= self.max_missed_frames:
                if state.state == "confirmed":
                    state.state = "held"
                retained.append(state)

        index_to_stable_id: dict[int, int] = {}
        for center_index, center in enumerate(centers):
            state_index = assignments.get(center_index)
            if state_index is None:
                state = _ConeState(
                    stable_id=self._next_id,
                    center=center,
                    votes=deque(maxlen=self.confirm_window),
                )
                self._next_id += 1
                retained.append(state)
            else:
                state = self._states[state_index]
                elapsed = state.missed_frames + 1
                measured_velocity = (
                    (center[0] - state.center[0]) / elapsed,
                    (center[1] - state.center[1]) / elapsed,
                )
                state.velocity = (
                    state.velocity[0] * 0.5 + measured_velocity[0] * 0.5,
                    state.velocity[1] * 0.5 + measured_velocity[1] * 0.5,
                )
                state.center = center
                state.missed_frames = 0
                state.hits += 1
                state.hit_streak += 1
                if state.state == "held":
                    state.state = "confirmed"
            state.votes.append(True)
            if (
                state.state == "tentative"
                and len(state.votes) == self.confirm_window
                and sum(state.votes) >= self.confirm_votes
            ):
                state.state = "confirmed"
            if state.state == "confirmed" and not state.counted:
                state.counted = True
                self._unique_seen_count += 1
                new_ids.append(state.stable_id)
            index_to_stable_id[center_index] = state.stable_id

        self._states = retained
        active_ids = tuple(
            state.stable_id
            for state in self._states
            if state.state == "confirmed" and state.missed_frames == 0
        )
        held_ids = tuple(
            state.stable_id
            for state in self._states
            if state.state == "held"
        )
        tracked_cones = tuple(
            {
                "stable_id": state.stable_id,
                "state": state.state,
                "matched": state.missed_frames == 0,
                "counted": state.counted,
                "hits": state.hits,
                "hit_streak": state.hit_streak,
                "age": state.age,
                "missed_frames": state.missed_frames,
                "center_x": state.center[0],
                "center_y": state.center[1],
            }
            for state in self._states
        )
        return ConeCountingResult(
            stable_visible_count=len(active_ids),
            unique_seen_count=self._unique_seen_count,
            new_cone_ids=tuple(new_ids),
            active_cone_ids=active_ids,
            held_cone_ids=held_ids,
            index_to_stable_id=index_to_stable_id,
            stable_cone_ids=tuple(
                index_to_stable_id.get(index) for index in range(len(centers))
            ),
            tracked_cones=tracked_cones,
            policy={
                "confirm_window": self.confirm_window,
                "confirm_votes": self.confirm_votes,
                "max_missed_frames": self.max_missed_frames,
                "match_distance_ratio": self.match_distance_ratio,
                "min_match_distance": self.min_match_distance,
                "max_match_distance": self.max_match_distance,
            },
        )
