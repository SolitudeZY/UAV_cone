"""Stable cone identities and temporal FALLEN confirmation."""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

from route_tracker import RouteTrackingResult
from traffic_rules import AnalysisResult


Point = tuple[float, float]


def _distance(first: Point, second: Point) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


@dataclass(frozen=True)
class FallenObservation:
    index: int
    stable_cone_id: int
    angle_degrees: float
    difference_degrees: float


@dataclass(frozen=True)
class FallenTrackingResult:
    active: tuple[FallenObservation, ...]
    suspect: tuple[FallenObservation, ...]
    index_to_stable_id: dict[int, int]
    policy: dict[str, int | float]


@dataclass
class _ConeState:
    stable_id: int
    center: Point
    votes: deque[bool]
    velocity: Point = (0.0, 0.0)
    active: bool = False
    normal_streak: int = 0
    missed_frames: int = 0
    last_fallen: tuple[float, float] | None = None
    last_vote_angle: float | None = None
    cooldown_remaining: int = 0


class ConeTrackRegistry:
    """Match all cone detections, including fallen cones, across frames."""

    def __init__(
        self,
        *,
        vote_window: int = 7,
        confirm_votes: int = 5,
        clear_frames: int = 3,
        max_missed_frames: int = 5,
        reactivation_cooldown_frames: int = 30,
        match_distance_ratio: float = 0.6,
        max_angle_jump_degrees: float = 20.0,
    ) -> None:
        if not 1 <= confirm_votes <= vote_window:
            raise ValueError("confirm_votes must be within vote_window")
        if clear_frames < 1 or max_missed_frames < 0:
            raise ValueError("clear_frames and max_missed_frames are invalid")
        if reactivation_cooldown_frames < 0:
            raise ValueError("reactivation_cooldown_frames must not be negative")
        if match_distance_ratio <= 0.0:
            raise ValueError("match_distance_ratio must be positive")
        if not 0.0 < max_angle_jump_degrees <= 90.0:
            raise ValueError("max_angle_jump_degrees must be within (0, 90]")
        self.vote_window = vote_window
        self.confirm_votes = confirm_votes
        self.clear_frames = clear_frames
        self.max_missed_frames = max_missed_frames
        self.reactivation_cooldown_frames = reactivation_cooldown_frames
        self.match_distance_ratio = match_distance_ratio
        self.max_angle_jump_degrees = max_angle_jump_degrees
        self._states: list[_ConeState] = []
        self._next_id = 1

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
                        state.center[0] + state.velocity[0] * (state.missed_frames + 1),
                        state.center[1] + state.velocity[1] * (state.missed_frames + 1),
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

    @staticmethod
    def _angle_difference(first: float, second: float) -> float:
        difference = abs(first - second) % 180.0
        return min(difference, 180.0 - difference)

    def update(
        self,
        analysis: AnalysisResult,
        tracking: RouteTrackingResult | None = None,
    ) -> FallenTrackingResult:
        centers = [detection.center for detection in analysis.ordered]
        fallen_by_index = {alert.index: alert for alert in analysis.fallen}
        maximum_distance = max(
            15.0,
            min(100.0, self._spacing(centers) * self.match_distance_ratio),
        )
        assignments = self._assign(centers, maximum_distance)
        matched_states = set(assignments.values())
        held_active: list[FallenObservation] = []
        for state_index, state in enumerate(self._states):
            if state_index not in matched_states:
                state.missed_frames += 1
                state.votes.append(False)
                if state.active and state.missed_frames <= self.max_missed_frames and state.last_fallen is not None:
                    angle_degrees, difference_degrees = state.last_fallen
                    held_active.append(
                        FallenObservation(
                            index=-1,
                            stable_cone_id=state.stable_id,
                            angle_degrees=angle_degrees,
                            difference_degrees=difference_degrees,
                        )
                    )
                if state.missed_frames > self.max_missed_frames:
                    state.active = False
                    state.votes.clear()

        index_to_stable_id: dict[int, int] = {}
        active: list[FallenObservation] = held_active
        suspect: list[FallenObservation] = []
        for index, center in enumerate(centers):
            state_index = assignments.get(index)
            if state_index is None:
                state = _ConeState(
                    stable_id=self._next_id,
                    center=center,
                    votes=deque(maxlen=self.vote_window),
                )
                self._next_id += 1
                self._states.append(state)
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
            index_to_stable_id[index] = state.stable_id
            fallen = fallen_by_index.get(index)
            angle_continuous = (
                fallen is not None
                and (
                    state.last_vote_angle is None
                    or self._angle_difference(
                        fallen.angle_degrees, state.last_vote_angle
                    )
                    <= self.max_angle_jump_degrees
                )
            )
            is_fallen = fallen is not None and angle_continuous
            state.votes.append(is_fallen)
            if is_fallen:
                state.normal_streak = 0
                state.last_vote_angle = fallen.angle_degrees
                state.last_fallen = (
                    fallen.angle_degrees,
                    fallen.difference_degrees,
                )
            else:
                state.normal_streak += 1
            cooldown_blocked = state.cooldown_remaining > 0
            if cooldown_blocked:
                state.cooldown_remaining -= 1
            if (
                not state.active
                and not cooldown_blocked
                and state.cooldown_remaining == 0
                and len(state.votes) == self.vote_window
                and sum(state.votes) >= self.confirm_votes
                and is_fallen
            ):
                state.active = True
            if state.active and state.normal_streak >= self.clear_frames:
                state.active = False
                state.votes.clear()
                state.cooldown_remaining = self.reactivation_cooldown_frames
            if fallen is not None and angle_continuous:
                observation = FallenObservation(
                    index=index,
                    stable_cone_id=state.stable_id,
                    angle_degrees=fallen.angle_degrees,
                    difference_degrees=fallen.difference_degrees,
                )
                if state.active:
                    active.append(observation)
                elif sum(state.votes) > 0:
                    suspect.append(observation)
            elif state.active and state.last_fallen is not None:
                # Keep a confirmed FALLEN identity alive through a short
                # normal/lost observation gap; this prevents alert toggling.
                angle_degrees, difference_degrees = state.last_fallen
                active.append(
                    FallenObservation(
                        index=index,
                        stable_cone_id=state.stable_id,
                        angle_degrees=angle_degrees,
                        difference_degrees=difference_degrees,
                    )
                )

        self._states = [
            state
            for state in self._states
            if state.missed_frames <= self.max_missed_frames
        ]
        return FallenTrackingResult(
            active=tuple(active),
            suspect=tuple(suspect),
            index_to_stable_id=index_to_stable_id,
            policy={
                "vote_window": self.vote_window,
                "confirm_votes": self.confirm_votes,
                "clear_frames": self.clear_frames,
                "max_missed_frames": self.max_missed_frames,
                "reactivation_cooldown_frames": self.reactivation_cooldown_frames,
                "match_distance_ratio": self.match_distance_ratio,
                "max_angle_jump_degrees": self.max_angle_jump_degrees,
            },
        )
