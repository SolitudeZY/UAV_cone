"""Temporal per-route gap identity, thresholds, and alert voting."""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

from route_tracker import RouteTrackingResult, TrackedRoute
from traffic_rules import AnalysisResult, MissingAlert


Point = tuple[float, float]
GapKey = tuple[int, int]


def _distance(first: Point, second: Point) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _gap_key(first_id: int, second_id: int) -> GapKey:
    return tuple(sorted((first_id, second_id)))


def _point_in_convex_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    signs: list[bool] = []
    for first, second in zip(polygon, (*polygon[1:], polygon[0])):
        cross = (second[0] - first[0]) * (point[1] - first[1]) - (
            second[1] - first[1]
        ) * (point[0] - first[0])
        if abs(cross) > 1e-9:
            signs.append(cross > 0.0)
    return not signs or all(sign == signs[0] for sign in signs)


def _route_gap_items(
    analysis: AnalysisResult, route: Sequence[int]
) -> list[tuple[int, int, float, float]]:
    items: list[tuple[int, int, float, float]] = []
    for first_index, second_index in zip(route, route[1:]):
        pixel_gap = _distance(
            analysis.ordered[first_index].center,
            analysis.ordered[second_index].center,
        )
        first_ground = (
            analysis.ground_centers[first_index]
            if first_index < len(analysis.ground_centers)
            else None
        )
        second_ground = (
            analysis.ground_centers[second_index]
            if second_index < len(analysis.ground_centers)
            else None
        )
        corrected_gap = (
            _distance(first_ground, second_ground)
            if first_ground is not None and second_ground is not None
            else pixel_gap
        )
        items.append((first_index, second_index, pixel_gap, corrected_gap))
    return items


@dataclass(frozen=True)
class GapRouteThreshold:
    stable_id: int
    current_route_index: int
    nominal_gap_pixels: float | None
    frame_average_pixels: float | None
    average_of_averages_pixels: float | None
    history_size: int
    confidence: float
    excluded_close_gaps: tuple[float, ...]
    threshold_source: str
    distance_unit: str = "pixels"


@dataclass(frozen=True)
class TrackedGap:
    stable_id: int
    stable_route_id: int
    after_index: int
    endpoint_ids: GapKey
    candidate_missing: bool
    active_missing: bool
    positive_votes: int
    vote_count: int
    normal_streak: int
    cooldown_remaining: int
    business_eligible: bool
    alert_level: str
    suspect_frames: int
    strong_alert_streak: int
    distance_ratio: float | None
    pixel_ratio: float | None
    measurement_conflict: bool
    position_conflict: bool


@dataclass(frozen=True)
class SuppressedMissing:
    """Raw MISSING candidate suppressed by a confirmed FALLEN cone."""

    alert: MissingAlert
    stable_route_id: int
    fallen_stable_ids: tuple[int, ...]
    reason: str = "fallen_occupies_expected_position"


@dataclass(frozen=True)
class GapTrackingResult:
    missing: tuple[MissingAlert, ...]
    routes: tuple[GapRouteThreshold, ...]
    gaps: tuple[TrackedGap, ...] = ()
    missing_gap_ids: tuple[int, ...] = ()
    suspect: tuple[MissingAlert, ...] = ()
    suspect_gap_ids: tuple[int, ...] = ()
    policy: GapAlertPolicy | None = None
    suppressed_by_fallen: tuple[SuppressedMissing, ...] = ()


@dataclass(frozen=True)
class GapAlertPolicy:
    suspect_ratio: float
    alert_ratio: float
    vote_window: int
    confirm_votes: int
    suspect_frames: int
    alert_confirm_frames: int
    clear_frames: int
    max_ratio_disagreement: float


@dataclass(frozen=True)
class _MissingCandidate:
    alert: MissingAlert
    distance_ratio: float
    pixel_ratio: float
    measurement_conflict: bool
    position_conflict: bool

    @property
    def credible(self) -> bool:
        return not self.measurement_conflict and not self.position_conflict


@dataclass
class _StableGapState:
    stable_id: int
    endpoint_ids: GapKey
    votes: deque[bool]
    level: str = "normal"
    suspect_frames: int = 0
    strong_alert_streak: int = 0
    normal_streak: int = 0
    cooldown_remaining: int = 0
    last_alert: MissingAlert | None = None


@dataclass
class _GapState:
    history: deque[float]
    nominal_gap: float | None = None
    cone_ids: tuple[int, ...] = ()
    centers: tuple[Point, ...] = ()
    gaps: dict[GapKey, _StableGapState] = field(default_factory=dict)
    history_cooldown_remaining: int = 0


@dataclass
class GapThresholdTracker:
    history_frames: int = 30
    min_history_frames: int = 3
    close_ratio: float = 0.55
    max_drop_ratio: float = 0.05
    max_rise_ratio: float = 0.10
    minimum_gap_pixels: float = 0.0
    vote_window: int = 5
    confirm_votes: int = 3
    alert_ratio: float = 2.2
    suspect_frames: int = 20
    alert_confirm_frames: int = 5
    max_ratio_disagreement: float = 0.6
    clear_frames: int = 3
    membership_cooldown_frames: int = 5
    min_alert_cones: int = 3
    history_min_cones: int = 5
    _states: dict[int, _GapState] = field(default_factory=dict, init=False)
    _next_cone_id: int = field(default=1, init=False)
    _next_gap_id: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        if self.history_frames < 1:
            raise ValueError("history_frames must be at least 1")
        if not 1 <= self.min_history_frames <= self.history_frames:
            raise ValueError("min_history_frames must be within history_frames")
        if not 0.0 < self.close_ratio < 1.0:
            raise ValueError("close_ratio must be in (0, 1)")
        if not 0.0 <= self.max_drop_ratio < 1.0:
            raise ValueError("max_drop_ratio must be in [0, 1)")
        if self.max_rise_ratio < 0.0:
            raise ValueError("max_rise_ratio must not be negative")
        if self.minimum_gap_pixels < 0.0:
            raise ValueError("minimum_gap_pixels must not be negative")
        if self.vote_window < 1:
            raise ValueError("vote_window must be at least 1")
        if not 1 <= self.confirm_votes <= self.vote_window:
            raise ValueError("confirm_votes must be within vote_window")
        if self.alert_ratio <= 1.0:
            raise ValueError("alert_ratio must be greater than 1")
        if self.suspect_frames < 1:
            raise ValueError("suspect_frames must be at least 1")
        if self.alert_confirm_frames < 1:
            raise ValueError("alert_confirm_frames must be at least 1")
        if self.max_ratio_disagreement < 0.0:
            raise ValueError("max_ratio_disagreement must be non-negative")
        if self.clear_frames < 1:
            raise ValueError("clear_frames must be at least 1")
        if not 3 <= self.membership_cooldown_frames <= 5:
            raise ValueError("membership_cooldown_frames must be between 3 and 5")
        if self.min_alert_cones < 3:
            raise ValueError("min_alert_cones must be at least 3")
        if self.history_min_cones < 3:
            raise ValueError("history_min_cones must be at least 3")

    def _state(self, stable_id: int) -> _GapState:
        state = self._states.get(stable_id)
        if state is None:
            state = _GapState(deque(maxlen=self.history_frames))
            self._states[stable_id] = state
        return state

    def _new_cone_id(self) -> int:
        stable_id = self._next_cone_id
        self._next_cone_id += 1
        return stable_id

    def _new_gap_state(self, key: GapKey) -> _StableGapState:
        state = _StableGapState(
            stable_id=self._next_gap_id,
            endpoint_ids=key,
            votes=deque(maxlen=self.vote_window),
        )
        self._next_gap_id += 1
        return state

    def _update_state(
        self,
        state: _GapState,
        gaps: Sequence[float],
        *,
        allow_history_update: bool,
    ) -> tuple[float | None, tuple[float, ...]]:
        frame_average = statistics.median(gaps) if gaps else None
        excluded_close = (
            tuple(gap for gap in gaps if gap < state.nominal_gap * self.close_ratio)
            if state.nominal_gap is not None
            else ()
        )
        if frame_average is None or not allow_history_update:
            return frame_average, excluded_close
        if (
            state.nominal_gap is not None
            and frame_average < state.nominal_gap * self.close_ratio
        ):
            return frame_average, excluded_close

        state.history.append(frame_average)
        if state.nominal_gap is None and len(state.history) < self.min_history_frames:
            return frame_average, excluded_close
        target = statistics.median(state.history)
        if state.nominal_gap is None:
            state.nominal_gap = target
        else:
            lower = state.nominal_gap * (1.0 - self.max_drop_ratio)
            upper = state.nominal_gap * (1.0 + self.max_rise_ratio)
            state.nominal_gap = min(upper, max(lower, target))
        return frame_average, excluded_close

    def _match_cones(
        self,
        state: _GapState,
        centers: Sequence[Point],
        track: TrackedRoute,
    ) -> tuple[tuple[int, ...], set[GapKey]]:
        if not state.cone_ids:
            return tuple(self._new_cone_id() for _ in centers), set()

        predicted = [
            (center[0] + track.velocity[0], center[1] + track.velocity[1])
            for center in state.centers
        ]
        maximum_distance = max(20.0, track.spacing * 0.75)
        candidates = sorted(
            (
                _distance(previous, current),
                previous_index,
                current_index,
            )
            for previous_index, previous in enumerate(predicted)
            for current_index, current in enumerate(centers)
        )
        previous_matches: set[int] = set()
        current_matches: set[int] = set()
        current_ids: list[int | None] = [None] * len(centers)
        for distance, previous_index, current_index in candidates:
            if distance > maximum_distance:
                break
            if previous_index in previous_matches or current_index in current_matches:
                continue
            previous_matches.add(previous_index)
            current_matches.add(current_index)
            current_ids[current_index] = state.cone_ids[previous_index]

        for current_index, stable_id in enumerate(current_ids):
            if stable_id is None:
                current_ids[current_index] = self._new_cone_id()

        resolved_ids = tuple(int(stable_id) for stable_id in current_ids)
        added_ids = set(resolved_ids) - set(state.cone_ids)
        removed_ids = set(state.cone_ids) - set(resolved_ids)
        affected = {
            _gap_key(first_id, second_id)
            for first_id, second_id in zip(resolved_ids, resolved_ids[1:])
            if first_id in added_ids or second_id in added_ids
        }
        current_positions = {
            stable_id: index for index, stable_id in enumerate(resolved_ids)
        }
        for removed_id in removed_ids:
            removed_index = state.cone_ids.index(removed_id)
            left = next(
                (
                    state.cone_ids[index]
                    for index in range(removed_index - 1, -1, -1)
                    if state.cone_ids[index] in current_positions
                ),
                None,
            )
            right = next(
                (
                    state.cone_ids[index]
                    for index in range(removed_index + 1, len(state.cone_ids))
                    if state.cone_ids[index] in current_positions
                ),
                None,
            )
            if (
                left is not None
                and right is not None
                and abs(current_positions[left] - current_positions[right]) == 1
            ):
                affected.add(_gap_key(left, right))
        return resolved_ids, affected

    def _candidate_alert(
        self,
        analysis: AnalysisResult,
        first_index: int,
        second_index: int,
        gap_pixels: float,
        gap_distance: float,
        raw_alert: MissingAlert | None,
        nominal_gap: float,
        gap_ratio: float,
    ) -> _MissingCandidate | None:
        if raw_alert is None:
            return None
        raw_reference = (
            raw_alert.reference_gap_corrected
            if raw_alert.reference_gap_corrected is not None
            else raw_alert.reference_gap_pixels
        )
        reference = max(
            nominal_gap,
            raw_reference,
            self.minimum_gap_pixels if analysis.distance_unit == "pixels" else 0.0,
        )
        distance_ratio = gap_distance / reference
        if distance_ratio <= gap_ratio:
            return None
        pixel_reference = max(raw_alert.reference_gap_pixels, 1e-9)
        pixel_ratio = gap_pixels / pixel_reference
        measurement_conflict = False
        if analysis.distance_unit != "pixels":
            smaller_ratio = min(distance_ratio, pixel_ratio)
            measurement_conflict = (
                smaller_ratio <= 1e-9
                or abs(distance_ratio - pixel_ratio) / smaller_ratio
                > self.max_ratio_disagreement
            )
        alert = self._alert_for_gap(
            analysis,
            first_index,
            second_index,
            gap_pixels,
            gap_distance,
            reference,
            raw_alert.reference_gap_pixels,
            raw_alert.positions,
        )
        endpoint_polygons = (
            analysis.ordered[first_index].polygon,
            analysis.ordered[second_index].polygon,
        )
        position_conflict = any(
            _point_in_convex_polygon(position, polygon)
            for position in alert.positions
            for polygon in endpoint_polygons
        )
        return _MissingCandidate(
            alert,
            distance_ratio,
            pixel_ratio,
            measurement_conflict,
            position_conflict,
        )

    @staticmethod
    def _alert_for_gap(
        analysis: AnalysisResult,
        first_index: int,
        second_index: int,
        gap_pixels: float,
        gap_distance: float,
        reference: float,
        reference_pixels: float,
        positions: tuple[Point, ...] | None = None,
    ) -> MissingAlert:
        estimated_count = max(1, round(gap_distance / reference) - 1)
        first = analysis.ordered[first_index].center
        second = analysis.ordered[second_index].center
        if positions is None or len(positions) != estimated_count:
            positions = tuple(
                (
                    first[0]
                    + (second[0] - first[0]) * step / (estimated_count + 1),
                    first[1]
                    + (second[1] - first[1]) * step / (estimated_count + 1),
                )
                for step in range(1, estimated_count + 1)
            )
        return MissingAlert(
            first_index,
            gap_pixels,
            reference_pixels,
            estimated_count,
            positions,
            gap_corrected=(
                gap_distance if analysis.distance_unit != "pixels" else None
            ),
            reference_gap_corrected=(
                reference if analysis.distance_unit != "pixels" else None
            ),
            distance_unit=analysis.distance_unit,
        )

    def _vote(
        self,
        gap: _StableGapState,
        candidate: _MissingCandidate | None,
        *,
        allowed: bool,
        can_activate: bool,
    ) -> None:
        if not allowed:
            self._reset_gap_level(gap)
            return
        if gap.cooldown_remaining > 0:
            gap.cooldown_remaining -= 1
            self._reset_gap_level(gap)
            return

        positive = candidate is not None
        gap.votes.append(positive)
        if positive:
            gap.last_alert = candidate.alert
            gap.normal_streak = 0
            if (
                can_activate
                and candidate.credible
                and candidate.distance_ratio >= self.alert_ratio
            ):
                gap.strong_alert_streak += 1
            else:
                gap.strong_alert_streak = 0
        else:
            gap.normal_streak += 1
            gap.strong_alert_streak = 0

        if (
            gap.level == "normal"
            and can_activate
            and sum(gap.votes) >= self.confirm_votes
        ):
            gap.level = "suspect"
            gap.suspect_frames = 1
            # Strong evidence collected while the route history was warming up
            # must not skip the visible suspect stage on the first eligible frame.
            gap.strong_alert_streak = 0
        elif gap.level in {"suspect", "alert"}:
            gap.suspect_frames += 1

        if (
            gap.level == "suspect"
            and candidate is not None
            and candidate.credible
            and (
                gap.strong_alert_streak >= self.alert_confirm_frames
                or gap.suspect_frames >= self.suspect_frames
            )
        ):
            gap.level = "alert"

        if gap.level != "normal" and gap.normal_streak >= self.clear_frames:
            self._reset_gap_level(gap)

    @staticmethod
    def _reset_gap_level(gap: _StableGapState) -> None:
        gap.votes.clear()
        gap.level = "normal"
        gap.suspect_frames = 0
        gap.strong_alert_streak = 0
        gap.normal_streak = 0
        gap.last_alert = None

    def update(
        self,
        analysis: AnalysisResult,
        tracking: RouteTrackingResult,
        *,
        gap_ratio: float,
        fallen_indices: Sequence[int] = (),
        fallen_stable_ids: dict[int, int] | None = None,
    ) -> GapTrackingResult:
        if gap_ratio <= 1.0:
            raise ValueError("gap_ratio must be greater than 1")
        if self.alert_ratio < gap_ratio:
            raise ValueError("alert_ratio must be greater than or equal to gap_ratio")
        active_ids = {track.stable_id for track in tracking.tracks}
        self._states = {
            stable_id: state
            for stable_id, state in self._states.items()
            if stable_id in active_ids
        }
        tracks_by_route = {
            track.current_route_index: track
            for track in tracking.tracks
            if track.current_route_index is not None
        }
        alerts: list[MissingAlert] = []
        alert_gap_ids: list[int] = []
        suspects: list[MissingAlert] = []
        suspect_gap_ids: list[int] = []
        suppressed_by_fallen: list[SuppressedMissing] = []
        fallen_stable_ids = fallen_stable_ids or {}
        fallen_centers = {
            index: analysis.ordered[index].center
            for index in fallen_indices
            if 0 <= index < len(analysis.ordered)
        }
        diagnostics: list[GapRouteThreshold] = []
        gap_diagnostics: list[TrackedGap] = []

        for route_index, route in enumerate(analysis.routes):
            track = tracks_by_route.get(route_index)
            if track is None:
                continue
            gap_items = _route_gap_items(analysis, route)
            state = self._state(track.stable_id)
            raw_by_after = {
                alert.after_index: alert
                for alert in analysis.missing
                if alert.after_index in route
            }
            # Keep the raw candidate in events.jsonl, but prevent this gap
            # from entering the MISSING state when a confirmed fallen cone is
            # sitting at one of its expected positions.
            suppressed_after: set[int] = set()
            for alert in analysis.missing:
                if alert.after_index not in route:
                    continue
                threshold = max(
                    20.0,
                    (state.nominal_gap or alert.reference_gap_pixels) * 0.80,
                )
                matching_fallen = tuple(
                    index
                    for index, center in fallen_centers.items()
                    if any(_distance(center, position) <= threshold for position in alert.positions)
                )
                if not matching_fallen:
                    continue
                suppressed_after.add(alert.after_index)
                suppressed_by_fallen.append(
                    SuppressedMissing(
                        alert=alert,
                        stable_route_id=track.business_stable_id,
                        fallen_stable_ids=tuple(
                            sorted(
                                fallen_stable_ids[index]
                                for index in matching_fallen
                                if index in fallen_stable_ids
                            )
                        ),
                    )
                )
            centers = tuple(analysis.ordered[index].center for index in route)
            cone_ids, affected_keys = self._match_cones(state, centers, track)
            if affected_keys:
                state.history_cooldown_remaining = self.membership_cooldown_frames
            history_in_cooldown = state.history_cooldown_remaining > 0
            history_update_allowed = (
                track.alert_ready
                and len(route) >= self.history_min_cones
                and not history_in_cooldown
            )
            ignored_after_indices = set(analysis.missing_edge_ignored)
            reliable_gaps = [
                gap_distance
                for first_index, _, _, gap_distance in gap_items
                if first_index not in raw_by_after
                and first_index not in ignored_after_indices
            ]
            frame_average, excluded_close = self._update_state(
                state,
                reliable_gaps,
                allow_history_update=history_update_allowed,
            )
            if state.history_cooldown_remaining > 0:
                state.history_cooldown_remaining -= 1

            enough_history = len(state.history) >= self.min_history_frames
            if state.nominal_gap is None and state.history:
                source = "warming_up"
            elif state.nominal_gap is None and len(route) < self.history_min_cones:
                source = "insufficient_cones"
            elif state.nominal_gap is None:
                source = "insufficient_history"
            elif history_in_cooldown:
                source = "frozen_membership_change"
            elif len(route) < self.history_min_cones:
                source = "frozen_low_support"
            elif not track.alert_ready:
                source = "frozen_unconfirmed_route"
            else:
                source = "history"
            current_keys = {
                _gap_key(first_id, second_id)
                for first_id, second_id in zip(cone_ids, cone_ids[1:])
            }
            for key in affected_keys:
                gap_state = state.gaps.get(key)
                if gap_state is None:
                    gap_state = self._new_gap_state(key)
                    state.gaps[key] = gap_state
                gap_state.cooldown_remaining = self.membership_cooldown_frames
                self._reset_gap_level(gap_state)

            eligible = len(route) >= self.min_alert_cones
            for gap_index, (
                first_index,
                second_index,
                gap_pixels,
                gap_distance,
            ) in enumerate(gap_items):
                key = _gap_key(cone_ids[gap_index], cone_ids[gap_index + 1])
                gap_state = state.gaps.get(key)
                if gap_state is None:
                    gap_state = self._new_gap_state(key)
                    state.gaps[key] = gap_state
                raw_alert = raw_by_after.get(first_index)
                if first_index in suppressed_after:
                    self._vote(
                        gap_state,
                        None,
                        allowed=False,
                        can_activate=False,
                    )
                    candidate = None
                else:
                    provisional_reference = (
                        state.nominal_gap
                        or (
                            raw_alert.reference_gap_corrected
                            if raw_alert is not None
                            and raw_alert.reference_gap_corrected is not None
                            else raw_alert.reference_gap_pixels
                            if raw_alert is not None
                            else None
                        )
                    )
                    candidate = (
                        self._candidate_alert(
                            analysis,
                            first_index,
                            second_index,
                            gap_pixels,
                            gap_distance,
                            raw_alert,
                            provisional_reference,
                            gap_ratio,
                        )
                        if provisional_reference is not None
                        else None
                    )
                if first_index not in suppressed_after:
                    self._vote(
                        gap_state,
                        candidate,
                        allowed=track.alert_ready and eligible,
                        can_activate=enough_history and state.nominal_gap is not None,
                    )
                if gap_state.level != "normal":
                    reference = max(
                        state.nominal_gap or 0.0,
                        self.minimum_gap_pixels
                        if analysis.distance_unit == "pixels"
                        else 0.0,
                    )
                    reference_pixels = (
                        gap_state.last_alert.reference_gap_pixels
                        if gap_state.last_alert is not None
                        else gap_pixels
                    )
                    alert = (
                        candidate.alert
                        if candidate is not None
                        else (
                            self._alert_for_gap(
                                analysis,
                                first_index,
                                second_index,
                                gap_pixels,
                                gap_distance,
                                reference,
                                reference_pixels,
                            )
                            if reference > 0.0
                            else gap_state.last_alert
                        )
                    )
                    if alert is not None:
                        if gap_state.level == "alert":
                            alerts.append(alert)
                            alert_gap_ids.append(gap_state.stable_id)
                        else:
                            suspects.append(alert)
                            suspect_gap_ids.append(gap_state.stable_id)
                gap_diagnostics.append(
                    TrackedGap(
                        stable_id=gap_state.stable_id,
                        stable_route_id=track.business_stable_id,
                        after_index=first_index,
                        endpoint_ids=key,
                        candidate_missing=candidate is not None,
                        active_missing=gap_state.level == "alert",
                        positive_votes=sum(gap_state.votes),
                        vote_count=len(gap_state.votes),
                        normal_streak=gap_state.normal_streak,
                        cooldown_remaining=gap_state.cooldown_remaining,
                        business_eligible=eligible,
                        alert_level=gap_state.level,
                        suspect_frames=gap_state.suspect_frames,
                        strong_alert_streak=gap_state.strong_alert_streak,
                        distance_ratio=(
                            candidate.distance_ratio if candidate is not None else None
                        ),
                        pixel_ratio=(
                            candidate.pixel_ratio if candidate is not None else None
                        ),
                        measurement_conflict=(
                            candidate.measurement_conflict
                            if candidate is not None
                            else False
                        ),
                        position_conflict=(
                            candidate.position_conflict
                            if candidate is not None
                            else False
                        ),
                    )
                )

            state.cone_ids = cone_ids
            state.centers = centers
            state.gaps = {
                key: gap_state
                for key, gap_state in state.gaps.items()
                if key in current_keys
            }
            diagnostics.append(
                GapRouteThreshold(
                    stable_id=track.business_stable_id,
                    current_route_index=route_index,
                    nominal_gap_pixels=state.nominal_gap,
                    frame_average_pixels=frame_average,
                    average_of_averages_pixels=(
                        statistics.fmean(state.history) if state.history else None
                    ),
                    history_size=len(state.history),
                    confidence=min(1.0, len(state.history) / self.min_history_frames),
                    excluded_close_gaps=excluded_close,
                    threshold_source=source,
                    distance_unit=analysis.distance_unit,
                )
            )

        return GapTrackingResult(
            missing=tuple(alerts),
            routes=tuple(diagnostics),
            gaps=tuple(gap_diagnostics),
            missing_gap_ids=tuple(alert_gap_ids),
            suspect=tuple(suspects),
            suspect_gap_ids=tuple(suspect_gap_ids),
            policy=GapAlertPolicy(
                suspect_ratio=gap_ratio,
                alert_ratio=self.alert_ratio,
                vote_window=self.vote_window,
                confirm_votes=self.confirm_votes,
                suspect_frames=self.suspect_frames,
                alert_confirm_frames=self.alert_confirm_frames,
                clear_frames=self.clear_frames,
                max_ratio_disagreement=self.max_ratio_disagreement,
            ),
            suppressed_by_fallen=tuple(suppressed_by_fallen),
        )
