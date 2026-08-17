"""Route-level temporal confirmation for SHIFT business alerts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class ShiftTracker:
    confirm_frames: int = 3
    membership_stable_frames: int = 10
    _streaks: dict[int, int] = field(default_factory=dict, init=False)
    _member_counts: dict[int, int] = field(default_factory=dict, init=False)
    _stable_frames: dict[int, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.confirm_frames < 1:
            raise ValueError("confirm_frames must be at least 1")
        if self.membership_stable_frames < 1:
            raise ValueError("membership_stable_frames must be at least 1")

    def update(
        self,
        candidate_route_ids: Iterable[int],
        route_member_counts: dict[int, int],
    ) -> frozenset[int]:
        candidates = set(candidate_route_ids)
        for route_id in set(self._member_counts) - set(route_member_counts):
            self._stable_frames[route_id] = 0
        for route_id, member_count in route_member_counts.items():
            previous_count = self._member_counts.get(route_id)
            if previous_count == member_count:
                self._stable_frames[route_id] = self._stable_frames.get(route_id, 0) + 1
            else:
                self._member_counts[route_id] = member_count
                # A newly observed route has only one stable membership frame.
                # Do not grant it the full warm-up period on first sight.
                self._stable_frames[route_id] = 1
        eligible = {
            route_id
            for route_id in candidates
            if self._stable_frames.get(route_id, 0) >= self.membership_stable_frames
        }
        self._streaks = {
            route_id: self._streaks.get(route_id, 0) + 1
            for route_id in eligible
        }
        return frozenset(
            route_id
            for route_id, streak in self._streaks.items()
            if streak >= self.confirm_frames
        )
