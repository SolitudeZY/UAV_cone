#!/usr/bin/env python3
"""Focused regression tests for UAV physical IDs and grouped ROI behavior."""

from __future__ import annotations

import unittest
from collections import deque

import numpy as np

from infer_obb import _extract_detections, build_parser
from gap_tracker import GapThresholdTracker, _MissingCandidate, _StableGapState
from traffic_rules import MissingAlert
from uav_physical_cone_counter import (
    MotionEstimate,
    PhysicalRoiTracker,
    PhysicalTrack,
    UavPhysicalCounter,
    build_parser as build_physical_parser,
)


def motion(valid: bool = True) -> MotionEstimate:
    return MotionEstimate(
        valid=valid,
        method="test",
        inliers=100 if valid else 0,
        points=100 if valid else 0,
        inlier_ratio=1.0 if valid else 0.0,
        reprojection_error=0.0 if valid else None,
        reference_from_current=np.eye(3, dtype=np.float64),
    )


class UavPhysicalCounterTest(unittest.TestCase):
    def test_confirms_one_physical_id_and_holds_it_during_degraded_motion(self) -> None:
        counter = UavPhysicalCounter(
            confirm_window=3,
            confirm_votes=2,
            max_missed_frames=8,
            match_distance_pixels=20.0,
        )
        counter.update([(100.0, 100.0)], [(100.0, 90.0)], motion())
        counter.update([(102.0, 100.0)], [(102.0, 90.0)], motion())
        confirmed = counter.update(
            [(104.0, 100.0)], [(104.0, 90.0)], motion()
        )

        self.assertEqual(confirmed.unique_seen_count, 1)
        self.assertEqual(confirmed.active_physical_ids, (1,))
        self.assertEqual(confirmed.detection_to_physical_id, (1,))

        degraded = counter.update(
            [(400.0, 400.0)], [(400.0, 390.0)], motion(False)
        )
        self.assertEqual(degraded.unique_seen_count, 1)
        self.assertEqual(degraded.held_physical_ids, (1,))
        self.assertEqual(degraded.detection_to_physical_id, (None,))

    def test_low_confidence_detection_rematches_confirmed_id(self) -> None:
        counter = UavPhysicalCounter(
            confirm_window=1,
            confirm_votes=1,
            max_missed_frames=8,
            match_distance_pixels=20.0,
        )
        confirmed = counter.update(
            [(100.0, 100.0)],
            [(100.0, 90.0)],
            motion(),
            can_create_physical_id=[True],
        )
        recovered = counter.update(
            [(103.0, 100.0)],
            [(103.0, 90.0)],
            motion(),
            can_create_physical_id=[False],
        )

        self.assertEqual(confirmed.new_physical_ids, (1,))
        self.assertEqual(recovered.detection_to_physical_id, (1,))
        self.assertEqual(recovered.recovery_detection_indices, (0,))
        self.assertEqual(recovered.unique_seen_count, 1)

    def test_far_high_confidence_candidate_does_not_block_low_confidence_recovery(
        self,
    ) -> None:
        counter = UavPhysicalCounter(
            confirm_window=1,
            confirm_votes=1,
            max_missed_frames=8,
            match_distance_pixels=20.0,
        )
        counter.update(
            [(100.0, 100.0)],
            [(100.0, 90.0)],
            motion(),
            can_create_physical_id=[True],
        )

        recovered = counter.update(
            [(1000.0, 1000.0), (103.0, 100.0)],
            [(1000.0, 990.0), (103.0, 90.0)],
            motion(),
            can_create_physical_id=[True, False],
        )

        self.assertEqual(recovered.detection_to_physical_id, (2, 1))
        self.assertEqual(recovered.recovery_detection_indices, (1,))
        self.assertEqual(recovered.unique_seen_count, 2)

    def test_low_confidence_detection_cannot_create_or_confirm_id(self) -> None:
        counter = UavPhysicalCounter(
            confirm_window=3,
            confirm_votes=2,
            max_missed_frames=8,
            match_distance_pixels=20.0,
        )
        first = counter.update(
            [(100.0, 100.0)],
            [(100.0, 90.0)],
            motion(),
            can_create_physical_id=[False],
        )
        counter.update(
            [(200.0, 200.0)],
            [(200.0, 190.0)],
            motion(),
            can_create_physical_id=[True],
        )
        tentative_recovery = counter.update(
            [(202.0, 200.0)],
            [(202.0, 190.0)],
            motion(),
            can_create_physical_id=[False],
        )

        self.assertEqual(first.detection_to_physical_id, (None,))
        self.assertEqual(first.tracks, ())
        self.assertEqual(tentative_recovery.detection_to_physical_id, (None,))
        self.assertEqual(tentative_recovery.recovery_detection_indices, ())
        self.assertEqual(tentative_recovery.unique_seen_count, 0)


class _FakeTensor:
    def __init__(self, value) -> None:
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _FakeObb:
    def __init__(self) -> None:
        self.xyxyxyxy = _FakeTensor(
            [
                [[90, 80], [110, 80], [110, 120], [90, 120]],
                [[190, 180], [210, 180], [210, 220], [190, 220]],
            ]
        )
        self.conf = _FakeTensor([0.30, 0.20])
        self.cls = _FakeTensor([0, 0])

    def __len__(self) -> int:
        return 2


class _FakeResult:
    obb = _FakeObb()
    names = {0: "cone"}


class _SmallFakeObb:
    xyxyxyxy = _FakeTensor(
        [
            [[90, 80], [102, 80], [102, 92], [90, 92]],
            [[190, 180], [202, 180], [202, 192], [190, 192]],
        ]
    )
    conf = _FakeTensor([0.35, 0.20])
    cls = _FakeTensor([0, 0])

    def __len__(self) -> int:
        return 2


class _SmallFakeResult:
    obb = _SmallFakeObb()
    names = {0: "cone"}


class DetectionThresholdSplitTest(unittest.TestCase):
    def test_low_confidence_cone_is_physical_only(self) -> None:
        (
            anomaly_cones,
            physical_candidates,
            detections,
            person_suppressed,
            small_suppressed,
        ) = _extract_detections(
            _FakeResult(),
            "cone",
            None,
            image_size=(1280, 720),
            primary_confidence=0.25,
        )

        self.assertEqual(len(anomaly_cones), 1)
        self.assertEqual(len(physical_candidates), 2)
        self.assertEqual([cone.confidence for cone in physical_candidates], [0.30, 0.20])
        self.assertEqual(
            [item.get("postprocess_status") for item in detections],
            ["accepted", "physical_recovery_candidate"],
        )
        self.assertEqual(person_suppressed, 0)
        self.assertEqual(small_suppressed, 0)

    def test_high_confidence_too_small_cone_is_physical_only(self) -> None:
        (
            anomaly_cones,
            physical_candidates,
            detections,
            _,
            small_suppressed,
        ) = _extract_detections(
            _SmallFakeResult(),
            "cone",
            None,
            image_size=(1280, 720),
            min_cone_long_side_ratio=16.0 / 720.0,
            primary_confidence=0.25,
            physical_small_cone_conf=0.30,
        )

        self.assertEqual(anomaly_cones, [])
        self.assertEqual(len(physical_candidates), 1)
        self.assertEqual(physical_candidates[0].confidence, 0.35)
        self.assertEqual(small_suppressed, 2)
        self.assertEqual(
            [item.get("postprocess_status") for item in detections],
            ["too_small", "too_small"],
        )
        self.assertTrue(detections[0].get("physical_only"))
        self.assertNotIn("physical_only", detections[1])


class InferParserTest(unittest.TestCase):
    def test_physical_roi_is_default_and_can_be_disabled(self) -> None:
        parser = build_parser()
        default = parser.parse_args(["--weights", "model.pt", "--source", "video.mp4"])
        legacy = parser.parse_args(
            ["--weights", "model.pt", "--source", "video.mp4", "--no-physical-roi"]
        )

        self.assertTrue(default.physical_roi)
        self.assertFalse(legacy.physical_roi)
        self.assertEqual(default.conf, 0.25)
        self.assertEqual(default.physical_match_conf, 0.20)

    def test_standalone_physical_counter_uses_the_same_threshold_defaults(
        self,
    ) -> None:
        args = build_physical_parser().parse_args(
            ["--weights", "model.pt", "--source", "video.mp4", "--output", "out"]
        )

        self.assertEqual(args.conf, 0.25)
        self.assertEqual(args.physical_match_conf, 0.20)


class PhysicalRoiTrackerTest(unittest.TestCase):
    @staticmethod
    def track(
        physical_id: int,
        map_point: tuple[float, float],
        *,
        state: str = "confirmed",
        missed_frames: int = 0,
    ) -> PhysicalTrack:
        return PhysicalTrack(
            physical_id=physical_id,
            map_point=map_point,
            image_point=map_point,
            state=state,
            age=10,
            hits=9,
            missed_frames=missed_frames,
            counted=True,
        )

    @staticmethod
    def bounds(
        tracks: list[PhysicalTrack],
    ) -> dict[int, tuple[float, float, float, float]]:
        return {
            track.physical_id: (
                track.image_point[0] - 5.0,
                track.image_point[1] - 5.0,
                track.image_point[0] + 5.0,
                track.image_point[1] + 5.0,
            )
            for track in tracks
            if track.state == "confirmed"
        }

    @staticmethod
    def confirm(
        tracker: PhysicalRoiTracker,
        tracks: list[PhysicalTrack],
        frames: int = 3,
    ):
        result = None
        for _ in range(frames):
            result = tracker.update(tracks, PhysicalRoiTrackerTest.bounds(tracks), (1280, 720))
        assert result is not None
        return result

    def test_parallel_cone_rows_form_two_rois(self) -> None:
        tracker = PhysicalRoiTracker(
            smoothing=0.25,
            padding=0.0,
            hold_frames=8,
            confirm_frames=3,
        )
        tracks = [
            *[
                self.track(index + 1, (100.0, 100.0 + index * 50.0))
                for index in range(5)
            ],
            *[
                self.track(index + 6, (260.0, 120.0 + index * 50.0))
                for index in range(3)
            ],
        ]

        grouped = self.confirm(tracker, tracks)

        self.assertEqual(len(grouped.rois), 2)
        self.assertEqual(
            {roi.member_ids for roi in grouped.rois},
            {(1, 2, 3, 4, 5), (6, 7, 8)},
        )
        self.assertEqual(len({roi.roi_id for roi in grouped.rois}), 2)

    def test_collinear_batches_with_large_gap_form_two_rois(self) -> None:
        tracker = PhysicalRoiTracker(
            smoothing=0.25,
            padding=0.0,
            hold_frames=8,
            confirm_frames=3,
        )
        tracks = [
            *[
                self.track(index + 1, (100.0, 100.0 + index * 50.0))
                for index in range(3)
            ],
            *[
                self.track(index + 4, (100.0, 400.0 + index * 50.0))
                for index in range(3)
            ],
        ]

        grouped = self.confirm(tracker, tracks)

        self.assertEqual(
            {roi.member_ids for roi in grouped.rois},
            {(1, 2, 3), (4, 5, 6)},
        )

    def test_overlapping_member_cannot_reassociate_a_distant_roi(self) -> None:
        tracker = PhysicalRoiTracker(
            smoothing=0.25,
            padding=0.0,
            hold_frames=0,
            confirm_frames=3,
            identity_memory_frames=3,
        )
        initial = [
            self.track(1, (100.0, 100.0)),
            self.track(2, (100.0, 150.0)),
        ]
        original = self.confirm(tracker, initial)
        original_roi_id = original.rois[0].roi_id
        distant = [
            self.track(2, (100.0, 400.0)),
            self.track(3, (100.0, 450.0)),
        ]

        reassociated = self.confirm(tracker, distant)

        self.assertEqual(len(reassociated.rois), 1)
        self.assertNotEqual(reassociated.rois[0].roi_id, original_roi_id)
        self.assertEqual(reassociated.rois[0].member_ids, (2, 3))

    def test_stale_held_roi_does_not_overlap_a_confirmed_roi(self) -> None:
        tracker = PhysicalRoiTracker(
            smoothing=0.25,
            padding=0.0,
            hold_frames=8,
            confirm_frames=3,
            identity_memory_frames=30,
        )
        initial = [
            self.track(1, (100.0, 100.0)),
            self.track(2, (100.0, 150.0)),
        ]
        original = self.confirm(tracker, initial)
        original_roi_id = original.rois[0].roi_id
        distant = [
            self.track(2, (100.0, 400.0)),
            self.track(3, (100.0, 450.0)),
        ]

        regrouped = self.confirm(tracker, distant)

        self.assertEqual(len(regrouped.rois), 1)
        self.assertEqual(regrouped.rois[0].state, "confirmed")
        self.assertNotEqual(regrouped.rois[0].roi_id, original_roi_id)
        self.assertEqual(regrouped.rois[0].member_ids, (2, 3))

    def test_one_frame_miss_keeps_the_last_observed_roi(self) -> None:
        tracker = PhysicalRoiTracker(
            smoothing=0.25,
            padding=0.0,
            hold_frames=8,
            confirm_frames=3,
        )
        tracks = [
            self.track(1, (100.0, 100.0)),
            self.track(2, (100.0, 150.0)),
            self.track(3, (100.0, 200.0)),
        ]
        observed = self.confirm(tracker, tracks)
        held_tracks = [
            tracks[0],
            self.track(2, (100.0, 150.0), state="held", missed_frames=1),
            tracks[2],
        ]
        held = tracker.update(
            held_tracks,
            self.bounds(held_tracks),
            (1280, 720),
        )

        self.assertEqual(len(held.rois), 1)
        self.assertEqual(held.rois[0].roi_id, observed.rois[0].roi_id)
        self.assertEqual(held.rois[0].bounds, observed.rois[0].bounds)
        self.assertEqual(held.rois[0].held_ids, (2,))

    def test_left_group_gets_a_new_stable_roi_after_confirmation(self) -> None:
        tracker = PhysicalRoiTracker(
            smoothing=0.25,
            padding=0.0,
            hold_frames=8,
            confirm_frames=3,
        )
        main = [
            self.track(index + 1, (100.0, 100.0 + index * 50.0))
            for index in range(5)
        ]
        initial = self.confirm(tracker, main)
        main_roi_id = initial.rois[0].roi_id
        left = [
            self.track(index + 6, (260.0, 120.0 + index * 50.0))
            for index in range(3)
        ]

        first = tracker.update(main + left, self.bounds(main + left), (1280, 720))
        second = tracker.update(main + left, self.bounds(main + left), (1280, 720))
        confirmed = tracker.update(main + left, self.bounds(main + left), (1280, 720))

        self.assertEqual([roi.roi_id for roi in first.rois], [main_roi_id])
        self.assertEqual([roi.roi_id for roi in second.rois], [main_roi_id])
        self.assertEqual(len(confirmed.rois), 2)
        main_roi = next(roi for roi in confirmed.rois if 1 in roi.member_ids)
        left_roi = next(roi for roi in confirmed.rois if 6 in roi.member_ids)
        self.assertEqual(main_roi.roi_id, main_roi_id)
        self.assertNotEqual(left_roi.roi_id, main_roi_id)

    def test_hidden_route_recovers_its_roi_id_after_member_replacement(self) -> None:
        tracker = PhysicalRoiTracker(
            smoothing=0.25,
            padding=0.0,
            hold_frames=8,
            confirm_frames=3,
            identity_memory_frames=30,
        )
        initial_tracks = [
            self.track(1, (260.0, 100.0)),
            self.track(2, (260.0, 150.0)),
        ]
        initial = self.confirm(tracker, initial_tracks)
        roi_id = initial.rois[0].roi_id

        single_track = [self.track(1, (260.0, 100.0))]
        hidden = None
        for _ in range(9):
            hidden = tracker.update(
                single_track, self.bounds(single_track), (1280, 720)
            )
        assert hidden is not None
        self.assertEqual(hidden.rois, ())

        replacement = [
            self.track(1, (260.0, 100.0)),
            self.track(3, (260.0, 155.0)),
        ]
        recovered = None
        for _ in range(3):
            recovered = tracker.update(
                replacement, self.bounds(replacement), (1280, 720)
            )
        assert recovered is not None

        self.assertEqual(len(recovered.rois), 1)
        self.assertEqual(recovered.rois[0].roi_id, roi_id)
        self.assertEqual(recovered.rois[0].member_ids, (1, 3))

    def test_roi_expires_after_hold_window(self) -> None:
        tracker = PhysicalRoiTracker(
            smoothing=0.25,
            padding=0.0,
            hold_frames=8,
            confirm_frames=3,
        )
        tracks = [
            self.track(1, (100.0, 100.0)),
            self.track(2, (100.0, 150.0)),
        ]
        self.confirm(tracker, tracks)
        expired = tracker.update(
            [
                self.track(1, (100.0, 100.0), state="held", missed_frames=9),
                self.track(2, (100.0, 150.0), state="held", missed_frames=9),
            ],
            {},
            (1280, 720),
        )

        self.assertEqual(expired.rois, ())


class GapAlertTransitionTest(unittest.TestCase):
    @staticmethod
    def strong_candidate() -> _MissingCandidate:
        return _MissingCandidate(
            alert=MissingAlert(0, 250.0, 100.0, 1, ((125.0, 100.0),)),
            distance_ratio=2.5,
            pixel_ratio=2.5,
            measurement_conflict=False,
            position_conflict=False,
        )

    def test_strong_missing_must_enter_suspect_before_alert(self) -> None:
        tracker = GapThresholdTracker(
            vote_window=5,
            confirm_votes=3,
            suspect_frames=24,
            alert_confirm_frames=5,
        )
        gap = _StableGapState(1, (1, 2), deque(maxlen=5))
        candidate = self.strong_candidate()

        for _ in range(8):
            tracker._vote(gap, candidate, allowed=True, can_activate=False)
        tracker._vote(gap, candidate, allowed=True, can_activate=True)

        self.assertEqual(gap.level, "suspect")
        self.assertEqual(gap.strong_alert_streak, 0)
        for _ in range(4):
            tracker._vote(gap, candidate, allowed=True, can_activate=True)
            self.assertEqual(gap.level, "suspect")
        tracker._vote(gap, candidate, allowed=True, can_activate=True)
        self.assertEqual(gap.level, "alert")


if __name__ == "__main__":
    unittest.main()
