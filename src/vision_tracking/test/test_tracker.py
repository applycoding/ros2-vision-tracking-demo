# Copyright 2026 Arpan Bom
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the two-stage IoU tracker."""

from __future__ import annotations

import pytest

from vision_tracking.records import Detection
from vision_tracking.tracker import MultiObjectTracker


def _det(box, score=0.9, class_id=0, class_name='person'):
    return Detection(bbox=box, score=score, class_id=class_id, class_name=class_name)


def test_parallel_motion_keeps_ids():
    tracker = MultiObjectTracker(min_hits=1, coast_frames=0)
    first = tracker.update([
        _det((0.0, 0.0, 10.0, 10.0)),
        _det((100.0, 0.0, 110.0, 10.0), class_name='car', class_id=2),
    ], 0.1)
    second = tracker.update([
        _det((2.0, 0.0, 12.0, 10.0)),
        _det((102.0, 0.0, 112.0, 10.0), class_name='car', class_id=2),
    ], 0.1)
    assert [track.track_id for track in first] == [1, 2]
    assert [track.track_id for track in second] == [1, 2]
    assert second[0].class_name == 'person'
    assert second[1].class_name == 'car'


def test_low_score_does_not_start_a_track():
    tracker = MultiObjectTracker(high_thresh=0.5, low_thresh=0.1)
    assert tracker.update([_det((0.0, 0.0, 10.0, 10.0), score=0.2)], 0.1) == []


def test_low_score_can_keep_an_existing_id():
    tracker = MultiObjectTracker(high_thresh=0.5, low_thresh=0.1, coast_frames=0)
    first = tracker.update([_det((0.0, 0.0, 20.0, 20.0), score=0.9)], 0.1)
    second = tracker.update([_det((1.0, 1.0, 21.0, 21.0), score=0.2)], 0.1)
    assert first[0].track_id == second[0].track_id
    assert second[0].score == pytest.approx(0.2)


def test_gap_shorter_than_max_age_keeps_id():
    tracker = MultiObjectTracker(max_age=5, coast_frames=0, min_hits=1)
    born = tracker.update([_det((0.0, 0.0, 30.0, 30.0))], 0.1)
    for _ in range(3):
        assert tracker.update([], 0.1) == []
    back = tracker.update([_det((0.0, 0.0, 30.0, 30.0))], 0.1)
    assert back[0].track_id == born[0].track_id


def test_track_expires_after_max_age():
    tracker = MultiObjectTracker(max_age=2, coast_frames=0, min_hits=1)
    tracker.update([_det((0.0, 0.0, 10.0, 10.0))], 0.1)
    tracker.update([], 0.1)
    tracker.update([], 0.1)
    tracker.update([], 0.1)
    restarted = tracker.update([_det((0.0, 0.0, 10.0, 10.0))], 0.1)
    assert restarted[0].track_id == 2


def test_min_hits_hides_young_tracks():
    tracker = MultiObjectTracker(min_hits=3, coast_frames=0)
    box = (0.0, 0.0, 12.0, 12.0)
    assert tracker.update([_det(box)], 0.1) == []
    assert tracker.update([_det((1.0, 0.0, 13.0, 12.0))], 0.1) == []
    visible = tracker.update([_det((2.0, 0.0, 14.0, 12.0))], 0.1)
    assert len(visible) == 1
    assert visible[0].hits >= 3


def test_invalid_thresholds_raise():
    with pytest.raises(ValueError):
        MultiObjectTracker(high_thresh=0.2, low_thresh=0.4)
