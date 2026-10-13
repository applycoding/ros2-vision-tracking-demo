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

"""
ByteTrack-style multi-object tracker.

High-score detections are matched first and may start tracks. Low-score
detections only revive tracks that the first stage missed. Association is
IoU plus a constant-velocity Kalman filter. This is an original implementation
of that idea, not a copy of the ByteTrack or SORT reference code.
"""

from __future__ import annotations

from vision_tracking.assignment import associate_boxes
from vision_tracking.geometry import as_box
from vision_tracking.kalman_box import KalmanBox
from vision_tracking.records import Detection
from vision_tracking.records import TrackSnapshot


class _Track:
    """Mutable track kept inside the tracker."""

    def __init__(self, track_id: int, detection: Detection) -> None:
        self.track_id = track_id
        self.score = float(detection.score)
        self.class_id = int(detection.class_id)
        self.class_name = detection.class_name
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self._filter = KalmanBox(detection.bbox)

    def predict(self, dt: float) -> None:
        """Coast the box forward and count the frame."""
        self._filter.predict(dt)
        self.age += 1

    def update(self, detection: Detection) -> None:
        """Correct the box and refresh the label from this detection."""
        self._filter.update(detection.bbox)
        self.score = float(detection.score)
        self.class_id = int(detection.class_id)
        self.class_name = detection.class_name
        self.hits += 1
        self.time_since_update = 0

    def mark_missed(self) -> None:
        """Record a frame with no matching detection."""
        self.time_since_update += 1

    def box(self):
        """Return the current xyxy box."""
        return as_box(self._filter.box())

    def snapshot(self) -> TrackSnapshot:
        """Return a frozen view of the track."""
        return TrackSnapshot(
            track_id=self.track_id,
            bbox=self.box(),
            score=self.score,
            class_id=self.class_id,
            class_name=self.class_name,
            age=self.age,
            hits=self.hits,
            time_since_update=self.time_since_update,
        )


class MultiObjectTracker:
    """Keep stable ids across frames with two-stage IoU association."""

    def __init__(
        self,
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 1,
        coast_frames: int = 1,
    ) -> None:
        if not 0.0 <= low_thresh <= high_thresh <= 1.0:
            raise ValueError('Require 0 <= low_thresh <= high_thresh <= 1')
        if iou_threshold < 0.0 or iou_threshold > 1.0:
            raise ValueError('iou_threshold must be between 0 and 1')
        if max_age < 1 or min_hits < 1 or coast_frames < 0:
            raise ValueError('max_age and min_hits start at 1; coast_frames at 0')
        self._high_thresh = float(high_thresh)
        self._low_thresh = float(low_thresh)
        self._iou_threshold = float(iou_threshold)
        self._max_age = int(max_age)
        self._min_hits = int(min_hits)
        self._coast_frames = int(coast_frames)
        self._tracks: list[_Track] = []
        self._next_id = 1

    def update(
        self,
        detections: list[Detection],
        dt: float,
    ) -> list[TrackSnapshot]:
        """Predict, associate, and return tracks that should be drawn."""
        step = min(max(float(dt), 1.0e-3), 1.0)
        for track in self._tracks:
            track.predict(step)

        high = [item for item in detections if item.score >= self._high_thresh]
        low = [
            item for item in detections
            if self._low_thresh <= item.score < self._high_thresh
        ]

        matches, unmatched, unmatched_high = associate_boxes(
            [track.box() for track in self._tracks],
            [item.bbox for item in high],
            self._iou_threshold,
        )
        for track_index, det_index in matches:
            self._tracks[track_index].update(high[det_index])

        remaining = [self._tracks[index] for index in unmatched]
        low_matches, still_open, _unused = associate_boxes(
            [track.box() for track in remaining],
            [item.bbox for item in low],
            self._iou_threshold,
        )
        recovered = set()
        for track_index, det_index in low_matches:
            remaining[track_index].update(low[det_index])
            recovered.add(track_index)
        for index, track in enumerate(remaining):
            if index not in recovered:
                track.mark_missed()

        for det_index in unmatched_high:
            self._tracks.append(self._start(high[det_index]))

        self._tracks = [
            track for track in self._tracks
            if track.time_since_update <= self._max_age
        ]
        return [track.snapshot() for track in self._tracks if self._visible(track)]

    def _start(self, detection: Detection) -> _Track:
        track = _Track(self._next_id, detection)
        self._next_id += 1
        return track

    def _visible(self, track: _Track) -> bool:
        if track.hits < self._min_hits:
            return False
        return track.time_since_update <= self._coast_frames
