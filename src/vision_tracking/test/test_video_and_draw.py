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

"""Tests for the file reader and the overlay."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from vision_tracking.drawing import draw_tracks
from vision_tracking.records import TrackSnapshot
from vision_tracking.video_source import VideoSource


def test_video_source_loops(tmp_path):
    path = tmp_path / 'clip.avi'
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*'MJPG'),
        10.0,
        (32, 24),
    )
    if not writer.isOpened():
        pytest.skip('OpenCV could not open an MJPG writer')
    first = np.full((24, 32, 3), 10, dtype=np.uint8)
    second = np.full((24, 32, 3), 200, dtype=np.uint8)
    writer.write(first)
    writer.write(second)
    writer.release()

    source = VideoSource(str(path), loop=True)
    seen = [source.read(), source.read(), source.read()]
    source.close()
    assert seen[0] is not None and seen[1] is not None and seen[2] is not None
    assert abs(int(seen[0][0, 0, 0]) - int(seen[2][0, 0, 0])) <= 3


def test_missing_video_raises(tmp_path):
    with pytest.raises(RuntimeError):
        VideoSource(str(tmp_path / 'missing.mp4'), loop=False)


def test_draw_tracks_changes_pixels():
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    track = TrackSnapshot(
        track_id=3,
        bbox=(5.0, 6.0, 30.0, 40.0),
        score=0.8,
        class_id=0,
        class_name='person',
        age=2,
        hits=2,
        time_since_update=0,
    )
    annotated = draw_tracks(image, [track])
    assert annotated.shape == image.shape
    assert int(annotated.sum()) > 0
    assert int(image.sum()) == 0
