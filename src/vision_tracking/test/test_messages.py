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

"""Round-trip tests for vision_msgs conversion. Skipped without ROS."""

from __future__ import annotations

import pytest

from vision_tracking.records import Detection
from vision_tracking.records import TrackSnapshot

try:
    from std_msgs.msg import Header

    from vision_tracking.messages import detections_to_msg
    from vision_tracking.messages import msg_to_detections
    from vision_tracking.messages import msg_to_tracks
    from vision_tracking.messages import tracks_to_msg
except ImportError as exc:
    if 'vision_msgs' not in str(exc) and 'std_msgs' not in str(exc):
        raise
    Header = None
    detections_to_msg = None
    msg_to_detections = None
    msg_to_tracks = None
    tracks_to_msg = None


def test_detection_and_track_messages_roundtrip():
    if Header is None:
        pytest.skip('vision_msgs is not installed')
    header = Header()
    header.frame_id = 'camera'
    header.stamp.sec = 3
    detection = Detection(
        bbox=(10.0, 20.0, 30.0, 50.0),
        score=0.75,
        class_id=0,
        class_name='person',
    )
    restored = msg_to_detections(detections_to_msg([detection], header))
    assert len(restored) == 1
    assert restored[0].class_name == 'person'
    assert restored[0].score == pytest.approx(0.75)
    assert restored[0].bbox == pytest.approx((10.0, 20.0, 30.0, 50.0))

    track = TrackSnapshot(
        track_id=4,
        bbox=(10.0, 20.0, 30.0, 50.0),
        score=0.75,
        class_id=0,
        class_name='person',
        age=2,
        hits=2,
        time_since_update=0,
    )
    tracks = msg_to_tracks(tracks_to_msg([track], header))
    assert [item.track_id for item in tracks] == [4]
    assert tracks[0].class_name == 'person'
