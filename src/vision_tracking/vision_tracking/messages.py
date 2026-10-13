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

"""Convert detections and tracks to vision_msgs/Detection2DArray."""

from __future__ import annotations

from vision_msgs.msg import Detection2D
from vision_msgs.msg import Detection2DArray
from vision_msgs.msg import ObjectHypothesisWithPose

from vision_tracking.coco import class_id_for_name
from vision_tracking.geometry import as_box
from vision_tracking.geometry import xyxy_to_cxcywh
from vision_tracking.records import Detection
from vision_tracking.records import TrackSnapshot


def detections_to_msg(detections: list[Detection], header) -> Detection2DArray:
    """Build a detection array. Track ids are left empty."""
    message = Detection2DArray()
    message.header = header
    for detection in detections:
        message.detections.append(_box_message(
            detection.bbox,
            detection.class_name,
            detection.score,
            header,
            '',
        ))
    return message


def tracks_to_msg(tracks: list[TrackSnapshot], header) -> Detection2DArray:
    """Build a detection array whose ``id`` field is the track id."""
    message = Detection2DArray()
    message.header = header
    for track in tracks:
        message.detections.append(_box_message(
            track.bbox,
            track.class_name,
            track.score,
            header,
            str(track.track_id),
        ))
    return message


def msg_to_detections(message: Detection2DArray) -> list[Detection]:
    """Read detector output back into Detection records."""
    detections = []
    for item in message.detections:
        box, name, score = _read_box(item)
        detections.append(Detection(
            bbox=box,
            score=score,
            class_id=class_id_for_name(name),
            class_name=name,
        ))
    return detections


def msg_to_tracks(message: Detection2DArray) -> list[TrackSnapshot]:
    """Read tracker output. Entries without a numeric id are skipped."""
    tracks = []
    for item in message.detections:
        if not item.id:
            continue
        try:
            track_id = int(item.id)
        except ValueError:
            continue
        box, name, score = _read_box(item)
        tracks.append(TrackSnapshot(
            track_id=track_id,
            bbox=box,
            score=score,
            class_id=class_id_for_name(name),
            class_name=name,
            age=0,
            hits=0,
            time_since_update=0,
        ))
    return tracks


def _box_message(box, class_name: str, score: float, header, track_id: str) -> Detection2D:
    cx, cy, width, height = xyxy_to_cxcywh(as_box(box))
    detection = Detection2D()
    detection.header = header
    detection.bbox.center.position.x = float(cx)
    detection.bbox.center.position.y = float(cy)
    detection.bbox.center.theta = 0.0
    detection.bbox.size_x = float(width)
    detection.bbox.size_y = float(height)
    hypothesis = ObjectHypothesisWithPose()
    hypothesis.hypothesis.class_id = class_name
    hypothesis.hypothesis.score = float(score)
    detection.results.append(hypothesis)
    detection.id = track_id
    return detection


def _read_box(detection: Detection2D):
    cx = float(detection.bbox.center.position.x)
    cy = float(detection.bbox.center.position.y)
    width = float(detection.bbox.size_x)
    height = float(detection.bbox.size_y)
    box = (
        cx - width * 0.5,
        cy - height * 0.5,
        cx + width * 0.5,
        cy + height * 0.5,
    )
    if detection.results:
        name = str(detection.results[0].hypothesis.class_id)
        score = float(detection.results[0].hypothesis.score)
    else:
        name = 'object'
        score = 0.0
    return as_box(box), name, score
