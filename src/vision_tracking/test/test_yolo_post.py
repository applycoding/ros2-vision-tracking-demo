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

"""Tests for YOLO decoding without a model file."""

from __future__ import annotations

import numpy as np

from vision_tracking.coco import COCO_CLASSES
from vision_tracking.detectors.yolo_post import detections_from_network
from vision_tracking.detectors.yolo_post import letterbox


def _darknet_row(cx, cy, width, height, score, class_id):
    row = np.zeros(85, dtype=np.float64)
    row[:4] = (cx, cy, width, height)
    row[4] = 1.0
    row[5 + class_id] = score
    return row


def test_coco_class_list_has_eighty_names():
    assert len(COCO_CLASSES) == 80
    assert COCO_CLASSES[0] == 'person'
    assert COCO_CLASSES[2] == 'car'


def test_letterbox_preserves_aspect():
    image = np.zeros((50, 100, 3), dtype=np.uint8)
    canvas, scale, pad_x, pad_y = letterbox(image, 80)
    assert canvas.shape == (80, 80, 3)
    assert abs(scale - 0.8) < 1.0e-9
    assert pad_x == 0
    assert pad_y == 20


def test_darknet_row_decodes_to_image_pixels():
    raw = [_darknet_row(0.5, 0.5, 0.2, 0.4, 0.9, 0).reshape(1, 85)]
    detections = detections_from_network(
        raw,
        (100, 100, 3),
        input_size=100,
        scale=1.0,
        pad_x=0,
        pad_y=0,
        confidence=0.25,
        iou_threshold=0.45,
    )
    assert len(detections) == 1
    assert detections[0].class_name == 'person'
    x1, y1, x2, y2 = detections[0].bbox
    assert abs(x1 - 40.0) < 1.0e-6
    assert abs(y1 - 30.0) < 1.0e-6
    assert abs(x2 - 60.0) < 1.0e-6
    assert abs(y2 - 70.0) < 1.0e-6


def test_yolov8_tensor_and_nms():
    prediction = np.zeros((1, 84, 100), dtype=np.float32)
    prediction[0, 0, 0] = 50.0
    prediction[0, 1, 0] = 50.0
    prediction[0, 2, 0] = 20.0
    prediction[0, 3, 0] = 40.0
    prediction[0, 4, 0] = 0.92
    # A second, lower-score box that covers the same object.
    prediction[0, 0, 1] = 51.0
    prediction[0, 1, 1] = 50.0
    prediction[0, 2, 1] = 20.0
    prediction[0, 3, 1] = 40.0
    prediction[0, 4, 1] = 0.4
    detections = detections_from_network(
        prediction,
        (100, 100, 3),
        input_size=100,
        scale=1.0,
        pad_x=0,
        pad_y=0,
        confidence=0.25,
        iou_threshold=0.45,
    )
    assert len(detections) == 1
    assert detections[0].score == np.float32(0.92) or abs(detections[0].score - 0.92) < 1e-5
    assert detections[0].class_name == 'person'


def test_allowed_classes_filter():
    person = _darknet_row(0.2, 0.2, 0.1, 0.1, 0.95, 0)
    car = _darknet_row(0.7, 0.7, 0.2, 0.2, 0.88, 2)
    raw = [np.vstack([person, car])]
    detections = detections_from_network(
        raw,
        (100, 100, 3),
        input_size=100,
        scale=1.0,
        pad_x=0,
        pad_y=0,
        confidence=0.25,
        iou_threshold=0.45,
        allowed={'car'},
    )
    assert [item.class_name for item in detections] == ['car']
