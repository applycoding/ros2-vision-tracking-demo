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

"""Tests for box geometry."""

from __future__ import annotations

from vision_tracking.geometry import clip_box
from vision_tracking.geometry import cxcywh_to_xyxy
from vision_tracking.geometry import iou
from vision_tracking.geometry import xyxy_to_cxcywh


def test_iou_identical_and_disjoint():
    box = (0.0, 0.0, 10.0, 10.0)
    assert iou(box, box) == 1.0
    assert iou(box, (20.0, 20.0, 30.0, 30.0)) == 0.0


def test_iou_partial_overlap():
    # Intersection is 5x10, union is 100+100-50.
    overlap = iou((0.0, 0.0, 10.0, 10.0), (5.0, 0.0, 15.0, 10.0))
    assert abs(overlap - (50.0 / 150.0)) < 1.0e-9


def test_iou_zero_area_is_zero():
    assert iou((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 10.0, 10.0)) == 0.0


def test_center_conversion_roundtrip():
    box = (2.0, 4.0, 12.0, 14.0)
    converted = cxcywh_to_xyxy(xyxy_to_cxcywh(box))
    assert converted == box


def test_clip_box_orders_and_limits():
    assert clip_box((15.0, -4.0, 8.0, 3.0), 10, 6) == (8.0, 0.0, 9.0, 3.0)
