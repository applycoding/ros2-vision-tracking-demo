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

"""Box geometry used by the detector and the tracker."""

from __future__ import annotations

Box = tuple[float, float, float, float]


def as_box(values) -> Box:
    """Return an xyxy box as plain floats."""
    x1, y1, x2, y2 = values
    return (float(x1), float(y1), float(x2), float(y2))


def xyxy_to_cxcywh(box: Box) -> Box:
    """Convert an xyxy box to center, width, and height."""
    x1, y1, x2, y2 = box
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5, width, height)


def cxcywh_to_xyxy(box: Box) -> Box:
    """Convert a center box to xyxy, keeping a minimum size of one pixel."""
    cx, cy, width, height = box
    width = max(float(width), 1.0)
    height = max(float(height), 1.0)
    return (cx - width * 0.5, cy - height * 0.5, cx + width * 0.5, cy + height * 0.5)


def clip_box(box: Box, width: int, height: int) -> Box:
    """Clamp an xyxy box to the image."""
    x1, y1, x2, y2 = box
    x1 = min(max(x1, 0.0), max(width - 1, 0))
    y1 = min(max(y1, 0.0), max(height - 1, 0))
    x2 = min(max(x2, 0.0), max(width - 1, 0))
    y2 = min(max(y2, 0.0), max(height - 1, 0))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def iou(first: Box, second: Box) -> float:
    """Return the intersection-over-union of two xyxy boxes."""
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union
