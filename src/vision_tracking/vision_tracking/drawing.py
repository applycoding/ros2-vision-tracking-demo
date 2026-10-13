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

"""Draw track ids onto a BGR image."""

from __future__ import annotations

import cv2
import numpy as np

from vision_tracking.records import TrackSnapshot


def color_for_id(track_id: int) -> tuple[int, int, int]:
    """Return a stable BGR color for a track id."""
    hue = int((track_id * 47) % 180)
    pixel = np.uint8([[[hue, 220, 255]]])
    blue, green, red = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0]
    return int(blue), int(green), int(red)


def draw_tracks(image, tracks: list[TrackSnapshot]):
    """Return a copy of ``image`` with one labeled box per track."""
    canvas = image.copy()
    height, width = canvas.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for track in tracks:
        x1, y1, x2, y2 = [int(round(value)) for value in track.bbox]
        x1 = min(max(x1, 0), width - 1)
        y1 = min(max(y1, 0), height - 1)
        x2 = min(max(x2, 0), width - 1)
        y2 = min(max(y2, 0), height - 1)
        color = color_for_id(track.track_id)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f'{track.track_id} {track.class_name} {track.score:.2f}'
        scale = 0.5
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)
        top = max(y1 - text_h - baseline - 4, 0)
        cv2.rectangle(
            canvas,
            (x1, top),
            (min(x1 + text_w + 6, width - 1), min(top + text_h + baseline + 4, height - 1)),
            color,
            -1,
        )
        cv2.putText(
            canvas,
            label,
            (x1 + 2, min(top + text_h + 1, height - 2)),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return canvas
