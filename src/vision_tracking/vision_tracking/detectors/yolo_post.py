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

"""Letterbox, decode, and suppress YOLO network outputs."""

from __future__ import annotations

import cv2
import numpy as np

from vision_tracking.coco import COCO_CLASSES
from vision_tracking.geometry import clip_box
from vision_tracking.geometry import iou
from vision_tracking.records import Detection


def letterbox(image, new_size: int):
    """
    Fit ``image`` into a square canvas without stretching it.

    The canvas is filled with gray (114), matching common YOLO preprocessing.
    Returns the canvas, the resize scale, and the left and top padding.
    """
    height, width = image.shape[:2]
    scale = min(new_size / float(height), new_size / float(width))
    resized_w = max(int(round(width * scale)), 1)
    resized_h = max(int(round(height * scale)), 1)
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    pad_x = (new_size - resized_w) // 2
    pad_y = (new_size - resized_h) // 2
    canvas = np.full((new_size, new_size, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = resized
    return canvas, scale, pad_x, pad_y


def detections_from_network(
    raw,
    image_shape,
    input_size: int,
    scale: float,
    pad_x: int,
    pad_y: int,
    confidence: float,
    iou_threshold: float,
    class_names: tuple[str, ...] = COCO_CLASSES,
    allowed: set[str] | None = None,
) -> list[Detection]:
    """Turn an OpenCV DNN forward() result into detections in image pixels."""
    rows, layout = _rows_from_raw(raw)
    if rows.size == 0:
        return []
    height, width = image_shape[:2]
    allowed_names = {name.lower() for name in (allowed or set())}
    boxes = []
    scores = []
    class_ids = []
    labels = []
    for row in rows:
        parsed = _parse_row(row, input_size, confidence, layout)
        if parsed is None:
            continue
        x1, y1, x2, y2, score, class_id = parsed
        name = class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id)
        if allowed_names and name.lower() not in allowed_names:
            continue
        box = _unmap_box(
            (x1, y1, x2, y2), scale, pad_x, pad_y, width, height,
        )
        if box[2] - box[0] < 1.0 or box[3] - box[1] < 1.0:
            continue
        boxes.append(box)
        scores.append(score)
        class_ids.append(class_id)
        labels.append(name)

    if not boxes:
        return []
    if len(scores) > 300:
        chosen = np.argpartition(-np.asarray(scores), 300)[:300]
        boxes = [boxes[int(index)] for index in chosen]
        scores = [scores[int(index)] for index in chosen]
        class_ids = [class_ids[int(index)] for index in chosen]
        labels = [labels[int(index)] for index in chosen]

    keep = _nms_indices(boxes, scores, iou_threshold)
    detections = []
    for index in keep:
        detections.append(Detection(
            bbox=boxes[index],
            score=float(scores[index]),
            class_id=int(class_ids[index]),
            class_name=labels[index],
        ))
    detections.sort(key=lambda item: item.score, reverse=True)
    return detections


def _rows_from_raw(raw) -> tuple[np.ndarray, str]:
    """Return prediction rows and a layout name, ``darknet`` or ``yolov8``."""
    if isinstance(raw, (list, tuple)):
        parts = []
        for item in raw:
            array = np.asarray(item)
            if array.size == 0:
                continue
            parts.append(array.reshape(-1, array.shape[-1]))
        if not parts:
            return np.zeros((0, 6), dtype=np.float64), 'darknet'
        return np.vstack(parts), 'darknet'

    array = np.asarray(raw)
    if array.size == 0:
        return np.zeros((0, 6), dtype=np.float64), 'darknet'
    layout = 'darknet'
    # YOLOv8 is (batch, 4 + classes, anchors) or (4 + classes, anchors).
    if array.ndim == 3 and 6 <= array.shape[1] < array.shape[2] <= 200000:
        if array.shape[1] <= 200:
            array = np.transpose(array, (0, 2, 1))
            layout = 'yolov8'
    elif array.ndim == 2 and 6 <= array.shape[0] < array.shape[1] and array.shape[0] <= 200:
        array = array.T
        layout = 'yolov8'
    return array.reshape(-1, array.shape[-1]), layout


def _parse_row(row, input_size: int, confidence: float, layout: str):
    values = np.asarray(row, dtype=np.float64).reshape(-1)
    if values.size < 6:
        return None
    cx, cy, width, height = [float(values[index]) for index in range(4)]
    if layout == 'yolov8':
        objectness = 1.0
        class_scores = values[4:]
    else:
        objectness = float(values[4])
        class_scores = values[5:]
    if class_scores.size == 0:
        return None
    class_id = int(np.argmax(class_scores))
    score = float(objectness * class_scores[class_id])
    if score < confidence:
        return None
    x1, y1, x2, y2 = _xywh_to_input_xyxy(cx, cy, width, height, input_size)
    return x1, y1, x2, y2, score, class_id


def _xywh_to_input_xyxy(cx, cy, width, height, input_size: int):
    """Accept either normalized or pixel boxes on the network canvas."""
    if max(abs(cx), abs(cy), abs(width), abs(height)) <= 1.5:
        cx *= input_size
        cy *= input_size
        width *= input_size
        height *= input_size
    return (
        cx - width * 0.5,
        cy - height * 0.5,
        cx + width * 0.5,
        cy + height * 0.5,
    )


def _unmap_box(box, scale, pad_x, pad_y, width, height):
    x1, y1, x2, y2 = box
    if scale <= 0.0:
        scale = 1.0
    mapped = (
        (x1 - pad_x) / scale,
        (y1 - pad_y) / scale,
        (x2 - pad_x) / scale,
        (y2 - pad_y) / scale,
    )
    return clip_box(mapped, width, height)


def _nms_indices(boxes, scores, iou_threshold: float) -> list[int]:
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    keep = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        overlaps = np.array([
            iou(boxes[current], boxes[int(index)]) for index in rest
        ])
        order = rest[overlaps < iou_threshold]
    return keep
