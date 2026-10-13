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
YOLO via the optional ultralytics package.

The import of ultralytics is intentional and local to this module so the rest
of the demo still runs when that package is not installed.
"""

from __future__ import annotations

from ultralytics import YOLO

from vision_tracking.coco import COCO_CLASSES
from vision_tracking.geometry import clip_box
from vision_tracking.records import Detection


class UltralyticsDetector:
    """Run an Ultralytics YOLO checkpoint and return image-space boxes."""

    def __init__(
        self,
        model_path: str,
        config_path: str = '',
        confidence: float = 0.35,
        iou: float = 0.45,
        input_size: int = 640,
        device: str = 'cpu',
        allowed_classes: str = '',
        class_names: tuple[str, ...] = COCO_CLASSES,
    ) -> None:
        del config_path
        if input_size < 32:
            raise ValueError('input_size must be at least 32')
        self._confidence = float(confidence)
        self._iou = float(iou)
        self._input_size = int(input_size)
        self._device = device or 'cpu'
        self._fallback_names = class_names
        self._allowed = {
            part.strip().lower()
            for part in allowed_classes.split(',')
            if part.strip()
        }
        self._model = YOLO(model_path)

    def detect(self, image) -> list[Detection]:
        """Detect objects in a BGR image."""
        results = self._model.predict(
            source=image,
            conf=self._confidence,
            iou=self._iou,
            imgsz=self._input_size,
            device=self._device,
            verbose=False,
        )
        if not results:
            return []
        height, width = image.shape[:2]
        names = getattr(self._model, 'names', None) or {}
        detections = []
        boxes = results[0].boxes
        if boxes is None:
            return []
        for box in boxes:
            xyxy = [float(value) for value in box.xyxy[0].tolist()]
            score = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = _class_name(names, class_id, self._fallback_names)
            if self._allowed and class_name.lower() not in self._allowed:
                continue
            detections.append(Detection(
                bbox=clip_box(tuple(xyxy), width, height),
                score=score,
                class_id=class_id,
                class_name=class_name,
            ))
        detections.sort(key=lambda item: item.score, reverse=True)
        return detections


def _class_name(names, class_id: int, fallback: tuple[str, ...]) -> str:
    if isinstance(names, dict) and class_id in names:
        return str(names[class_id])
    if 0 <= class_id < len(fallback):
        return fallback[class_id]
    return str(class_id)
