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
YOLO via OpenCV DNN.

Darknet YOLOv4-tiny (``.cfg`` + ``.weights``) is the portable default: it runs
on the OpenCV build shipped with ROS 2 Humble. ONNX exports are accepted when
that same OpenCV build can import them.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from vision_tracking.coco import COCO_CLASSES
from vision_tracking.detectors.yolo_post import detections_from_network
from vision_tracking.detectors.yolo_post import letterbox
from vision_tracking.records import Detection


class OpenCvDnnDetector:
    """Run a YOLO network with ``cv2.dnn`` and return image-space boxes."""

    def __init__(
        self,
        model_path: str,
        config_path: str = '',
        confidence: float = 0.35,
        iou: float = 0.45,
        input_size: int = 416,
        device: str = 'cpu',
        allowed_classes: str = '',
        class_names: tuple[str, ...] = COCO_CLASSES,
    ) -> None:
        if device not in ('cpu', '', 'CPU'):
            raise RuntimeError(
                'The OpenCV DNN backend in this demo runs on CPU. '
                'For CUDA, launch with backend:=ultralytics and device:=0.'
            )
        if input_size < 32:
            raise ValueError('input_size must be at least 32')
        if not os.path.isfile(model_path):
            raise RuntimeError(
                f'Model file not found: {model_path}. '
                'Run scripts/fetch_assets.sh or pass model_path to a local file.'
            )
        self._input_size = int(input_size)
        self._confidence = float(confidence)
        self._iou = float(iou)
        self._class_names = class_names
        self._allowed = _parse_allowed(allowed_classes)
        self._net = _load_net(model_path, config_path)
        self._output_names = _output_layer_names(self._net)

    def detect(self, image) -> list[Detection]:
        """Detect objects in a BGR image."""
        canvas, scale, pad_x, pad_y = letterbox(image, self._input_size)
        blob = cv2.dnn.blobFromImage(
            canvas,
            scalefactor=1.0 / 255.0,
            size=(self._input_size, self._input_size),
            swapRB=True,
            crop=False,
        )
        self._net.setInput(blob)
        if self._output_names:
            raw = self._net.forward(self._output_names)
        else:
            raw = self._net.forward()
        return detections_from_network(
            raw,
            image.shape,
            self._input_size,
            scale,
            pad_x,
            pad_y,
            self._confidence,
            self._iou,
            self._class_names,
            self._allowed,
        )


def _load_net(model_path: str, config_path: str):
    lower = model_path.lower()
    if lower.endswith('.weights'):
        if not config_path or not os.path.isfile(config_path):
            raise RuntimeError(
                'Darknet weights need config_path pointing at the matching .cfg file.'
            )
        return cv2.dnn.readNetFromDarknet(config_path, model_path)
    if lower.endswith('.onnx'):
        return cv2.dnn.readNetFromONNX(model_path)
    raise RuntimeError(
        'OpenCV DNN backend expects a .weights (Darknet) or .onnx model. '
        f'Got {model_path!r}.'
    )


def _output_layer_names(net) -> list[str]:
    layer_names = net.getLayerNames()
    if not layer_names:
        return []
    raw = np.array(net.getUnconnectedOutLayers()).flatten().tolist()
    if not raw:
        return []
    # Humble-era OpenCV reports 1-based layer indexes.
    if max(raw) == len(layer_names):
        return [layer_names[int(index) - 1] for index in raw]
    return [layer_names[int(index)] for index in raw]


def _parse_allowed(allowed_classes: str) -> set[str]:
    if not allowed_classes.strip():
        return set()
    return {part.strip().lower() for part in allowed_classes.split(',') if part.strip()}
