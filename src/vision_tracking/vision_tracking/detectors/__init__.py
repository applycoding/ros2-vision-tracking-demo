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

"""Detector backends. Ultralytics is optional; OpenCV DNN is the default."""

from __future__ import annotations

from typing import Literal
from typing import NoReturn

from vision_tracking.detectors.opencv_dnn import OpenCvDnnDetector

try:
    from vision_tracking.detectors.ultralytics_detector import UltralyticsDetector
except ImportError as exc:
    if 'ultralytics' not in str(exc):
        raise
    UltralyticsDetector = None

BackendName = Literal['opencv_dnn', 'ultralytics']


def create_detector(backend: str, **options):
    """
    Build the detector named by ``backend``.

    ``opencv_dnn`` loads a Darknet or ONNX YOLO model.
    ``ultralytics`` loads a ``.pt`` checkpoint and needs that package installed.
    """
    selected = _require_backend(backend)
    if selected == 'opencv_dnn':
        return OpenCvDnnDetector(**options)
    if selected == 'ultralytics':
        if UltralyticsDetector is None:
            raise RuntimeError(
                "Detector backend 'ultralytics' needs the ultralytics package. "
                'Install it with: pip install ultralytics'
            )
        return UltralyticsDetector(**options)
    return _never(selected)


def _require_backend(backend: str) -> BackendName:
    if backend == 'opencv_dnn' or backend == 'ultralytics':
        return backend
    raise ValueError(
        f"Unknown detector backend '{backend}'. Use 'opencv_dnn' or 'ultralytics'."
    )


def _never(value: NoReturn) -> NoReturn:
    """Fail when a backend variant has no branch."""
    raise RuntimeError(f'Unhandled detector backend: {value!r}')
