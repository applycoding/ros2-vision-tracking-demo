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

"""Tests for detector construction. Model inference is optional."""

from __future__ import annotations

import os

import numpy as np
import pytest

from vision_tracking.detectors import create_detector


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        create_detector('motion', model_path='unused.weights')


def test_missing_darknet_files_raise(tmp_path):
    with pytest.raises(RuntimeError):
        create_detector(
            'opencv_dnn',
            model_path=str(tmp_path / 'missing.weights'),
            config_path=str(tmp_path / 'missing.cfg'),
        )


def test_cuda_request_is_rejected_on_opencv_backend(tmp_path):
    weights = tmp_path / 'model.weights'
    config = tmp_path / 'model.cfg'
    weights.write_bytes(b'not a real model')
    config.write_text('not a real config', encoding='utf-8')
    with pytest.raises(RuntimeError):
        create_detector(
            'opencv_dnn',
            model_path=str(weights),
            config_path=str(config),
            device='cuda',
        )


def test_optional_yolov4_tiny_on_blank_image():
    weights = os.environ.get('VISION_TRACKING_WEIGHTS', '')
    config = os.environ.get('VISION_TRACKING_CFG', '')
    if not weights or not config or not os.path.isfile(weights):
        pytest.skip('Set VISION_TRACKING_WEIGHTS and VISION_TRACKING_CFG to run')
    detector = create_detector(
        'opencv_dnn',
        model_path=weights,
        config_path=config,
        confidence=0.35,
        input_size=416,
    )
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    assert detector.detect(frame) == []
