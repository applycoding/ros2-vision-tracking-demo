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

"""Constant-velocity Kalman filter on an image-plane box."""

from __future__ import annotations

import numpy as np

from vision_tracking.geometry import Box
from vision_tracking.geometry import cxcywh_to_xyxy
from vision_tracking.geometry import xyxy_to_cxcywh


class KalmanBox:
    """Filter center, size, and their velocities. The measurement is cx, cy, w, h."""

    def __init__(self, box: Box) -> None:
        cx, cy, width, height = xyxy_to_cxcywh(box)
        self._state = np.array(
            [cx, cy, width, height, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        self._covariance = np.eye(8, dtype=np.float64)
        self._covariance[:4, :4] *= 10.0
        self._covariance[4:, 4:] *= 100.0
        self._measurement_noise = np.eye(4, dtype=np.float64)
        self._measurement = np.zeros((4, 8), dtype=np.float64)
        self._measurement[0, 0] = 1.0
        self._measurement[1, 1] = 1.0
        self._measurement[2, 2] = 1.0
        self._measurement[3, 3] = 1.0

    def predict(self, dt: float) -> None:
        """Advance the box by ``dt`` seconds."""
        transition = np.eye(8, dtype=np.float64)
        transition[0, 4] = dt
        transition[1, 5] = dt
        transition[2, 6] = dt
        transition[3, 7] = dt
        process = np.diag(
            [1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1],
        ).astype(np.float64)
        self._state = transition @ self._state
        self._covariance = transition @ self._covariance @ transition.T + process
        self._state[2] = max(float(self._state[2]), 1.0)
        self._state[3] = max(float(self._state[3]), 1.0)

    def update(self, box: Box) -> None:
        """Correct the filter with an xyxy measurement."""
        cx, cy, width, height = xyxy_to_cxcywh(box)
        residual = np.array([cx, cy, width, height], dtype=np.float64)
        residual = residual - self._measurement @ self._state
        innovation = (
            self._measurement @ self._covariance @ self._measurement.T
            + self._measurement_noise
        )
        gain = self._covariance @ self._measurement.T @ np.linalg.inv(innovation)
        self._state = self._state + gain @ residual
        identity = np.eye(8, dtype=np.float64)
        self._covariance = (identity - gain @ self._measurement) @ self._covariance
        self._state[2] = max(float(self._state[2]), 1.0)
        self._state[3] = max(float(self._state[3]), 1.0)

    def box(self) -> Box:
        """Return the filtered xyxy box."""
        return cxcywh_to_xyxy(self._state[:4])
