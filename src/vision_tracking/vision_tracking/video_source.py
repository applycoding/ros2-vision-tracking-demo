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

"""Read frames from a video file or a V4L2 camera."""

from __future__ import annotations

import cv2


def _is_live(path: str) -> bool:
    text = str(path)
    return text.isdigit() or text.startswith('/dev/video')


class VideoSource:
    """OpenCV capture that can loop a file and leave a camera alone."""

    def __init__(self, path: str, loop: bool = True) -> None:
        self.path = str(path)
        self._live = _is_live(self.path)
        if self._live and self.path.isdigit():
            target = int(self.path)
        else:
            target = self.path
        self._capture = cv2.VideoCapture(target)
        if not self._capture.isOpened():
            raise RuntimeError(f'Could not open video source {self.path!r}.')
        self._loop = bool(loop) and not self._live
        measured = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps = measured if measured > 1.0e-3 else 30.0

    def read(self):
        """Return the next BGR frame, or None when a non-looping source ends."""
        ok, frame = self._capture.read()
        if ok:
            return frame
        if not self._loop:
            return None
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = self._capture.read()
        if not ok:
            return None
        return frame

    def close(self) -> None:
        """Release the capture."""
        self._capture.release()
