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

"""Shared detection and track records."""

from __future__ import annotations

from dataclasses import dataclass

from vision_tracking.geometry import Box


@dataclass(frozen=True)
class Detection:
    """One detector hit, before a track id exists."""

    bbox: Box
    score: float
    class_id: int
    class_name: str


@dataclass(frozen=True)
class TrackSnapshot:
    """A track published for one frame."""

    track_id: int
    bbox: Box
    score: float
    class_id: int
    class_name: str
    age: int
    hits: int
    time_since_update: int
