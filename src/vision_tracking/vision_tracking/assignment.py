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

"""Minimum-cost bipartite matching for track-to-detection association."""

from __future__ import annotations

import numpy as np

from vision_tracking.geometry import Box
from vision_tracking.geometry import iou

_INF = 1.0e18


def hungarian(cost) -> list[tuple[int, int]]:
    """
    Return a minimum-cost assignment for a rectangular cost matrix.

    Pairs cover the smaller side. Dummy padding never appears in the result.
    """
    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError('cost must be a 2-D matrix')
    n_rows, n_cols = matrix.shape
    if n_rows == 0 or n_cols == 0:
        return []

    size = max(n_rows, n_cols)
    peak = float(np.max(np.abs(matrix))) if matrix.size else 0.0
    pad_cost = peak + 1.0
    padded = np.full((size, size), pad_cost, dtype=np.float64)
    padded[:n_rows, :n_cols] = matrix

    row_potential = np.zeros(size + 1, dtype=np.float64)
    col_potential = np.zeros(size + 1, dtype=np.float64)
    column_match = np.zeros(size + 1, dtype=np.int64)
    predecessor = np.zeros(size + 1, dtype=np.int64)

    for row in range(1, size + 1):
        column_match[0] = row
        current_col = 0
        minimum = np.full(size + 1, _INF, dtype=np.float64)
        used = np.zeros(size + 1, dtype=bool)
        while True:
            used[current_col] = True
            current_row = int(column_match[current_col])
            delta = _INF
            next_col = 0
            for col in range(1, size + 1):
                if used[col]:
                    continue
                reduced = (
                    padded[current_row - 1, col - 1]
                    - row_potential[current_row]
                    - col_potential[col]
                )
                if reduced < minimum[col]:
                    minimum[col] = reduced
                    predecessor[col] = current_col
                if minimum[col] < delta:
                    delta = float(minimum[col])
                    next_col = col
            for col in range(size + 1):
                if used[col]:
                    row_potential[column_match[col]] += delta
                    col_potential[col] -= delta
                else:
                    minimum[col] -= delta
            current_col = next_col
            if column_match[current_col] == 0:
                break
        while True:
            previous = int(predecessor[current_col])
            column_match[current_col] = column_match[previous]
            current_col = previous
            if current_col == 0:
                break

    pairs = []
    for col in range(1, size + 1):
        row = int(column_match[col])
        if row == 0:
            continue
        row_index = row - 1
        col_index = col - 1
        if row_index < n_rows and col_index < n_cols:
            pairs.append((row_index, col_index))
    return pairs


def associate_boxes(
    tracks: list[Box],
    detections: list[Box],
    iou_threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Match track boxes to detection boxes with a minimum IoU.

    The cost is ``1 - IoU``. Pairs below ``iou_threshold`` are left unmatched.
    """
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    cost = np.empty((len(tracks), len(detections)), dtype=np.float64)
    overlaps = np.empty_like(cost)
    for i, track_box in enumerate(tracks):
        for j, detection_box in enumerate(detections):
            overlap = iou(track_box, detection_box)
            overlaps[i, j] = overlap
            if overlap < iou_threshold:
                cost[i, j] = 1.0e3
            else:
                cost[i, j] = 1.0 - overlap

    matched_tracks = set()
    matched_detections = set()
    matches = []
    for track_index, detection_index in hungarian(cost):
        if overlaps[track_index, detection_index] < iou_threshold:
            continue
        matches.append((track_index, detection_index))
        matched_tracks.add(track_index)
        matched_detections.add(detection_index)

    unmatched_tracks = [
        index for index in range(len(tracks)) if index not in matched_tracks
    ]
    unmatched_detections = [
        index for index in range(len(detections)) if index not in matched_detections
    ]
    return matches, unmatched_tracks, unmatched_detections
