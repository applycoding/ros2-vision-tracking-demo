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

"""Tests for the Hungarian assignment and IoU gating."""

from __future__ import annotations

import itertools

import numpy as np

from vision_tracking.assignment import associate_boxes
from vision_tracking.assignment import hungarian


def _assignment_cost(cost, pairs):
    return sum(cost[row][col] for row, col in pairs)


def _brute_cost(cost):
    matrix = np.asarray(cost, dtype=float)
    n_rows, n_cols = matrix.shape
    best = None
    if n_rows <= n_cols:
        choices = itertools.permutations(range(n_cols), n_rows)
        for choice in choices:
            total = sum(matrix[row, choice[row]] for row in range(n_rows))
            if best is None or total < best:
                best = total
    else:
        choices = itertools.permutations(range(n_rows), n_cols)
        for choice in choices:
            total = sum(matrix[choice[col], col] for col in range(n_cols))
            if best is None or total < best:
                best = total
    return best


def test_hungarian_known_matrix():
    cost = [
        [4.0, 1.0, 3.0],
        [2.0, 0.0, 5.0],
        [3.0, 2.0, 2.0],
    ]
    pairs = hungarian(cost)
    assert abs(_assignment_cost(cost, pairs) - 5.0) < 1.0e-6
    assert len(pairs) == 3


def test_hungarian_matches_brute_force():
    rng = np.random.default_rng(7)
    for rows, cols in ((1, 1), (2, 3), (3, 2), (4, 4), (3, 5)):
        cost = rng.normal(size=(rows, cols))
        pairs = hungarian(cost)
        assert len(pairs) == min(rows, cols)
        assert abs(_assignment_cost(cost, pairs) - _brute_cost(cost)) < 1.0e-6


def test_hungarian_empty():
    assert hungarian(np.zeros((0, 3))) == []
    assert hungarian(np.zeros((2, 0))) == []


def test_associate_rejects_distant_boxes():
    tracks = [(0.0, 0.0, 10.0, 10.0), (100.0, 100.0, 110.0, 110.0)]
    detections = [(100.0, 100.0, 112.0, 110.0), (1.0, 1.0, 11.0, 11.0)]
    matches, unmatched_tracks, unmatched_detections = associate_boxes(
        tracks, detections, 0.3,
    )
    assert unmatched_tracks == []
    assert unmatched_detections == []
    assert sorted(matches) == [(0, 1), (1, 0)]


def test_associate_leaves_low_iou_unmatched():
    matches, unmatched_tracks, unmatched_detections = associate_boxes(
        [(0.0, 0.0, 10.0, 10.0)],
        [(50.0, 50.0, 60.0, 60.0)],
        0.3,
    )
    assert matches == []
    assert unmatched_tracks == [0]
    assert unmatched_detections == [0]
