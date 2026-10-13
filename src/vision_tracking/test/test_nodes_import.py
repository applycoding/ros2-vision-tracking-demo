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

"""Import the ROS nodes when Humble is on PYTHONPATH."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ROS_MODULES = (
    'rclpy',
    'cv_bridge',
    'vision_msgs',
    'message_filters',
    'launch',
    'launch_ros',
)

try:
    from vision_tracking.annotate_node import AnnotateNode
    from vision_tracking.detector_node import DetectorNode
    from vision_tracking.tracker_node import TrackerNode
    from vision_tracking.video_publisher import VideoPublisher

    _LAUNCH_PATH = Path(__file__).resolve().parents[1] / 'launch' / 'tracking_demo.launch.py'
    _LAUNCH_SPEC = importlib.util.spec_from_file_location(
        'tracking_demo_launch', _LAUNCH_PATH,
    )
    _LAUNCH_MODULE = importlib.util.module_from_spec(_LAUNCH_SPEC)
    _LAUNCH_SPEC.loader.exec_module(_LAUNCH_MODULE)
except ImportError as exc:
    if not any(name in str(exc) for name in _ROS_MODULES):
        raise
    AnnotateNode = None
    DetectorNode = None
    TrackerNode = None
    VideoPublisher = None
    _LAUNCH_MODULE = None


def test_nodes_import():
    if VideoPublisher is None:
        pytest.skip('ROS 2 Python packages are not installed')
    assert VideoPublisher.__name__ == 'VideoPublisher'
    assert DetectorNode.__name__ == 'DetectorNode'
    assert TrackerNode.__name__ == 'TrackerNode'
    assert AnnotateNode.__name__ == 'AnnotateNode'


def test_launch_file_imports():
    if _LAUNCH_MODULE is None:
        pytest.skip('ROS 2 launch packages are not installed')
    assert callable(_LAUNCH_MODULE.generate_launch_description)
