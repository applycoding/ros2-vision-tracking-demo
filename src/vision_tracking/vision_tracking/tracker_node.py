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

"""Associate detections into tracks with stable ids."""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from vision_msgs.msg import Detection2DArray

from vision_tracking.messages import msg_to_detections
from vision_tracking.messages import tracks_to_msg
from vision_tracking.tracker import MultiObjectTracker


class TrackerNode(Node):
    """Subscribe to detections and publish ``/tracks``."""

    def __init__(self) -> None:
        super().__init__('tracker')
        self.declare_parameter('detections_topic', 'detections')
        self.declare_parameter('tracks_topic', 'tracks')
        self.declare_parameter('high_thresh', 0.5)
        self.declare_parameter('low_thresh', 0.1)
        self.declare_parameter('iou_threshold', 0.3)
        self.declare_parameter('max_age', 30)
        self.declare_parameter('min_hits', 1)
        self.declare_parameter('coast_frames', 1)
        self.declare_parameter('fallback_fps', 30.0)

        self._tracker = MultiObjectTracker(
            high_thresh=float(self.get_parameter('high_thresh').value),
            low_thresh=float(self.get_parameter('low_thresh').value),
            iou_threshold=float(self.get_parameter('iou_threshold').value),
            max_age=int(self.get_parameter('max_age').value),
            min_hits=int(self.get_parameter('min_hits').value),
            coast_frames=int(self.get_parameter('coast_frames').value),
        )
        self._fallback_dt = 1.0 / float(self.get_parameter('fallback_fps').value)
        self._previous_stamp = None
        qos = _qos()
        tracks_topic = str(self.get_parameter('tracks_topic').value)
        detections_topic = str(self.get_parameter('detections_topic').value)
        self._publisher = self.create_publisher(Detection2DArray, tracks_topic, qos)
        self.create_subscription(
            Detection2DArray,
            detections_topic,
            self._on_detections,
            qos,
        )
        self.get_logger().info(f'Tracking {detections_topic} -> {tracks_topic}')

    def _on_detections(self, message: Detection2DArray) -> None:
        detections = msg_to_detections(message)
        tracks = self._tracker.update(detections, self._dt(message.header.stamp))
        self._publisher.publish(tracks_to_msg(tracks, message.header))
        self.get_logger().info(
            f'{len(tracks)} tracks',
            throttle_duration_sec=2.0,
        )

    def _dt(self, stamp) -> float:
        current = float(stamp.sec) + float(stamp.nanosec) * 1.0e-9
        if self._previous_stamp is None or current <= self._previous_stamp:
            dt = self._fallback_dt
        else:
            dt = current - self._previous_stamp
        self._previous_stamp = current
        return dt


def _qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
    )


def main(args=None) -> None:
    """Run the tracker until interrupted."""
    rclpy.init(args=args)
    node = TrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
