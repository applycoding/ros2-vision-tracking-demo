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

"""Draw track boxes onto the camera image."""

from __future__ import annotations

from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer
from message_filters import Subscriber
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

from vision_tracking.drawing import draw_tracks
from vision_tracking.messages import msg_to_tracks


class AnnotateNode(Node):
    """Publish ``image_annotated`` for RViz and rqt_image_view."""

    def __init__(self) -> None:
        super().__init__('annotate')
        self.declare_parameter('image_topic', 'image_raw')
        self.declare_parameter('tracks_topic', 'tracks')
        self.declare_parameter('annotated_topic', 'image_annotated')
        self.declare_parameter('slop', 0.3)

        qos = _qos()
        image_topic = str(self.get_parameter('image_topic').value)
        tracks_topic = str(self.get_parameter('tracks_topic').value)
        annotated_topic = str(self.get_parameter('annotated_topic').value)
        self._bridge = CvBridge()
        self._publisher = self.create_publisher(Image, annotated_topic, qos)
        self._image_sub = Subscriber(self, Image, image_topic, qos_profile=qos)
        self._tracks_sub = Subscriber(
            self, Detection2DArray, tracks_topic, qos_profile=qos,
        )
        self._sync = ApproximateTimeSynchronizer(
            [self._image_sub, self._tracks_sub],
            queue_size=10,
            slop=float(self.get_parameter('slop').value),
        )
        self._sync.registerCallback(self._on_pair)
        self.get_logger().info(
            f'Annotating {image_topic} + {tracks_topic} -> {annotated_topic}'
        )

    def _on_pair(self, image_msg: Image, tracks_msg: Detection2DArray) -> None:
        frame = self._bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        annotated = draw_tracks(frame, msg_to_tracks(tracks_msg))
        message = self._bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        message.header = image_msg.header
        self._publisher.publish(message)


def _qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
    )


def main(args=None) -> None:
    """Run the annotator until interrupted."""
    rclpy.init(args=args)
    node = AnnotateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
