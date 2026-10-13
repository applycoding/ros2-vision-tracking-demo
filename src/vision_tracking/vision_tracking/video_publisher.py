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

"""Publish frames from a video file or V4L2 device."""

from __future__ import annotations

from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import Image

from vision_tracking.video_source import VideoSource


class VideoPublisher(Node):
    """Publish ``sensor_msgs/Image`` at the file frame rate, looping by default."""

    def __init__(self) -> None:
        super().__init__('video_publisher')
        self.declare_parameter('video_path', '')
        self.declare_parameter('image_topic', 'image_raw')
        self.declare_parameter('fps', 0.0)
        self.declare_parameter('loop', True)
        self.declare_parameter('frame_id', 'camera')

        video_path = str(self.get_parameter('video_path').value)
        if not video_path:
            raise RuntimeError(
                'video_path is empty. Pass a video file or a camera such as /dev/video0.'
            )
        self._frame_id = str(self.get_parameter('frame_id').value)
        self._source = VideoSource(
            video_path,
            loop=bool(self.get_parameter('loop').value),
        )
        requested = float(self.get_parameter('fps').value)
        fps = requested if requested > 0.0 else self._source.fps
        if fps < 1.0e-3:
            fps = 30.0
        topic = str(self.get_parameter('image_topic').value)
        self._bridge = CvBridge()
        self._publisher = self.create_publisher(Image, topic, _image_qos())
        self._timer = self.create_timer(1.0 / fps, self._on_timer)
        self.get_logger().info(
            f'Publishing {video_path} on {topic} at {fps:.2f} fps '
            f'(frame_id={self._frame_id}).'
        )

    def _on_timer(self) -> None:
        frame = self._source.read()
        if frame is None:
            self.get_logger().error('Video source returned no frame. Stopping.')
            self._timer.cancel()
            return
        message = self._bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self._frame_id
        self._publisher.publish(message)

    def destroy_node(self):
        """Release the capture, then the node."""
        self._source.close()
        return super().destroy_node()


def _image_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
    )


def main(args=None) -> None:
    """Run the video publisher until interrupted."""
    rclpy.init(args=args)
    node = VideoPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
