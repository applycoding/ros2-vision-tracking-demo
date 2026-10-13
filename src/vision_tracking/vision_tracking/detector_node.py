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

"""Subscribe to images and publish YOLO detections."""

from __future__ import annotations

from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

from vision_tracking.detectors import create_detector
from vision_tracking.messages import detections_to_msg


class DetectorNode(Node):
    """Run the configured YOLO backend on ``image_raw``."""

    def __init__(self) -> None:
        super().__init__('detector')
        self.declare_parameter('image_topic', 'image_raw')
        self.declare_parameter('detections_topic', 'detections')
        self.declare_parameter('backend', 'opencv_dnn')
        self.declare_parameter('model_path', '')
        self.declare_parameter('config_path', '')
        self.declare_parameter('confidence', 0.35)
        self.declare_parameter('iou', 0.45)
        self.declare_parameter('input_size', 416)
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('allowed_classes', '')

        model_path = str(self.get_parameter('model_path').value)
        if not model_path:
            raise RuntimeError(
                'model_path is empty. Point it at yolov4-tiny.weights or a YOLO checkpoint.'
            )
        self._bridge = CvBridge()
        self._detector = create_detector(
            backend=str(self.get_parameter('backend').value),
            model_path=model_path,
            config_path=str(self.get_parameter('config_path').value),
            confidence=float(self.get_parameter('confidence').value),
            iou=float(self.get_parameter('iou').value),
            input_size=int(self.get_parameter('input_size').value),
            device=str(self.get_parameter('device').value),
            allowed_classes=str(self.get_parameter('allowed_classes').value),
        )
        qos = _qos()
        image_topic = str(self.get_parameter('image_topic').value)
        detections_topic = str(self.get_parameter('detections_topic').value)
        self._publisher = self.create_publisher(Detection2DArray, detections_topic, qos)
        self.create_subscription(Image, image_topic, self._on_image, qos)
        self.get_logger().info(
            f"Detector backend={self.get_parameter('backend').value} "
            f'model={model_path} {image_topic} -> {detections_topic}'
        )

    def _on_image(self, message: Image) -> None:
        frame = self._bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
        detections = self._detector.detect(frame)
        self._publisher.publish(detections_to_msg(detections, message.header))
        self.get_logger().info(
            f'{len(detections)} detections',
            throttle_duration_sec=2.0,
        )


def _qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
    )


def main(args=None) -> None:
    """Run the detector until interrupted."""
    rclpy.init(args=args)
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
