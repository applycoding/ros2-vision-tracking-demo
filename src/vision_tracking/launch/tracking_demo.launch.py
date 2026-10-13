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

"""Launch the video, detector, tracker, annotator, and optional RViz."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Create the tracking demo launch description."""
    share = get_package_share_directory('vision_tracking')
    default_params = os.path.join(share, 'config', 'tracking.yaml')
    default_rviz = os.path.join(share, 'rviz', 'tracking.rviz')

    video_path = LaunchConfiguration('video_path')
    model_path = LaunchConfiguration('model_path')
    config_path = LaunchConfiguration('config_path')
    backend = LaunchConfiguration('backend')
    device = LaunchConfiguration('device')
    params_file = LaunchConfiguration('params_file')
    rviz = LaunchConfiguration('rviz')

    return LaunchDescription([
        DeclareLaunchArgument(
            'video_path',
            default_value='',
            description='Video file or V4L2 device, for example /dev/video0.',
        ),
        DeclareLaunchArgument(
            'model_path',
            default_value='',
            description='YOLO weights (.weights) or an Ultralytics checkpoint (.pt).',
        ),
        DeclareLaunchArgument(
            'config_path',
            default_value='',
            description='Darknet .cfg used with model_path when backend is opencv_dnn.',
        ),
        DeclareLaunchArgument(
            'backend',
            default_value='opencv_dnn',
            description="Detector backend: 'opencv_dnn' or 'ultralytics'.",
        ),
        DeclareLaunchArgument(
            'device',
            default_value='cpu',
            description="Inference device. OpenCV DNN accepts 'cpu'. Ultralytics accepts '0'.",
        ),
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='YAML file of node parameters.',
        ),
        DeclareLaunchArgument(
            'rviz',
            default_value='false',
            description='Start RViz with the annotated image display.',
        ),
        Node(
            package='vision_tracking',
            executable='video_publisher',
            name='video_publisher',
            output='screen',
            parameters=[
                params_file,
                {'video_path': video_path},
            ],
        ),
        Node(
            package='vision_tracking',
            executable='detector',
            name='detector',
            output='screen',
            parameters=[
                params_file,
                {
                    'model_path': model_path,
                    'config_path': config_path,
                    'backend': backend,
                    'device': device,
                },
            ],
        ),
        Node(
            package='vision_tracking',
            executable='tracker',
            name='tracker',
            output='screen',
            parameters=[params_file],
        ),
        Node(
            package='vision_tracking',
            executable='annotate',
            name='annotate',
            output='screen',
            parameters=[params_file],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', default_rviz],
            condition=IfCondition(rviz),
        ),
    ])
