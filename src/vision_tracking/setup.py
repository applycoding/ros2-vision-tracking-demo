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

"""Install the vision_tracking ROS 2 package."""

from glob import glob
import os

from setuptools import find_packages
from setuptools import setup

package_name = 'vision_tracking'


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Arpan Bom',
    maintainer_email='applycoding@gmail.com',
    description='ROS 2 Humble video detection and multi-object tracking demo.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'video_publisher = vision_tracking.video_publisher:main',
            'detector = vision_tracking.detector_node:main',
            'tracker = vision_tracking.tracker_node:main',
            'annotate = vision_tracking.annotate_node:main',
        ],
    },
)
