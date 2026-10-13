#!/bin/bash
# Source ROS 2 Humble and this workspace, then run the container command.

set -e
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
if [[ -f /opt/ws/install/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /opt/ws/install/setup.bash
fi
exec "$@"
