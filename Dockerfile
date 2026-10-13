FROM ros:humble-ros-base-jammy

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
        python3-colcon-common-extensions \
        python3-numpy \
        python3-opencv \
        python3-pip \
        python3-pytest \
        ros-humble-ament-copyright \
        ros-humble-ament-flake8 \
        ros-humble-ament-pep257 \
        ros-humble-cv-bridge \
        ros-humble-message-filters \
        ros-humble-rqt-image-view \
        ros-humble-rviz2 \
        ros-humble-vision-msgs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ws
COPY . /opt/ws

RUN chmod +x /opt/ws/docker/entrypoint.sh /opt/ws/scripts/fetch_assets.sh \
    && bash /opt/ws/scripts/fetch_assets.sh \
    && source /opt/ros/humble/setup.bash \
    && colcon build --packages-select vision_tracking

ENTRYPOINT ["/opt/ws/docker/entrypoint.sh"]
CMD ["ros2", "launch", "vision_tracking", "tracking_demo.launch.py", \
     "video_path:=/opt/ws/media/sample_crossing.mp4", \
     "model_path:=/opt/ws/models/yolov4-tiny.weights", \
     "config_path:=/opt/ws/models/yolov4-tiny.cfg", \
     "rviz:=false"]
