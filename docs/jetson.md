# NVIDIA Jetson

This demo does not need a GPU. On a Jetson the same nodes run, and the ultralytics backend can use the board's CUDA PyTorch build.

These notes were written from the Humble/JetPack platform docs. They were not executed on Jetson hardware in this repository.

## Platform

- JetPack 6 is Ubuntu 22.04, which matches ROS 2 Humble's apt target. Add the ROS 2 apt repository the same way as on an x86 Ubuntu 22.04 machine, then install the packages in the README.
- JetPack 5 is Ubuntu 20.04. Prefer an L4T container that already pairs Humble with Jetson's libraries (NVIDIA publishes `dustynv` ROS images for this) instead of the desktop Humble apt line.
- Use the `opencv_dnn` backend and YOLOv4-tiny when you want the same CPU path as the x86 Docker image.
- Use `backend:=ultralytics` and `device:=0` after installing Ultralytics against NVIDIA's PyTorch wheel. The PyPI CPU wheel will not see the GPU.

## Camera

A CSI sensor is not a normal V4L2 device until Argus is bridged. Two practical options:

- USB webcam: `video_path:=/dev/video0` and `loop:=false`.
- CSI: run `nvarguscamerasrc` into a `v4l2loopback` device, then point `video_path` at that device.

`video_publisher` treats paths under `/dev/video` and numeric indexes as live cameras and does not rewind them.

## Power

For a live demo you can raise the clocks. This is optional and draws more power:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

OpenCV DNN in this package stays on CPU. A JetPack OpenCV build with the CUDA DNN backend is not required.
