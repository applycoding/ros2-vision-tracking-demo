#!/usr/bin/env bash
# Download the YOLOv4-tiny files used by the OpenCV DNN backend.
# Weights stay out of git. The sample clip in media/ is already included.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${ROOT}/models"
mkdir -p "${MODEL_DIR}"

CFG="${MODEL_DIR}/yolov4-tiny.cfg"
WEIGHTS="${MODEL_DIR}/yolov4-tiny.weights"
CFG_URL="https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
WEIGHTS_URL="https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights"

if [[ ! -f "${CFG}" ]]; then
  curl -fsSL -o "${CFG}" "${CFG_URL}"
fi
if [[ ! -f "${WEIGHTS}" ]]; then
  curl -fL --retry 3 -o "${WEIGHTS}" "${WEIGHTS_URL}"
fi

echo "Ready:"
echo "  ${CFG}"
echo "  ${WEIGHTS}"
