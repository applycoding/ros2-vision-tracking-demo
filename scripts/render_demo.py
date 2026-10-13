#!/usr/bin/env python3
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

"""Render an annotated GIF from a video using the same tracker as the ROS nodes."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / 'src' / 'vision_tracking'))

from vision_tracking.detectors import create_detector  # noqa: E402
from vision_tracking.drawing import draw_tracks  # noqa: E402
from vision_tracking.tracker import MultiObjectTracker  # noqa: E402
from vision_tracking.video_source import VideoSource  # noqa: E402


def _parse_args():
    parser = argparse.ArgumentParser(description='Render a tracking demo GIF.')
    parser.add_argument('--video', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--still', default='')
    parser.add_argument('--backend', default='opencv_dnn')
    parser.add_argument('--model', required=True)
    parser.add_argument('--config', default='')
    parser.add_argument('--confidence', type=float, default=0.35)
    parser.add_argument('--input-size', type=int, default=416)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output-fps', type=float, default=8.0)
    parser.add_argument('--max-seconds', type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    """Detect, track, and write a GIF plus an optional still frame."""
    args = _parse_args()
    if shutil.which('ffmpeg') is None:
        raise SystemExit('ffmpeg is required to encode the GIF.')

    detector = create_detector(
        backend=args.backend,
        model_path=args.model,
        config_path=args.config,
        confidence=args.confidence,
        input_size=args.input_size,
        device=args.device,
    )
    tracker = MultiObjectTracker(min_hits=1, coast_frames=1)
    source = VideoSource(args.video, loop=False)
    stride = max(int(round(source.fps / args.output_fps)), 1)
    limit = max(int(args.max_seconds * source.fps), 1)

    best_count = -1
    best_frame = None
    written = 0
    with tempfile.TemporaryDirectory(prefix='vision-tracking-') as temp:
        folder = Path(temp)
        index = 0
        kept = 0
        while index < limit:
            frame = source.read()
            if frame is None:
                break
            if index % stride == 0:
                detections = detector.detect(frame)
                dt = stride / source.fps
                tracks = tracker.update(detections, dt)
                annotated = draw_tracks(frame, tracks)
                path = folder / f'frame_{kept:04d}.png'
                _write_png(path, annotated)
                if len(tracks) > best_count:
                    best_count = len(tracks)
                    best_frame = annotated
                kept += 1
                print(f'frame {kept}: {len(detections)} detections, {len(tracks)} tracks')
            index += 1
        source.close()
        if kept == 0:
            raise SystemExit('No frames were decoded from the video.')
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        _encode_gif(folder, output, args.output_fps)
        written = kept
        if args.still and best_frame is not None:
            still = Path(args.still)
            still.parent.mkdir(parents=True, exist_ok=True)
            _write_png(still, best_frame)
    print(f'Wrote {written} frames to {args.output}')


def _write_png(path: Path, image) -> None:
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f'Could not write {path}')


def _encode_gif(folder: Path, output: Path, fps: float) -> None:
    palette = folder / 'palette.png'
    pattern = str(folder / 'frame_%04d.png')
    subprocess.run(
        [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-framerate', str(fps), '-i', pattern,
            '-vf', 'fps={0},scale=480:-1:flags=lanczos,palettegen'.format(fps),
            str(palette),
        ],
        check=True,
    )
    subprocess.run(
        [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-framerate', str(fps), '-i', pattern,
            '-i', str(palette),
            '-lavfi', 'fps={0},scale=480:-1:flags=lanczos[x];[x][1:v]paletteuse'.format(fps),
            str(output),
        ],
        check=True,
    )


if __name__ == '__main__':
    main()
