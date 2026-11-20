#!/usr/bin/env python
"""Extract one frame from a (drone) video, e.g. as the source image of a grid map.

    python scripts/tools/extract_frame.py DJI_0008.MOV --time 167 --output site_frame.jpg

Requires OpenCV: pip install -e ".[tools]"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def extract_frame(video_path: Path, target_time: float, output_path: Path) -> None:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"Cannot open video {video_path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"FPS: {fps:.2f}, frames: {total_frames}, duration: {total_frames / fps:.2f} s")
        frame_number = int(target_time * fps)
        if frame_number >= total_frames:
            raise SystemExit("The requested time exceeds the video duration.")
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = capture.read()
        if not success:
            raise SystemExit("Failed to read the frame.")
        cv2.imwrite(str(output_path), frame)
        height, width = frame.shape[:2]
        print(f"Saved the frame at {target_time:.2f} s ({width} x {height}) to {output_path}")
    finally:
        capture.release()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("video", type=Path, help="input video file")
    parser.add_argument("--time", type=float, required=True, help="time of the frame in seconds")
    parser.add_argument("--output", type=Path, default=Path("frame.jpg"), help="output image (default: frame.jpg)")
    args = parser.parse_args(argv)
    extract_frame(args.video, args.time, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
