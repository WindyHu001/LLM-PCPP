#!/usr/bin/env python
"""Overlay a coordinate grid on an image to read off obstacle and POI coordinates.

Labels are grid coordinates as used in the configs: x to the right and y up,
with the origin at the bottom-left corner of the image.

    python scripts/tools/draw_pixel_grid.py site_frame.jpg --rows 9 --cols 9 --output site_frame_grid.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional


def draw_grid(image_path: Path, output_path: Path, rows: int, cols: int, font_path: Optional[str] = None) -> None:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    print(f"Image size: {width} x {height}")
    draw = ImageDraw.Draw(image)
    cell_width, cell_height = width / cols, height / rows
    line_width = max(1, round(min(width, height) / 1000))

    for i in range(1, cols):
        x = int(i * cell_width)
        draw.line([(x, 0), (x, height)], fill="red", width=line_width)
    for j in range(1, rows):
        y = int(j * cell_height)
        draw.line([(0, y), (width, y)], fill="red", width=line_width)

    font_size = max(int(min(cell_width, cell_height) * 0.2), 20)
    font = None
    for candidate in [font_path, "DejaVuSans.ttf", "Arial.ttf", "/Library/Fonts/Arial.ttf"]:
        if not candidate:
            continue
        try:
            font = ImageFont.truetype(candidate, font_size)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()

    for i in range(rows + 1):
        for j in range(cols + 1):
            x = int(j * cell_width)
            y_image = int(i * cell_height)
            label = f"({x},{height - y_image})"  # y measured from the bottom edge
            draw.text((x + 5, y_image + 5), label, fill="blue", font=font)

    image.save(output_path)
    print(f"Saved {output_path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("image", type=Path, help="input image")
    parser.add_argument("--rows", type=int, default=9, help="number of grid rows (default: 9)")
    parser.add_argument("--cols", type=int, default=9, help="number of grid columns (default: 9)")
    parser.add_argument("--output", type=Path, default=None, help="output image (default: <image>_grid.jpg)")
    parser.add_argument("--font", default=None, help="path to a .ttf font for the labels")
    args = parser.parse_args(argv)
    output = args.output or args.image.with_name(f"{args.image.stem}_grid.jpg")
    draw_grid(args.image, output, args.rows, args.cols, args.font)
    return 0


if __name__ == "__main__":
    sys.exit(main())
