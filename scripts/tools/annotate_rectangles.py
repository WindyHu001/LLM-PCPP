#!/usr/bin/env python
"""Draw obstacle rectangles on an image with the mouse and print them as config regions.

Drag with the left mouse button to add a rectangle, press "u" to undo the
last one, and press "q" or Esc to finish. The rectangles are printed as
``[[x_min, x_max], [y_min, y_max]]`` grid regions (origin at the bottom-left
corner, y up), ready to paste into a scene config. Optionally, an abstract
image with each rectangle filled with its mean colour is saved.

    python scripts/tools/annotate_rectangles.py site_frame.jpg --abstract site_abstract.png

Requires OpenCV with GUI support: pip install -e ".[tools]"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def annotate(image_path: Path):
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Cannot read image {image_path}")
    window = "Annotate obstacles (drag: add, u: undo, q/Esc: finish)"
    rectangles = []
    state = {"drawing": False, "origin": (0, 0)}

    def redraw(extra=None):
        canvas = image.copy()
        for x1, y1, x2, y2 in rectangles + ([extra] if extra else []):
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.imshow(window, canvas)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"], state["origin"] = True, (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            redraw((*state["origin"], x, y))
        elif event == cv2.EVENT_LBUTTONUP and state["drawing"]:
            state["drawing"] = False
            x0, y0 = state["origin"]
            if abs(x - x0) > 1 and abs(y - y0) > 1:
                rectangles.append((x0, y0, x, y))
            redraw()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    redraw()
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("u") and rectangles:
            rectangles.pop()
            redraw()
    cv2.destroyAllWindows()
    return image, rectangles


def to_region(rect, image_height, image_width):
    x1, y1, x2, y2 = rect
    x_min, x_max = sorted((min(max(x1, 0), image_width), min(max(x2, 0), image_width)))
    top, bottom = sorted((min(max(y1, 0), image_height), min(max(y2, 0), image_height)))
    return [[x_min, x_max], [image_height - bottom, image_height - top]]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("image", type=Path, help="input image")
    parser.add_argument("--abstract", type=Path, default=None,
                        help="also save an image with the rectangles filled with their mean colour")
    args = parser.parse_args(argv)

    image, rectangles = annotate(args.image)
    height, width = image.shape[:2]
    print(f"# {len(rectangles)} regions on a {width} x {height} grid (x right, y up)")
    print("regions:")
    for rect in rectangles:
        print(f"  - {to_region(rect, height, width)}")

    if args.abstract:
        import cv2
        import numpy as np

        abstract = np.zeros_like(image)
        for x1, y1, x2, y2 in rectangles:
            (xa, xb), (ya, yb) = sorted((x1, x2)), sorted((y1, y2))
            color = image[ya:yb, xa:xb].reshape(-1, 3).mean(axis=0)
            cv2.rectangle(abstract, (xa, ya), (xb, yb), color.tolist(), -1)
        cv2.imwrite(str(args.abstract), abstract)
        print(f"Saved {args.abstract}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
