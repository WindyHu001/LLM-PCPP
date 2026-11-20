"""Figures in the style of the paper: hatched semantic grid map, POIs and paths."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np

from .scene import Scene

Point = Tuple[int, int]

#: Hatch pattern per obstacle label (label 1 -> HATCHES[0], ...), as in the paper.
HATCHES = ("///", "...", "xxx", "--", "\\\\\\", "++", "oo", "**")


def block_labels(grid: np.ndarray, block: int) -> np.ndarray:
    """Downsample a label grid into ``block x block`` cells.

    Each cell takes the most frequent obstacle label inside it (the smaller
    label on ties), or 0 if the cell contains no obstacle. Incomplete cells at
    the right and top border are dropped.
    """
    height, width = grid.shape
    rows, cols = height // block, width // block
    cells = (
        grid[: rows * block, : cols * block]
        .reshape(rows, block, cols, block)
        .swapaxes(1, 2)
        .reshape(rows, cols, block * block)
    )
    labels = np.zeros((rows, cols), dtype=grid.dtype)
    best = np.zeros((rows, cols), dtype=np.int64)
    for label in np.unique(grid[grid > 0]):
        count = (cells == label).sum(axis=2)
        better = count > best
        labels[better] = label
        best[better] = count[better]
    return labels


def plot_scene(
    scene: Scene,
    grid: Optional[np.ndarray] = None,
    points: Optional[Sequence[Point]] = None,
    point_names: Optional[Sequence[str]] = None,
    path: Optional[Sequence[Point]] = None,
    ax: Any = None,
    block: int = 20,
    title: Optional[str] = None,
    annotate: bool = True,
    figsize: Tuple[float, float] = (14, 8),
) -> Tuple[Any, Any]:
    """Draw the semantic grid map with the start point, the POIs and an optional path.

    ``points[0]`` is the start point; by default the scene's start and valid
    POIs are drawn. Returns ``(figure, axes)``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    grid = scene.build_grid() if grid is None else grid
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    labels = block_labels(grid, block)
    names = scene.class_names()
    handles = []
    for label in sorted(names):
        hatch = HATCHES[(label - 1) % len(HATCHES)]
        ys, xs = np.nonzero(labels == label)
        if len(xs):
            cells = [Rectangle((x * block, y * block), block, block) for x, y in zip(xs, ys)]
            ax.add_collection(
                PatchCollection(cells, facecolor="white", edgecolor="black", hatch=hatch, linewidth=0.5)
            )
        handles.append(Patch(facecolor="white", edgecolor="black", hatch=hatch, label=names[label]))

    if points is None:
        valid, _ = scene.split_pois(grid)
        points = [scene.start] + [poi.xy for poi in valid]
        point_names = ["Start"] + [poi.name for poi in valid]

    if path:
        xs, ys = zip(*path)
        ax.plot(xs, ys, color="blue", linewidth=2, zorder=3)
    ax.plot(points[0][0], points[0][1], "b*", markersize=10, zorder=4)
    if len(points) > 1:
        ax.plot([p[0] for p in points[1:]], [p[1] for p in points[1:]], "r*", markersize=8,
                linestyle="None", zorder=4)
    if annotate and point_names:
        for name, (x, y) in zip(point_names, points):
            ax.text(x + 10, y + 10, name, fontsize=8, color="black", alpha=0.8, zorder=5)

    handles.append(Line2D([0], [0], color="blue", marker="*", linestyle="None", markersize=10, label="Start"))
    handles.append(Line2D([0], [0], color="red", marker="*", linestyle="None", markersize=8, label="POIs"))
    if path:
        handles.append(Line2D([0], [0], color="blue", linewidth=2, label="Coverage path"))

    ax.set_xlim(0, scene.width)
    ax.set_ylim(0, scene.height)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if title:
        ax.set_title(title)
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    return fig, ax
