"""Scene definition: a 2D semantic grid map with rectangular obstacles and POIs.

Coordinates follow the paper. The grid has the same resolution as the source
image (one cell per pixel), the origin ``(0, 0)`` is the bottom-left corner,
``x`` points right and ``y`` points up. An obstacle region
``[[x_min, x_max], [y_min, y_max]]`` blocks the cells
``x_min <= x < x_max`` and ``y_min <= y < y_max``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

Point = Tuple[int, int]


@dataclass(frozen=True)
class Region:
    """Axis-aligned rectangular obstacle region on the grid."""

    x_min: int
    x_max: int
    y_min: int
    y_max: int

    @classmethod
    def from_ranges(cls, ranges: Sequence[Sequence[int]]) -> "Region":
        """Build a region from ``[[x_min, x_max], [y_min, y_max]]``."""
        try:
            (x_min, x_max), (y_min, y_max) = ranges
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Region must be [[x_min, x_max], [y_min, y_max]], got {ranges!r}"
            ) from exc
        region = cls(int(x_min), int(x_max), int(y_min), int(y_max))
        if region.x_min >= region.x_max or region.y_min >= region.y_max:
            raise ValueError(f"Region bounds must be increasing, got {ranges!r}")
        return region

    def as_xyxy(self) -> List[int]:
        """Return ``[x_min, y_min, x_max, y_max]``, the format given to the LLM."""
        return [self.x_min, self.y_min, self.x_max, self.y_max]


@dataclass(frozen=True)
class ObstacleClass:
    """A semantic obstacle type (e.g. "Construction Sites") and its regions."""

    name: str
    label: int
    regions: Tuple[Region, ...]


@dataclass(frozen=True)
class POI:
    """A point of interest the robot has to visit."""

    name: str
    x: int
    y: int
    note: str = ""

    @property
    def xy(self) -> Point:
        return (self.x, self.y)


@dataclass
class Scene:
    """A complete planning problem: grid size, obstacles, start point and POIs."""

    name: str
    width: int
    height: int
    start: Point
    obstacle_classes: List[ObstacleClass]
    pois: List[POI]
    description: str = ""
    task_guidance: str = ""
    planners: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    paper_results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    source: Optional[Path] = None

    # ------------------------------------------------------------------ grid
    def build_grid(self) -> np.ndarray:
        """Rasterise the obstacles into a ``(height, width)`` label grid.

        ``0`` marks free space and ``k > 0`` the obstacle class with label
        ``k``. Classes are painted in ascending label order, so where regions
        overlap the larger label wins (this only affects plots, not planning).
        """
        grid = np.zeros((self.height, self.width), dtype=np.uint8)
        for obstacle_class in sorted(self.obstacle_classes, key=lambda c: c.label):
            for r in obstacle_class.regions:
                grid[r.y_min : r.y_max, r.x_min : r.x_max] = obstacle_class.label
        return grid

    def obstacle_rectangles(self) -> List[List[int]]:
        """All obstacles as ``[x_min, y_min, x_max, y_max]``, in config order.

        This is the obstacle list that is written into the LLM-A* prompt, so
        the order matters for reproducing the paper's LLM queries.
        """
        return [r.as_xyxy() for c in self.obstacle_classes for r in c.regions]

    def class_names(self) -> Dict[int, str]:
        return {c.label: c.name for c in self.obstacle_classes}

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def split_pois(self, grid: Optional[np.ndarray] = None) -> Tuple[List[POI], List[POI]]:
        """Separate valid POIs from POIs outside the map or inside an obstacle."""
        grid = self.build_grid() if grid is None else grid
        valid: List[POI] = []
        rejected: List[POI] = []
        for poi in self.pois:
            if self.in_bounds(poi.x, poi.y) and grid[poi.y, poi.x] == 0:
                valid.append(poi)
            else:
                rejected.append(poi)
        return valid, rejected

    # ------------------------------------------------------------------- I/O
    @classmethod
    def from_dict(cls, data: Dict[str, Any], source: Optional[Path] = None) -> "Scene":
        try:
            grid_cfg = data["grid"]
            width, height = int(grid_cfg["width"]), int(grid_cfg["height"])
            start = data["start"]
            start_xy = (int(start[0]), int(start[1]))
        except (KeyError, TypeError, IndexError) as exc:
            raise ValueError(f"Scene config is missing 'grid' or 'start': {exc}") from exc

        classes: List[ObstacleClass] = []
        for entry in data.get("obstacle_classes", []):
            label = int(entry["label"])
            if not 1 <= label <= 255:
                raise ValueError(f"Obstacle label must be in 1..255, got {label}")
            regions = tuple(Region.from_ranges(r) for r in entry.get("regions", []))
            for r in regions:
                if r.x_min < 0 or r.y_min < 0 or r.x_max > width or r.y_max > height:
                    raise ValueError(
                        f"Region {r.as_xyxy()} of '{entry['name']}' lies outside "
                        f"the {width} x {height} grid"
                    )
            classes.append(ObstacleClass(name=str(entry["name"]), label=label, regions=regions))

        pois = [
            POI(
                name=str(p.get("name", f"POI_{k}")),
                x=int(p["x"]),
                y=int(p["y"]),
                note=str(p.get("note", "")),
            )
            for k, p in enumerate(data.get("pois", []))
        ]

        scene = cls(
            name=str(data.get("name", source.stem if source else "scene")),
            width=width,
            height=height,
            start=start_xy,
            obstacle_classes=classes,
            pois=pois,
            description=str(data.get("description", "")),
            task_guidance=str(data.get("task_guidance", "")).strip(),
            planners=dict(data.get("planners", {}) or {}),
            paper_results=dict(data.get("paper_results", {}) or {}),
            source=source,
        )
        if not scene.in_bounds(*scene.start):
            raise ValueError(f"Start point {scene.start} lies outside the grid")
        return scene


def load_scene(path: "str | Path") -> Scene:
    """Load a scene from a YAML config file (see ``configs/``)."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a YAML mapping")
    return Scene.from_dict(data, source=path)
