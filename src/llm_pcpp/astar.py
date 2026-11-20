"""8-connected grid A* with the operation and storage counters used in the paper.

* ``operations`` counts every pop from the open list (including stale entries),
  which is the "Operations" metric of the paper.
* ``storage`` is the number of grid cells that ever received a cost-to-come
  value, which is the "Storage" metric of the paper.
* Path lengths are reported with a cost of 1 per axis-aligned step and 1.414 per
  diagonal step (:func:`path_length`), as in the paper.

The search reproduces the paper code exactly (same heap entries, same
tie-breaking), so the counters match the published numbers. It reads the
occupancy grid from a flat byte string instead of indexing the NumPy array
cell by cell, which gives the same result considerably faster.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

import numpy as np

Point = Tuple[int, int]

#: Neighbour expansion orders. They only change tie-breaking between equally
#: good nodes (and therefore the operation/storage counts), not path costs.
NEIGHBOR_ORDERS: Dict[str, Tuple[Tuple[int, int], ...]] = {
    "axis_first": ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)),
    "diagonal_first": ((-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)),
}

#: Cost of a diagonal step when reporting path lengths (paper convention).
DIAGONAL_STEP = 1.414


@dataclass(frozen=True)
class AStarConfig:
    """Settings of the grid A* search.

    Attributes:
        neighbor_order: ``"axis_first"`` or ``"diagonal_first"``.
        diagonal_cost: cost of a diagonal move during the search, either a
            number (1.414 in most paper runs) or ``"exact"`` for sqrt(2).
    """

    neighbor_order: str = "axis_first"
    diagonal_cost: Union[float, str] = DIAGONAL_STEP

    def __post_init__(self) -> None:
        if self.neighbor_order not in NEIGHBOR_ORDERS:
            raise ValueError(
                f"neighbor_order must be one of {sorted(NEIGHBOR_ORDERS)}, "
                f"got {self.neighbor_order!r}"
            )
        if isinstance(self.diagonal_cost, str):
            if self.diagonal_cost != "exact":
                object.__setattr__(self, "diagonal_cost", float(self.diagonal_cost))
        elif self.diagonal_cost <= 0:
            raise ValueError("diagonal_cost must be positive")

    @property
    def diagonal_step(self) -> float:
        return math.sqrt(2) if self.diagonal_cost == "exact" else float(self.diagonal_cost)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AStarConfig":
        kwargs = {k: data[k] for k in ("neighbor_order", "diagonal_cost") if k in data}
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return {"neighbor_order": self.neighbor_order, "diagonal_cost": self.diagonal_cost}


@dataclass
class SearchResult:
    """Result of one A* query (or of a chain of queries through waypoints)."""

    path: List[Point] = field(default_factory=list)
    operations: int = 0
    storage: int = 0

    @property
    def found(self) -> bool:
        return bool(self.path)


class GridAStar:
    """A* search on an occupancy grid where cells with value ``> 0`` are blocked."""

    def __init__(self, grid: np.ndarray, config: AStarConfig = AStarConfig()) -> None:
        if grid.ndim != 2:
            raise ValueError("grid must be a 2D array indexed as grid[y, x]")
        self.height, self.width = (int(n) for n in grid.shape)
        self.config = config
        # Row-major occupancy bytes: cell (x, y) is blocked iff blocked[y * width + x].
        self._blocked = np.ascontiguousarray(grid > 0, dtype=np.uint8).tobytes()
        diagonal = config.diagonal_step
        self._moves = tuple(
            (dx, dy, diagonal if dx != 0 and dy != 0 else 1.0)
            for dx, dy in NEIGHBOR_ORDERS[config.neighbor_order]
        )

    def is_blocked(self, x: int, y: int) -> bool:
        return not (0 <= x < self.width and 0 <= y < self.height) or bool(
            self._blocked[y * self.width + x]
        )

    def search(self, start: Point, goal: Point) -> SearchResult:
        """Find a shortest path from ``start`` to ``goal`` (both ``(x, y)``)."""
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))
        width, height = self.width, self.height
        blocked = self._blocked
        moves = self._moves
        goal_x, goal_y = goal
        sqrt = math.sqrt
        heappush, heappop = heapq.heappush, heapq.heappop

        # Heap entries are (f, g, node), exactly as in the paper code.
        open_heap = [(0 + sqrt((start[0] - goal_x) ** 2 + (start[1] - goal_y) ** 2), 0, start)]
        came_from: Dict[Point, Point] = {}
        g_score: Dict[Point, float] = {start: 0}
        closed = set()
        operations = 0

        while open_heap:
            operations += 1
            _, g, current = heappop(open_heap)
            if current == goal:
                return SearchResult(_reconstruct(came_from, start, goal), operations, len(g_score))
            if current in closed:
                continue
            closed.add(current)
            cx, cy = current
            for dx, dy, step in moves:
                x = cx + dx
                y = cy + dy
                if 0 <= x < width and 0 <= y < height:
                    if blocked[y * width + x]:
                        continue
                    neighbor = (x, y)
                    tentative = g + step
                    previous = g_score.get(neighbor)
                    if previous is not None and tentative >= previous:
                        continue
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    h = sqrt((x - goal_x) ** 2 + (y - goal_y) ** 2)
                    heappush(open_heap, (tentative + h, tentative, neighbor))

        return SearchResult([], operations, len(g_score))


def _reconstruct(came_from: Mapping[Point, Point], start: Point, goal: Point) -> List[Point]:
    path = []
    current = goal
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path


def search_through_waypoints(astar: GridAStar, waypoints: Sequence[Point]) -> SearchResult:
    """Run A* segment by segment through ``waypoints`` (the LLM-A* scheme used here).

    The segment paths are concatenated, and the operation and storage counts
    of all segments are summed. With ``waypoints = [start, goal]`` this is plain
    A*. If a segment is unreachable an empty path is returned.
    """
    if len(waypoints) < 2:
        raise ValueError("need at least a start and a goal waypoint")
    full_path: List[Point] = []
    operations = 0
    storage = 0
    for k in range(len(waypoints) - 1):
        result = astar.search(waypoints[k], waypoints[k + 1])
        operations += result.operations
        storage += result.storage
        if not result.found:
            return SearchResult([], operations, storage)
        full_path.extend(result.path if k == 0 else result.path[1:])
    return SearchResult(full_path, operations, storage)


def path_length(path: Sequence[Point]) -> float:
    """Length of a grid path: 1 per axis-aligned step, 1.414 per diagonal step."""
    total = 0
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        total += DIAGONAL_STEP if (x1 != x0 and y1 != y0) else 1
    return total
