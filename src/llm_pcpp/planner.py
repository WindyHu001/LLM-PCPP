"""The LLM-PCPP pipeline (paper Sec. 3).

1. Take the start point and the (validated) POIs of a scene.
2. For every pair of points, compute a path with A* (baseline) or with
   LLM-A* (A* through LLM-generated waypoints), and record its length,
   operations and storage.
3. Build the pairwise distance matrix and solve the TSP with OR-Tools.
4. Concatenate the pairwise paths along the TSP tour into the coverage path.
"""

from __future__ import annotations

import logging
import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .astar import AStarConfig, GridAStar, Point, SearchResult, path_length, search_through_waypoints
from .scene import POI, Scene
from .tsp import route_length, solve_tsp

log = logging.getLogger(__name__)

#: ``waypoint_fn(start, goal) -> [start, ..., goal]``
WaypointFn = Callable[[Point, Point], List[Point]]

PLANNERS = ("astar", "llm-astar")


@dataclass
class PairResult:
    """Path between point ``i`` and point ``j`` (``i < j``)."""

    i: int
    j: int
    waypoints: List[Point]
    path: List[Point]
    length: float
    operations: int
    storage: int

    @property
    def reachable(self) -> bool:
        return bool(self.path)


@dataclass
class CoverageResult:
    """Output of :func:`plan_coverage`."""

    scene: str
    planner: str
    astar_config: AStarConfig
    point_names: List[str]
    points: List[Point]
    pairs: List[PairResult]
    distance_matrix: np.ndarray
    route: List[int]
    route_length: float
    final_path: List[Point]
    rejected_pois: List[POI] = field(default_factory=list)
    runtime_seconds: float = 0.0

    @property
    def total_operations(self) -> int:
        return sum(p.operations for p in self.pairs)

    @property
    def total_storage(self) -> int:
        return sum(p.storage for p in self.pairs)

    def summary(self) -> Dict[str, Any]:
        return {
            "scene": self.scene,
            "planner": self.planner,
            "astar": self.astar_config.to_dict(),
            "num_pois": len(self.points) - 1,
            "num_pairs": len(self.pairs),
            "route": list(self.route),
            "route_names": [self.point_names[k] for k in self.route],
            "route_length": self.route_length,
            "total_operations": self.total_operations,
            "total_storage": self.total_storage,
            "rejected_pois": [poi.name for poi in self.rejected_pois],
            "runtime_seconds": round(self.runtime_seconds, 1),
        }


def coverage_points(scene: Scene, grid: np.ndarray) -> Tuple[List[str], List[Point], List[POI]]:
    """Start point followed by the valid POIs; also returns the rejected POIs."""
    valid, rejected = scene.split_pois(grid)
    names = ["Start"] + [poi.name for poi in valid]
    points = [scene.start] + [poi.xy for poi in valid]
    return names, points, rejected


def pair_waypoints(
    points: Sequence[Point], waypoint_fn: Optional[WaypointFn] = None
) -> Dict[Tuple[int, int], List[Point]]:
    """Waypoint chains for all pairs ``i < j`` (just ``[start, goal]`` without an LLM)."""
    chains: Dict[Tuple[int, int], List[Point]] = {}
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            if waypoint_fn is None:
                chains[(i, j)] = [points[i], points[j]]
            else:
                chains[(i, j)] = waypoint_fn(points[i], points[j])
                log.info("waypoints %d -> %d: %s", i, j, chains[(i, j)])
    return chains


_WORKER_ASTAR: Optional[GridAStar] = None


def _init_worker(grid: np.ndarray, config: AStarConfig) -> None:
    global _WORKER_ASTAR
    _WORKER_ASTAR = GridAStar(grid, config)


def _solve_in_worker(task: Tuple[int, int, List[Point]]) -> Tuple[int, int, SearchResult]:
    i, j, chain = task
    assert _WORKER_ASTAR is not None
    return i, j, search_through_waypoints(_WORKER_ASTAR, chain)


def compute_pairwise_paths(
    grid: np.ndarray,
    chains: Dict[Tuple[int, int], List[Point]],
    config: AStarConfig,
    workers: int = 1,
) -> List[PairResult]:
    """Run (segmented) A* for every pair. Results do not depend on ``workers``."""
    tasks = [(i, j, chains[(i, j)]) for (i, j) in sorted(chains)]
    results: Dict[Tuple[int, int], PairResult] = {}

    def record(i: int, j: int, res: SearchResult) -> None:
        length = path_length(res.path) if res.found else math.inf
        results[(i, j)] = PairResult(i, j, chains[(i, j)], res.path, length, res.operations, res.storage)
        log.info(
            "[%d/%d] pair %d -> %d: length=%.3f operations=%d storage=%d",
            len(results), len(tasks), i, j, length, res.operations, res.storage,
        )

    if workers <= 1:
        astar = GridAStar(grid, config)
        for i, j, chain in tasks:
            record(i, j, search_through_waypoints(astar, chain))
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_init_worker,
            initargs=(grid, config),
        ) as pool:
            futures = [pool.submit(_solve_in_worker, task) for task in tasks]
            for future in as_completed(futures):
                record(*future.result())
    return [results[key] for key in sorted(results)]


def distance_matrix_from_pairs(num_points: int, pairs: Sequence[PairResult]) -> np.ndarray:
    matrix = np.full((num_points, num_points), np.inf)
    np.fill_diagonal(matrix, 0)
    for p in pairs:
        matrix[p.i, p.j] = matrix[p.j, p.i] = p.length
    return matrix


def assemble_path(route: Sequence[int], pairs: Sequence[PairResult]) -> List[Point]:
    """Concatenate the pairwise paths along ``route`` without repeating joints."""
    paths: Dict[Tuple[int, int], List[Point]] = {}
    for p in pairs:
        paths[(p.i, p.j)] = p.path
        paths[(p.j, p.i)] = p.path[::-1]
    full: List[Point] = []
    for k, (a, b) in enumerate(zip(route, route[1:])):
        segment = paths.get((a, b), [])
        full.extend(segment if k == 0 else segment[1:])
    return full


def plan_coverage(
    scene: Scene,
    planner: str = "astar",
    astar_config: Optional[AStarConfig] = None,
    waypoint_fn: Optional[WaypointFn] = None,
    workers: int = 1,
) -> CoverageResult:
    """Plan a prioritized coverage path over the scene's POIs.

    Args:
        scene: the scene to plan in.
        planner: ``"astar"`` (baseline) or ``"llm-astar"``.
        astar_config: A* settings; defaults to ``scene.planners[planner]``.
        waypoint_fn: waypoint generator, required for ``"llm-astar"`` (e.g.
            :class:`llm_pcpp.waypoints.LLMWaypointGenerator`).
        workers: number of processes for the pairwise A* searches.
    """
    if planner not in PLANNERS:
        raise ValueError(f"planner must be one of {PLANNERS}, got {planner!r}")
    if planner == "llm-astar" and waypoint_fn is None:
        raise ValueError("planner 'llm-astar' needs a waypoint_fn")

    t0 = time.perf_counter()
    grid = scene.build_grid()
    if grid[scene.start[1], scene.start[0]] > 0:
        log.warning("Start point %s lies inside an obstacle", scene.start)
    names, points, rejected = coverage_points(scene, grid)
    for poi in rejected:
        log.warning("Skipping %s at (%d, %d): outside the map or inside an obstacle", poi.name, poi.x, poi.y)
    if astar_config is None:
        astar_config = AStarConfig.from_mapping(scene.planners.get(planner, {}))

    chains = pair_waypoints(points, waypoint_fn if planner == "llm-astar" else None)
    pairs = compute_pairwise_paths(grid, chains, astar_config, workers=workers)
    unreachable = [(p.i, p.j) for p in pairs if not p.reachable]
    if unreachable:
        raise RuntimeError(f"No path found for point pairs {unreachable}; check the POIs and obstacles.")

    matrix = distance_matrix_from_pairs(len(points), pairs)
    route = solve_tsp(matrix, depot=0)
    return CoverageResult(
        scene=scene.name,
        planner=planner,
        astar_config=astar_config,
        point_names=names,
        points=points,
        pairs=pairs,
        distance_matrix=matrix,
        route=route,
        route_length=route_length(route, matrix),
        final_path=assemble_path(route, pairs),
        rejected_pois=rejected,
        runtime_seconds=time.perf_counter() - t0,
    )
