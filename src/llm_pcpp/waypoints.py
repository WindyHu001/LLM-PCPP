"""LLM waypoint generation for LLM-A* with rectangular obstacles.

For every pair of points, the LLM is asked for a coarse list of waypoints
between them; A* then searches segment by segment through those waypoints
(:func:`llm_pcpp.astar.search_through_waypoints`).

The prompt and the response parsing are adapted from LLM-A* (Meng et al., 2024,
https://github.com/SilinMeng0510/llm-astar, MIT License, Copyright (c) 2024
Silin Meng; see ``third_party/llm-astar/LICENSE``). This is the version used for
the paper, in which obstacles are rectangles ``[x_min, y_min, x_max, y_max]``.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .llm import ChatModel

Point = Tuple[int, int]
log = logging.getLogger(__name__)

# The prompt used in the paper, byte for byte (tests/test_waypoints.py checks
# its SHA-256). Lines 1, 4 and the last line end with a space.
_WAYPOINT_PROMPT_LINES = [
    "",
    "Identify a path between the start and goal points to navigate around obstacles and find the shortest path to the goal. ",
    "Obstacles are represented as rectangular blocks defined by their bottom-left and top-right corners, given in the format [x_min, y_min, x_max, y_max].",
    "You need to complete: 1. Generate waypoints to guide the algorithm towards the goal. 2. Ensure that the target does not fall into the obstacle block area.",
    "Conclude your response with the generated path in the format \"Generated Path: [[x1, y1], [x2, y2], ...]\". ",
    "Warning!: Do not add any additional formats, follow 'Generated Path: [[x1, y1], [x2, y2],...]'",
    "Prompt: Please consider how to keep the length of the path short, for example, if a straight line can be reached, try to keep it straight as much as possible without generating too many additional waypoints",
    "",
    "Start Point: [5, 5]",
    "Goal Point: [20, 20]",
    "Obstacles: [[10, 10, 15, 15], [18, 5, 22, 8]]",
    "Generated Path: [[5, 5], [9, 9], [15, 16], [20, 20]]",
    "",
    "Start Point: [2, 3]",
    "Goal Point: [18, 8]",
    "Obstacles: [[10, 6, 14, 12], [6, 7, 9, 10]]",
    "Generated Path: [[2, 3], [9, 6], [18, 8]]",
    "",
    "Start Point: [40, 20]",
    "Goal Point: [10, 10]",
    "Obstacles: [[20, 15, 30, 25], [15, 5, 25, 10]]",
    "Generated Path: [[40, 20], [35, 12], [12, 10], [10, 10]]",
    "",
    "Start Point: [4, 4]",
    "Goal Point: [16, 16]",
    "Obstacles: [[8, 8, 12, 12], [5, 5, 7, 7]]",
    "Generated Path: [[4, 4], [7, 7], [13, 13], [16, 16]]",
    "",
    "Start Point: [25, 25]",
    "Goal Point: [4, 4]",
    "Obstacles: [[10, 10, 20, 20], [5, 5, 15, 15]]",
    "Generated Path: [[25, 25], [21, 9], [10, 6], [4, 4]]",
    "",
    "Start Point: {start}",
    "Goal Point: {goal}",
    "Obstacles: {obstacles}",
    "Generated Path: ",
    "",
]
WAYPOINT_PROMPT = "\n".join(_WAYPOINT_PROMPT_LINES)


def format_waypoint_prompt(start: Point, goal: Point, obstacles: Sequence[Sequence[int]]) -> str:
    """Fill the waypoint prompt for one start/goal pair.

    ``obstacles`` is a list of ``[x_min, y_min, x_max, y_max]`` rectangles
    (see :meth:`llm_pcpp.scene.Scene.obstacle_rectangles`).
    """
    return WAYPOINT_PROMPT.format(
        start=[int(start[0]), int(start[1])],
        goal=[int(goal[0]), int(goal[1])],
        obstacles=[[int(v) for v in rect] for rect in obstacles],
    )


def parse_waypoints(text: str) -> List[List[float]]:
    """Extract the last ``[[x, y], ...]`` list from an LLM response.

    Simple ``a - b.c`` expressions inside the list are evaluated, as in LLM-A*.
    Returns an empty list if no list can be parsed.
    """
    lists = re.findall(r"\[\[.*?\]\]", text or "", re.DOTALL)
    if not lists:
        return []
    candidate = re.sub(
        r"(\d+) - (\d+\.\d+)",
        lambda m: str(int(m.group(1)) - float(m.group(2))),
        lists[-1],
    )
    try:
        nodes = ast.literal_eval(candidate)
    except (ValueError, SyntaxError) as exc:
        log.warning("Could not parse waypoints from %r: %s", candidate, exc)
        return []
    return [list(node) for node in nodes if isinstance(node, (list, tuple)) and len(node) >= 2]


def _to_point(node: Sequence[Any]) -> Optional[Point]:
    try:
        x, y = float(node[0]), float(node[1])
    except (TypeError, ValueError, IndexError):
        return None
    return (int(round(x)), int(round(y)))


def _inside_any(point: Point, rectangles: Sequence[Sequence[int]]) -> bool:
    x, y = point
    return any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in rectangles)


def filter_waypoints(
    nodes: Sequence[Sequence[Any]],
    start: Point,
    goal: Point,
    obstacles: Sequence[Sequence[int]],
    width: int,
    height: int,
    grid: Optional[np.ndarray] = None,
) -> List[Point]:
    """Turn raw LLM waypoints into a valid waypoint chain from ``start`` to ``goal``.

    1. As in LLM-A*, drop waypoints that lie inside an obstacle rectangle
       (bounds inclusive) or within two cells of the map border, then make sure
       the chain starts at ``start`` and ends at ``goal``.
    2. If ``grid`` is given, also drop waypoints on blocked grid cells and
       restore ``start``/``goal`` at the ends again.
    """
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))

    chain: List[Point] = []
    for node in nodes:
        point = _to_point(node)
        if point is None or _inside_any(point, obstacles):
            continue
        if 1 < point[0] < width - 2 and 1 < point[1] < height - 2:
            chain.append(point)
    chain = _anchor(chain, start, goal)

    if grid is not None:
        chain = [p for p in chain if not grid[p[1], p[0]] > 0]
        chain = _anchor(chain, start, goal)
    return chain


def _anchor(chain: List[Point], start: Point, goal: Point) -> List[Point]:
    if not chain or chain[0] != start:
        chain.insert(0, start)
    if chain[-1] != goal:
        chain.append(goal)
    return chain


class WaypointCache:
    """JSON cache of raw LLM responses, keyed by model and prompt.

    Saving the raw responses makes LLM-A* runs replayable without API calls,
    and lets an interrupted run resume where it stopped.
    """

    VERSION = 1

    def __init__(self, path: "str | Path") -> None:
        self.path = Path(path)
        self._entries: Dict[str, Dict[str, Any]] = {}
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            self._entries = dict(data.get("entries", {}))

    @staticmethod
    def key(model: str, prompt: str) -> str:
        return hashlib.sha256(f"{model}\n{prompt}".encode("utf-8")).hexdigest()

    def get(self, model: str, prompt: str) -> Optional[str]:
        entry = self._entries.get(self.key(model, prompt))
        return None if entry is None else entry["response"]

    def put(self, model: str, prompt: str, response: str, start: Point, goal: Point) -> None:
        self._entries[self.key(model, prompt)] = {
            "model": model,
            "start": [int(start[0]), int(start[1])],
            "goal": [int(goal[0]), int(goal[1])],
            "response": response,
        }
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": self.VERSION, "entries": self._entries}
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1)
        os.replace(tmp, self.path)

    def __len__(self) -> int:
        return len(self._entries)


class LLMWaypointGenerator:
    """Query an LLM for the waypoints between two points of a scene."""

    def __init__(
        self,
        model: ChatModel,
        obstacles: Sequence[Sequence[int]],
        width: int,
        height: int,
        grid: Optional[np.ndarray] = None,
        cache: Optional[WaypointCache] = None,
    ) -> None:
        self.model = model
        self.obstacles = [list(rect) for rect in obstacles]
        self.width = width
        self.height = height
        self.grid = grid
        self.cache = cache
        self.num_queries = 0
        self.num_cache_hits = 0

    def raw_response(self, start: Point, goal: Point) -> str:
        prompt = format_waypoint_prompt(start, goal, self.obstacles)
        if self.cache is not None:
            cached = self.cache.get(self.model.name, prompt)
            if cached is not None:
                self.num_cache_hits += 1
                return cached
        response = self.model.complete(prompt)
        self.num_queries += 1
        if self.cache is not None:
            self.cache.put(self.model.name, prompt, response, start, goal)
        return response

    def __call__(self, start: Point, goal: Point) -> List[Point]:
        response = self.raw_response(start, goal)
        nodes = parse_waypoints(response)
        return filter_waypoints(
            nodes, start, goal, self.obstacles, self.width, self.height, self.grid
        )
