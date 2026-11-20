"""LLM-based identification of points of interest (paper Sec. 3.2).

For the paper, the POIs were generated interactively with ChatGPT (GPT-4o for
Exp. 1, o1 for Exp. 2 and 3) and are stored verbatim in ``configs/``. This
module runs the same step through the API for new scenes: the LLM receives a
text description of the semantic grid map and the task guidance, proposes
POIs, and POIs outside the map or inside an obstacle are discarded. The prompt
follows the description in the paper; it is not the exact chat prompt used
for the published POIs.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import numpy as np

from .llm import ChatModel
from .scene import POI, Scene

POI_PROMPT = """You are helping a mobile robot plan a prioritized coverage path.

Map: a 2D grid of {width} x {height} cells. The origin (0, 0) is the bottom-left corner; x increases to the right and y increases upwards.
Obstacles are axis-aligned rectangles [x_min, y_min, x_max, y_max], grouped by type:
{obstacles}

The robot starts at {start}.
Task guidance: {guidance}

Propose {num_pois} points of interest (POIs) that the robot should visit to fulfil the task guidance. Every POI must be inside the map and outside every obstacle.
Answer with one POI per line, formatted as (x, y, "short reason"), and nothing else."""

_POI_PATTERN = re.compile(
    r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*(?:,\s*[\"'“]?(.*?)[\"'”]?\s*)?\)"
)


def build_poi_prompt(scene: Scene, num_pois: int, guidance: Optional[str] = None) -> str:
    obstacles = "\n".join(
        f"- {c.name}: {[r.as_xyxy() for r in c.regions]}" for c in scene.obstacle_classes
    )
    return POI_PROMPT.format(
        width=scene.width,
        height=scene.height,
        obstacles=obstacles,
        start=list(scene.start),
        guidance=guidance or scene.task_guidance or "Avoid obstacles and cover most of the map.",
        num_pois=num_pois,
    )


def parse_pois(text: str) -> List[POI]:
    """Parse ``(x, y, "reason")`` tuples from an LLM answer."""
    pois: List[POI] = []
    for match in _POI_PATTERN.finditer(text or ""):
        x, y = int(round(float(match.group(1)))), int(round(float(match.group(2))))
        pois.append(POI(name=f"POI_{len(pois)}", x=x, y=y, note=(match.group(3) or "").strip()))
    return pois


def validate_pois(
    scene: Scene, pois: List[POI], grid: Optional[np.ndarray] = None
) -> Tuple[List[POI], List[POI]]:
    """Keep POIs inside the map, outside obstacles and not duplicated; renumber them."""
    grid = scene.build_grid() if grid is None else grid
    valid: List[POI] = []
    rejected: List[POI] = []
    seen = {scene.start}
    for poi in pois:
        if scene.in_bounds(poi.x, poi.y) and grid[poi.y, poi.x] == 0 and poi.xy not in seen:
            seen.add(poi.xy)
            valid.append(POI(name=f"POI_{len(valid)}", x=poi.x, y=poi.y, note=poi.note))
        else:
            rejected.append(poi)
    return valid, rejected


def identify_pois(
    scene: Scene, model: ChatModel, num_pois: int, guidance: Optional[str] = None
) -> Tuple[List[POI], List[POI], str]:
    """Ask the LLM for POIs. Returns ``(valid, rejected, raw_answer)``."""
    answer = model.complete(build_poi_prompt(scene, num_pois, guidance))
    valid, rejected = validate_pois(scene, parse_pois(answer))
    return valid, rejected, answer
