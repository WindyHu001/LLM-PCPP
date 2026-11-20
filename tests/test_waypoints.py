import hashlib

import numpy as np
import pytest

import reference_impl as ref
from llm_pcpp.scene import load_scene
from llm_pcpp.waypoints import (
    WAYPOINT_PROMPT,
    LLMWaypointGenerator,
    WaypointCache,
    filter_waypoints,
    format_waypoint_prompt,
    parse_waypoints,
)
from helpers import CONFIG_DIR, FakeModel

# SHA-256 of the LLM-A* prompt used for the paper (`standard_gpt`).
PAPER_PROMPT_SHA256 = "f088b6e1ff141741fd8395483e7069762d40370d353ec966836e6521d412eb24"


def test_prompt_is_the_paper_prompt():
    assert hashlib.sha256(WAYPOINT_PROMPT.encode("utf-8")).hexdigest() == PAPER_PROMPT_SHA256


def test_prompt_formatting():
    prompt = format_waypoint_prompt((1920, 0), (1493, 1715), [[0, 650, 450, 2160], [3000, 0, 3840, 1400]])
    assert prompt.endswith(
        "Start Point: [1920, 0]\n"
        "Goal Point: [1493, 1715]\n"
        "Obstacles: [[0, 650, 450, 2160], [3000, 0, 3840, 1400]]\n"
        "Generated Path: \n"
    )


@pytest.mark.parametrize(
    "text",
    [
        "Generated Path: [[1920, 0], [1500, 1200], [1493, 1715]]",
        "Some reasoning [[1, 2], [3, 4]] then Generated Path: [[5, 6], [7, 8]]",
        "Generated Path: [[10 - 2.5, 3], [4, 5]]",
        "Generated Path: [[0, 0],\n [12, 30]]",
    ],
)
def test_parse_matches_llm_astar(text):
    assert parse_waypoints(text) == ref.list_parse(text)


def test_parse_without_list():
    # LLM-A* falls back to [[0, 0], [0, 0]], which the filter always removes.
    assert parse_waypoints("I cannot find a path.") == []
    assert parse_waypoints("") == []


def test_filter_matches_paper_runs():
    scene = load_scene(CONFIG_DIR / "exp1_complex_outdoor.yaml")
    grid = scene.build_grid()
    obstacles = scene.obstacle_rectangles()
    rng = np.random.default_rng(0)
    points = [scene.start] + [poi.xy for poi in scene.pois]
    for _ in range(200):
        i, j = rng.choice(len(points), size=2, replace=False)
        start, goal = points[i], points[j]
        nodes = [[int(rng.integers(-5, scene.width + 5)), int(rng.integers(-5, scene.height + 5))]
                 for _ in range(rng.integers(0, 6))]
        response = f"Generated Path: {nodes}" if nodes else "no path"
        expected = ref.llm_waypoints(
            response, start, goal, obstacles, [0, scene.width], [0, scene.height], grid
        )
        got = filter_waypoints(
            parse_waypoints(response), start, goal, obstacles, scene.width, scene.height, grid
        )
        assert got == expected


def test_filter_drops_invalid_points():
    obstacles = [[10, 10, 20, 20]]
    chain = filter_waypoints([[15, 15], [30, 5], [1, 30], [40, 40]], (5, 5), (45, 45), obstacles, 50, 50)
    assert chain == [(5, 5), (30, 5), (40, 40), (45, 45)]


def test_generator_and_cache(tmp_path):
    scene = load_scene(CONFIG_DIR / "exp2_simple_outdoor.yaml")
    grid = scene.build_grid()
    cache_file = tmp_path / "waypoints.json"
    model = FakeModel("Generated Path: [[2970, 50], [2970, 1000], [2970, 1850]]")

    generator = LLMWaypointGenerator(model, scene.obstacle_rectangles(), scene.width, scene.height,
                                     grid, WaypointCache(cache_file))
    first = generator((2970, 50), (2970, 1850))
    assert first == [(2970, 50), (2970, 1000), (2970, 1850)]
    assert model.calls == 1

    replay_model = FakeModel("this response must not be used")
    replay = LLMWaypointGenerator(replay_model, scene.obstacle_rectangles(), scene.width, scene.height,
                                  grid, WaypointCache(cache_file))
    assert replay((2970, 50), (2970, 1850)) == first
    assert replay_model.calls == 0
    assert replay.num_cache_hits == 1
