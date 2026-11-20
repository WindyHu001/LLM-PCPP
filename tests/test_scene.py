import numpy as np
import pytest

from helpers import CONFIG_DIR
from llm_pcpp.scene import Region, Scene, load_scene

CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))

# Obstacle lists sent to the LLM in the paper runs (format_obstacles(...) in the
# original scripts), in the original order.
PAPER_PROMPT_OBSTACLES = {
    "exp1_complex_outdoor": [
        [0, 650, 450, 2160], [3000, 0, 3840, 1400],
        [1700, 400, 2100, 1100], [400, 0, 1100, 700], [2300, 1550, 3840, 2160],
        [1900, 1300, 2100, 1600], [2100, 1200, 2300, 1500], [2400, 700, 2800, 1500],
        [0, 0, 400, 650], [450, 800, 1000, 1900],
    ],
    "exp2_simple_outdoor": [
        [1900, 1240, 2200, 1600], [2200, 1020, 3024, 1790], [2120, 90, 3024, 930],
        [0, 590, 700, 1650],
        [1200, 530, 1850, 1190],
    ],
    "exp3_indoor_warehouse": [
        [100, 100, 500, 800], [700, 100, 1100, 800], [100, 1000, 500, 1700], [700, 1000, 1100, 1700],
        [1300, 800, 1600, 1400],
        [2000, 100, 2800, 600],
        [2500, 1400, 2900, 1800],
    ],
}

PAPER_SCENES = {
    # name: (width, height, start, number of POIs)
    "exp1_complex_outdoor": (3840, 2160, (1920, 0), 15),
    "exp2_simple_outdoor": (3024, 1890, (1512, 0), 12),
    "exp3_indoor_warehouse": (3024, 1890, (1512, 0), 13),
}


def test_all_paper_configs_present():
    assert {path.stem for path in CONFIGS} >= set(PAPER_SCENES)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_config_is_valid(path):
    scene = load_scene(path)
    grid = scene.build_grid()
    valid, rejected = scene.split_pois(grid)
    assert not rejected, f"POIs inside obstacles: {rejected}"
    assert grid[scene.start[1], scene.start[0]] == 0
    assert len({poi.xy for poi in valid}) == len(valid)


@pytest.mark.parametrize("name", sorted(PAPER_SCENES))
def test_paper_scene_matches_original_scripts(name):
    scene = load_scene(CONFIG_DIR / f"{name}.yaml")
    width, height, start, num_pois = PAPER_SCENES[name]
    assert (scene.width, scene.height, scene.start, len(scene.pois)) == (width, height, start, num_pois)
    assert scene.obstacle_rectangles() == PAPER_PROMPT_OBSTACLES[name]
    assert set(scene.planners) == {"astar", "llm-astar"}
    assert set(scene.paper_results) == {"astar", "llm-astar"}


def test_grid_painting_order():
    scene = Scene.from_dict(
        {
            "grid": {"width": 10, "height": 10},
            "start": [0, 0],
            "obstacle_classes": [
                {"name": "B", "label": 2, "regions": [[[0, 6], [0, 6]]]},
                {"name": "A", "label": 1, "regions": [[[4, 10], [4, 10]]]},
            ],
        }
    )
    grid = scene.build_grid()
    assert grid[5, 5] == 2  # overlap: larger label painted last
    assert grid[9, 9] == 1 and grid[0, 0] == 2 and grid[0, 9] == 0
    assert int((grid > 0).sum()) == 36 + 36 - 4


def test_region_validation():
    assert Region.from_ranges([[1, 5], [2, 8]]).as_xyxy() == [1, 2, 5, 8]
    with pytest.raises(ValueError):
        Region.from_ranges([[5, 1], [2, 8]])
    with pytest.raises(ValueError):
        Scene.from_dict(
            {"grid": {"width": 10, "height": 10}, "start": [0, 0],
             "obstacle_classes": [{"name": "X", "label": 1, "regions": [[[0, 11], [0, 5]]]}]}
        )
    with pytest.raises(ValueError):
        Scene.from_dict({"grid": {"width": 10, "height": 10}, "start": [10, 0]})


def test_grid_dtype_and_shape():
    scene = load_scene(CONFIG_DIR / "exp1_complex_outdoor.yaml")
    grid = scene.build_grid()
    assert grid.shape == (2160, 3840) and grid.dtype == np.uint8
