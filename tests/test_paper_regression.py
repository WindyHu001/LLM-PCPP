"""Single path queries checked against the paper's run logs (paper_logs/exp2_*).

The full experiments take minutes per planner; reproduce them with
``scripts/run_experiment.py``. These two queries take a few seconds.
"""

import pytest

from helpers import CONFIG_DIR
from llm_pcpp.astar import AStarConfig, GridAStar, path_length, search_through_waypoints
from llm_pcpp.scene import load_scene

POI_10 = (2970, 50)
POI_11 = (2970, 1850)


@pytest.fixture(scope="module")
def exp2():
    scene = load_scene(CONFIG_DIR / "exp2_simple_outdoor.yaml")
    return scene, scene.build_grid()


def test_exp2_astar_pair_matches_log(exp2):
    # paper_logs/exp2_simple_outdoor.txt: "POI 11 to POI 12 is {'operation': 1182978, 'storage': 837981}",
    # distance matrix entry 3642.554
    scene, grid = exp2
    astar = GridAStar(grid, AStarConfig.from_mapping(scene.planners["astar"]))
    result = astar.search(POI_10, POI_11)
    assert (result.operations, result.storage) == (1182978, 837981)
    assert path_length(result.path) == pytest.approx(3642.554, abs=1e-6)


def test_exp2_llm_astar_pair_matches_log(exp2):
    # Same pair with the GPT-4o waypoints recorded in the log:
    # "POI 11 to POI 12 is {'operation': 965893, 'storage': 745223, 'length': 5291.22799999999}"
    scene, grid = exp2
    astar = GridAStar(grid, AStarConfig.from_mapping(scene.planners["llm-astar"]))
    result = search_through_waypoints(astar, [POI_10, (2970, 1000), POI_11])
    assert (result.operations, result.storage) == (965893, 745223)
    assert path_length(result.path) == pytest.approx(5291.22799999999, abs=1e-9)
