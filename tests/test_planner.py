import re

import pytest

from helpers import FakeModel, tiny_scene
from llm_pcpp.astar import AStarConfig, path_length
from llm_pcpp.planner import plan_coverage
from llm_pcpp.waypoints import LLMWaypointGenerator


def assert_valid_coverage(result, scene):
    grid = scene.build_grid()
    n = len(result.points)
    assert result.route[0] == result.route[-1] == 0
    assert sorted(result.route[:-1]) == list(range(n))
    path = result.final_path
    assert path[0] == scene.start and path[-1] == scene.start
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        assert max(abs(x1 - x0), abs(y1 - y0)) == 1
    assert all(grid[y, x] == 0 for x, y in path)
    visited = set(path)
    assert all(p in visited for p in result.points)
    assert path_length(path) == pytest.approx(result.route_length)
    assert result.total_operations == sum(p.operations for p in result.pairs)
    assert len(result.pairs) == n * (n - 1) // 2


def test_astar_pipeline():
    scene = tiny_scene()
    result = plan_coverage(scene, "astar")
    assert [poi.name for poi in result.rejected_pois] == ["Blocked"]
    assert result.point_names == ["Start", "A", "B", "C", "D"]
    assert (result.distance_matrix == result.distance_matrix.T).all()
    assert_valid_coverage(result, scene)


def test_llm_astar_pipeline_with_fake_llm():
    scene = tiny_scene()

    def respond(prompt):
        # Detour through the top-middle of the map for every pair.
        start = re.search(r"Start Point: \[(\d+), (\d+)\]\nGoal", prompt).groups()
        goal = re.search(r"Goal Point: \[(\d+), (\d+)\]\nObstacles", prompt).groups()
        return f"Generated Path: [[{start[0]}, {start[1]}], [30, 36], [{goal[0]}, {goal[1]}]]"

    model = FakeModel(respond)
    generator = LLMWaypointGenerator(model, scene.obstacle_rectangles(), scene.width, scene.height,
                                     scene.build_grid())
    result = plan_coverage(scene, "llm-astar", astar_config=AStarConfig(), waypoint_fn=generator)
    assert model.calls == 10
    assert all((30, 36) in pair.waypoints for pair in result.pairs)
    assert all((30, 36) in pair.path for pair in result.pairs)
    assert_valid_coverage(result, scene)


def test_llm_astar_requires_waypoints():
    with pytest.raises(ValueError):
        plan_coverage(tiny_scene(), "llm-astar")


def test_plot_coverage(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from llm_pcpp.visualization import block_labels, plot_scene

    scene = tiny_scene()
    result = plan_coverage(scene, "astar")
    fig, _ = plot_scene(scene, points=result.points, point_names=result.point_names,
                        path=result.final_path, block=5)
    fig.savefig(tmp_path / "coverage.png")
    assert (tmp_path / "coverage.png").stat().st_size > 0
    labels = block_labels(scene.build_grid(), 5)
    assert labels.shape == (8, 12) and set(labels.ravel().tolist()) == {0, 1, 2}


def test_workers_do_not_change_results():
    scene = tiny_scene()
    serial = plan_coverage(scene, "astar", workers=1)
    parallel = plan_coverage(scene, "astar", workers=2)
    assert serial.route == parallel.route
    assert serial.final_path == parallel.final_path
    assert [(p.i, p.j, p.length, p.operations, p.storage) for p in serial.pairs] == [
        (p.i, p.j, p.length, p.operations, p.storage) for p in parallel.pairs
    ]
