import numpy as np
import pytest

import reference_impl as ref
from llm_pcpp.astar import AStarConfig, GridAStar, path_length, search_through_waypoints

PAPER_VARIANTS = [
    (AStarConfig("diagonal_first", "exact"), ref.astar_diagonal_first_exact),
    (AStarConfig("axis_first", 1.414), ref.astar_axis_first_1414),
]


def random_grid(rng, height=48, width=64, num_rects=8):
    grid = np.zeros((height, width), dtype=np.uint8)
    for label in rng.integers(1, 5, size=num_rects):
        x0, y0 = rng.integers(0, width - 4), rng.integers(0, height - 4)
        x1, y1 = x0 + rng.integers(2, 20), y0 + rng.integers(2, 16)
        grid[y0:y1, x0:x1] = label
    return grid


def random_free_cell(rng, grid):
    free = np.argwhere(grid == 0)
    y, x = free[rng.integers(len(free))]
    return (int(x), int(y))


@pytest.mark.parametrize("config,reference", PAPER_VARIANTS)
def test_matches_paper_implementation(config, reference):
    """Same path, operations and storage as the code used for the paper."""
    rng = np.random.default_rng(42)
    for _ in range(40):
        grid = random_grid(rng)
        start, goal = random_free_cell(rng, grid), random_free_cell(rng, grid)
        expected_path, stats = reference(start, goal, grid)
        result = GridAStar(grid, config).search(start, goal)
        assert result.path == expected_path
        assert result.operations == stats["operation"]
        assert result.storage == stats["storage"]
        assert path_length(result.path) == ref.calculate_path_length(expected_path)


def test_open_grid_path_is_optimal():
    grid = np.zeros((20, 20), dtype=np.uint8)
    result = GridAStar(grid).search((0, 0), (5, 3))
    assert result.path[0] == (0, 0) and result.path[-1] == (5, 3)
    assert path_length(result.path) == pytest.approx(3 * 1.414 + 2)


def test_path_avoids_obstacles_and_is_connected():
    grid = np.zeros((30, 30), dtype=np.uint8)
    grid[0:25, 10:12] = 1  # wall with a gap at the top
    result = GridAStar(grid).search((2, 2), (25, 2))
    assert result.found
    for (x0, y0), (x1, y1) in zip(result.path, result.path[1:]):
        assert max(abs(x1 - x0), abs(y1 - y0)) == 1
    assert all(grid[y, x] == 0 for x, y in result.path)
    assert max(y for _, y in result.path) >= 25


def test_unreachable_goal():
    grid = np.zeros((20, 20), dtype=np.uint8)
    grid[8:13, 8] = grid[8:13, 12] = 1
    grid[8, 8:13] = grid[12, 8:13] = 1  # closed box around (10, 10)
    result = GridAStar(grid).search((1, 1), (10, 10))
    assert not result.found
    assert result.operations > 0


def test_start_equals_goal():
    grid = np.zeros((5, 5), dtype=np.uint8)
    result = GridAStar(grid).search((2, 2), (2, 2))
    assert result.path == [(2, 2)]
    assert (result.operations, result.storage) == (1, 1)


def test_segmented_search_sums_counters():
    grid = np.zeros((30, 30), dtype=np.uint8)
    astar = GridAStar(grid)
    first = astar.search((0, 0), (10, 20))
    second = astar.search((10, 20), (25, 5))
    chained = search_through_waypoints(astar, [(0, 0), (10, 20), (25, 5)])
    assert chained.path == first.path + second.path[1:]
    assert chained.operations == first.operations + second.operations
    assert chained.storage == first.storage + second.storage


def test_config_validation():
    with pytest.raises(ValueError):
        AStarConfig(neighbor_order="random")
    assert AStarConfig(diagonal_cost="1.5").diagonal_step == 1.5
    assert AStarConfig(diagonal_cost="exact").diagonal_step == pytest.approx(2**0.5)
