import itertools
import math

import numpy as np
import pytest

from llm_pcpp.tsp import route_length, solve_tsp


def euclidean_matrix(points):
    return np.array([[math.dist(a, b) for b in points] for a in points])


def test_square_tour():
    points = [(0, 0), (100, 100), (100, 0), (0, 100)]
    matrix = euclidean_matrix(points)
    route = solve_tsp(matrix)
    assert route[0] == route[-1] == 0
    assert sorted(route[:-1]) == [0, 1, 2, 3]
    assert route_length(route, matrix) == pytest.approx(400.0)


def test_tour_is_optimal_on_small_instance():
    rng = np.random.default_rng(1)
    points = [tuple(p) for p in rng.integers(0, 1000, size=(7, 2))]
    matrix = euclidean_matrix(points)
    route = solve_tsp(matrix)
    best = min(
        route_length((0,) + perm + (0,), matrix) for perm in itertools.permutations(range(1, 7))
    )
    # OR-Tools works on integer-truncated costs, so allow one unit per edge.
    assert route_length(route, matrix) <= best + 7


def test_unreachable_pairs_are_rejected():
    matrix = np.array([[0, 1, np.inf], [1, 0, 2], [np.inf, 2, 0]])
    with pytest.raises(ValueError):
        solve_tsp(matrix)
