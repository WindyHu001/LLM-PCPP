"""Traveling Salesman Problem over the POI distance matrix (Google OR-Tools).

As in the paper, the first solution is built with the ``PATH_CHEAPEST_ARC``
heuristic and refined by OR-Tools' default local search. Arc costs are the
pairwise path lengths truncated to integers; the returned route length uses
the exact (floating-point) distances.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


def solve_tsp(distance_matrix: Sequence[Sequence[float]], depot: int = 0) -> List[int]:
    """Return a closed tour ``[depot, ..., depot]`` visiting every node once."""
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    matrix = np.asarray(distance_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("distance_matrix must be square")
    if not np.isfinite(matrix).all():
        raise ValueError("distance_matrix contains unreachable pairs (inf); TSP cannot be solved")

    n = matrix.shape[0]
    costs = [[int(value) for value in row] for row in matrix.tolist()]

    manager = pywrapcp.RoutingIndexManager(n, 1, depot)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index: int, to_index: int) -> int:
        return costs[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    transit_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_index)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(params)
    if solution is None:
        raise RuntimeError("OR-Tools did not find a TSP solution")

    route = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        route.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    route.append(manager.IndexToNode(index))
    return route


def route_length(route: Sequence[int], distance_matrix: Sequence[Sequence[float]]) -> float:
    """Total length of a route given as a sequence of node indices."""
    matrix = np.asarray(distance_matrix, dtype=float)
    total = 0
    for a, b in zip(route, route[1:]):
        total += matrix[a, b]
    return float(total)
