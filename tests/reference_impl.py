"""Verbatim copies of the code that produced the paper's numbers.

They are used as references in the tests: the package must reproduce their
outputs exactly. Only comments and the function names were changed.
"""

import ast
import heapq
import math
import re


def astar_diagonal_first_exact(start, goal, grid):
    """A* of the Exp. 1 / Exp. 3 baselines (diagonal neighbours first, sqrt(2) cost)."""

    def heuristic(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    neighbors = [(-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)]

    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}
    closed_set = set()

    operation = 0
    storage = 1

    while open_set:
        operation += 1
        current_f, current_g, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, {"operation": operation, "storage": storage}

        if current in closed_set:
            continue
        closed_set.add(current)

        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            x, y = neighbor
            if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]:
                if grid[y, x] > 0:
                    continue
                tentative_g = current_g + math.sqrt(dx**2 + dy**2)
                if neighbor in g_score and tentative_g >= g_score[neighbor]:
                    continue
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                storage = len(g_score)
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, tentative_g, neighbor))

    return [], {"operation": operation, "storage": storage}


def astar_axis_first_1414(start, goal, grid):
    """A* of the Exp. 2 baseline and of all LLM-A* runs (axis neighbours first, 1.414 cost)."""

    def heuristic(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}
    closed_set = set()

    operation = 0
    storage = 1

    while open_set:
        operation += 1
        current_f, current_g, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, {"operation": operation, "storage": storage}

        if current in closed_set:
            continue
        closed_set.add(current)

        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            x, y = neighbor
            if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]:
                if grid[y, x] > 0:
                    continue
                tentative_g = current_g + (1.414 if dx != 0 and dy != 0 else 1)
                if neighbor in g_score and tentative_g >= g_score[neighbor]:
                    continue
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                storage = len(g_score)
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, tentative_g, neighbor))

    return [], {"operation": operation, "storage": storage}


def calculate_path_length(path):
    total_length = 0
    for i in range(1, len(path)):
        dx = abs(path[i][0] - path[i - 1][0])
        dy = abs(path[i][1] - path[i - 1][1])
        if dx != 0 and dy != 0:
            total_length += 1.414
        else:
            total_length += 1
    return total_length


def list_parse(text):
    """llmastar.utils.list_parse (LLM-A*)."""
    lists = re.findall(r"\[\[.*?\]\]", text, re.DOTALL)
    if lists:
        last_list_str = lists[-1]
        evaluated_list_str = re.sub(
            r"(\d+ - \d+\.\d+)", lambda x: str(eval(x.group(1))), last_list_str
        )
        try:
            return ast.literal_eval(evaluated_list_str)
        except Exception as e:
            return f"Error evaluating list: {e}"
    else:
        return [[0, 0], [0, 0]]


def llm_waypoints(response, start, goal, obstacles, range_x, range_y, grid):
    """Waypoint filtering of the paper runs (LLM-A* filter, then the grid filter)."""
    # LLMAStar._initialize_llm_paths (range limits are decremented in _initialize_parameters)
    range_x = [range_x[0], range_x[1] - 1]
    range_y = [range_y[0], range_y[1] - 1]

    def is_point_in_obstacle(point, obstacles):
        x, y = point
        for obs in obstacles:
            x_min, y_min, x_max, y_max = obs
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return True
        return False

    nodes = list_parse(response)
    target_list = [
        (node[0], node[1])
        for node in nodes
        if not is_point_in_obstacle(node, obstacles)
        and range_x[0] + 1 < node[0] < range_x[1] - 1
        and range_y[0] + 1 < node[1] < range_y[1] - 1
    ]
    if not target_list or target_list[0] != tuple(start):
        target_list.insert(0, tuple(start))
    if not target_list or target_list[-1] != tuple(goal):
        target_list.append(tuple(goal))

    # get_llm_waypoints in tsp_storage_LLM_astar.py
    waypoints = [p for p in target_list if not grid[p[1], p[0]] > 0]
    if waypoints[0] != start:
        waypoints.insert(0, start)
    if waypoints[-1] != goal:
        waypoints.append(goal)
    return waypoints
