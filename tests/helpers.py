from pathlib import Path

from llm_pcpp.scene import Scene

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


class FakeModel:
    """Stand-in for an LLM: returns a fixed response (or one computed from the prompt)."""

    def __init__(self, response, name="fake-llm"):
        self.response = response
        self.name = name
        self.calls = 0

    def complete(self, prompt):
        self.calls += 1
        return self.response(prompt) if callable(self.response) else self.response


def tiny_scene():
    """A 60 x 40 map with two obstacles and four POIs."""
    return Scene.from_dict(
        {
            "name": "tiny",
            "grid": {"width": 60, "height": 40},
            "start": [30, 0],
            "obstacle_classes": [
                {"name": "Shelves", "label": 1, "regions": [[[10, 25], [8, 30]]]},
                {"name": "Hazard", "label": 2, "regions": [[[35, 50], [5, 20]]]},
            ],
            "pois": [
                {"name": "A", "x": 5, "y": 35},
                {"name": "B", "x": 55, "y": 35},
                {"name": "C", "x": 55, "y": 2},
                {"name": "D", "x": 30, "y": 25},
                {"name": "Blocked", "x": 15, "y": 15},
            ],
            "planners": {"astar": {"neighbor_order": "axis_first", "diagonal_cost": 1.414}},
        }
    )
