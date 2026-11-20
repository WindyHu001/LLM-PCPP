from helpers import FakeModel, tiny_scene
from llm_pcpp.poi import build_poi_prompt, identify_pois, parse_pois, validate_pois

ANSWER = """(5, 35, "Top-left corner coverage.")
(55, 35, "Top-right corner")
(15, 15, "Inside the shelves - invalid")
(70, 10, "Outside the map - invalid")
(55, 35, "Duplicate")
(30.4, 25, "Between the obstacles.")"""


def test_prompt_mentions_map_and_guidance():
    prompt = build_poi_prompt(tiny_scene(), 6, "Inspect the hazard area closely.")
    assert "60 x 40 cells" in prompt
    assert "- Shelves: [[10, 8, 25, 30]]" in prompt
    assert "Inspect the hazard area closely." in prompt
    assert "Propose 6 points of interest" in prompt


def test_parse_and_validate():
    scene = tiny_scene()
    parsed = parse_pois(ANSWER)
    assert [(p.x, p.y) for p in parsed] == [(5, 35), (55, 35), (15, 15), (70, 10), (55, 35), (30, 25)]
    assert parsed[0].note == "Top-left corner coverage."
    valid, rejected = validate_pois(scene, parsed)
    assert [(p.name, p.x, p.y) for p in valid] == [("POI_0", 5, 35), ("POI_1", 55, 35), ("POI_2", 30, 25)]
    assert len(rejected) == 3


def test_identify_pois_with_fake_model():
    model = FakeModel(ANSWER)
    valid, rejected, answer = identify_pois(tiny_scene(), model, 6)
    assert model.calls == 1 and answer == ANSWER
    assert len(valid) == 3 and len(rejected) == 3
