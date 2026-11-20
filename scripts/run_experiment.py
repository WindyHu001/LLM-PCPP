#!/usr/bin/env python
"""Run LLM-PCPP (LLM-A* + TSP) or the A* baseline on a scene.

Reproduce the A* baseline of Experiment 1 (no API key needed):

    python scripts/run_experiment.py --config configs/exp1_complex_outdoor.yaml --planner astar --workers 4

Run LLM-PCPP (needs OPENAI_API_KEY):

    python scripts/run_experiment.py --config configs/exp1_complex_outdoor.yaml --planner llm-astar --workers 4

Results are written to outputs/<scene>/<planner>/: summary.json, pairs.csv,
distance_matrix.csv, coverage_path.json and coverage_path.png. For llm-astar,
waypoints.json stores the raw LLM responses; running again with the same file
replays them without API calls.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))  # allow running from a source checkout without installing

from llm_pcpp import (  # noqa: E402
    AStarConfig,
    LLMWaypointGenerator,
    OpenAIChatModel,
    WaypointCache,
    __version__,
    load_scene,
    plan_coverage,
)
from llm_pcpp.planner import CoverageResult  # noqa: E402

log = logging.getLogger("llm_pcpp")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog=__doc__.split("\n\n", 1)[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True, help="scene config (YAML)")
    parser.add_argument("--planner", choices=("astar", "llm-astar"), default="llm-astar",
                        help="pairwise path planner (default: llm-astar)")
    parser.add_argument("--workers", type=int, default=1,
                        help="processes for the pairwise A* searches (default: 1)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="output folder (default: outputs/<scene>/<planner>)")
    parser.add_argument("--model", default=None,
                        help="LLM for the waypoints (default: from the config, else gpt-4o)")
    parser.add_argument("--waypoint-cache", type=Path, default=None,
                        help="JSON file with raw LLM responses to reuse and extend "
                             "(default: <output-dir>/waypoints.json)")
    parser.add_argument("--neighbor-order", choices=("axis_first", "diagonal_first"), default=None,
                        help="override the A* neighbour order from the config")
    parser.add_argument("--diagonal-cost", default=None,
                        help="override the A* diagonal cost from the config (a number or 'exact')")
    parser.add_argument("--no-plot", action="store_true", help="do not draw coverage_path.png")
    parser.add_argument("--quiet", action="store_true", help="only print the final summary")
    return parser.parse_args(argv)


def astar_config_for(scene, args) -> AStarConfig:
    settings = dict(scene.planners.get(args.planner, {}))
    if args.neighbor_order:
        settings["neighbor_order"] = args.neighbor_order
    if args.diagonal_cost:
        settings["diagonal_cost"] = args.diagonal_cost
    return AStarConfig.from_mapping(settings)


def write_outputs(result: CoverageResult, out_dir: Path, summary: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = result.point_names
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    with (out_dir / "pairs.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["i", "j", "from", "to", "length", "operations", "storage", "waypoints"])
        for p in result.pairs:
            writer.writerow([p.i, p.j, names[p.i], names[p.j], repr(p.length), p.operations,
                             p.storage, json.dumps([list(w) for w in p.waypoints])])

    with (out_dir / "distance_matrix.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + names)
        for name, row in zip(names, result.distance_matrix.tolist()):
            writer.writerow([name] + [repr(v) for v in row])

    coverage = {
        "points": {name: list(xy) for name, xy in zip(names, result.points)},
        "route": result.route,
        "route_names": [names[k] for k in result.route],
        "path": [list(xy) for xy in result.final_path],
    }
    (out_dir / "coverage_path.json").write_text(json.dumps(coverage) + "\n")


def print_report(result: CoverageResult, paper: dict) -> None:
    rows = [
        ("Path length (TSP)", f"{result.route_length:.2f}", paper.get("path_length")),
        ("Operations", str(result.total_operations), paper.get("operations")),
        ("Storage", str(result.total_storage), paper.get("storage")),
    ]
    print(f"\nScene: {result.scene} | planner: {result.planner} | "
          f"{len(result.points) - 1} POIs, {len(result.pairs)} pairs | {result.runtime_seconds:.0f} s")
    print(f"{'':20s}{'this run':>14s}{'paper' if paper else '':>14s}")
    for label, value, reference in rows:
        ref = "" if reference is None else (f"{reference:.2f}" if isinstance(reference, float) else str(reference))
        print(f"{label:20s}{value:>14s}{ref:>14s}")
    print(f"TSP tour: {result.route}")
    if paper.get("route"):
        print(f"Paper tour: {paper['route']}")


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    scene = load_scene(args.config)
    out_dir = args.output_dir or ROOT / "outputs" / scene.name / args.planner
    config = astar_config_for(scene, args)
    log.info("Scene %s (%d x %d, %d POIs), planner %s, A* %s",
             scene.name, scene.width, scene.height, len(scene.pois), args.planner, config.to_dict())

    waypoint_fn = None
    run_info = {}
    if args.planner == "llm-astar":
        model_name = args.model or scene.planners.get("llm-astar", {}).get("model", "gpt-4o")
        cache = WaypointCache(args.waypoint_cache or out_dir / "waypoints.json")
        log.info("LLM %s, waypoint cache %s (%d saved responses)", model_name, cache.path, len(cache))
        waypoint_fn = LLMWaypointGenerator(
            OpenAIChatModel(model_name), scene.obstacle_rectangles(), scene.width, scene.height,
            scene.build_grid(), cache,
        )
        run_info = {"llm_model": model_name, "waypoint_cache": str(cache.path)}

    try:
        result = plan_coverage(scene, args.planner, astar_config=config,
                               waypoint_fn=waypoint_fn, workers=args.workers)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        if "OPENAI_API_KEY" in str(exc):
            print("hint: --waypoint-cache replays saved LLM responses without API calls", file=sys.stderr)
        return 1

    import numpy
    import ortools

    summary = result.summary()
    summary.update(run_info)
    if waypoint_fn is not None:
        summary["llm_queries"] = waypoint_fn.num_queries
        summary["llm_cache_hits"] = waypoint_fn.num_cache_hits
    paper = scene.paper_results.get(args.planner, {})
    if paper:
        summary["paper"] = paper
    summary["environment"] = {
        "llm_pcpp": __version__,
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "ortools": ortools.__version__,
    }
    write_outputs(result, out_dir, summary)

    if not args.no_plot:
        import matplotlib

        matplotlib.use("Agg")
        from llm_pcpp.visualization import plot_scene

        fig, _ = plot_scene(scene, points=result.points, point_names=result.point_names,
                            path=result.final_path)
        fig.savefig(out_dir / "coverage_path.png", dpi=150, bbox_inches="tight")

    print_report(result, paper)
    print(f"Outputs: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
