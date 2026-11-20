#!/usr/bin/env python
"""Draw a scene's semantic grid map with its start point and POIs (paper Fig. 1b, 2b, 3).

    python scripts/plot_scene.py --config configs/exp1_complex_outdoor.yaml
    python scripts/plot_scene.py --config configs/exp2_simple_outdoor.yaml --output exp2_map.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))  # allow running from a source checkout without installing

from llm_pcpp import load_scene  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, required=True, help="scene config (YAML)")
    parser.add_argument("--output", type=Path, default=None,
                        help="image file (default: outputs/<scene>/scene.png)")
    parser.add_argument("--block", type=int, default=20,
                        help="size of the drawn grid cells in map cells (default: 20)")
    parser.add_argument("--no-labels", action="store_true", help="do not write POI names")
    parser.add_argument("--show", action="store_true", help="open a window instead of only saving")
    args = parser.parse_args(argv)

    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from llm_pcpp.visualization import plot_scene

    scene = load_scene(args.config)
    fig, _ = plot_scene(scene, block=args.block, annotate=not args.no_labels)
    output = args.output or ROOT / "outputs" / scene.name / "scene.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved {output}")
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
