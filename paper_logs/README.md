# Paper run logs

Console output saved from the runs behind the numbers in the paper (December
2024), kept for reference. `scripts/run_experiment.py --planner astar`
reproduces the A* numbers in these logs exactly.

| File | Contents |
|---|---|
| `exp1_complex_outdoor.txt` | End of the LLM-A* run (last POI pair, totals, TSP tour, length 10530.60), then the A* totals and tour (length 9270.45). |
| `exp2_simple_outdoor.txt` | A* totals, the 13 x 13 A* distance matrix and tour (length 13562.50), then the last LLM-A* query with its GPT-4o waypoints, totals and tour (length 16300.53). |
| `exp3_indoor_warehouse.txt` | LLM-A* totals and tour (length 15682.17). |

Point indices: `0` is the start point and `k` is `POI_{k-1}`. The `operation`
and `storage` values in `Updated Cumulative Data` are the totals over all
POI pairs reported in the paper.
