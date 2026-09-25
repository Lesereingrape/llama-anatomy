"""Run the full 3-seed ablation and write results/anatomy.json.

    python experiments/run_study.py

Everything in the README comes from that file. Re-running recomputes the whole
matrix; ``tests/test_readme.py`` then fails if the prose and the numbers have
drifted apart. The artifact also records the environment the fits ran in
(``study._environment``), because that is the condition under which a rerun is
bit-exact rather than merely close.
"""

from __future__ import annotations

import json
from pathlib import Path

from llamanat.study import build_results

OUT = Path(__file__).resolve().parents[1] / "results" / "anatomy.json"


def main() -> None:
    results = build_results()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(f"wrote {OUT} in {results['runtime_sec']}s")
    for task, cells in results["in_domain"].items():
        for variant, cell in cells.items():
            curve = cell["curve"]
            line = " ".join(f"{step}:{curve[step]['mean']:.3f}"
                            for step in sorted(curve, key=int))
            print(f"  {task:<9} {variant:<10} {cell['params']:>6}p  {line}")


if __name__ == "__main__":
    main()
