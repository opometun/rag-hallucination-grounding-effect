"""Pick the learning rate from the grid results and record how the choice was made.

Reads every <cond>_seed<seed>_result.json under the grid directory, averages
best_val_macro_f1 across conditions for each rate, and writes lr_decision.json next to
them. Exits non-zero rather than guessing if the grid is incomplete.
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

CONDITIONS = ("A", "B")
TIE_BAND = 0.005


def main():
    grid = Path(sys.argv[1] if len(sys.argv) > 1 else "results/grid")
    # rglob, not glob: each rate gets its own subdirectory because the run stem carries
    # only condition and seed, so a flat directory would have them overwrite each other.
    files = sorted(p for p in grid.rglob("*_result.json"))
    if len(files) < 4:
        sys.exit(f"expected 4 grid results under {grid}, found {len(files)}")

    runs = []
    for path in files:
        row = json.loads(path.read_text())
        runs.append({
            "condition": row["condition"],
            "learning_rate": float(row["learning_rate"]),
            "macro_f1": row["best_val_macro_f1"],
            "file": str(path),
        })

    by_lr = defaultdict(dict)
    for run in runs:
        score = run["macro_f1"]
        if score is None or math.isnan(float(score)):
            sys.exit(f"best_val_macro_f1 is missing or NaN in {run['file']}")
        by_lr[run["learning_rate"]][run["condition"]] = float(score)

    if len(by_lr) != 2:
        sys.exit(f"expected 2 learning rates in the grid, found {sorted(by_lr)}")
    for rate, cells in by_lr.items():
        missing = [c for c in CONDITIONS if c not in cells]
        if missing:
            sys.exit(f"learning rate {rate:g} has no result for condition {missing}")

    means = {rate: sum(cells[c] for c in CONDITIONS) / len(CONDITIONS)
             for rate, cells in by_lr.items()}
    lower, higher = sorted(means)

    # Inside the tie band the grid cannot tell the two rates apart, so take the smaller
    # one. Outside it, take the better mean.
    if abs(means[higher] - means[lower]) <= TIE_BAND:
        chosen = lower
        branch = "tie, took the smaller rate"
    else:
        chosen = max(means, key=means.__getitem__)
        branch = "clear winner on mean macro f1"

    decision = {
        "chosen_lr": chosen,
        "branch": branch,
        "tie_band": TIE_BAND,
        "means": {f"{rate:g}": means[rate] for rate in sorted(means)},
        "runs": runs,
    }
    (grid / "lr_decision.json").write_text(json.dumps(decision, indent=2) + "\n")

    print(f"branch: {branch}")
    for rate in sorted(means):
        print(f"lr {rate:g} mean macro f1 {means[rate]:.4f}")
    print(chosen)


if __name__ == "__main__":
    main()
