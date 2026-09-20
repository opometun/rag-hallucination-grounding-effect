"""Score the saved final-run logits and aggregate the A versus B comparison.

Run from anywhere: python analysis/score.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from load_logits import load_logits

REPO_ROOT = Path(__file__).resolve().parents[1]
FINALS = REPO_ROOT / "results" / "finals"

CONDITIONS = ("A", "B")
SEEDS = (42, 1337, 2026)
CLEAN, HALLUCINATED = 0, 1
CLASSES = (CLEAN, HALLUCINATED)
TOLERANCE = 1e-6


def predictions_and_labels(rows):
    """Take the loader's dict, return (predictions, labels) aligned by sorted id."""
    preds, labels = [], []
    for key in sorted(rows):
        row = rows[key]
        # An exact tie goes to the hallucinated class, as in the training loop.
        hit = row["logit_hallucinated"] >= row["logit_clean"]
        preds.append(HALLUCINATED if hit else CLEAN)
        labels.append(row["label"])
    return preds, labels


def prf(preds, labels, positive):
    """Precision, recall and F1 for one class, counted over the whole set."""
    tp = sum(1 for p, y in zip(preds, labels) if p == positive and y == positive)
    fp = sum(1 for p, y in zip(preds, labels) if p == positive and y != positive)
    fn = sum(1 for p, y in zip(preds, labels) if p != positive and y == positive)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def macro_f1(preds, labels):
    """Unweighted mean of the per-class F1 for clean and for hallucinated."""
    return sum(prf(preds, labels, c)[2] for c in CLASSES) / len(CLASSES)


def score_file(path):
    """Score one logits file, returning the positive-class metrics and macro F1."""
    preds, labels = predictions_and_labels(load_logits(path))
    precision, recall, f1 = prf(preds, labels, HALLUCINATED)
    return {
        "n": len(labels),
        "positive_precision": precision,
        "positive_recall": recall,
        "positive_f1": f1,
        "macro_f1": macro_f1(preds, labels),
    }


def run_name(condition, seed):
    return f"{condition}_seed{seed}"


def cross_check(runs):
    print("validation cross-check, recomputed here against the value stored at selection")
    print(f"{'run':16}{'recomputed':>13}{'stored':>13}{'abs diff':>12}")
    mismatches = []
    for condition, seed in runs:
        name = run_name(condition, seed)
        recomputed = score_file(FINALS / f"{name}_val_logits.jsonl")["macro_f1"]
        saved = json.loads((FINALS / f"{name}_result.json").read_text())
        stored = saved["best_val_macro_f1"]
        diff = abs(recomputed - stored)
        print(f"{name:16}{recomputed:>13.6f}{stored:>13.6f}{diff:>12.2e}")
        if diff > TOLERANCE:
            mismatches.append((name, diff))

    if mismatches:
        for name, diff in mismatches:
            print(f"WARNING {name}: recomputed val macro F1 is off by {diff:.2e}")
        raise SystemExit(
            f"{len(mismatches)} of {len(runs)} runs disagree by more than {TOLERANCE:g}. "
            f"Scoring here is not identical to what selected the checkpoints, so the "
            f"test numbers are not reported."
        )
    print(f"all {len(runs)} runs match within {TOLERANCE:g}")


def main():
    runs = [(c, s) for c in CONDITIONS for s in SEEDS]
    cross_check(runs)

    test = {}
    print()
    print("test set, per run")
    header = (f"{'run':16}{'n':>7}{'precision':>12}{'recall':>10}"
              f"{'pos F1':>10}{'macro F1':>11}")
    print(header)
    for condition, seed in runs:
        name = run_name(condition, seed)
        row = score_file(FINALS / f"{name}_test_logits.jsonl")
        test[(condition, seed)] = row
        print(f"{name:16}{row['n']:>7,}{row['positive_precision']:>12.4f}"
              f"{row['positive_recall']:>10.4f}{row['positive_f1']:>10.4f}"
              f"{row['macro_f1']:>11.4f}")

    print()
    print("test set, aggregated over the three seeds")
    print(f"{'condition':12}{'metric':20}{'mean':>10}{'min':>10}{'max':>10}")
    for condition in CONDITIONS:
        # Same per-run values the table above printed; score_file already returns
        # precision and recall alongside the two F1 figures.
        for metric in ("positive_precision", "positive_recall",
                       "positive_f1", "macro_f1"):
            values = [test[(condition, seed)][metric] for seed in SEEDS]
            print(f"{condition:12}{metric:20}{sum(values) / len(values):>10.4f}"
                  f"{min(values):>10.4f}{max(values):>10.4f}")

    print()
    print("paired delta on positive F1, B minus A at the same seed")
    print(f"{'seed':10}{'A':>10}{'B':>10}{'delta':>10}")
    deltas = []
    for seed in SEEDS:
        a = test[("A", seed)]["positive_f1"]
        b = test[("B", seed)]["positive_f1"]
        deltas.append(b - a)
        print(f"{seed:<10}{a:>10.4f}{b:>10.4f}{b - a:>+10.4f}")
    print(f"{'mean':10}{'':>10}{'':>10}{sum(deltas) / len(deltas):>+10.4f}")


if __name__ == "__main__":
    main()
