"""Table II: per task family counts, prevalence, and A versus B positive class F1."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from load_logits import load_logits
from score import HALLUCINATED, predictions_and_labels, prf

REPO_ROOT = Path(__file__).resolve().parents[1]
FINALS = REPO_ROOT / "results" / "finals"
DATASET_DIR = REPO_ROOT / "data" / "raw" / "RAGTruth" / "dataset"

CONDITIONS = ("A", "B")
SEEDS = (42, 1337, 2026)
FAMILIES = ("QA", "Summary", "Data2txt")


def _read_jsonl(path):
    if not path.is_file():
        raise FileNotFoundError(f"no file at {path}")
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_id_to_family(dataset_dir=DATASET_DIR):
    """Return {response id as str: task family}, joining response.jsonl to
    source_info.jsonl.

    Raises ValueError on an unexpected task_type or a response whose source_id has no
    matching source row.
    """
    dataset_dir = Path(dataset_dir)
    family_of_source = {
        str(row["source_id"]): row["task_type"]
        for row in _read_jsonl(dataset_dir / "source_info.jsonl")
    }
    unknown = sorted(set(family_of_source.values()) - set(FAMILIES))
    if unknown:
        raise ValueError(f"unexpected task_type values in source_info.jsonl: {unknown}")

    mapping = {}
    for row in _read_jsonl(dataset_dir / "response.jsonl"):
        source_id = str(row["source_id"])
        if source_id not in family_of_source:
            raise ValueError(
                f"response {row['id']} has source_id {source_id}, which has no source row"
            )
        mapping[str(row["id"])] = family_of_source[source_id]
    return mapping


def test_path(condition, seed):
    return FINALS / f"{condition}_seed{seed}_test_logits.jsonl"


def family_f1(rows, ids):
    """Positive class F1 over the subset of rows whose id is in ids."""
    preds, labels = predictions_and_labels({i: rows[i] for i in rows if i in ids})
    return prf(preds, labels, HALLUCINATED)[2]


def main():
    id_to_family = build_id_to_family()
    runs = {(c, s): load_logits(test_path(c, s)) for c in CONDITIONS for s in SEEDS}

    reference = runs[("A", SEEDS[0])]
    for key, rows in runs.items():
        if set(rows) != set(reference):
            raise SystemExit(f"{key} does not cover the same test ids as the reference run")
        if any(rows[i]["label"] != reference[i]["label"] for i in reference):
            raise SystemExit(f"{key} disagrees with the reference run on the true labels")

    missing = sorted(i for i in reference if i not in id_to_family)
    if missing:
        raise SystemExit(
            f"{len(missing)} test id(s) have no task family, for example {missing[:5]}"
        )

    groups = {}
    for family in FAMILIES:
        ids = {i for i in reference if id_to_family[i] == family}
        labels = [reference[i]["label"] for i in ids]
        groups[family] = {"ids": ids, "n": len(ids), "positives": sum(labels)}

    print(f"{'family':12}{'n':>7}{'prevalence':>12}{'A F1':>9}{'B F1':>9}{'delta':>10}")
    notes = []
    for family in FAMILIES:
        group = groups[family]
        ids = group["ids"]
        a_scores = [family_f1(runs[("A", seed)], ids) for seed in SEEDS]
        b_scores = [family_f1(runs[("B", seed)], ids) for seed in SEEDS]
        a_mean = sum(a_scores) / len(a_scores)
        b_mean = sum(b_scores) / len(b_scores)
        delta = sum(b - a for a, b in zip(a_scores, b_scores)) / len(SEEDS)
        prevalence = group["positives"] / group["n"] if group["n"] else 0.0
        print(f"{family:12}{group['n']:>7,}{prevalence:>12.3f}"
              f"{a_mean:>9.4f}{b_mean:>9.4f}{delta:>+10.4f}")

        if group["n"] < 50:
            notes.append(f"{family} has only {group['n']} responses")
        for label, value in (("A", a_mean), ("B", b_mean)):
            if value in (0.0, 1.0):
                notes.append(f"{family} {label} F1 is exactly {value:.1f}")
        if group["positives"] in (0, group["n"]):
            notes.append(
                f"{family} has a degenerate label split, "
                f"{group['positives']} positive"
            )

    total_n = sum(groups[f]["n"] for f in FAMILIES)
    total_positives = sum(groups[f]["positives"] for f in FAMILIES)
    print(f"{'total':12}{total_n:>7,}{total_positives / total_n:>12.3f}")

    print()
    print(f"family counts sum to {total_n}, test files hold {len(reference)} rows")
    if total_n != len(reference):
        raise SystemExit("family counts do not cover every test row")
    for note in notes:
        print(f"note: {note}")


if __name__ == "__main__":
    main()
