"""Entry point for the step 3 data pipeline: load, label, split, tokenize, and check.

The only thing written to disk is splits/val_source_ids.json. Tokenized datasets stay in
memory, training rebuilds them.
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import MAX_LENGTH
from src.data.dataset import RagTruthDataset, run_token_checks
from src.data.labels import EXCLUDED_QUALITY, label_records
from src.data.load import load_joined
from src.data.splits import (
    SEED,
    VAL_FRACTION,
    assert_disjoint,
    assign_split,
    make_val_split,
)

MODEL = "answerdotai/ModernBERT-base"
SPLIT_PATH = REPO_ROOT / "splits" / "val_source_ids.json"
FAMILIES = ("QA", "Summary", "Data2txt")

# These totals live here rather than in the modules. The modules have to stay usable on a
# subset, and a whole dataset number would be wrong the moment someone passes a slice.
EXPECTED_RETAINED = {"train": 14942, "test": 2675, "total": 17617}
EXPECTED_DROPPED = {"train": 148, "test": 25}

# The five token level guards are structural properties of the tokenizer and the builder,
# so a stratified sample settles them. Running them over the whole train fold would
# tokenize it about ten times over for no extra information.
CHECK_SAMPLE_PER_FAMILY = 100


def fail(message):
    raise SystemExit(f"prepare_data stopped: {message}")


def check_counts(before, after, total):
    for split in ("train", "test"):
        want = EXPECTED_RETAINED[split]
        if after[split] != want:
            fail(f"retained {split} is {after[split]:,}, expected {want:,}")
    if total != EXPECTED_RETAINED["total"]:
        fail(f"retained total is {total:,}, expected {EXPECTED_RETAINED['total']:,}")
    for split, want in EXPECTED_DROPPED.items():
        dropped = before[split] - after[split]
        if dropped != want:
            fail(f"dropped {dropped:,} flagged rows from {split}, expected {want:,}")


def report_sources(assigned):
    sources = defaultdict(lambda: defaultdict(set))
    for r in assigned:
        sources[r["task_type"]][r["fold"]].add(r["source_id"])

    print(f"{'family':<10}{'train src':>11}{'val src':>10}{'val share':>12}")
    for family in FAMILIES:
        train = len(sources[family]["train"])
        val = len(sources[family]["val"])
        print(f"{family:<10}{train:>11,}{val:>10,}{val / (train + val):>12.4f}")
    total_train = sum(len(sources[f]["train"]) for f in FAMILIES)
    total_val = sum(len(sources[f]["val"]) for f in FAMILIES)
    print(f"{'overall':<10}{total_train:>11,}{total_val:>10,}"
          f"{total_val / (total_train + total_val):>12.4f}")


def report_labels(assigned):
    by_fold = defaultdict(list)
    for r in assigned:
        by_fold[r["fold"]].append(r["label"])
    print(f"{'fold':<10}{'responses':>11}{'positive':>10}{'rate':>12}")
    for fold in ("train", "val", "test"):
        labels = by_fold[fold]
        positive = sum(labels)
        print(f"{fold:<10}{len(labels):>11,}{positive:>10,}{positive / len(labels):>12.4f}")


def build_fold(records, tokenizer, fold):
    blind = RagTruthDataset(records, tokenizer, "A", MAX_LENGTH, fold=fold)
    grounded = RagTruthDataset(records, tokenizer, "B", MAX_LENGTH, fold=fold)
    if blind.ids != grounded.ids or blind.labels != grounded.labels:
        fail(f"conditions A and B disagree on ids or labels in the {fold} fold")
    return {
        "examples": len(blind),
        "longest_a": max(len(x) for x in blind.input_ids),
        "longest_b": max(len(x) for x in grounded.input_ids),
        "at_cap": sum(1 for x in grounded.input_ids if len(x) >= MAX_LENGTH),
    }


def sample_for_checks(assigned, fold="train"):
    picked = []
    for family in FAMILIES:
        same = [r for r in assigned if r["fold"] == fold and r["task_type"] == family]
        picked.extend(same[:CHECK_SAMPLE_PER_FAMILY])
    return picked


def main():
    records = load_joined().to_dict("records")
    before = Counter(r["split"] for r in records)
    print(f"loaded {len(records):,} responses, "
          f"{before['train']:,} train and {before['test']:,} test")

    # Order matters here. Drop the flagged rows and label first, then draw the split, so
    # the split is drawn over exactly the rows that end up being trained on.
    kept = label_records(records)
    after = Counter(r["split"] for r in kept)
    check_counts(before, after, len(kept))
    print(f"dropped {sum(before.values()) - len(kept):,} responses with quality in "
          f"{EXCLUDED_QUALITY}: {before['train'] - after['train']:,} train, "
          f"{before['test'] - after['test']:,} test")
    print(f"retained {after['train']:,} train and {after['test']:,} test, "
          f"{len(kept):,} total, all three counts as expected")

    existed = SPLIT_PATH.exists()
    val_ids = make_val_split(kept, out_path=SPLIT_PATH)
    assigned = assign_split(kept, val_ids)
    assert_disjoint(kept, val_ids)
    where = "loaded the existing" if existed else "drew a fresh"
    print(f"\n{where} split of {len(val_ids):,} validation sources "
          f"(seed {SEED}, fraction {VAL_FRACTION}) at {SPLIT_PATH.relative_to(REPO_ROOT)}")
    print("train, val and test source ids are mutually disjoint")

    print()
    report_sources(assigned)
    print()
    report_labels(assigned)

    from transformers import AutoTokenizer

    print(f"\nloading tokenizer {MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    sample = sample_for_checks(assigned)
    run_token_checks(tokenizer, sample, MAX_LENGTH)
    print(f"five token level checks passed on {len(sample):,} real records")

    header = (f"{'fold':<10}{'examples':>10}{'longest A':>11}"
              f"{'longest B':>11}{'B at cap':>10}")
    print(f"\n{header}")
    for fold in ("train", "val"):
        stats = build_fold(assigned, tokenizer, fold)
        print(f"{fold:<10}{stats['examples']:>10,}{stats['longest_a']:>11,}"
              f"{stats['longest_b']:>11,}{stats['at_cap']:>10,}")

    print(f"\nall checks passed. max_length {MAX_LENGTH}, one artifact written, "
          f"{SPLIT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
