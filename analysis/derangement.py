"""Source substitution check: re-score the test set with each response given a wrong source.

Loads the three trained condition B checkpoints and runs fresh fp32 inference twice per
model, once with the true source and once with a within-family derangement of the sources.
Tokenization goes through the project's own RagTruthDataset so the inputs match training.
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from load_logits import load_logits
from score import CLEAN, HALLUCINATED, prf
from src.config import MAX_LENGTH
from src.data.dataset import RagTruthDataset, make_collate_fn
from src.model import HallucinationClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
FINALS = REPO_ROOT / "results" / "finals"
DATASET_DIR = REPO_ROOT / "data" / "raw" / "RAGTruth" / "dataset"
MODEL_DIR = REPO_ROOT / "ModernBERT-base"

SEEDS = (42, 1337, 2026)
FAMILIES = ("QA", "Summary", "Data2txt")
PERMUTATION_SEED = 0
BATCH_SIZE = 16


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_jsonl(path):
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_test_records(limit=None):
    """Test records keyed to the ids in the saved B logits.

    Labels come from those logits, which the scoring step already cross-checked.
    """
    rows = read_jsonl(DATASET_DIR / "source_info.jsonl")
    sources = {str(r["source_id"]): r for r in rows}
    truth = load_logits(FINALS / f"B_seed{SEEDS[0]}_test_logits.jsonl")
    records = []
    for row in read_jsonl(DATASET_DIR / "response.jsonl"):
        key = str(row["id"])
        if key not in truth:
            continue
        source = sources[str(row["source_id"])]
        records.append({
            "id": key,
            "source_id": str(row["source_id"]),
            "task_type": source["task_type"],
            "response": row["response"],
            "source_info": source["source_info"],
            "label": truth[key]["label"],
        })
    records.sort(key=lambda r: r["id"])
    if limit:
        per_family = defaultdict(list)
        for r in records:
            per_family[r["task_type"]].append(r)
        records = [r for f in FAMILIES for r in per_family[f][: limit // len(FAMILIES)]]
    return records, sources


def build_derangement(records, seed):
    """Map each source id to a different source id in the same family, no fixed points."""
    by_family = defaultdict(set)
    for r in records:
        by_family[r["task_type"]].add(r["source_id"])
    rng = random.Random(seed)
    mapping = {}
    for family in sorted(by_family):
        ids = sorted(by_family[family])
        if len(ids) < 2:
            raise SystemExit(f"family {family} has {len(ids)} source(s), cannot derange")
        order = ids[:]
        rng.shuffle(order)
        for i, source_id in enumerate(order):
            mapping[source_id] = order[(i + 1) % len(order)]
    return mapping


def substitute(records, mapping, sources):
    """Swap in the donor source, leaving the task string and the response untouched."""
    out = []
    for r in records:
        donor = sources[mapping[r["source_id"]]]
        if donor["task_type"] != r["task_type"]:
            raise SystemExit(f"donor for {r['source_id']} is from another family")
        if r["task_type"] == "QA":
            swapped = {**r["source_info"], "passages": donor["source_info"]["passages"]}
        else:
            swapped = donor["source_info"]
        out.append({**r, "source_info": swapped})
    return out


def assert_derangement(records, mapping, sources):
    """Every source moves, and every donor is in the same family."""
    fixed = [s for s in mapping if mapping[s] == s]
    if fixed:
        raise SystemExit(
            f"{len(fixed)} fixed point(s) in the permutation, for example {fixed[:5]}"
        )
    for r in records:
        donor = sources[mapping[r["source_id"]]]
        if donor["task_type"] != r["task_type"]:
            raise SystemExit(
                f"source {r['source_id']} was given a {donor['task_type']} donor"
            )
    print(f"  derangement: {len(mapping)} sources permuted, 0 fixed points, "
          f"families preserved")


def assert_inputs_moved(original, swapped, tokenizer):
    """The B token ids must change, and the A token ids (task plus response) must not."""
    true_b = RagTruthDataset(original, tokenizer, "B", MAX_LENGTH)
    swap_b = RagTruthDataset(swapped, tokenizer, "B", MAX_LENGTH)
    identical = [i for i in range(len(true_b))
                 if true_b.input_ids[i] == swap_b.input_ids[i]]
    if identical:
        raise SystemExit(
            f"{len(identical)} example(s) tokenize identically after substitution, so the "
            f"swap is a no-op there, for example index {identical[:5]}"
        )

    true_a = RagTruthDataset(original, tokenizer, "A", MAX_LENGTH)
    swap_a = RagTruthDataset(swapped, tokenizer, "A", MAX_LENGTH)
    moved_a = [i for i in range(len(true_a))
               if true_a.input_ids[i] != swap_a.input_ids[i]]
    if moved_a:
        raise SystemExit(
            f"{len(moved_a)} example(s) changed their task plus response text, "
            f"only the source segment may move"
        )

    print(f"  token ids: all {len(true_b)} B inputs differ after substitution, "
          f"all A inputs unchanged")
    for i in range(min(3, len(true_b))):
        print(f"    {original[i]['id']:>6} {original[i]['task_type']:<9} "
              f"true {len(true_b.input_ids[i]):>5} tokens, "
              f"deranged {len(swap_b.input_ids[i]):>5} tokens, "
              f"first differing position "
              f"{_first_diff(true_b.input_ids[i], swap_b.input_ids[i])}")
    return true_b, swap_b


def _first_diff(left, right):
    for i, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return i
    return min(len(left), len(right))


@torch.no_grad()
def predict(model, dataset, tokenizer, device):
    """Predicted class per response id, batching longest-last to limit padding waste."""
    order = sorted(range(len(dataset)), key=lambda i: len(dataset.input_ids[i]))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=order,
                        collate_fn=make_collate_fn(tokenizer.pad_token_id))
    preds = {}
    for batch in loader:
        logits = model(input_ids=batch["input_ids"].to(device),
                       attention_mask=batch["attention_mask"].to(device))
        for rid, row in zip(batch["ids"], logits.detach().float().cpu().tolist()):
            preds[rid] = HALLUCINATED if row[HALLUCINATED] >= row[CLEAN] else CLEAN
    return preds


def saved_predictions(seed):
    rows = load_logits(FINALS / f"B_seed{seed}_test_logits.jsonl")
    return {
        i: HALLUCINATED
        if rows[i]["logit_hallucinated"] >= rows[i]["logit_clean"]
        else CLEAN
        for i in rows
    }


def rate(values):
    return sum(values) / len(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="source substitution check for condition B")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap records per family, for a quick smoke run")
    args = parser.parse_args()

    device = pick_device()
    print(f"device {device}, fp32, max_length {MAX_LENGTH}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    records, sources = load_test_records(args.limit)
    print(f"test records {len(records):,}")

    mapping = build_derangement(records, PERMUTATION_SEED)
    swapped = substitute(records, mapping, sources)
    assert_derangement(records, mapping, sources)
    true_set, swap_set = assert_inputs_moved(records, swapped, tokenizer)

    labels = {r["id"]: r["label"] for r in records}
    family_of = {r["id"]: r["task_type"] for r in records}
    clean_ids = [i for i in labels if labels[i] == CLEAN]
    print(f"originally clean responses {len(clean_ids):,}")

    per_model = {}
    for seed in SEEDS:
        started = time.time()
        model = HallucinationClassifier(str(MODEL_DIR)).to(device)
        model.load_state_dict(
            torch.load(FINALS / f"B_seed{seed}_best.pt", map_location=device,
                       weights_only=True)
        )
        model.eval()

        true_preds = predict(model, true_set, tokenizer, device)
        swap_preds = predict(model, swap_set, tokenizer, device)
        per_model[seed] = (true_preds, swap_preds)

        ids = sorted(true_preds)
        rerun = prf([true_preds[i] for i in ids], [labels[i] for i in ids], HALLUCINATED)[2]
        saved = saved_predictions(seed)
        stored = prf([saved[i] for i in ids], [labels[i] for i in ids], HALLUCINATED)[2]
        agree = sum(1 for i in ids if saved[i] == true_preds[i]) / len(ids)
        print(f"\nB_seed{seed}: true source rerun positive F1 {rerun:.4f}, "
              f"saved {stored:.4f}, difference {abs(rerun - stored):.4f}")
        print(f"  prediction agreement with the saved logits {agree:.4f}, "
              f"{time.time() - started:.0f}s")
        if abs(rerun - stored) > 0.02:
            print("  WARNING large gap, the reload or the tokenization may not "
                  "match training")
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    scopes = [("overall", clean_ids)] + [
        (f, [i for i in clean_ids if family_of[i] == f]) for f in FAMILIES
    ]

    print()
    print("originally clean responses, per model")
    header = (f"{'model':12}{'scope':12}{'n':>7}{'pos true':>10}"
              f"{'pos deranged':>14}{'flip':>9}")
    print(header)
    summary = defaultdict(list)
    for seed in SEEDS:
        true_preds, swap_preds = per_model[seed]
        for name, ids in scopes:
            pos_true = rate([true_preds[i] for i in ids])
            pos_swap = rate([swap_preds[i] for i in ids])
            flips = rate([1 if true_preds[i] != swap_preds[i] else 0 for i in ids])
            summary[name].append((pos_true, pos_swap, flips))
            print(f"B_seed{seed:<6}{name:12}{len(ids):>7,}{pos_true:>10.4f}"
                  f"{pos_swap:>14.4f}{flips:>9.4f}")

    print()
    print("originally clean responses, mean over the three models")
    print(f"{'scope':12}{'n':>7}{'pos true':>10}{'pos deranged':>14}"
          f"{'flip':>9}{'shift':>9}")
    for name, ids in scopes:
        rows = summary[name]
        pos_true = rate([r[0] for r in rows])
        pos_swap = rate([r[1] for r in rows])
        flips = rate([r[2] for r in rows])
        print(f"{name:12}{len(ids):>7,}{pos_true:>10.4f}{pos_swap:>14.4f}"
              f"{flips:>9.4f}{pos_swap - pos_true:>+9.4f}")

    print()
    print("a higher predicted positive rate under a wrong source means B is sensitive to")
    print("source substitution, not that its verification is correct")


if __name__ == "__main__":
    main()
