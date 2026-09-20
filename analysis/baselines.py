"""Non-model baselines: how much of condition A comes from prevalence and metadata priors.

Response length is measured in characters. The metadata regression is fit on the train
split only and scored on test.
"""

import json
import sys
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from family_breakdown import build_id_to_family
from load_logits import load_logits
from score import CLEAN, HALLUCINATED, macro_f1, prf
from src.data.labels import label_records

REPO_ROOT = Path(__file__).resolve().parents[1]
FINALS = REPO_ROOT / "results" / "finals"
RESPONSE_FILE = REPO_ROOT / "data" / "raw" / "RAGTruth" / "dataset" / "response.jsonl"

SEEDS = (42, 1337, 2026)
FAMILIES = ("QA", "Summary", "Data2txt")
RANDOM_STATE = 0


def read_responses():
    with RESPONSE_FILE.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def train_rows(records, id_to_family):
    """Train split rows after the project quality exclusion.

    Each row carries the derived label, task family, response length and generator.
    """
    raw = [r for r in records if r["split"] == "train"]
    kept = label_records(raw)
    rows = [
        {
            "id": str(r["id"]),
            "label": r["label"],
            "family": id_to_family[str(r["id"])],
            "length": len(r["response"]),
            "generator": r["model"],
        }
        for r in kept
    ]
    return rows, len(raw) - len(kept)


def test_rows(id_to_family, records):
    """Test rows keyed to the ids that actually appear in the saved logits."""
    truth = load_logits(FINALS / f"A_seed{SEEDS[0]}_test_logits.jsonl")
    by_id = {str(r["id"]): r for r in records}
    ids = sorted(truth)
    rows = [
        {
            "id": i,
            "label": truth[i]["label"],
            "family": id_to_family[i],
            "length": len(by_id[i]["response"]),
            "generator": by_id[i]["model"],
        }
        for i in ids
    ]
    return rows


def design_matrix(rows, families, generators, scaler, fit):
    """One-hot family (plus generator when given) alongside the standardized length."""
    blocks = [np.array([[r["family"] == f for f in families] for r in rows], dtype=float)]
    if generators:
        blocks.append(
            np.array([[r["generator"] == g for g in generators] for r in rows], dtype=float)
        )
    length = np.array([[r["length"]] for r in rows], dtype=float)
    length = scaler.fit_transform(length) if fit else scaler.transform(length)
    blocks.append(length)
    return np.hstack(blocks)


def scope_masks(rows):
    """(name, mask) for the whole set and for each family."""
    yield "overall", [True] * len(rows)
    for family in FAMILIES:
        yield family, [r["family"] == family for r in rows]


def subset(values, mask):
    return [v for v, keep in zip(values, mask) if keep]


def condition_a_f1(id_to_family):
    """A mean positive F1 and macro F1 over the seeds, overall and per family.

    Recomputed from the six A test logits files, not read from anywhere.
    """
    runs = [load_logits(FINALS / f"A_seed{seed}_test_logits.jsonl") for seed in SEEDS]
    out = {}
    for name in ("overall",) + FAMILIES:
        positive, macro = [], []
        for rows in runs:
            ids = sorted(i for i in rows if name == "overall" or id_to_family[i] == name)
            preds = [
                HALLUCINATED
                if rows[i]["logit_hallucinated"] >= rows[i]["logit_clean"]
                else CLEAN
                for i in ids
            ]
            truth = [rows[i]["label"] for i in ids]
            positive.append(prf(preds, truth, HALLUCINATED)[2])
            macro.append(macro_f1(preds, truth))
        out[name] = {
            "positive_f1": sum(positive) / len(positive),
            "macro_f1": sum(macro) / len(macro),
        }
    return out


def main():
    print(f"sklearn {sklearn.__version__}, numpy {np.__version__}")

    id_to_family = build_id_to_family()
    records = read_responses()
    train, train_dropped = train_rows(records, id_to_family)
    test = test_rows(id_to_family, records)

    raw_test = [r for r in records if r["split"] == "test"]
    print(f"train rows after exclusion {len(train):,}, dropped {train_dropped}")
    print(f"test rows after exclusion {len(test):,}, dropped {len(raw_test) - len(test)}")
    print(f"total dropped {train_dropped + len(raw_test) - len(test)}")

    labels = [r["label"] for r in test]
    print()
    print("test prevalence")
    print(f"{'scope':12}{'n':>7}{'prevalence':>12}")
    for name, mask in scope_masks(test):
        part = subset(labels, mask)
        print(f"{name:12}{len(part):>7,}{sum(part) / len(part):>12.3f}")

    prevalence = sum(labels) / len(labels)
    closed = 2 * prevalence / (1 + prevalence)
    direct = prf([HALLUCINATED] * len(labels), labels, HALLUCINATED)[2]
    print()
    print("always predict hallucinated")
    print(f"  closed form 2p/(1+p) {closed:.6f}")
    print(f"  direct from prf      {direct:.6f}")
    print(f"  difference           {abs(closed - direct):.2e}")

    a_scores = condition_a_f1(id_to_family)

    majority = {}
    for family in FAMILIES:
        part = [r["label"] for r in train if r["family"] == family]
        majority[family] = HALLUCINATED if sum(part) * 2 > len(part) else CLEAN
    majority_preds = [majority[r["family"]] for r in test]
    print()
    print("majority label per task family, learned on train")
    print(f"{'scope':12}{'majority':>10}{'pos F1':>10}{'macro F1':>11}{'A macro F1':>13}")
    for name, mask in scope_masks(test):
        preds, truth = subset(majority_preds, mask), subset(labels, mask)
        shown = majority[name] if name in majority else ""
        print(f"{name:12}{str(shown):>10}{prf(preds, truth, HALLUCINATED)[2]:>10.4f}"
              f"{macro_f1(preds, truth):>11.4f}{a_scores[name]['macro_f1']:>13.4f}")

    generators = sorted({r["generator"] for r in train})
    variants = [
        ("family one-hot + length", None),
        ("family one-hot + length + generator one-hot", generators),
    ]
    for title, gens in variants:
        scaler = StandardScaler()
        x_train = design_matrix(train, FAMILIES, gens, scaler, fit=True)
        x_test = design_matrix(test, FAMILIES, gens, scaler, fit=False)
        model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
        model.fit(x_train, [r["label"] for r in train])
        preds = [int(p) for p in model.predict(x_test)]

        print()
        print(f"metadata logistic regression, fit on train, scored on test: {title}")
        header = (f"{'scope':12}{'n':>7}{'precision':>11}{'recall':>9}"
                  f"{'base F1':>10}{'A F1':>9}{'gap':>9}")
        print(header)
        for name, mask in scope_masks(test):
            part_preds, truth = subset(preds, mask), subset(labels, mask)
            precision, recall, f1 = prf(part_preds, truth, HALLUCINATED)
            a_positive = a_scores[name]["positive_f1"]
            print(f"{name:12}{len(truth):>7,}{precision:>11.4f}{recall:>9.4f}"
                  f"{f1:>10.4f}{a_positive:>9.4f}{a_positive - f1:>+9.4f}")
        overall_macro = macro_f1(preds, labels)
        print(f"  overall macro F1 {overall_macro:.4f}")

    print()
    print("generator identity is dataset metadata, a deployed detector may not have it")


if __name__ == "__main__":
    main()
