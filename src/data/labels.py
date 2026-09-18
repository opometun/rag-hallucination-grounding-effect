"""Turn the RAGTruth span annotations into a response level training target.

Any span makes the response positive, implicit_true and due_to_null spans included. The
target is faithfulness to the source, not whether the claim happens to be true in the
world, and due_to_null only records why an error happened. The two implicit_true fields
are diagnostics for a later sensitivity check, they do not change the label.
"""

from collections.abc import Iterable

EXCLUDED_QUALITY = ("incorrect_refusal", "truncated")


def drop_flagged(records: Iterable[dict]) -> list[dict]:
    return [r for r in records if r["quality"] not in EXCLUDED_QUALITY]


def derive_labels(record: dict) -> dict:
    labels = record["labels"]
    has_implicit_true = any(span.get("implicit_true", False) for span in labels)
    # all() on an empty list is True, so without the bool(labels) guard a clean response
    # would come out flagged implicit_true_only.
    implicit_true_only = bool(labels) and all(
        span.get("implicit_true", False) for span in labels
    )
    return {
        **record,
        "label": int(bool(labels)),
        "has_implicit_true": has_implicit_true,
        "implicit_true_only": implicit_true_only,
    }


def label_records(records: Iterable[dict]) -> list[dict]:
    return [derive_labels(r) for r in drop_flagged(records)]
