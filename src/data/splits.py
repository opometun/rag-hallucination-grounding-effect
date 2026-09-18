"""Carve a validation set out of the RAGTruth training split, by source and not by response.

All six responses of a source share one grounding text, so splitting on responses would put
the same source on both sides and flatter the grounded condition. The official test split is
never touched here.
"""

import json
import random
from collections.abc import Iterable
from pathlib import Path

DEFAULT_PATH = "splits/val_source_ids.json"
SEED = 42
VAL_FRACTION = 0.15


def _train_sources_by_family(records: Iterable[dict]) -> dict[str, list[str]]:
    family_of = {}
    for r in records:
        if r["split"] != "train":
            continue
        sid, task_type = r["source_id"], r["task_type"]
        if sid in family_of and family_of[sid] != task_type:
            raise ValueError(
                f"source_id {sid!r} appears under two task types, "
                f"{family_of[sid]!r} and {task_type!r}"
            )
        family_of[sid] = task_type

    by_family: dict[str, list[str]] = {}
    for sid, task_type in family_of.items():
        by_family.setdefault(task_type, []).append(sid)
    return by_family


def make_val_split(
    records: Iterable[dict],
    out_path: Path | str = DEFAULT_PATH,
    seed: int = SEED,
    val_fraction: float = VAL_FRACTION,
    force: bool = False,
) -> list[str]:
    by_family = _train_sources_by_family(records)
    path = Path(out_path)

    # An existing file wins, so a stray rerun cannot quietly move the split.
    if path.exists() and not force:
        return load_val_split(path)

    rng = random.Random(seed)
    chosen = []
    # Families in sorted order, ids sorted within each, so the draw does not depend on the
    # order the records happened to arrive in.
    for family in sorted(by_family):
        ids = sorted(by_family[family])
        chosen.extend(rng.sample(ids, round(val_fraction * len(ids))))

    chosen = sorted(chosen)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(chosen, indent=2) + "\n")
    return chosen


def load_val_split(out_path: Path | str = DEFAULT_PATH) -> list[str]:
    return json.loads(Path(out_path).read_text())


def assign_split(records: Iterable[dict], val_source_ids: Iterable[str]) -> list[dict]:
    val = set(val_source_ids)
    out = []
    for r in records:
        if r["split"] == "train":
            fold = "val" if r["source_id"] in val else "train"
        else:
            # Test rows stay in the list and carry their own split through as the fold.
            fold = r["split"]
        out.append({**r, "fold": fold})
    return out


def assert_disjoint(records: Iterable[dict], val_source_ids: Iterable[str]) -> None:
    val = set(val_source_ids)
    train_fold, val_fold, test = set(), set(), set()
    for r in records:
        sid = r["source_id"]
        if r["split"] == "train":
            (val_fold if sid in val else train_fold).add(sid)
        else:
            test.add(sid)

    pairs = [
        ("train fold", train_fold, "val fold", val_fold),
        ("train fold", train_fold, "test", test),
        ("val fold", val_fold, "test", test),
    ]
    for left_name, left, right_name, right in pairs:
        overlap = left & right
        if overlap:
            raise AssertionError(
                f"{len(overlap)} source_id(s) in both {left_name} and {right_name}: "
                f"{sorted(overlap)[:10]}"
            )
