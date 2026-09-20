"""Read a saved logits jsonl file into memory. Loading only, no scoring."""

import json
from pathlib import Path

FIELDS = ("id", "logit_clean", "logit_hallucinated", "label")


def _coerce(kind, row, path, number, field):
    try:
        return kind(row[field])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{path} line {number}: field {field} is not a {kind.__name__}, "
            f"got {row[field]!r}"
        ) from exc


def load_logits(path) -> dict:
    """Return {id: {"logit_clean": float, "logit_hallucinated": float, "label": int}}.

    Raises FileNotFoundError if there is no file at path. Raises ValueError if the file
    holds no rows, if a line is not valid json, if a line lacks one of the four expected
    fields, if a value will not coerce, or if an id appears more than once.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no logits file at {path}")

    rows = {}
    with path.open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {number}: not valid json, {exc}") from exc

            missing = [f for f in FIELDS if f not in row]
            if missing:
                raise ValueError(f"{path} line {number}: missing field(s) {missing}")

            key = str(row["id"])
            if key in rows:
                raise ValueError(
                    f"{path}: id {key!r} appears more than once, again at line {number}"
                )

            rows[key] = {
                "logit_clean": _coerce(float, row, path, number, "logit_clean"),
                "logit_hallucinated": _coerce(
                    float, row, path, number, "logit_hallucinated"
                ),
                "label": _coerce(int, row, path, number, "label"),
            }

    if not rows:
        raise ValueError(f"{path}: file holds no rows")
    return rows
