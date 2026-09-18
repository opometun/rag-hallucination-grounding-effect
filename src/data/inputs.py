"""Build model inputs from a RAGTruth record.

text_a is the task and response; the source is returned separately so tokenization
happens later. No tokenizer here.
"""

import json
from collections.abc import Iterable

TASK_TEMPLATES = {
    "QA": "Briefly answer the following question: {question}",
    "Summary": "Summarize the following news article.",
    "Data2txt": "Write an objective overview of a local business based on structured data.",
}

# All of the record that condition A is allowed to see. The record also carries prompt
# and source, and both of those embed the grounding, so text_a must never reach for them.
TEXT_A_KEYS = ("task_type", "response")

# Pinned rather than taken from the incoming dict. Truncation cuts the tail, so if field
# order followed the loader then which fields survive would vary between records.
DATA2TXT_FIELDS = (
    "name",
    "address",
    "city",
    "state",
    "categories",
    "hours",
    "attributes",
    "business_stars",
    "review_info",
)


def _text_a_fields(record: dict) -> dict:
    fields = {key: record[key] for key in TEXT_A_KEYS}
    if record["task_type"] == "QA":
        # The question is part of the task, not part of the retrieved grounding, so it
        # shows up in both conditions. The passages never do.
        fields["question"] = record["source_info"]["question"]
    return fields


def build_task(record: dict) -> str:
    fields = _text_a_fields(record)
    task_type = fields["task_type"]
    if task_type == "QA":
        return TASK_TEMPLATES["QA"].format(question=fields["question"])
    if task_type in TASK_TEMPLATES:
        return TASK_TEMPLATES[task_type]
    raise ValueError(f"unknown task_type: {task_type!r}")


def _require_dict(task_type: str, source_info) -> None:
    if not isinstance(source_info, dict):
        raise TypeError(
            f"{task_type} source_info should be a dict, got {type(source_info).__name__}"
        )


def serialize_source(record: dict) -> str:
    task_type = record["task_type"]
    source_info = record["source_info"]
    if task_type == "Summary":
        if not isinstance(source_info, str):
            raise TypeError(
                f"Summary source_info should be a str, got {type(source_info).__name__}"
            )
        return source_info
    if task_type == "QA":
        _require_dict(task_type, source_info)
        missing = [k for k in ("question", "passages") if k not in source_info]
        if missing:
            raise KeyError(f"QA source_info is missing {missing}")
        return source_info["passages"]
    if task_type == "Data2txt":
        _require_dict(task_type, source_info)
        return flatten_data2txt(source_info)
    raise ValueError(f"unknown task_type: {task_type!r}")


def flatten_data2txt(d: dict) -> str:
    """Flatten the Data2txt structured source into one "qualified.key: value" line per leaf.

    Top level order is fixed by DATA2TXT_FIELDS; nested keys keep their own order.
    Leaves go through compact JSON, which keeps null distinct from the string "null"
    and keeps a review that contains newlines on a single line.
    """
    missing = [k for k in DATA2TXT_FIELDS if k not in d]
    unexpected = [k for k in d if k not in DATA2TXT_FIELDS]
    if missing or unexpected:
        raise ValueError(
            f"Data2txt schema drift, missing={missing} unexpected={unexpected}"
        )
    lines: list[str] = []
    for key in DATA2TXT_FIELDS:
        _walk(d[key], key, lines)
    return "\n".join(lines)


def _walk(value, prefix: str, lines: list[str]) -> None:
    if isinstance(value, dict) and value:
        for key, child in value.items():
            _walk(child, f"{prefix}.{key}", lines)
    elif isinstance(value, list) and value:
        for i, child in enumerate(value):
            _walk(child, f"{prefix}[{i}]", lines)
    else:
        # Empty dicts and lists land here too and come out as {} and [].
        lines.append(f"{prefix}: {_render_leaf(value)}")


def _render_leaf(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def build_inputs(record: dict) -> dict:
    fields = _text_a_fields(record)
    text_a = f"Task: {build_task(record)}\n\nResponse: {fields['response']}"
    return {"text_a": text_a, "source": serialize_source(record)}


def _scrambled_source_info(task_type: str, source_info):
    if task_type == "QA":
        return {**source_info, "passages": "scrambled passages"}
    if task_type == "Summary":
        return "scrambled article"
    return {
        "name": "scrambled",
        "address": "scrambled",
        "city": "scrambled",
        "state": "scrambled",
        "categories": "scrambled",
        "hours": {},
        "attributes": {},
        "business_stars": 0,
        "review_info": [],
    }


def _scrambled_copy(record: dict) -> dict:
    scrambled = {}
    for key, value in record.items():
        if key in TEXT_A_KEYS:
            scrambled[key] = value
        elif key == "source_info":
            scrambled[key] = _scrambled_source_info(record["task_type"], value)
        else:
            scrambled[key] = f"scrambled {key}"
    return scrambled


def assert_no_leak(record: dict) -> None:
    original = build_inputs(record)
    scrambled = build_inputs(_scrambled_copy(record))
    if original["source"] == scrambled["source"]:
        raise AssertionError(
            "scrambling left the source unchanged, so this check would pass for free"
        )
    if original["text_a"] != scrambled["text_a"]:
        raise AssertionError(
            "text_a changed when the non-whitelisted fields changed, so condition A is "
            "leaking the source"
        )


def assert_paired(records: Iterable[dict]) -> None:
    for i, record in enumerate(records):
        # Two separate calls on purpose, the way the pipeline will build the conditions,
        # so a build_inputs that is not deterministic shows up here rather than later.
        text_a = build_inputs(record)["text_a"]
        built = build_inputs(record)
        if text_a != built["text_a"]:
            raise AssertionError(f"record {i}: text_a is not stable across builds")
        if not built["source"].strip():
            raise AssertionError(f"record {i}: condition B has an empty source segment")
        assert_no_leak(record)
