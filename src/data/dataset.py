"""Tokenize labeled records into the two conditions, blind (text_a) and grounded (pair).

torch is not imported at module level. The dataset itself holds plain lists, which is all a
map style Dataset needs, and only the collate function pulls torch in to build tensors.
"""

from collections.abc import Iterable, Sequence

from src.data.inputs import scrambled_copy, build_inputs

CONDITIONS = ("A", "B")
TOKENIZE_BATCH = 256


def _encode_single(tokenizer, texts: Sequence[str], batch_size: int) -> list[list[int]]:
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[i:i + batch_size],
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
        out.extend(enc["input_ids"])
    return out


def _encode_pair(tokenizer, texts_a, sources, max_length, batch_size) -> list[list[int]]:
    out = []
    for i in range(0, len(texts_a), batch_size):
        enc = tokenizer(
            texts_a[i:i + batch_size],
            sources[i:i + batch_size],
            add_special_tokens=True,
            truncation="only_second",
            max_length=max_length,
            padding=False,
        )
        out.extend(enc["input_ids"])
    return out


class RagTruthDataset:
    def __init__(
        self,
        records: Iterable[dict],
        tokenizer,
        condition: str,
        max_length: int,
        fold: str | None = None,
        batch_size: int = TOKENIZE_BATCH,
    ):
        if condition not in CONDITIONS:
            raise ValueError(f"condition should be one of {CONDITIONS}, got {condition!r}")
        records = [r for r in records if fold is None or r["fold"] == fold]

        self.condition = condition
        self.max_length = max_length
        self.ids = [r["id"] for r in records]
        self.labels = [int(r["label"]) for r in records]

        built = [build_inputs(r) for r in records]
        texts_a = [b["text_a"] for b in built]

        # Tokenized once here instead of in __getitem__, so no epoch pays for it twice and
        # the ids are fixed for the whole run.
        if condition == "A":
            self.input_ids = _encode_single(tokenizer, texts_a, batch_size)
            self._guard_answer_block()
        else:
            # only_second belongs to B alone. A has no second segment to cut, and cutting
            # it would mean cutting the answer.
            self.input_ids = _encode_pair(
                tokenizer, texts_a, [b["source"] for b in built], max_length, batch_size
            )

    def _guard_answer_block(self):
        for rid, ids in zip(self.ids, self.input_ids):
            if len(ids) > self.max_length:
                raise ValueError(
                    f"response {rid} tokenizes to {len(ids)} tokens, over max_length "
                    f"{self.max_length}, and condition A is never truncated"
                )

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, i: int) -> dict:
        ids = self.input_ids[i]
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "label": self.labels[i],
            "id": self.ids[i],
        }


def pad_batch(examples: Sequence[dict], pad_token_id: int) -> dict:
    width = max(len(e["input_ids"]) for e in examples)
    input_ids, attention_mask = [], []
    for e in examples:
        ids = e["input_ids"]
        gap = width - len(ids)
        input_ids.append(list(ids) + [pad_token_id] * gap)
        attention_mask.append([1] * len(ids) + [0] * gap)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": [e["label"] for e in examples],
        "ids": [e["id"] for e in examples],
    }


def make_collate_fn(pad_token_id: int, as_tensors: bool = True):
    def collate(examples):
        batch = pad_batch(examples, pad_token_id)
        if not as_tensors:
            return batch
        import torch

        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long),
            "ids": batch["ids"],
        }

    return collate


def check_text_a_survives(tokenizer, records, max_length) -> None:
    built = [build_inputs(r) for r in records]
    raw_a = tokenizer([b["text_a"] for b in built], add_special_tokens=False)["input_ids"]
    grounded = RagTruthDataset(records, tokenizer, "B", max_length)
    for rid, a_ids, b_ids in zip(grounded.ids, raw_a, grounded.input_ids):
        if b_ids[1:1 + len(a_ids)] != a_ids:
            raise AssertionError(
                f"response {rid}: text_a did not survive condition B intact"
            )


def check_same_examples(tokenizer, records, max_length) -> None:
    blind = RagTruthDataset(records, tokenizer, "A", max_length)
    grounded = RagTruthDataset(records, tokenizer, "B", max_length)
    if blind.ids != grounded.ids:
        raise AssertionError("A and B do not carry the same response ids in the same order")
    if blind.labels != grounded.labels:
        raise AssertionError("A and B do not carry the same labels")


def check_a_is_prefix_of_b(tokenizer, records, max_length) -> None:
    blind = RagTruthDataset(records, tokenizer, "A", max_length)
    grounded = RagTruthDataset(records, tokenizer, "B", max_length)
    # A is [CLS] text_a [SEP] and B repeats that through the same [SEP], so A should be a
    # straight prefix of B. If it is not, the two conditions drifted somewhere upstream.
    for rid, a_ids, b_ids in zip(blind.ids, blind.input_ids, grounded.input_ids):
        if b_ids[:len(a_ids)] != a_ids:
            raise AssertionError(
                f"response {rid}: condition A is not a prefix of condition B"
            )


def check_source_scramble_leaves_a_alone(tokenizer, records, max_length) -> None:
    scrambled = []
    for r in records:
        s = scrambled_copy(r)
        # scrambled_copy overwrites everything outside the text_a whitelist, so put back
        # the two fields the dataset reads.
        s["id"], s["label"] = r["id"], r["label"]
        scrambled.append(s)

    grounded = RagTruthDataset(records, tokenizer, "B", max_length)
    grounded_scrambled = RagTruthDataset(scrambled, tokenizer, "B", max_length)
    if grounded.input_ids == grounded_scrambled.input_ids:
        raise AssertionError(
            "scrambling did not move condition B, so this check proves nothing"
        )

    blind = RagTruthDataset(records, tokenizer, "A", max_length)
    blind_scrambled = RagTruthDataset(scrambled, tokenizer, "A", max_length)
    if blind.input_ids != blind_scrambled.input_ids:
        raise AssertionError("condition A token ids moved when the source changed")


def check_labels_match_by_id(tokenizer, records, max_length) -> None:
    want = {r["id"]: int(r["label"]) for r in records}
    for condition in CONDITIONS:
        ds = RagTruthDataset(records, tokenizer, condition, max_length)
        for i in range(len(ds)):
            example = ds[i]
            if example["label"] != want[example["id"]]:
                raise AssertionError(
                    f"response {example['id']} in condition {condition} carries label "
                    f"{example['label']}, the record says {want[example['id']]}"
                )


def run_token_checks(tokenizer, records, max_length) -> None:
    check_text_a_survives(tokenizer, records, max_length)
    check_same_examples(tokenizer, records, max_length)
    check_a_is_prefix_of_b(tokenizer, records, max_length)
    check_source_scramble_leaves_a_alone(tokenizer, records, max_length)
    check_labels_match_by_id(tokenizer, records, max_length)
