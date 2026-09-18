"""Measure tokenized input lengths on the RAGTruth training split, to help pick max_length.

Measurement only: trains nothing, writes nothing, just prints a table.
"""

import sys
from pathlib import Path

import numpy as np
from src.config  import MAX_LENGTH

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.inputs import build_inputs
from src.data.load import load_joined

MODEL = "answerdotai/ModernBERT-base"
PRIMARY_CAP = MAX_LENGTH
LARGER_CAP = 8192
FAMILIES = ("QA", "Summary", "Data2txt")
DROP_QUALITY = ("incorrect_refusal", "truncated")
BATCH = 256


def token_lengths(tok, texts):
    lengths = []
    for i in range(0, len(texts), BATCH):
        enc = tok(
            texts[i:i + BATCH],
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )
        lengths.extend(len(ids) for ids in enc["input_ids"])
    return np.array(lengths, dtype=np.int64)


def check_pair_accounting(tok, pairs, a_len, s_len, overhead):
    # Encoding all 15k pairs twice is slow and the pair length is just a + b + overhead
    # for a template post-processor, so derive it and spot check the derivation here
    # rather than trusting it.
    idx = np.linspace(0, len(pairs) - 1, num=min(32, len(pairs)), dtype=int)
    enc = tok(
        [pairs[i][0] for i in idx],
        [pairs[i][1] for i in idx],
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )
    for j, i in enumerate(idx):
        real = len(enc["input_ids"][j])
        derived = int(a_len[i]) + int(s_len[i]) + overhead
        if real != derived:
            raise SystemExit(
                f"pair length accounting is off at record {i}: tokenizer says {real}, "
                f"derived {derived}. This tokenizer does not build pairs by simple "
                f"concatenation, so tokenize the pairs directly instead of deriving."
            )


def fmt_pct(x):
    return f"{100 * x:.1f}%"


def main():
    df = load_joined()
    # Train only. The cap is a modelling choice, so reading test here would leak.
    train = df[(df["split"] == "train") & (~df["quality"].isin(DROP_QUALITY))]
    records = train.to_dict("records")
    built = [build_inputs(r) for r in records]
    task_types = np.array([r["task_type"] for r in records])
    print(f"model      {MODEL}")
    print(f"records    {len(records):,} train rows after dropping quality in {DROP_QUALITY}")

    from transformers import AutoTokenizer
    from transformers import logging as hf_logging

    # Untruncated encoding of long text warns on every batch otherwise.
    hf_logging.set_verbosity_error()
    tok = AutoTokenizer.from_pretrained(MODEL)

    # Ask the tokenizer how many specials the pair form adds instead of assuming 3.
    overhead = tok.num_special_tokens_to_add(pair=True)
    print(f"specials   {overhead} added by the pair form, {tok.num_special_tokens_to_add(pair=False)} by a single")

    a_len = token_lengths(tok, [b["text_a"] for b in built])
    s_len = token_lengths(tok, [b["source"] for b in built])
    pairs = [(b["text_a"], b["source"]) for b in built]
    check_pair_accounting(tok, pairs, a_len, s_len, overhead)

    full = a_len + s_len + overhead
    # Training uses truncation="only_second", so text_a is never cut and the source
    # absorbs the whole overflow. That is what makes retention the number to look at.
    a_total = a_len + overhead
    budget = PRIMARY_CAP - a_total
    kept = np.minimum(np.clip(budget, 0, None), s_len)
    retention = kept / np.maximum(s_len, 1)

    groups = [(f, task_types == f) for f in FAMILIES] + [("overall", np.ones(len(full), bool))]

    print()
    print("full input length, text_a + source + specials, untruncated")
    head = f"{'family':<10}{'n':>7}{'p50':>8}{'p90':>8}{'p95':>8}{'p99':>8}{'max':>9}{'>4096':>9}{'>8192':>9}"
    print(head)
    print("-" * len(head))
    for name, m in groups:
        f = full[m]
        print(
            f"{name:<10}{len(f):>7,}"
            f"{int(np.percentile(f, 50)):>8,}{int(np.percentile(f, 90)):>8,}"
            f"{int(np.percentile(f, 95)):>8,}{int(np.percentile(f, 99)):>8,}"
            f"{int(f.max()):>9,}"
            f"{fmt_pct((f > PRIMARY_CAP).mean()):>9}{fmt_pct((f > LARGER_CAP).mean()):>9}"
        )

    print()
    print(f"answer safety at {PRIMARY_CAP}: text_a has to fit or the response itself gets cut")
    head = f"{'family':<10}{'max text_a':>12}{'headroom':>10}{'verdict':>9}"
    print(head)
    print("-" * len(head))
    for name, m in groups:
        mx = int(a_total[m].max())
        verdict = "OK" if mx <= PRIMARY_CAP else "FAIL"
        print(f"{name:<10}{mx:>12,}{PRIMARY_CAP - mx:>10,}{verdict:>9}")

    print()
    print(f"source retention at {PRIMARY_CAP}, over the records that overflow")
    head = f"{'family':<10}{'overflowing':>12}{'lose source':>13}{'median kept':>13}{'worst kept':>12}"
    print(head)
    print("-" * len(head))
    for name, m in groups:
        over = m & (full > PRIMARY_CAP)
        n_over = int(over.sum())
        if n_over == 0:
            print(f"{name:<10}{0:>12}   no record overflows {PRIMARY_CAP}")
            continue
        lost = int((m & (kept < s_len)).sum())
        r = retention[over]
        print(
            f"{name:<10}{n_over:>12,}{lost:>13,}"
            f"{fmt_pct(float(np.median(r))):>13}{fmt_pct(float(r.min())):>12}"
        )

    print()
    n = len(full)
    o4 = int((full > PRIMARY_CAP).sum())
    o8 = int((full > LARGER_CAP).sum())
    if o4 == 0:
        print(
            f"reading: {PRIMARY_CAP} already covers every training record "
            f"(longest is {int(full.max()):,}), so nothing is truncated and {LARGER_CAP} buys nothing."
        )
    else:
        r = retention[full > PRIMARY_CAP]
        print(
            f"reading: {PRIMARY_CAP} truncates source on {o4:,}/{n:,} records ({fmt_pct(o4 / n)}), "
            f"keeping a median {fmt_pct(float(np.median(r)))} and a worst {fmt_pct(float(r.min()))} "
            f"of their source; at {LARGER_CAP} that falls to {o8:,}/{n:,} ({fmt_pct(o8 / n)}), "
            f"longest input {int(full.max()):,} tokens."
        )


if __name__ == "__main__":
    main()
