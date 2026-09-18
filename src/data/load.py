"""Load, join and sanity-check the RAGTruth dataset.

Read-only inspection layer for the response-level hallucination-detection
experiment. Nothing here tokenizes, truncates or derives a training label --
those belong to later phases, which import from this module.

The dataset is vendored (gitignored) at data/raw/RAGTruth/dataset/ and is
treated as strictly read-only.

Schema, as verified against the vendored copy:
  source_info.jsonl : source_id, task_type, source, source_info, prompt
  response.jsonl    : id, source_id, model, temperature, labels, split,
                      quality, response

Grounding text and task_type live on the SOURCE side; `split` is a field on the
RESPONSE; `source` is provenance (MARCO, Yelp, CNN/DM, Recent News) while
`source_info` is the grounding payload; `model` is the generator.

Caveat that the rest of the pipeline must honour: `source_info` is NOT uniformly
a string. It is a str only for Summary; for QA it is a dict with keys
{question, passages} and for Data2txt a dict of Yelp business fields. See
SOURCE_INFO_TYPES. Rendering it to text is a later-phase decision.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------- locations

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = REPO_ROOT / "data" / "raw" / "RAGTruth" / "dataset"

RESPONSE_FILE = "response.jsonl"
SOURCE_FILE = "source_info.jsonl"

# ------------------------------------------------------- verified constants

RESPONSE_COLUMNS = [
    "id", "source_id", "model", "temperature", "labels", "split", "quality",
    "response",
]
SOURCE_COLUMNS = ["source_id", "task_type", "source", "source_info", "prompt"]

#: Every annotated span carries exactly these keys.
SPAN_KEYS = [
    "start", "end", "text", "label_type", "implicit_true", "due_to_null", "meta",
]

TASK_TYPES = ["QA", "Summary", "Data2txt"]

#: Python type of the `source_info` value, per task_type.
SOURCE_INFO_TYPES = {"QA": dict, "Summary": str, "Data2txt": dict}

#: Claims from the project brief, checked (not trusted) by `check_counts`.
EXPECTED_COUNTS = {
    "train_responses": 15_090,
    "test_responses": 2_700,
    "train_sources": 2_515,
    "test_sources": 450,
    "total_sources": 2_965,
}


# -------------------------------------------------------------------- load

def _read_jsonl(path: Path) -> pd.DataFrame:
    """Read a JSON-lines file into a DataFrame, preserving nested values.

    stdlib json rather than pd.read_json so that dict/list-valued fields
    (`source_info`, `labels`) survive as Python objects instead of being
    coerced or normalised.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"RAGTruth file not found: {path}\n"
            f"Expected the vendored dataset under {DEFAULT_DATASET_DIR}."
        )
    with path.open(encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    return pd.DataFrame(records)


def load_responses(dataset_dir: Path | str = DEFAULT_DATASET_DIR) -> pd.DataFrame:
    """Load response.jsonl. One row per generated response."""
    df = _read_jsonl(Path(dataset_dir) / RESPONSE_FILE)
    _assert_columns(df, RESPONSE_COLUMNS, RESPONSE_FILE)
    return df


def load_sources(dataset_dir: Path | str = DEFAULT_DATASET_DIR) -> pd.DataFrame:
    """Load source_info.jsonl. One row per grounding source / prompt."""
    df = _read_jsonl(Path(dataset_dir) / SOURCE_FILE)
    _assert_columns(df, SOURCE_COLUMNS, SOURCE_FILE)
    return df


def _assert_columns(df: pd.DataFrame, expected: list[str], name: str) -> None:
    missing = [c for c in expected if c not in df.columns]
    extra = [c for c in df.columns if c not in expected]
    if missing or extra:
        raise AssertionError(
            f"{name} schema drift -- missing={missing} unexpected={extra}"
        )


# -------------------------------------------------------------------- join

def join(responses: pd.DataFrame, sources: pd.DataFrame) -> pd.DataFrame:
    """LEFT join responses onto sources on `source_id`.

    Left (not inner) so that an orphan response -- one whose source_id is absent
    from source_info.jsonl -- surfaces as a null rather than being silently
    dropped. Raises AssertionError if any orphan exists.
    """
    collisions = (set(responses.columns) & set(sources.columns)) - {"source_id"}
    if collisions:
        raise AssertionError(
            f"column collision would create _x/_y suffixes: {sorted(collisions)}"
        )

    merged = responses.merge(
        sources, on="source_id", how="left", validate="many_to_one", indicator=True
    )

    orphans = merged.loc[merged["_merge"] == "left_only"]
    if len(orphans):
        raise AssertionError(
            f"{len(orphans)} response(s) have no matching source_id; "
            f"examples: {orphans['source_id'].head().tolist()}"
        )

    null_grounding = merged["source_info"].isna().sum()
    if null_grounding:
        raise AssertionError(f"{null_grounding} row(s) have null source_info after join")

    return merged.drop(columns="_merge")


def assert_source_disjoint(df: pd.DataFrame) -> None:
    """Assert no source_id appears in both the train and test splits.

    Guards against leakage: responses are grouped per source, so a shared
    source would put near-identical grounding on both sides of the split.
    """
    train = set(df.loc[df["split"] == "train", "source_id"])
    test = set(df.loc[df["split"] == "test", "source_id"])
    overlap = train & test
    if overlap:
        raise AssertionError(
            f"{len(overlap)} source_id(s) appear in BOTH splits; "
            f"examples: {sorted(overlap)[:10]}"
        )


def load_joined(dataset_dir: Path | str = DEFAULT_DATASET_DIR) -> pd.DataFrame:
    """Convenience: load both files, join, and assert split disjointness."""
    df = join(load_responses(dataset_dir), load_sources(dataset_dir))
    assert_source_disjoint(df)
    return df


def check_counts(df: pd.DataFrame, sources: pd.DataFrame) -> dict[str, tuple[int, int]]:
    """Compare observed counts against EXPECTED_COUNTS.

    Returns {name: (expected, observed)}. Reports rather than raises, so a
    discrepancy is visible without blocking the rest of the report.
    """
    observed = {
        "train_responses": int((df["split"] == "train").sum()),
        "test_responses": int((df["split"] == "test").sum()),
        "train_sources": df.loc[df["split"] == "train", "source_id"].nunique(),
        "test_sources": df.loc[df["split"] == "test", "source_id"].nunique(),
        "total_sources": len(sources),
    }
    return {k: (v, observed[k]) for k, v in EXPECTED_COUNTS.items()}


# ------------------------------------------------------------------ report

def _rule(n: int, title: str) -> None:
    print(f"\n{'=' * 78}\n[{n}] {title}\n{'=' * 78}")


def _trunc(value: object, width: int = 300) -> str:
    s = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    s = " ".join(s.split())
    return s if len(s) <= width else s[:width] + f" ... [+{len(s) - width} chars]"


def report(dataset_dir: Path | str = DEFAULT_DATASET_DIR) -> pd.DataFrame:
    """Print the Step-1 verification report. Returns the joined frame."""
    dataset_dir = Path(dataset_dir)
    print(f"RAGTruth Step 1 -- read-only inspection\ndataset_dir: {dataset_dir}")

    responses = load_responses(dataset_dir)
    sources = load_sources(dataset_dir)

    # 1 -- schema
    _rule(1, "SCHEMA")
    for name, df, expected in [
        (SOURCE_FILE, sources, SOURCE_COLUMNS),
        (RESPONSE_FILE, responses, RESPONSE_COLUMNS),
    ]:
        keys = list(df.columns)
        print(f"{name}: {keys}")
        print(f"  matches expected: {keys == expected}   rows: {len(df):,}")
    print("\nfirst record of each file (values truncated):")
    for name, df in [(SOURCE_FILE, sources), (RESPONSE_FILE, responses)]:
        print(f"  -- {name}")
        for k, v in df.iloc[0].items():
            print(f"     {k:13s} {type(v).__name__:6s} {_trunc(v, 110)}")

    # 2 -- join
    _rule(2, "JOIN (left, response -> source_info on source_id)")
    df = join(responses, sources)
    print(f"rows in: {len(responses):,}   rows out: {len(df):,}   "
          f"(left join preserves row count: {len(df) == len(responses)})")
    print(f"null source_info after join: {int(df['source_info'].isna().sum())}  -- asserted == 0  PASS")
    print(f"orphan responses (source_id absent from source_info.jsonl): 0  -- asserted  PASS")
    unused = set(sources["source_id"]) - set(responses["source_id"])
    print(f"sources with no response attached: {len(unused)}")

    # 3 -- counts
    _rule(3, "COUNTS vs EXPECTED (read from the `split` field)")
    results = check_counts(df, sources)
    ok = True
    for name, (exp, obs) in results.items():
        if exp == obs:
            print(f"  {name:18s} expected {exp:>7,}   observed {obs:>7,}   MATCH")
        else:
            ok = False
            print(f"  {name:18s} expected {exp:>7,}   observed {obs:>7,}   "
                  f"*** MISMATCH ({obs - exp:+,}) ***")
    print(f"\nall expected counts matched: {ok}")
    print(f"split value counts (responses): {df['split'].value_counts().to_dict()}")
    print(f"split value counts (sources)  : "
          f"{df.drop_duplicates('source_id')['split'].value_counts().to_dict()}")

    # 4 -- disjointness
    _rule(4, "SOURCE-ID DISJOINTNESS (train vs test)")
    assert_source_disjoint(df)
    tr = set(df.loc[df["split"] == "train", "source_id"])
    te = set(df.loc[df["split"] == "test", "source_id"])
    print(f"train source ids: {len(tr):,}   test source ids: {len(te):,}   "
          f"intersection: {len(tr & te)}")
    print("assert_source_disjoint(df) -- PASS (no leakage across splits)")

    # 5 -- responses per source
    _rule(5, "RESPONSES PER SOURCE (distribution)")
    per_source = df.groupby("source_id").size()
    print(f"min {per_source.min()}   max {per_source.max()}   "
          f"mean {per_source.mean():.3f}   median {per_source.median():.1f}")
    print("value_counts (responses-per-source -> number of sources):")
    for k, v in per_source.value_counts().sort_index().items():
        print(f"  {k} responses: {v:,} sources")
    print(f"\ngenerator models ({df['model'].nunique()} distinct):")
    for k, v in df["model"].value_counts().items():
        print(f"  {k:28s} {v:,}")
    print("\nresponses-per-source by split:")
    for sp in ["train", "test"]:
        s = df[df["split"] == sp].groupby("source_id").size()
        print(f"  {sp:5s} min {s.min()} max {s.max()} "
              f"counts {s.value_counts().sort_index().to_dict()}")

    # 6 -- source vs source_info
    _rule(6, "`source` (provenance) vs `source_info` (grounding payload)")
    print("NOTE: source_info is NOT uniformly text -- see the type column.\n")
    picks = (df.drop_duplicates("source_id")
               .groupby(["task_type", "source"], as_index=False)
               .head(1)
               .head(6))
    for _, r in picks.iterrows():
        print(f"  source_id   : {r['source_id']}")
        print(f"  task_type   : {r['task_type']}")
        print(f"  source      : {r['source']!r}   <- provenance (a corpus name)")
        print(f"  source_info : [{type(r['source_info']).__name__}] "
              f"{_trunc(r['source_info'], 240)}")
        print(f"  prompt      : {_trunc(r['prompt'], 160)}")
        print()
    print("source (provenance) x task_type:")
    print(df.drop_duplicates("source_id")
            .groupby(["task_type", "source"]).size().to_string())

    # 7 -- task_type
    _rule(7, "TASK_TYPE -- exact distinct string values")
    print("sources:")
    for k, v in sources["task_type"].value_counts().items():
        print(f"  {k!r:12s} {v:,}")
    print("responses:")
    for k, v in df["task_type"].value_counts().items():
        print(f"  {k!r:12s} {v:,}")
    print(f"\nmatches expected {TASK_TYPES}: "
          f"{sorted(df['task_type'].unique()) == sorted(TASK_TYPES)}")
    print("\nresponses by task_type x split:")
    print(pd.crosstab(df["task_type"], df["split"]).to_string())
    print("\nsource_info Python type by task_type (sources):")
    st = sources.assign(t=sources["source_info"].map(lambda v: type(v).__name__))
    print(st.groupby(["task_type", "t"]).size().to_string())
    for tt, expected_t in SOURCE_INFO_TYPES.items():
        sub = sources.loc[sources["task_type"] == tt, "source_info"]
        allmatch = sub.map(lambda v: isinstance(v, expected_t)).all()
        inner = sorted({k for v in sub for k in v}) if expected_t is dict else "-"
        print(f"  {tt:9s} all {expected_t.__name__:4s}: {allmatch}   inner keys: {inner}")

    # 8 -- labels and quality
    _rule(8, "LABELS (span annotations) and QUALITY")
    spans = [s for L in df["labels"] for s in L]
    print(f"labels container type(s): "
          f"{set(type(v).__name__ for v in df['labels'])}")
    print(f"total annotated spans: {len(spans):,}")
    keysets = pd.Series([tuple(sorted(s.keys())) for s in spans]).value_counts()
    print("distinct span key-sets:")
    for ks, n in keysets.items():
        print(f"  n={n:,}: {list(ks)}")
    print(f"matches SPAN_KEYS: "
          f"{sorted(keysets.index[0]) == sorted(SPAN_KEYS)}")

    print("\nexample span (first response that has one):")
    ex = next(L[0] for L in df["labels"] if L)
    for k, v in ex.items():
        print(f"  {k:14s} {type(v).__name__:5s} {_trunc(v, 130)}")

    print("\nlabel_type values:")
    for k, v in pd.Series([s["label_type"] for s in spans]).value_counts().items():
        print(f"  {k!r:26s} {v:,}")
    print("\nimplicit_true / due_to_null -- present on EVERY span, as bools:")
    for f in ["implicit_true", "due_to_null"]:
        vals = pd.Series([s[f] for s in spans])
        print(f"  {f:14s} present on {vals.notna().sum():,}/{len(spans):,} spans   "
              f"types={set(type(v).__name__ for v in vals)}   "
              f"{vals.value_counts().to_dict()}")
    print("  (these are per-SPAN flags -- not response-level fields)")
    print("\n  `meta` field values (sample):")
    metas = pd.Series([_trunc(s["meta"], 60) for s in spans]).value_counts()
    print(f"    distinct: {len(metas)}   null/empty: "
          f"{sum(1 for s in spans if s['meta'] in (None, '', []))}")
    for k, v in metas.head(5).items():
        print(f"    {v:>6,}  {k}")

    empties = df.loc[df["labels"].map(len) == 0, "labels"]
    reprs = set(repr(v) for v in empties)
    print(f"\nclean responses (len(labels) == 0): {len(empties):,}")
    print(f"  distinct representations: {reprs}  -- genuinely empty list, no sentinel")
    print(f"  any None/NaN labels: {df['labels'].isna().sum()}")
    print(f"  spans-per-response: min {df['labels'].map(len).min()}  "
          f"max {df['labels'].map(len).max()}")
    print(f"  responses with >=1 span: {int((df['labels'].map(len) > 0).sum()):,}")

    print(f"\nquality -- distinct values ({df['quality'].nunique()}):")
    for k, v in df["quality"].value_counts().items():
        print(f"  {k!r:20s} {v:,}")
    print("\nquality x has_span -- separate axes (a response can be low-quality"
          "\nAND clean, or good-quality AND hallucinated):")
    print(pd.crosstab(df["quality"], df["labels"].map(len) > 0)
            .rename(columns={False: "no_spans", True: "has_spans"}).to_string())
    print("\nquality x split:")
    print(pd.crosstab(df["quality"], df["split"]).to_string())
    print("\n(response-level binary label deliberately NOT derived here -- later step.)")

    # 9 -- one full record per family
    _rule(9, "ONE FULL JOINED RECORD PER FAMILY (untruncated)")
    for tt in TASK_TYPES:
        sub = df[(df["task_type"] == tt) & (df["labels"].map(len) > 0)]
        row = (sub if len(sub) else df[df["task_type"] == tt]).iloc[0]
        print(f"\n{'-' * 78}\n--- task_type = {tt} ---\n{'-' * 78}")
        print(json.dumps(row.to_dict(), indent=2, ensure_ascii=False,
                         default=str))

    print(f"\n{'=' * 78}\nEND OF REPORT\n{'=' * 78}")
    return df


if __name__ == "__main__":
    report()
