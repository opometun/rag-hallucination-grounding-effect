# rag-hallucination-grounding-effect

Does retrieved grounding help detect RAG hallucinations? A ModernBERT ablation on RAGTruth.

Two input conditions are compared at matched seeds. Condition A sees the task and the
response only. Condition B sees the same text plus the retrieved source appended as a
second segment. Everything else about the two runs is identical.

## Layout

    data/raw/RAGTruth/dataset/   response.jsonl and source_info.jsonl
    ModernBERT-base/             local encoder and tokenizer, used offline
    splits/val_source_ids.json   frozen validation split, committed
    src/                         data, model, metrics, training loop, CLI runner
    scripts/                     data preparation, smoke tests, LR selection
    run_all.sbatch               one SLURM job: smoke, LR grid, selection, final runs

## 1. Data

Put the RAGTruth files at these exact paths:

    data/raw/RAGTruth/dataset/response.jsonl
    data/raw/RAGTruth/dataset/source_info.jsonl

The loader joins the directory it is given with the filename, so --data-dir has to point
at the dataset/ directory, not at data/raw/RAGTruth/.

## 2. Model

Put a local copy of ModernBERT-base at the repo root:

    ModernBERT-base/config.json
    ModernBERT-base/model.safetensors
    ModernBERT-base/tokenizer.json
    ModernBERT-base/tokenizer_config.json
    ModernBERT-base/special_tokens_map.json

Nothing in the pipeline passes a hub name, so no download happens at run time.

## 3. Environment

    python -m venv myvenv
    source myvenv/bin/activate
    pip install -r requirements.txt

## 4. Offline mode

    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1

run_all.sbatch exports both itself. Set them by hand for single runs.

## 5. Data preparation

    python scripts/prepare_data.py

Loads and joins the two jsonl files, drops quality flagged responses, derives labels, and
writes splits/val_source_ids.json if that file does not already exist. It reads the data
from the default location above and takes no arguments.

That split file is frozen. The training runner loads it and exits non-zero if it is
missing, rather than drawing a new one, because a redraw would train on a different split
than the one already validated.

## 6. Running

logs/ has to exist before you submit, because SLURM opens the job log before the script
body runs:

    mkdir -p logs
    sbatch run_all.sbatch

Run from the repo root. The job runs the worst case smoke, then a four run LR grid at
seed 7 over 1e-5 and 3e-5, selects a rate, then six final runs (two conditions by seeds
42, 1337 and 2026, each with --final so the test set is scored once).

A single run without SLURM:

    python -m src.train \
      --model-dir ModernBERT-base \
      --data-dir data/raw/RAGTruth/dataset \
      --condition A --seed 42 --lr 1e-5 \
      --out results/dev

Add --final to also score the test set. Add --limit N to cap records per fold for a smoke
run.

## 7. Outputs

Grid:

    results/grid/lr<rate>/<condition>_seed7_result.json
    results/grid/lr<rate>/<condition>_seed7_val_logits.jsonl
    results/grid/lr_decision.json

Finals:

    results/finals/<condition>_seed<seed>_result.json
    results/finals/<condition>_seed<seed>_val_logits.jsonl
    results/finals/<condition>_seed<seed>_test_logits.jsonl
    results/finals/<condition>_seed<seed>_best.pt

Each grid rate gets its own directory because a run filename carries only the condition
and the seed, so two rates at the same seed would otherwise overwrite each other.

Every run writes its result.json last, so the presence of that file means the run
finished. Both loops in run_all.sbatch use it as the resume guard, which means a requeued
job skips runs that already completed.

Checkpoints, logs and results are gitignored.
