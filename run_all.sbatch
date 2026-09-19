#!/bin/bash
#SBATCH --job-name="AIP_C1_all"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs results results/grid results/finals

spack load /f6bu3ie && source myvenv/bin/activate

MODEL_DIR=ModernBERT-base
# src/data/load.py joins this directly with response.jsonl, and the two jsonl files sit
# under dataset/, so this has to point one level deeper than the repo folder.
DATA_DIR=data/raw/RAGTruth/dataset

echo "=== step 1: worst case condition B smoke ==="
python scripts/smoke_worstcase_b.py "$MODEL_DIR" || { echo "smoke failed, aborting"; exit 1; }

echo "=== step 2: LR grid, seed 7, validation only ==="
for COND in A B; do
  for LR in 1e-5 3e-5; do
    # The run stem is condition and seed only, so both rates would write the same
    # filename in one directory. Give each rate its own.
    OUT="results/grid/lr${LR}"
    if [ -f "${OUT}/${COND}_seed7_result.json" ]; then
      echo "skip grid ${COND} ${LR}"
      continue
    fi
    mkdir -p "$OUT"
    python -m src.train --model-dir "$MODEL_DIR" --data-dir "$DATA_DIR" \
      --condition "$COND" --seed 7 --lr "$LR" --out "$OUT" \
      --device auto --precision auto \
      || { echo "grid run failed: ${COND} lr=${LR}"; exit 1; }
  done
done

echo "=== step 3: select LR ==="
CHOSEN=$(python scripts/select_lr.py results/grid | tail -1) || { echo "select_lr failed"; exit 1; }
echo "chosen LR: ${CHOSEN}"

echo "=== step 4: final runs, 2 conditions by 3 seeds ==="
for COND in A B; do
  for SEED in 42 1337 2026; do
    if [ -f "results/finals/${COND}_seed${SEED}_result.json" ]; then
      echo "skip final ${COND} ${SEED}"
      continue
    fi
    python -m src.train --model-dir "$MODEL_DIR" --data-dir "$DATA_DIR" \
      --condition "$COND" --seed "$SEED" --lr "$CHOSEN" --out results/finals --final \
      --device auto --precision auto \
      || { echo "final run failed: ${COND} seed=${SEED}"; exit 1; }
  done
done

echo "all runs complete"
