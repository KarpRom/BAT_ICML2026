#!/bin/bash
# AudioSet sample-rate / bandwidth / time-scale study (scripts/study/audio_stats.py) as a CPU
# array job: one task per shard, shard = SLURM_ARRAY_TASK_ID. Finished shards are skipped, so
# resubmitting the same command only redoes what is missing (timeouts, node failures).
#
# Submit from the repo root. NUM_SHARDS must match the --array range (0..NUM_SHARDS-1):
#   sbatch --array=0-49 --export=ALL,AUDIOSET=/path/to/AudioSet,OUT_DIR=study,NUM_SHARDS=50 slurm/study_array.slurm
# Only the missing shards:
#   sbatch --array=$(python scripts/study/audio_stats.py missing --out-dir study --num-shards 50) \
#       --export=ALL,AUDIOSET=...,OUT_DIR=study,NUM_SHARDS=50 slurm/study_array.slurm
# Optional: ENV_SCRIPT (sourced instead of the module load below), ACCOUNT override with
# sbatch --account=<project>@cpu.

#SBATCH --job-name=as_study
#SBATCH --account=soz@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --time=00:20:00
#SBATCH --output=%x_%A_%a.out

set -euo pipefail

if [ -n "${ENV_SCRIPT:-}" ]; then
    source "$ENV_SCRIPT"
else
    module purge
    module load pytorch-gpu/py3/2.8.0
fi

: "${AUDIOSET:?Set AUDIOSET to the AudioSet root (eval_segments/, balanced_train_segments/, ...)}"
: "${NUM_SHARDS:?Set NUM_SHARDS (must match the --array range)}"
OUT_DIR="${OUT_DIR:-study}"
cd "$SLURM_SUBMIT_DIR"

# One file list for every task: task 0 builds it (a few minutes on Lustre), the others wait.
if [ ! -f "$OUT_DIR/files.csv" ]; then
    if [ "${SLURM_ARRAY_TASK_ID:-0}" = 0 ]; then
        python scripts/study/audio_stats.py list --dataset-dir "$AUDIOSET" --out-dir "$OUT_DIR"
    else
        while [ ! -f "$OUT_DIR/files.csv" ]; do sleep 10; done
    fi
fi

python scripts/study/audio_stats.py run --out-dir "$OUT_DIR" --num-shards "$NUM_SHARDS" \
    --shard "$SLURM_ARRAY_TASK_ID" --workers "$SLURM_CPUS_PER_TASK"
