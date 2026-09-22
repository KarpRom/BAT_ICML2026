#!/bin/bash
# Continued BAT SSL pretraining on bee audio, warm-started from BAT_base.pt.
#
# Before first use:
#   1. On the front/login node (compute nodes have no internet):
#        python scripts/download_bat_checkpoint.py
#   2. Fill in <your_account>/-C below from your actual Jean Zay allocation --
#      these are placeholders, not guessed values.
#
# IMPORTANT: submit this from the repo root (BAT_ICML2026/), not from slurm/ -- it cd's
# to $SLURM_SUBMIT_DIR (the directory `sbatch` was run from) to find pretrain.py/configs/.
# ($0 can't be used for this: sbatch copies the script into a SLURM-managed spool
# directory before running it, so $(dirname "$0") resolves to that spool path, not
# wherever the script actually lives in your checkout.)
#
# Usage (N_GPUS below must match --gres/--ntasks-per-node for a given submission):
#   Smoke test, 1 GPU, existing small labeled corpus (bbeedeep/data/Cuts_Database_ALL_SPECIES):
#     sbatch --export=DATA_DIR=/path/to/bbeedeep/data/Cuts_Database/Cuts_Database_ALL_SPECIES slurm/pretrain_bat.slurm
#   Real run, N GPUs, the 500k-clip corpus:
#     sbatch --export=DATA_DIR=/path/to/500k_corpus --gres=gpu:4 --ntasks-per-node=4 --cpus-per-task=<N> slurm/pretrain_bat.slurm
#   Continuing a run cut off by the time limit: find the latest checkpoint from the
#   previous job (logs/SSL/step(N)_BAT_state_*.pt, highest N) and pass it as RESUME_CKPT --
#   this restores optimizer/scheduler/step state too, not just model weights:
#     sbatch --export=DATA_DIR=/path/to/500k_corpus,RESUME_CKPT=/path/to/logs/SSL/step(50000)_BAT_state_....pt \
#         --gres=gpu:4 --ntasks-per-node=4 --cpus-per-task=<N> slurm/pretrain_bat.slurm

#SBATCH --job-name=bat_ssl_pretrain
#SBATCH --account=soz@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                # = number of GPUs used; override at submission for multi-GPU
#SBATCH --gres=gpu:1                       # override at submission for multi-GPU
#SBATCH --cpus-per-task=24                 # override to match what your account/partition allows per GPU
#SBATCH --time=20:00:00
#SBATCH --output=%x_%j.out

set -euo pipefail

module purge
module load pytorch-gpu/py3/2.8.0          # match whatever module the front-node download used

: "${DATA_DIR:?Set DATA_DIR, e.g. sbatch --export=DATA_DIR=/path/to/wavs pretrain_bat.slurm}"
INIT_CKPT="$HOME/.cache/bat_ssl/BAT_base.pt"
if [ ! -f "$INIT_CKPT" ]; then
    echo "Checkpoint not found at $INIT_CKPT -- run 'python scripts/download_bat_checkpoint.py' on the front node first." >&2
    exit 1
fi

cd "$SLURM_SUBMIT_DIR"  # repo root -- see the IMPORTANT note above; must submit from there

RESUME_ARGS=()
if [ -n "${RESUME_CKPT:-}" ]; then
    if [ ! -f "$RESUME_CKPT" ]; then
        echo "RESUME_CKPT set but not found: $RESUME_CKPT" >&2
        exit 1
    fi
    echo "Resuming from $RESUME_CKPT"
    RESUME_ARGS=(--resume-checkpoint "$RESUME_CKPT")
fi

torchrun --nproc_per_node="$SLURM_NTASKS_PER_NODE" --master_port=29500 pretrain.py \
    -c configs/ssl_bee.yaml \
    --dataset-dir "$DATA_DIR" \
    --init-checkpoint "$INIT_CKPT" \
    --num-workers "$SLURM_CPUS_PER_TASK" \
    "${RESUME_ARGS[@]}"
