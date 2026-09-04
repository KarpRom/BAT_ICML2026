#!/bin/bash
# Continued BAT SSL pretraining on bee audio, warm-started from BAT_base.pt.
#
# Before first use:
#   1. On the front/login node (compute nodes have no internet):
#        python scripts/download_bat_checkpoint.py
#   2. Fill in <your_account>/-C below from your actual Jean Zay allocation --
#      these are placeholders, not guessed values.
#
# Usage (N_GPUS below must match --gres/--ntasks-per-node for a given submission):
#   Smoke test, 1 GPU, existing small labeled corpus (bbeedeep/data/Cuts_Database_ALL_SPECIES):
#     sbatch --export=DATA_DIR=/path/to/bbeedeep/data/Cuts_Database/Cuts_Database_ALL_SPECIES pretrain_bat.slurm
#   Real run, N GPUs, the 500k-clip corpus:
#     sbatch --export=DATA_DIR=/path/to/500k_corpus --gres=gpu:4 --ntasks-per-node=4 --cpus-per-task=<N> pretrain_bat.slurm

#SBATCH --job-name=bat_ssl_pretrain
#SBATCH --account=<your_account>@h100      # fill in from your Jean Zay allocation
#SBATCH -C h100                            # or a100, matching --account above
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                # = number of GPUs used; override at submission for multi-GPU
#SBATCH --gres=gpu:1                       # override at submission for multi-GPU
#SBATCH --cpus-per-task=16                 # override to match what your account/partition allows per GPU
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

cd "$(dirname "$0")/.."  # repo root, so `pretrain.py`/`configs/` resolve regardless of submission cwd

torchrun --nproc_per_node="$SLURM_NTASKS_PER_NODE" --master_port=29500 pretrain.py \
    -c configs/ssl_bee.yaml \
    --dataset-dir "$DATA_DIR" \
    --init-checkpoint "$INIT_CKPT" \
    --num-workers "$SLURM_CPUS_PER_TASK"
