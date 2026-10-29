#!/bin/bash
# Downstream finetune / probe (main.py) in one run directory. Rerunnable: if RUN_DIR/DONE exists
# it exits; otherwise main.py resumes from RUN_DIR/ckpt.pt when a previous job was cut off by the
# time limit (checkpoint every --save-interval steps). Normally submitted by scripts/xp.py.
#
# Submit from the repo root:
#   sbatch --export=ALL,CONFIG=configs/as20k/as20k_finetune_config.yaml,RUN_DIR=runs/bat_as20k_ft,\
#   DATA_DIR=/path/to/AudioSet,INIT_CKPT=$HOME/.cache/bat_ssl/BAT_base.pt slurm/finetune_bat.slurm
# Optional: MODEL_NAME (default BAT), EXTRA (more main.py arguments, space-separated),
# ENV_SCRIPT (sourced instead of the module load), TEST_WORKERS (default 6).
#
# Host memory: JZ gives ~2.5 GB per requested CPU (60 GB for 24). The 24 training workers stay
# alive during evals, so the eval loader must stay small: 24 + 24 workers peaked at 77 GB and were
# OOM-killed at the first eval; 6 eval workers read the 18,886 eval clips in ~1-2 min.

#SBATCH --job-name=bat_finetune
#SBATCH --account=soz@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=20:00:00
#SBATCH --output=%x_%j.out

set -euo pipefail

if [ -n "${ENV_SCRIPT:-}" ]; then
    source "$ENV_SCRIPT"
else
    module purge
    module load arch/h100  # H100 nodes have their own module tree: without it torch fails on libmpi.so.40
    module load pytorch-gpu/py3/2.8.0
fi

: "${CONFIG:?Set CONFIG}"
: "${RUN_DIR:?Set RUN_DIR}"
: "${DATA_DIR:?Set DATA_DIR}"
cd "$SLURM_SUBMIT_DIR"

if [ -f "$RUN_DIR/DONE" ]; then
    echo "$RUN_DIR/DONE exists: nothing to do."
    exit 0
fi
[ -f "$RUN_DIR/ckpt.pt" ] && echo "Resuming from $RUN_DIR/ckpt.pt"

INIT_ARGS=()
if [ -n "${INIT_CKPT:-}" ]; then
    if [ ! -f "$INIT_CKPT" ]; then
        echo "INIT_CKPT not found: $INIT_CKPT" >&2
        exit 1
    fi
    INIT_ARGS=(--pretrained-path "$INIT_CKPT")
fi
read -r -a EXTRA_SPLIT <<< "${EXTRA:-}"

python main.py -c "$CONFIG" \
    --run-dir "$RUN_DIR" \
    --dataset-dir "$DATA_DIR" \
    --model-name "${MODEL_NAME:-BAT}" \
    --vit-size b \
    --train-num-workers "$SLURM_CPUS_PER_TASK" \
    --test-num-workers "${TEST_WORKERS:-6}" \
    "${INIT_ARGS[@]}" \
    "${EXTRA_SPLIT[@]}"
