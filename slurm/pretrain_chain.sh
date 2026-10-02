#!/bin/bash
# Run slurm/pretrain_bat.slurm as N chained 2 h jobs on the dev QOS: each job starts when the
# previous one ends (afterany: also after a timeout) and resumes from the latest checkpoint in
# the config's log_dir. Jobs left after the run reaches optimization_steps exit right away.
#
#   cd $WORK/BAT_ICML2026
#   slurm/pretrain_chain.sh <n_jobs> DATA_DIR=...,ENV_SCRIPT=...,CONFIG=configs/ssl_bee_lowlr.yaml
#
# Extra variables: SAVE_INTERVAL (default 1000 steps, so a timeout loses little), GPUS (default 4),
# QOS (default qos_gpu_h100-dev). The dev QOS counts pending jobs too: n_jobs <= 10.

set -euo pipefail

N_JOBS="${1:?usage: slurm/pretrain_chain.sh <n_jobs> DATA_DIR=...,CONFIG=...}"
EXPORTS="${2:?usage: slurm/pretrain_chain.sh <n_jobs> DATA_DIR=...,CONFIG=...}"
GPUS="${GPUS:-4}"
QOS="${QOS:-qos_gpu_h100-dev}"

args=(--qos="$QOS" --time=02:00:00 --gres=gpu:"$GPUS" --ntasks-per-node="$GPUS"
      --export=ALL,AUTO_RESUME=1,SAVE_INTERVAL="${SAVE_INTERVAL:-1000}","$EXPORTS")
job=$(sbatch --parsable "${args[@]}" slurm/pretrain_bat.slurm)
echo "job 1: $job"
for i in $(seq 2 "$N_JOBS"); do
    job=$(sbatch --parsable --dependency=afterany:"$job" "${args[@]}" slurm/pretrain_bat.slurm)
    echo "job $i: $job"
done
