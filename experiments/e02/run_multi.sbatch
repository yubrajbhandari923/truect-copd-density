#!/bin/bash
# One job, several runs side by side, assigned round-robin to the job's GPUs (owner preference: one job, parallel inside).
# Usage (from the staged snapshot dir): sbatch --gres=gpu:N -c <10*N> experiments/e02/run_multi.sbatch cfg1.json ... cfgN.json
# Runs resume from their own checkpoints on requeue. Per-run logs: logs/<jobname>-<jobid>-<config>.log
#SBATCH --partition=defq
#SBATCH --mem=400G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/railabs/yb107/research-output/truect-copd-density/logs/%x-%j.out
set -uo pipefail
W=/scratch/railabs/yb107/research-output/truect-copd-density
export TMPDIR=/tmp/yb107-$SLURM_JOB_ID XDG_CACHE_HOME=$W/cache MPLCONFIGDIR=$W/cache/mpl PYTHONPYCACHEPREFIX=$W/cache/pyc TORCH_HOME=$W/cache/torch
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$SLURM_SUBMIT_DIR PATH=$W/.venv/bin:$PATH
mkdir -p $TMPDIR; trap "rm -rf $TMPDIR" EXIT
cd "$SLURM_SUBMIT_DIR/experiments/e02"
echo "snapshot=$SLURM_SUBMIT_DIR job=$SLURM_JOB_ID host=$(hostname) restarts=${SLURM_RESTART_COUNT:-0} configs=$*"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
IFS=, read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
pids=()
i=0
for cfg in "$@"; do
  CUDA_VISIBLE_DEVICES=${GPUS[$((i % ${#GPUS[@]}))]} python train.py "$cfg" > "$W/logs/$SLURM_JOB_NAME-$SLURM_JOB_ID-$(basename "$cfg" .json).log" 2>&1 &
  pids+=($!); i=$((i + 1))
done
rc=0
for j in "${!pids[@]}"; do
  wait "${pids[$j]}"; s=$?; echo "run $(basename "${@:$((j + 1)):1}") exit $s"; [ $s -eq 0 ] || rc=1
done
exit $rc
