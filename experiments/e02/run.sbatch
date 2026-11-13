#!/bin/bash
# Usage (from the staged snapshot dir): sbatch [opts] experiments/e02/run.sbatch <cmd...>
#   e.g. sbatch experiments/e02/run.sbatch python train.py configs/u3_s11.json
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/railabs/yb107/research-output/truect-copd-density/logs/%x-%j.out
set -euo pipefail
W=/scratch/railabs/yb107/research-output/truect-copd-density
export TMPDIR=$W/tmp XDG_CACHE_HOME=$W/cache MPLCONFIGDIR=$W/cache/mpl PYTHONPYCACHEPREFIX=$W/cache/pyc TORCH_HOME=$W/cache/torch
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$SLURM_SUBMIT_DIR PATH=$W/.venv/bin:$PATH
cd "$SLURM_SUBMIT_DIR/experiments/e02"
echo "snapshot=$SLURM_SUBMIT_DIR job=$SLURM_JOB_ID host=$(hostname) restarts=${SLURM_RESTART_COUNT:-0} cmd=$*"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
exec "$@"
