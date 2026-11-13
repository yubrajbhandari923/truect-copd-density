#!/bin/bash
# Usage (from the staged snapshot dir): sbatch run.sbatch OUT_DIR [--smoke]
#SBATCH --partition=defq
#SBATCH --job-name=e01-baselines
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/railabs/yb107/research-output/truect-copd-density/logs/%x-%j.out
set -euo pipefail
W=/scratch/railabs/yb107/research-output/truect-copd-density
export TMPDIR=$W/tmp XDG_CACHE_HOME=$W/cache MPLCONFIGDIR=$W/cache/mpl PYTHONPYCACHEPREFIX=$W/cache/pyc
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$SLURM_SUBMIT_DIR SNAPSHOT=$SLURM_SUBMIT_DIR
echo "snapshot=$SLURM_SUBMIT_DIR job=$SLURM_JOB_ID host=$(hostname) args=$*"
exec "$W/.venv/bin/python" "$SLURM_SUBMIT_DIR/experiments/e01/baselines.py" "$@"
