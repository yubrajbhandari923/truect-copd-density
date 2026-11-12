#!/bin/bash
# Usage: sbatch [slurm opts] run.sbatch <script.py> [args]   (submit from the staged code snapshot dir)
#SBATCH --partition=defq
#SBATCH --output=/scratch/railabs/yb107/research-output/truect-copd-density/logs/%x-%j.out
set -euo pipefail
W=/scratch/railabs/yb107/research-output/truect-copd-density
export TMPDIR=$W/tmp XDG_CACHE_HOME=$W/cache MPLCONFIGDIR=$W/cache/mpl PYTHONPYCACHEPREFIX=$W/cache/pyc
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR/experiments/e00"
echo "snapshot=$SLURM_SUBMIT_DIR job=$SLURM_JOB_ID host=$(hostname) cmd=$*"
exec "$W/.venv/bin/python" "$@"
