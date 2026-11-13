#!/bin/bash
# One job, parallel inside (owner preference): repair organ_pt, then export all 67 subjects to NIfTI, 16 at a time.
# Submit from the staged code snapshot dir:  sbatch experiments/e00/export_all.sbatch [dest_root]
#SBATCH --partition=defq
#SBATCH -J e00-nifti-all
#SBATCH -c 64
#SBATCH --mem=400G
#SBATCH -t 06:00:00
#SBATCH --output=/scratch/railabs/yb107/research-output/truect-copd-density/logs/%x-%j.out
set -euo pipefail
W=/scratch/railabs/yb107/research-output/truect-copd-density
export TMPDIR=$W/tmp XDG_CACHE_HOME=$W/cache MPLCONFIGDIR=$W/cache/mpl PYTHONPYCACHEPREFIX=$W/cache/pyc
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR/experiments/e00"
PY=$W/.venv/bin/python
IDS=$($PY -c "from common import pairing; print(' '.join(s for s, _ in pairing()))")
echo "snapshot=$SLURM_SUBMIT_DIR job=$SLURM_JOB_ID host=$(hostname)"
echo $IDS | tr ' ' '\n' | xargs -P 16 -n 1 $PY fix_organ_pt.py
echo $IDS | tr ' ' '\n' | xargs -P 16 -I{} sh -c "$PY export_nifti.py {} ${1:-} > $W/logs/e00-nifti-all-$SLURM_JOB_ID-{}.log 2>&1 || echo FAILED {}"
echo finished
