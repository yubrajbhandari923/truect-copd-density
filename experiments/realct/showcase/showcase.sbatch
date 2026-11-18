#!/bin/bash
# Showcase package: u5a_wide on 3 TrueCT VAL subjects + TotalSeg, copy 3 real-CT cases, 3D renders of all.
# Run from the staged snapshot dir: sbatch experiments/realct/showcase/showcase.sbatch
#SBATCH --partition=defq
#SBATCH --nodelist=node001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --job-name=showcase
#SBATCH --output=/scratch/railabs/yb107/research-output/truect-copd-density/logs/%x-%j.out
set -euo pipefail
W=/scratch/railabs/yb107/research-output/truect-copd-density
OUT=${OUT:-$W/results/realct/showcase}
export TMPDIR=/tmp/yb107-$SLURM_JOB_ID XDG_CACHE_HOME=$W/cache MPLCONFIGDIR=$W/cache/mpl TOTALSEG_HOME_DIR=$W/cache/totalseg
export PYTHONPATH=$SLURM_SUBMIT_DIR OMP_NUM_THREADS=8
mkdir -p $TMPDIR $OUT; trap "rm -rf $TMPDIR" EXIT
VIS=/home/yb107/MedImgResearch/.venv/bin/python  # owner's env (pyvista 0.47.1, vtk 9.3.1), used read-only
cd $SLURM_SUBMIT_DIR/experiments/realct/showcase
echo "snapshot=$SLURM_SUBMIT_DIR job=$SLURM_JOB_ID host=$(hostname)"
TRUE=(10379 24803 21400)
# Real-CT cases by anonymous ID; DUHS series names live only in the private map (never commit them).
REAL=($(for a in RealCT-P25-s04 RealCT-P30-s04 RealCT-P45-s04; do awk -v a=$a '$1==a{print $2}' $W/private/realct_id_map.tsv; done))  # RealCT-P28 dropped: 164 mm FOV recon
[ ${#REAL[@]} = 3 ] || { echo "realct_id_map.tsv lookup failed" >&2; exit 1; }
[ "${SKIP_PREP:-0}" = 1 ] || {
  $W/.venv/bin/python prep.py $OUT truect "${TRUE[@]}"
  $W/.venv/bin/python prep.py $OUT realct "${REAL[@]}"
  for s in "${TRUE[@]}"; do for task in total body; do
    [ -s $OUT/TrueCT_$s/totalseg_$task.nii.gz ] || $W/envs/totalseg/bin/TotalSegmentator -i $OUT/TrueCT_$s/ct.nii.gz -o $OUT/TrueCT_$s/totalseg_$task.nii.gz --ml -ta $task -q
  done; done; }
for d in $OUT/TrueCT_* $OUT/RealCT_*; do $VIS render.py $d ${RENDER_ARGS:-}; done
ls -R $OUT | head -100 || true
