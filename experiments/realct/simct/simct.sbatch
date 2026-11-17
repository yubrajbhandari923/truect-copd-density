#!/bin/bash -l
#SBATCH --partition=defq
#SBATCH --no-requeue
#SBATCH --output=/scratch/railabs/yb107/research-output/truect-copd-density/logs/%x-%j.out
# Real-CT round trip through the drishya variant-3 simulator; one rank per GPU. Uses drishya's LEAP env (read-only).
# usage (from staged snapshot dir): sbatch -J simct --gres=gpu:G ... experiments/realct/simct/simct.sbatch [WORLD_OVERRIDE]
set -euo pipefail
W=/scratch/railabs/yb107/research-output/truect-copd-density; DW=/scratch/railabs/yb107/research-output/drishya
module load gcc/11.2.0 cuda12.2/toolkit/12.2.2
export LEAP_SRC=$DW/envs/leap/LEAP/src TMPDIR=$W/tmp PYTHONPYCACHEPREFIX=$W/cache/pyc MPLCONFIGDIR=$W/cache/mpl OMP_NUM_THREADS=4
DS=/scratch/railabs/yb107/Datasets/DUHS_CT_CHEST_thin; TS=$W/results/realct/totalseg; INF=$W/results/realct/inference; OUT=$W/results/realct/simct
cd "$SLURM_SUBMIT_DIR/experiments/realct/simct"
echo "snapshot=$SLURM_SUBMIT_DIR job=$SLURM_JOB_ID host=$(hostname) leap=$(git -C $DW/envs/leap/LEAP rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
IFS=, read -ra G <<< "$CUDA_VISIBLE_DEVICES"; WORLD=${1:-${#G[@]}}
for r in $(seq 0 $((${#G[@]} - 1))); do
  CUDA_VISIBLE_DEVICES=${G[$r]} $DW/.venv/bin/python simct.py $DS $TS $INF $OUT $r $WORLD &
done; wait
ls $OUT/metrics | wc -l
