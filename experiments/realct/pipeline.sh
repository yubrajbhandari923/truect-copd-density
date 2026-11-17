#!/bin/bash
# Real-CT test inference: copy+check -> TotalSegmentator (total, body) -> E02 model inference. 2 GPUs, resumable.
set -euo pipefail
W=/scratch/railabs/yb107/research-output/truect-copd-density
SRC=/scratch/railabs/ld258/dataset/DUHS/VLM/CT_CHEST
DST=/scratch/railabs/SHARED/data/ct_images/duke/DUHS_VLM_CT_CHEST_thin
TS=$W/results/realct/totalseg; OUT=$W/results/realct/inference
export TOTALSEG_HOME_DIR=$W/cache/totalseg
mkdir -p $TS $OUT
python copy_check.py $SRC $DST
$W/envs/totalseg/bin/python -c "import json; from totalsegmentator.map_to_binary import class_map as c; json.dump(c['total'], open('$TS/class_map_total.json','w'), indent=1)"
mapfile -t S < <(ls $DST/images/*.nii.gz | xargs -n1 basename | sed 's/.nii.gz$//')
seg() {  # $1 = GPU index, $2 = start, $3 = step
  for ((i=$2; i<${#S[@]}; i+=$3)); do s=${S[$i]}; mkdir -p $TS/$s
    for task in total body; do
      [ -s $TS/$s/$task.nii.gz ] || CUDA_VISIBLE_DEVICES=$1 $W/envs/totalseg/bin/TotalSegmentator -i $DST/images/$s.nii.gz \
        -o $TS/$s/$task.nii.gz --ml -ta $task -q > $TS/$s/$task.log 2>&1 || echo "TS FAILED $s $task"
    done; echo "ts done $s"
  done; }
IFS=, read -ra G <<< "$CUDA_VISIBLE_DEVICES"
seg ${G[0]} 0 2 & seg ${G[1]} 1 2 & wait
CUDA_VISIBLE_DEVICES=${G[0]} python infer.py $DST $TS $OUT 0 2 & CUDA_VISIBLE_DEVICES=${G[1]} python infer.py $DST $TS $OUT 1 2 & wait
ls $OUT/metrics | wc -l
