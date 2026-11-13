#!/bin/bash
# Smoke test (run inside run.sbatch): unit tests, GPU check, U1/U2/U4 short runs, U3 300 its with
# SIGKILL at 250 (checkpoint at 200) then resume to 300 + full-volume validation on 2 val subjects.
set -uxo pipefail
python -m pytest -q test_e02.py
python -c "import torch; a=torch.randn(4096,4096,device='cuda'); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(), float((a@a).abs().mean()))"
for c in smoke_u1 smoke_u2 smoke_u4; do python train.py configs/$c.json || echo "FAILED $c"; done
python train.py configs/smoke_u3.json --kill-at 250; echo "exit after kill: $?"
python train.py configs/smoke_u3.json
