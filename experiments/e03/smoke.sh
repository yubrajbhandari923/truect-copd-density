#!/bin/bash
# Smoke on 2 GPUs: unit tests (E02+E03), E02 A13 attention-U-Net memory vs DynUNet, D1 train 20 -> resume 40, D1 eval (DDIM 5).
set -e
W=/scratch/railabs/yb107/research-output/truect-copd-density
python probe_mem.py
rm -rf $W/checkpoints/e03/smoke_d1 $W/results/e03/smoke_d1
(cd ../e02 && python -m pytest -q test_e02.py) && python -m pytest -q test_e03.py
(cd ../e02 && python - <<'PY'
import json, torch
from train3d import build
for name in ("u5a_ref75k", "u5a_att"):
    cfg = json.load(open(f"configs/{name}.json")); m = build(cfg).cuda(); torch.cuda.reset_peak_memory_stats()
    x = torch.randn(2, 1, 128, 128, 128, device="cuda")
    with torch.autocast("cuda", torch.bfloat16):
        y = m(x)
    y.float().mean().backward()
    print(name, "params", sum(p.numel() for p in m.parameters()), "out", tuple(y.shape), "peak GB @batch2", torch.cuda.max_memory_allocated() / 2**30)
PY
)
torchrun --standalone --nproc_per_node=2 ddpm3d.py train configs/smoke_d1.json --stop-at 20
torchrun --standalone --nproc_per_node=2 ddpm3d.py train configs/smoke_d1.json
torchrun --standalone --nproc_per_node=2 ddpm3d.py eval configs/smoke_d1.json
python -c "import json;d=json.load(open('$W/results/e03/smoke_d1/eval_40.json'));s=next(iter(d['per_subject'].values()));print({k:(v.get('density_mae'),v.get('sec')) for k,v in s.items() if isinstance(v,dict) and 'density_mae' in v})"
