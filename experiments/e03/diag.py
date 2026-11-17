"""Diagnose D1 eval MAE ~0.36: one-step x0 recovery vs t (model/conditioning) and DDIM on one 96^3 patch with and
without sliding window, bf16 vs fp32 (sampler). One VAL subject, central patch. Usage: python diag.py <cfg.json> [sid]
"""
import json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e02"))
from ddpm3d import build, eval_region, net_eps, sample, scheduler  # noqa: E402
from train import NORM_PATH, normalize  # noqa: E402
from truect.data import WORK, load, split  # noqa: E402

cfg = json.load(open(sys.argv[1])); name = Path(sys.argv[1]).stem
dev = torch.device("cuda")
s = torch.load(WORK / "checkpoints/e03" / name / "last.pt", map_location="cpu", weights_only=False)
print("ckpt step", s["step"], "ema keys", list(s["ema"])[:2], flush=True)
model = build(cfg).to(dev).eval()
model.load_state_dict({k[len("module."):]: v for k, v in s["ema"].items() if k.startswith("module.")})
raw = build(cfg).to(dev).eval(); raw.load_state_dict(s["model"])
sid = sys.argv[2] if len(sys.argv) > 2 else split()["val"][0]
t, meta = load(sid)
z0, ys, xs = eval_region(t, meta)
cy, cx = (ys[0] + ys[1]) // 2, (xs[0] + xs[1]) // 2
sl = (slice(z0, z0 + 96), slice(cy - 48, cy + 48), slice(cx - 48, cx + 48))
norm = json.load(open(NORM_PATH))
ct = torch.from_numpy(normalize(t["ct"][sl], norm))[None, None].to(dev)
x0 = torch.from_numpy(np.asarray(t["density"][sl], np.float32) - 1)[None, None].to(dev)
print("x0 mean/min/max", float(x0.mean()), float(x0.min()), float(x0.max()), "ct mean", float(ct.mean()), flush=True)
acp = scheduler(cfg).alphas_cumprod.to(dev); V = cfg.get("pred") == "v"
g = torch.Generator(dev).manual_seed(0)
eps = torch.randn(x0.shape, generator=g, device=dev)


def fp32(m):
    return lambda inp, tt: m(inp, timesteps=torch.full((inp.shape[0],), tt, device=dev, dtype=torch.long)).float()


with torch.no_grad():
    for label, m, f in [("ema bf16", model, net_eps(model)), ("ema fp32", model, fp32(model)), ("raw fp32", raw, fp32(raw))]:
        out = []
        for tt in (10, 100, 300, 500, 700, 900, 999):
            a = acp[tt]; xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
            e = f(torch.cat([ct, xt], 1), tt)
            x0h = a.sqrt() * xt - (1 - a).sqrt() * e if V else (xt - (1 - a).sqrt() * e) / a.sqrt()
            tgt = a.sqrt() * eps - (1 - a).sqrt() * x0 if V else eps
            out.append(f"t{tt}: outMSE {float(((e - tgt) ** 2).mean()):.4f} x0MAE {float((x0h - x0).abs().mean()):.4f}")
        print(label, " | ".join(out), flush=True)
    for label, f in [("ema bf16", net_eps(model)), ("ema fp32", fp32(model))]:
        for steps in (25, 100):
            y_sw = sample(f, ct, steps, 0, (96, 96, 96), cfg=cfg)  # roi == patch: single window
            print(label, f"ddim{steps} one-patch MAE {float((y_sw - x0).abs().mean()):.4f}  pred mean {float(y_sw.mean()):.4f}", flush=True)
    # no sliding window, plain loop
    sch = scheduler(cfg); sch.set_timesteps(25); x = torch.randn(ct.shape, generator=torch.Generator(dev).manual_seed(0), device=dev)
    print("timesteps", sch.timesteps[:5].tolist(), "...", sch.timesteps[-3:].tolist(), flush=True)
    for tt in sch.timesteps:
        e = fp32(model)(torch.cat([ct, x], 1), int(tt)); x, _ = sch.step(e, int(tt), x)
    print("plain DDIM25 fp32 MAE", float((x - x0).abs().mean()), "pred mean", float(x.mean()), flush=True)
