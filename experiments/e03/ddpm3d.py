"""E03 D1: paired CT-conditional 3D DDPM for mass density (pilot). MONAI DiffusionModelUNet + DDIM sampler.

Train:  torchrun --standalone --nproc_per_node=G ddpm3d.py train configs/d1_pilot.json [--stop-at STEP]
Eval:   torchrun --standalone --nproc_per_node=G ddpm3d.py eval  configs/d1_pilot.json

Training data, patch sampler, CT normalisation and cosine LR are the E02 U5 ones (prep3d cache, TRAIN only).
Target x0 = density - 1 g/cm3 (air -1, soft tissue ~0, cortical bone ~+0.9), no clipping. Input = [CT, x_t];
epsilon prediction, MSE over the whole patch (outside the evaluation region the target is 0 density), linear beta
schedule 1e-4..0.02 over 1000 steps (Ho 2020). EMA weights are used for sampling.
D1b options (cfg): "pred": "v" -> v-prediction, target v = sqrt(a) eps - sqrt(1-a) x0; "zero_snr": true -> rescale the
schedule to zero terminal SNR (Lin et al. 2024, Alg. 1), so t=999 is pure noise and the model must predict x0 from CT.

Eval (prespecified, bounded pilot): per VAL subject, the 96 slices centred on the valid range, in-plane cropped to the
evaluation-mask bbox + 32 px. Every method sees exactly this input; metrics use its central 64 slices (no slab-edge
effects). Diffusion: DDIM 25/50/100 (eta 0), 4 fixed-seed samples per subject (seed = 10*sid + k, same noise for
every step count), each denoising step a Gaussian-blended sliding window over the slab. Reported: each sample, sample
mean (A10), sample spread, density-histogram distance and gradient sharpness; U5 3D U-Nets on the same input (A08).
"""
import argparse, datetime, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from monai.data import DataLoader
from monai.inferers import sliding_window_inference
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDIMScheduler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e02"))
from train import NORM_PATH, lr_at, normalize  # noqa: E402
from train3d import Patches3D, build as build_unet, split_out, train_subjects  # noqa: E402
from truect.data import WORK, eval_mask, load, split  # noqa: E402
from truect.metrics import subject_metrics  # noqa: E402

UNETS = ["u5_s11", "u5a_ref75k"]


def build(cfg):
    n = len(cfg["channels"])
    return DiffusionModelUNet(spatial_dims=3, in_channels=2, out_channels=1, channels=cfg["channels"],
                              attention_levels=[False] * (n - 1) + [cfg.get("attn_last", True)],
                              num_res_blocks=cfg["res_blocks"], num_head_channels=cfg.get("head_ch", 32))


def scheduler(cfg=None):
    cfg = cfg or {}
    s = DDIMScheduler(num_train_timesteps=1000, schedule="linear_beta", clip_sample=False,
                      prediction_type="v_prediction" if cfg.get("pred") == "v" else "epsilon")
    if cfg.get("zero_snr"):  # keep sqrt(abar_0), shift sqrt(abar_T) to 0; DDIM step only reads alphas_cumprod
        r = s.alphas_cumprod.sqrt()
        s.alphas_cumprod = ((r - r[-1]) * r[0] / (r[0] - r[-1])) ** 2
    return s


@torch.no_grad()
def sample(eps_fn, ct, steps, seed, roi, sw_batch=8, overlap=0.25, cfg=None):
    """DDIM (eta 0) from fixed noise. eps_fn(inp[N,2,...], t) -> model output (eps or v, per cfg); every step is a
    sliding window over the slab."""
    s = scheduler(cfg); s.set_timesteps(steps)
    g = torch.Generator(ct.device).manual_seed(seed)
    x = torch.randn(ct.shape, generator=g, device=ct.device)
    for t in s.timesteps:
        eps = sliding_window_inference(torch.cat([ct, x], 1), roi, sw_batch, lambda inp: eps_fn(inp, int(t)),
                                       overlap=overlap, mode="gaussian")
        x, _ = s.step(eps, int(t), x)
    return x


def net_eps(model):
    def f(inp, t):
        with torch.autocast("cuda", torch.bfloat16):
            return model(inp, timesteps=torch.full((inp.shape[0],), t, device=inp.device, dtype=torch.long)).float()
    return f


def ddp_setup():
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=6))
    r, w, lr_ = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(lr_)
    return r, w, lr_, torch.device("cuda", lr_)


def train(cfg, name, stop_at):
    rank, world, lr_, dev = ddp_setup()
    out_dir, ck_dir = WORK / "results/e03" / name, WORK / "checkpoints/e03" / name
    out_dir.mkdir(parents=True, exist_ok=True); ck_dir.mkdir(parents=True, exist_ok=True)
    ck = ck_dir / "last.pt"
    torch.manual_seed(cfg["seed"] + rank); torch.backends.cudnn.benchmark = True
    norm = json.load(open(NORM_PATH))
    model = build(cfg).to(dev)
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(cfg["ema"]))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_at(s, cfg))
    step = 0
    if ck.exists():
        s = torch.load(ck, map_location="cpu", weights_only=False)
        assert s["cfg"] == cfg and s["norm"] == norm and s["world"] == world, "config/normalisation/GPU count changed"
        model.load_state_dict(s["model"]); ema.load_state_dict(s["ema"]); opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"]); step = s["step"]
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[lr_])
    attempt = dict(start_step=step, resumed=ck.exists(), job=os.environ.get("SLURM_JOB_ID"), host=os.uname()[1],
                   world=world, snapshot=os.environ.get("SLURM_SUBMIT_DIR", os.getcwd()), torch=torch.__version__,
                   monai=__import__("monai").__version__, gpu=torch.cuda.get_device_name(), cfg=cfg,
                   n_params=sum(p.numel() for p in model.parameters()), time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    if rank == 0:
        print(json.dumps(attempt), flush=True)
        with open(out_dir / "attempts.jsonl", "a") as f:
            f.write(json.dumps(attempt) + "\n")
    B, G = cfg["batch"], world
    idx = [st * B * G + rank * B + j for st in range(step, cfg["iters"]) for j in range(B)]
    ucfg = {**cfg, "seed": cfg["seed"]}  # same (seed, i) -> patch mapping as E02
    dl = DataLoader(Patches3D(train_subjects(cfg), ucfg, norm), batch_size=B, sampler=idx, num_workers=cfg["workers"],
                    pin_memory=True, prefetch_factor=4, persistent_workers=True)
    ns = scheduler(cfg)
    acp = ns.alphas_cumprod.to(dev)
    log = open(out_dir / "log.jsonl", "a") if rank == 0 else None
    t_last, acc = time.time(), []
    for x, d, _, _ in dl:
        x, d = x.to(dev, non_blocking=True), d.to(dev, non_blocking=True)[:, None]
        x0 = d - 1.0
        g = torch.Generator(dev).manual_seed(cfg["seed"] * 1_000_003 + step * G + rank)  # resume-exact noise/timesteps
        t = torch.randint(0, 1000, (x.shape[0],), device=dev, generator=g)
        eps = torch.randn(x0.shape, device=dev, generator=g)
        a = acp[t].view(-1, 1, 1, 1, 1)
        xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", torch.bfloat16):
            pred = ddp(torch.cat([x, xt], 1), timesteps=t)
        target = a.sqrt() * eps - (1 - a).sqrt() * x0 if cfg.get("pred") == "v" else eps
        loss = ((pred.float() - target) ** 2).mean()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(ddp.parameters(), cfg.get("clip", 1.0))
        opt.step(); ema.update_parameters(model); sched.step(); step += 1
        acc.append(float(loss.detach()))
        if rank == 0 and (step % cfg["log_every"] == 0 or step == cfg["iters"]):
            torch.cuda.synchronize(); dt = time.time() - t_last
            rec = dict(step=step, loss=float(np.mean(acc)), gnorm=float(gn), lr=sched.get_last_lr()[0],
                       sec_per_it=dt / len(acc), peak_vram_gb=torch.cuda.max_memory_allocated() / 2**30, job=attempt["job"])
            log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
        if step % cfg["log_every"] == 0:
            t_last, acc = time.time(), []
        if rank == 0 and (step % cfg["ckpt_every"] == 0 or step == cfg["iters"] or step == stop_at):
            tmp = ck.with_suffix(".tmp")
            torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                            step=step, cfg=cfg, norm=norm, world=world), tmp)
            os.replace(tmp, ck)
        if step == stop_at:
            print(f"--stop-at {step}", flush=True)
            break
    dist.barrier(); dist.destroy_process_group()


def eval_region(t, meta, margin=32):
    """Prespecified pilot region: 96 slices centred on the valid range; in-plane eval-mask bbox + margin."""
    a, b = meta["valid_slices"]; z0 = (a + b + 1) // 2 - 48
    m = eval_mask(t, slice(z0 + 16, z0 + 80)).any(0)
    r, c = np.where(m)
    H, W = m.shape
    return int(z0), (int(max(r.min() - margin, 0)), int(min(r.max() + margin + 1, H))), (int(max(c.min() - margin, 0)), int(min(c.max() + margin + 1, W)))


def slab_metrics(t, z0, ys, xs, pred):
    """subject_metrics on the central 64 slices; pred is the (96, h, w) crop."""
    ts = {k: v[z0 + 16:z0 + 80] for k, v in t.items()}
    full = np.zeros(ts["valid"].shape, np.float32)
    full[:, ys[0]:ys[1], xs[0]:xs[1]] = pred[16:80]
    res = subject_metrics(ts, full)
    m = eval_mask(ts); true = np.asarray(ts["density"], np.float32)
    q = np.linspace(0.001, 0.999, 999)
    res["hist_w1"] = float(np.abs(np.quantile(full[m], q) - np.quantile(true[m], q)).mean())  # 1-D Wasserstein approx
    gp = np.hypot(*np.gradient(full, axis=(1, 2)))[m].mean(); gt = np.hypot(*np.gradient(true, axis=(1, 2)))[m].mean()
    res["grad_ratio"] = float(gp / gt)  # in-plane sharpness vs truth (1 = same, <1 smoother)
    return res


def evaluate(cfg, name):
    rank, world, _, dev = ddp_setup()
    out_dir = WORK / "results/e03" / name
    norm = json.load(open(NORM_PATH))
    s = torch.load(WORK / "checkpoints/e03" / name / "last.pt", map_location="cpu", weights_only=False)
    model = build(cfg).to(dev)
    model.load_state_dict({k[len("module."):]: v for k, v in s["ema"].items() if k.startswith("module.")})
    model.eval()
    unets = {}
    for u in UNETS:
        su = torch.load(WORK / "checkpoints/e02" / u / "last.pt", map_location="cpu", weights_only=False)
        unets[u] = build_unet(su["cfg"]).to(dev).eval(); unets[u].load_state_dict(su["model"])
        unets[u].step = su["step"]
    res = {}
    for sid in split()["val"][rank::world]:
        t, meta = load(sid)
        z0, ys, xs = eval_region(t, meta)
        ct = torch.from_numpy(normalize(t["ct"][z0:z0 + 96, ys[0]:ys[1], xs[0]:xs[1]], norm))[None, None].to(dev)
        r = res[sid] = dict(region=dict(z0=z0, ys=ys, xs=xs, eval_slices=[z0 + 16, z0 + 80]))
        for u, net in unets.items():
            torch.cuda.synchronize(); t0 = time.time()
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                y = sliding_window_inference(ct, (96, 128, 128), 4, lambda i: split_out(net(i).float())["density"][:, None],
                                             overlap=0.25, mode="gaussian")
            torch.cuda.synchronize()
            r[u] = slab_metrics(t, z0, ys, xs, y[0, 0].float().cpu().numpy()); r[u]["sec"] = time.time() - t0
            r[u]["step"] = net.step
        for steps in cfg["eval_steps"]:
            S = []
            for k in range(4):
                torch.cuda.synchronize(); t0 = time.time()
                x = sample(net_eps(model), ct, steps, 10 * int(sid) + k, (cfg["patch"],) * 3, cfg.get("sw_batch", 8), cfg=cfg)
                torch.cuda.synchronize()
                S.append(x[0, 0].float().cpu().numpy() + 1.0)
                r[f"ddim{steps}_s{k}"] = slab_metrics(t, z0, ys, xs, S[-1]); r[f"ddim{steps}_s{k}"]["sec"] = time.time() - t0
                print(f"{sid} ddim{steps} s{k}: mae={r[f'ddim{steps}_s{k}']['density_mae']:.4f} "
                      f"sec={time.time() - t0:.0f}", flush=True)
            S = np.stack(S)
            r[f"ddim{steps}_mean"] = slab_metrics(t, z0, ys, xs, S.mean(0))
            ts = {k: v[z0 + 16:z0 + 80] for k, v in t.items()}
            m = eval_mask(ts)[:, ys[0]:ys[1], xs[0]:xs[1]]
            r[f"ddim{steps}_mean"]["sample_sd"] = float(S[:, 16:80].std(0, ddof=1)[m].mean())
        print(f"eval {sid}: " + json.dumps({k: v.get("density_mae") for k, v in r.items() if isinstance(v, dict)}), flush=True)
    allres = [None] * world
    dist.all_gather_object(allres, res)
    if rank == 0:
        json.dump(dict(step=s["step"], job=os.environ.get("SLURM_JOB_ID"), cfg=cfg,
                       per_subject={k: v for r in allres for k, v in sorted(r.items())}),
                  open(out_dir / f"eval_{s['step']}.json", "w"), indent=1)
    dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval"])
    ap.add_argument("config")
    ap.add_argument("--stop-at", type=int)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    train(cfg, Path(a.config).stem, a.stop_at) if a.mode == "train" else evaluate(cfg, Path(a.config).stem)
