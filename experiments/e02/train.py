"""E02 2D/2.5D U-Net training (U1 density, U2 class, U3 joint, U4 = context>0), resumable, with full-volume validation.

Usage: python train.py configs/<run>.json [--kill-at STEP]
Outputs: results/e02/<run>/ (log.jsonl, attempts.jsonl, val_<step>.json, overlay_<step>.png, norm.json),
         checkpoints/e02/<run>/last.pt (overwritten atomically; resumed automatically).
Loss region == truect.data.eval_mask (valid phantom voxel, in body, inside recon FOV). CT normalization uses
only TRAIN CT inside the CT-derived FOV (no phantom data), so inference needs nothing but the CT.
"""
import argparse, json, math, os, signal, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from truect.data import CLASSES, WORK, eval_mask, fov, load, split  # noqa: E402
from truect.metrics import subject_metrics  # noqa: E402
from unet import UNet  # noqa: E402

NORM_PATH = WORK / "results/e02/ct_norm_train.json"  # shared by all E02 runs (train split frozen)


def _ct_sample(sid, step=16, per_slice=4000):
    t, meta = load(sid, ("ct",))
    a, b = meta["valid_slices"]
    m = fov(t["ct"].shape)
    rng = np.random.default_rng(int(sid))
    out = [rng.choice(np.asarray(t["ct"][z])[m], per_slice, replace=False) for z in range(a, b + 1, step)]
    return np.concatenate(out).astype(np.float32)


def ct_norm(workers):
    """Train-only CT normalization: clip to the fixed 12-bit CT range [-1024, 3071] HU, then z-score with the mean/sd
    of train in-FOV CT. Computed once, saved. (A percentile clip is wrong here: p99.5 of the in-FOV CT is ~227 HU,
    which would erase all bone contrast; smoke job 204988.)"""
    if NORM_PATH.exists():
        return json.load(open(NORM_PATH))
    with Pool(workers) as p:
        v = np.concatenate(p.map(_ct_sample, split()["train"]))
    lo, hi = -1024.0, 3071.0
    pct = dict(zip(["p0", "p0.5", "p99.5", "p99.99", "p100"], np.percentile(v, [0, 0.5, 99.5, 99.99, 100]).tolist()))
    v = np.clip(v, lo, hi)
    norm = dict(lo=lo, hi=hi, mean=float(v.mean()), std=float(v.std()), n_samples=int(v.size), percentiles=pct,
                subjects="train (configs/splits_copd.json)", region="truect.data.fov, every 16th valid slice")
    NORM_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(norm, open(NORM_PATH, "w"), indent=1)
    return norm


def normalize(ct, norm):
    return (np.clip(np.asarray(ct, np.float32), norm["lo"], norm["hi"]) - norm["mean"]) / norm["std"]


def ct_stack(ct, z, k):
    """Slices z-k..z+k as channels, clamped at the volume ends (k=0 is plain 2D). z may be an array of slices."""
    idx = np.clip(np.asarray(z)[..., None] + np.arange(-k, k + 1), 0, ct.shape[0] - 1)
    return ct[idx.ravel()].reshape(*idx.shape, *ct.shape[1:])


class Patches(torch.utils.data.Dataset):
    """Sample i is fully determined by (seed, i): patient-balanced (subject uniform), slice uniform in the subject's
    valid range, patch position uniform (re-drawn up to 4x if <5% of the patch is in the loss region).
    Index-determinism makes resume exact without saving loader state."""

    def __init__(self, sids, cfg, norm):
        self.sids, self.cfg, self.norm, self.t = sids, cfg, norm, {}

    def __len__(self):
        return self.cfg["iters"] * self.cfg["batch"]

    def subject(self, sid):
        if sid not in self.t:  # opened lazily so each worker has its own memmaps
            t, meta = load(sid, ("ct", "density", "cls", "valid", "organ_pt"))
            self.t[sid] = t, meta["valid_slices"]
        return self.t[sid]

    def __getitem__(self, i):
        rng = np.random.default_rng([self.cfg["seed"], i])
        t, (a, b) = self.subject(self.sids[rng.integers(len(self.sids))])
        P, k = self.cfg["patch"], self.cfg["context"]
        z = rng.integers(a, b + 1)
        mz = eval_mask(t, z)  # full slice, so the FOV disc is the real one
        H, W = mz.shape
        for _ in range(4):
            r, c = rng.integers(0, H - P + 1), rng.integers(0, W - P + 1)
            m = mz[r:r + P, c:c + P]
            if m.mean() >= 0.05:
                break
        x = normalize(ct_stack(t["ct"], z, k)[:, r:r + P, c:c + P], self.norm)
        d = np.where(m, np.asarray(t["density"][z, r:r + P, c:c + P], np.float32), 0)
        cl = np.where(m, t["cls"][z, r:r + P, c:c + P], 255).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(d), torch.from_numpy(cl), torch.from_numpy(m)


def losses(out, d, cl, m, cfg):
    n = m.sum().clamp(min=1)
    L = {}
    if "density" in out:
        L["l1"] = ((out["density"].float() - d).abs() * m).sum() / n
    if "cls" in out:
        L["ce"] = F.cross_entropy(out["cls"].float(), cl, ignore_index=255, reduction="sum") / n
    L["loss"] = L.get("l1", 0) + cfg.get("w_cls", 1.0) * L.get("ce", 0)
    return L


def build(cfg):
    return UNet(in_ch=2 * cfg["context"] + 1, base=cfg["base"], depth=cfg["depth"],
                density=cfg["heads"] in ("density", "both"),
                n_cls=len(CLASSES) if cfg["heads"] in ("cls", "both") else 0)


@torch.no_grad()
def predict_volume(model, ct, cfg, norm, z_range, bs=8):
    """Full-slice inference on CT only. Returns float16 density / uint8 class volumes (zeros outside z_range)."""
    dev = next(model.parameters()).device
    pd = np.zeros(ct.shape, np.float16) if model.density is not None else None
    pc = np.zeros(ct.shape, np.uint8) if model.cls is not None else None
    for z0 in range(z_range[0], z_range[1] + 1, bs):
        zs = np.arange(z0, min(z0 + bs, z_range[1] + 1))
        x = torch.from_numpy(normalize(ct_stack(ct, zs, cfg["context"]), norm)).to(dev, non_blocking=True)
        with torch.autocast("cuda", torch.bfloat16):
            out = model(x)
        if pd is not None:
            pd[zs] = out["density"].float().cpu().numpy()
        if pc is not None:
            pc[zs] = out["cls"].argmax(1).byte().cpu().numpy()
    return pd, pc


def validate(model, cfg, norm, step, out_dir):
    model.eval()
    res = {}
    for j, sid in enumerate(cfg["val_subjects"] or split()["val"]):
        assert sid in split()["val"], sid
        t, meta = load(sid)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        pd, pc = predict_volume(model, t["ct"], cfg, norm, meta["valid_slices"])
        torch.cuda.synchronize()
        sec = time.time() - t0
        res[sid] = subject_metrics(t, pd, pc)
        res[sid].update(infer_sec=sec, infer_slices=meta["valid_slices"][1] - meta["valid_slices"][0] + 1,
                        infer_peak_vram_gb=torch.cuda.max_memory_allocated() / 2**30)
        if j == 0:
            overlay(t, pd, pc, (meta["valid_slices"][0] + meta["valid_slices"][1]) // 2, norm,
                    out_dir / f"overlay_{step}_{sid}.png")
        print(f"val step={step} {sid}: " + json.dumps({k: v for k, v in res[sid].items() if not isinstance(v, dict)}),
              flush=True)
    json.dump(res, open(out_dir / f"val_{step}.json", "w"), indent=1)
    model.train()
    return res


def overlay(t, pd, pc, z, norm, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = eval_mask(t, z)
    ims = [("CT (HU)", np.asarray(t["ct"][z]), dict(cmap="gray", vmin=-1000, vmax=400))]
    dk = dict(cmap="magma", vmin=0, vmax=1.5)
    ims.append(("target density", np.where(m, t["density"][z], np.nan), dk))
    if pd is not None:
        ims += [("pred density", np.asarray(pd[z], np.float32), dk),
                ("|pred-target| in eval mask", np.where(m, np.abs(pd[z] - np.asarray(t["density"][z], np.float32)),
                                                         np.nan), dict(cmap="viridis", vmin=0, vmax=0.3))]
    ck = dict(cmap="tab20", vmin=0, vmax=19, interpolation="nearest")
    ims.append(("target class", np.where(m, t["cls"][z], np.nan), ck))
    if pc is not None:
        ims.append(("pred class", np.asarray(pc[z], np.float32), ck))
    fig, ax = plt.subplots(1, len(ims), figsize=(4 * len(ims), 4.4))
    for a, (title, im, kw) in zip(ax, ims):
        h = a.imshow(im, **kw); a.set_title(f"{title}\nz={z}", fontsize=9); a.axis("off")
        fig.colorbar(h, ax=a, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)


def lr_at(step, cfg):
    if step < cfg["warmup"]:
        return (step + 1) / cfg["warmup"]
    return 0.5 * (1 + math.cos(math.pi * (step - cfg["warmup"]) / max(cfg["iters"] - cfg["warmup"], 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--kill-at", type=int, help="smoke test only: SIGKILL self at this step (preemption test)")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    name = Path(args.config).stem
    out_dir, ck_dir = WORK / "results/e02" / name, WORK / "checkpoints/e02" / name
    out_dir.mkdir(parents=True, exist_ok=True); ck_dir.mkdir(parents=True, exist_ok=True)
    ck = ck_dir / "last.pt"

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda")
    norm = ct_norm(cfg["workers"])
    json.dump(norm, open(out_dir / "norm.json", "w"), indent=1)
    model = build(cfg).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_at(s, cfg))
    step, micro, n_skipped = 0, 1, 0
    if ck.exists():
        s = torch.load(ck, map_location="cpu", weights_only=False)
        assert s["cfg"] == cfg and s["norm"] == norm, "config/normalization changed since checkpoint"
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        step, micro, n_skipped = s["step"], s["micro"], s["n_skipped"]
        torch.set_rng_state(s["rng_cpu"]); torch.cuda.set_rng_state(s["rng_cuda"])
    attempt = dict(start_step=step, resumed=ck.exists(), job=os.environ.get("SLURM_JOB_ID"), host=os.uname()[1],
                   snapshot=os.environ.get("SLURM_SUBMIT_DIR", os.getcwd()), torch=torch.__version__,
                   cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), cfg=cfg,
                   n_params=sum(p.numel() for p in model.parameters()), time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    print(json.dumps(attempt), flush=True)
    with open(out_dir / "attempts.jsonl", "a") as f:
        f.write(json.dumps(attempt) + "\n")

    def save():
        tmp = ck.with_suffix(".tmp")
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step,
                        micro=micro, n_skipped=n_skipped, cfg=cfg, norm=norm, rng_cpu=torch.get_rng_state(),
                        rng_cuda=torch.cuda.get_rng_state()), tmp)
        os.replace(tmp, ck)

    ds = Patches(split()["train"], cfg, norm)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg["batch"], sampler=range(step * cfg["batch"], len(ds)),
                                     num_workers=cfg["workers"], pin_memory=True, prefetch_factor=4,
                                     persistent_workers=False)
    log = open(out_dir / "log.jsonl", "a")
    torch.cuda.reset_peak_memory_stats()
    t_last, t_wait, acc = time.time(), 0.0, {}
    t_fetch = time.time()
    model.train()
    for x, d, cl, m in dl:
        t_wait += time.time() - t_fetch
        x = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
        d, cl, m = d.to(dev, non_blocking=True), cl.to(dev, non_blocking=True), m.to(dev, non_blocking=True)
        while True:  # automatic OOM back-off: split the batch into more micro-batches (same effective batch)
            try:
                opt.zero_grad(set_to_none=True)
                for xs, ds_, cs, ms in zip(*(v.chunk(micro) for v in (x, d, cl, m))):
                    with torch.autocast("cuda", torch.bfloat16):
                        out = model(xs)
                    L = losses(out, ds_, cs, ms, cfg)
                    (L["loss"] * ms.numel() / m.numel()).backward()
                break
            except torch.OutOfMemoryError:
                micro *= 2
                print(f"OOM at step {step}: micro-batches -> {micro}", flush=True)
                log.write(json.dumps(dict(step=step, event="oom", micro=micro)) + "\n")
                opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
                if micro > cfg["batch"]:
                    raise
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("clip", float("inf")))
        if torch.isfinite(gn):
            opt.step()
        else:
            n_skipped += 1
        sched.step()
        step += 1
        for kk, v in L.items():  # note: last micro-batch only when micro > 1
            acc.setdefault(kk, []).append(float(v))
        acc.setdefault("gnorm", []).append(float(gn))
        if step % cfg["log_every"] == 0 or step == cfg["iters"]:
            torch.cuda.synchronize()
            dt = time.time() - t_last
            rec = dict(step=step, **{kk: float(np.mean(v)) for kk, v in acc.items()}, lr=sched.get_last_lr()[0],
                       sec_per_it=dt / len(acc["gnorm"]), data_wait_frac=t_wait / dt, n_skipped=n_skipped,
                       micro=micro, peak_vram_gb=torch.cuda.max_memory_allocated() / 2**30, job=attempt["job"])
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(json.dumps(rec), flush=True)
            t_last, t_wait, acc = time.time(), 0.0, {}
        if step % cfg["ckpt_every"] == 0 or step == cfg["iters"]:
            save()
        if args.kill_at and step == args.kill_at:
            print(f"--kill-at {step}: SIGKILL", flush=True)
            os.kill(os.getpid(), signal.SIGKILL)
        if step % cfg["val_every"] == 0 or step == cfg["iters"]:
            if cfg["val_subjects"] != []:
                validate(model, cfg, norm, step, out_dir)
            t_last = time.time()
        t_fetch = time.time()
    print(f"done step={step}", flush=True)


if __name__ == "__main__":
    main()
