"""E02 U5: 3D U-Net (MONAI DynUNet, nnU-Net-style) for joint density + class, DDP over all GPUs of one job.

Launch: torchrun --standalone --nproc_per_node=<gpus> train3d.py configs/u5_s11.json [--stop-at STEP]
Data: the prep3d.py cache (TRAIN only). Same targets, loss (masked L1 + CE, w_cls), CT normalisation, body loss region,
cosine schedule and full-volume validation metrics (truect.metrics on eval_mask) as the 2D runs, so U5 vs U3 isolates 2D->3D.
Sample i is fully determined by (seed, i) and the global step owns indices [step*B*G, (step+1)*B*G), so resume is exact.
Outputs: results/e02/<run>/, checkpoints/e02/<run>/last.pt (rank 0 writes; all ranks resume).
"""
import argparse, datetime, json, os, time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from monai.data import DataLoader
from monai.inferers import sliding_window_inference
from monai.losses import MaskedDiceLoss
from monai.networks.nets import DynUNet

from prep3d import CACHE
from train import NORM_PATH, losses, lr_at, normalize, overlay  # also sets the file_system sharing strategy
from truect.data import CLASSES, WORK, load, split
from truect.metrics import subject_metrics

C = len(CLASSES)
DICE = MaskedDiceLoss(include_background=True, to_onehot_y=True, softmax=True, batch=True)


def loss3d(out, d, cl, m, cfg):
    """2D losses (masked L1 + w_cls * CE) plus optional cfg["dice"] * soft Dice over the 19 classes in the loss region."""
    L = losses(out, d, cl, m, cfg)
    if cfg.get("dice"):
        L["dice"] = DICE(out["cls"].float(), torch.where(m, cl, 0)[:, None], m[:, None].float())
        L["loss"] = L["loss"] + cfg["dice"] * L["dice"]
    return L


def crop(a, start, P, fill):
    """a[start:start+P] in 3D, padded with fill where the box leaves the array (cache crops can be < P in-plane)."""
    sl = tuple(slice(max(s, 0), min(s + P, n)) for s, n in zip(start, a.shape))
    out = np.full((P,) * 3, fill, a.dtype)
    out[tuple(slice(x.start - s, x.stop - s) for x, s in zip(sl, start))] = a[sl]
    return out


class Patches3D(torch.utils.data.Dataset):
    def __init__(self, sids, cfg, norm):
        self.sids, self.cfg, self.norm, self.t = sids, cfg, norm, {}

    def __len__(self):
        return 2**62

    def subject(self, sid):
        if sid not in self.t:  # lazily, per worker; memmaps share the page cache across workers and ranks
            self.t[sid] = {k: np.load(CACHE / sid / f"{k}.npy", mmap_mode="r") for k in ("ct", "density", "lab", "centers")}
        return self.t[sid]

    def __getitem__(self, i):
        rng = np.random.default_rng([self.cfg["seed"], i])
        t = self.subject(self.sids[rng.integers(len(self.sids))])  # patient-balanced
        P = self.cfg["patch"]
        c = t["centers"][rng.integers(len(t["centers"]))] + rng.integers(-P // 4, P // 4 + 1, 3)  # in-body centre, jittered
        s = np.clip(c - P // 2, 0, np.maximum(np.array(t["lab"].shape) - P, 0))
        lab = crop(t["lab"], s, P, 255)
        m = lab != 255
        x = normalize(crop(t["ct"], s, P, -1024), self.norm)[None]
        d = np.where(m, crop(t["density"], s, P, 0).astype(np.float32), 0)
        return torch.from_numpy(x), torch.from_numpy(d), torch.from_numpy(lab.astype(np.int64)), torch.from_numpy(m)


def build(cfg):
    f, n = cfg["filters"], len(cfg["filters"])
    return DynUNet(spatial_dims=3, in_channels=1, out_channels=1 + C, kernel_size=[3] * n, strides=[1] + [2] * (n - 1),
                   upsample_kernel_size=[2] * (n - 1), filters=f, norm_name="instance", res_block=cfg.get("res_block", False))


def split_out(y):
    return {"density": y[:, 0], "cls": y[:, 1:]}


@torch.no_grad()
def predict_volume(model, ct, cfg, norm, z_range, slab=96, halo=32):
    """CT-only full-volume inference: MONAI Gaussian-weighted sliding window over z-slabs (bounded GPU memory)."""
    dev = next(model.parameters()).device
    pd, pc = np.zeros(ct.shape, np.float16), np.zeros(ct.shape, np.uint8)
    a, b = z_range[0], z_range[1] + 1
    P = cfg["patch"]

    def net(x):
        with torch.autocast("cuda", torch.bfloat16):
            return model(x).float()

    for z0 in range(a, b, slab):
        lo, hi = max(a, z0 - halo), min(b, z0 + slab + halo)
        x = torch.from_numpy(normalize(ct[lo:hi], norm))[None, None].to(dev)
        y = sliding_window_inference(x, (P, P, P), cfg.get("sw_batch", 4), net, overlap=cfg.get("sw_overlap", 0.25),
                                     mode="gaussian")[0]
        k0, k1 = z0 - lo, min(z0 + slab, b) - lo
        pd[z0:z0 + k1 - k0] = y[0, k0:k1].half().cpu().numpy()
        pc[z0:z0 + k1 - k0] = y[1:, k0:k1].argmax(0).byte().cpu().numpy()
    return pd, pc


def validate(model, cfg, norm, step, out_dir, rank, world):
    model.eval()
    sids = cfg["val_subjects"] or split()["val"]
    res = {}
    for j, sid in enumerate(sids[rank::world]):
        assert sid in split()["val"], sid
        t, meta = load(sid)
        torch.cuda.synchronize(); t0 = time.time()
        pd, pc = predict_volume(model, t["ct"], cfg, norm, meta["valid_slices"])
        res[sid] = subject_metrics(t, pd, pc)
        res[sid]["infer_sec"] = time.time() - t0
        if rank == 0 and j == 0:
            overlay(t, pd, pc, sum(meta["valid_slices"]) // 2, norm, out_dir / f"overlay_{step}_{sid}.png")
        print(f"val step={step} {sid}: " + json.dumps({k: v for k, v in res[sid].items() if not isinstance(v, dict)}),
              flush=True)
    allres = [None] * world
    dist.all_gather_object(allres, res)
    if rank == 0:
        json.dump({k: v for r in allres for k, v in sorted(r.items())}, open(out_dir / f"val_{step}.json", "w"), indent=1)
    model.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--stop-at", type=int, help="smoke test only: save and exit cleanly at this step (resume test)")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    name = Path(args.config).stem
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=2))  # validation is uneven across ranks
    rank, world, lr_ = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(lr_); dev = torch.device("cuda", lr_)
    out_dir, ck_dir = WORK / "results/e02" / name, WORK / "checkpoints/e02" / name
    out_dir.mkdir(parents=True, exist_ok=True); ck_dir.mkdir(parents=True, exist_ok=True)
    ck = ck_dir / "last.pt"
    torch.manual_seed(cfg["seed"]); torch.backends.cudnn.benchmark = True
    norm = json.load(open(NORM_PATH))  # the shared train-only normalisation of all E02 runs
    model = build(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_at(s, cfg))
    step, n_skipped = 0, 0
    if ck.exists():
        s = torch.load(ck, map_location="cpu", weights_only=False)
        assert s["cfg"] == cfg and s["norm"] == norm and s["world"] == world, "config/normalisation/GPU count changed"
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        step, n_skipped = s["step"], s["n_skipped"]
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[lr_])
    attempt = dict(start_step=step, resumed=ck.exists(), job=os.environ.get("SLURM_JOB_ID"), host=os.uname()[1],
                   world=world, snapshot=os.environ.get("SLURM_SUBMIT_DIR", os.getcwd()), torch=torch.__version__,
                   monai=__import__("monai").__version__, gpu=torch.cuda.get_device_name(), cfg=cfg,
                   n_params=sum(p.numel() for p in model.parameters()), time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    if rank == 0:
        print(json.dumps(attempt), flush=True)
        with open(out_dir / "attempts.jsonl", "a") as f:
            f.write(json.dumps(attempt) + "\n")

    B, G = cfg["batch"], world  # batch is per GPU
    idx = [st * B * G + rank * B + j for st in range(step, cfg["iters"]) for j in range(B)]
    dl = DataLoader(Patches3D(split()["train"], cfg, norm), batch_size=B, sampler=idx, num_workers=cfg["workers"],
                    pin_memory=True, prefetch_factor=4, persistent_workers=True)
    log = open(out_dir / "log.jsonl", "a") if rank == 0 else None
    torch.cuda.reset_peak_memory_stats()
    t_last, t_wait, acc, t_fetch = time.time(), 0.0, {}, time.time()
    ddp.train()
    for x, d, cl, m in dl:
        t_wait += time.time() - t_fetch
        x, d, cl, m = (v.to(dev, non_blocking=True) for v in (x, d, cl, m))
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", torch.bfloat16):
            out = split_out(ddp(x))
        L = loss3d(out, d, cl, m, cfg)
        L["loss"].backward()
        gn = torch.nn.utils.clip_grad_norm_(ddp.parameters(), cfg.get("clip", float("inf")))
        if torch.isfinite(gn):
            opt.step()
        else:
            n_skipped += 1
        sched.step(); step += 1
        for k, v in L.items():
            acc.setdefault(k, []).append(float(v.detach()))
        acc.setdefault("gnorm", []).append(float(gn))
        if rank == 0 and (step % cfg["log_every"] == 0 or step == cfg["iters"]):
            torch.cuda.synchronize(); dt = time.time() - t_last
            rec = dict(step=step, **{k: float(np.mean(v)) for k, v in acc.items()}, lr=sched.get_last_lr()[0],
                       sec_per_it=dt / len(acc["gnorm"]), data_wait_frac=t_wait / dt, n_skipped=n_skipped,
                       peak_vram_gb=torch.cuda.max_memory_allocated() / 2**30, job=attempt["job"])
            log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
        if step % cfg["log_every"] == 0:
            t_last, t_wait, acc = time.time(), 0.0, {}
        if rank == 0 and (step % cfg["ckpt_every"] == 0 or step == cfg["iters"] or step == args.stop_at):
            tmp = ck.with_suffix(".tmp")
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step,
                            n_skipped=n_skipped, cfg=cfg, norm=norm, world=world), tmp)
            os.replace(tmp, ck)
        if step == args.stop_at:
            dist.barrier()
            print(f"--stop-at {step}", flush=True)
            break
        if step % cfg["val_every"] == 0 or step == cfg["iters"]:
            validate(model, cfg, norm, step, out_dir, rank, world)
            t_last = time.time()
        t_fetch = time.time()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
