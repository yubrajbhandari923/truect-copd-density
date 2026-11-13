"""E02 U5 preprocessing: compact per-subject 3D training cache (TRAIN subjects only), built once, reused by every 3D run.

Per subject, cropped to the valid slices x (in-plane bounding box of the evaluation region + 32 px margin):
  ct.npy int16 HU, density.npy float16 g/cm3, lab.npy uint8 = cls inside truect.data.eval_mask, 255 elsewhere
  (255 = no loss: the 3D runs use the "body" loss region, as the best 2D run u3_s11), centers.npy int32 (N, 3)
  = up to 500k uniform random eval-region voxels (crop coordinates) for patch sampling, meta.json (crop box).
Usage: python prep3d.py [n_procs]. Idempotent: finished subjects (meta.json present) are skipped.
"""
import json, os, shutil, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from truect.data import WORK, eval_mask, load, split  # noqa: E402

CACHE = WORK / "cache/e02_3d"
MARGIN, N_CENTERS = 32, 500_000


def prep(sid):
    out = CACHE / sid
    if (out / "meta.json").exists():
        return sid, "cached"
    t, meta = load(sid, ("ct", "density", "cls", "valid", "organ_pt"))
    a, b = meta["valid_slices"]
    m = np.stack([eval_mask(t, z) for z in range(a, b + 1)])
    rows, cols = np.flatnonzero(m.any((0, 2))), np.flatnonzero(m.any((0, 1)))
    H, W = m.shape[1:]
    r0, r1 = max(rows[0] - MARGIN, 0), min(rows[-1] + 1 + MARGIN, H)
    c0, c1 = max(cols[0] - MARGIN, 0), min(cols[-1] + 1 + MARGIN, W)
    m = m[:, r0:r1, c0:c1]
    box = (slice(a, b + 1), slice(r0, r1), slice(c0, c1))
    tmp = CACHE / f".{sid}.tmp"
    shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    np.save(tmp / "ct.npy", np.asarray(t["ct"][box]))
    np.save(tmp / "density.npy", np.asarray(t["density"][box]))
    np.save(tmp / "lab.npy", np.where(m, t["cls"][box], 255).astype(np.uint8))
    idx = np.flatnonzero(m)
    rng = np.random.default_rng(int(sid))
    idx = rng.choice(idx, min(N_CENTERS, idx.size), replace=False)
    np.save(tmp / "centers.npy", np.stack(np.unravel_index(idx, m.shape), 1).astype(np.int32))
    json.dump(dict(id=sid, box_zrc=[[a, b + 1], [int(r0), int(r1)], [int(c0), int(c1)]], shape=list(m.shape),
                   n_eval_voxels=int(m.sum()), margin_px=MARGIN, loss_region="body (truect.data.eval_mask)"),
              open(tmp / "meta.json", "w"), indent=1)
    os.replace(tmp, out)
    return sid, list(m.shape)


if __name__ == "__main__":
    CACHE.mkdir(parents=True, exist_ok=True)
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 16) as p:
        for r in p.imap_unordered(prep, split()["train"]):
            print(*r, flush=True)
