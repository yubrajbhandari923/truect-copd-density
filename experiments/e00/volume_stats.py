"""E00 stage 2: read every phantom and recon once (streamed) for content-level facts.

Per subject -> OUT/volume_stats/<id>.json: phantom sha256 + label histogram, recon shape,
intensity histogram, per-slice mean (edge-band detection). Resumable: existing files are skipped.
"""
import hashlib
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pydicom

from common import OUT, SRC, pairing, phantom_file

D = OUT / "volume_stats"


def one(sid):
    dst = D / f"{sid}.json"
    if dst.exists():
        return sid
    pf, shape = phantom_file(sid)
    h, hist = hashlib.sha256(), np.zeros(256, np.int64)
    zocc = []  # per-z fraction of non-air voxels (labels 0/70 are air in the 10313 table)
    plane = shape[1] * shape[2]
    with open(pf, "rb") as f:
        while b := f.read(plane * 16):
            h.update(b)
            a = np.frombuffer(b, np.uint8)
            hist += np.bincount(a, minlength=256)
            zocc += [float(((s != 0) & (s != 70)).mean()) for s in a.reshape(-1, plane)]
    d = SRC / f"recons/dcm_recons_copd/dcm_recon_{sid}"
    files = sorted(f for f in os.listdir(d) if f.endswith(".dcm"))
    rh = np.zeros(8192, np.int64)  # raw stored value + 4096, clipped
    smean, sfrac = [], []
    shp, lo, hi = None, 1 << 30, -(1 << 30)
    for fn in files:
        px = pydicom.dcmread(d / fn).pixel_array
        shp = px.shape
        lo, hi = min(lo, int(px.min())), max(hi, int(px.max()))
        rh += np.bincount((px.ravel().astype(np.int32) + 4096).clip(0, 8191), minlength=8192)
        smean.append(float(px.mean()))
        sfrac.append(float((px > -500).mean()))
    nz = np.flatnonzero(rh)
    out = dict(id=sid, phantom_sha256=h.hexdigest(), phantom_shape_zyx=shape,
               label_hist={int(k): int(hist[k]) for k in np.flatnonzero(hist)},
               phantom_z_nonair_frac=zocc,
               recon_shape=[len(files), *shp], recon_dtype=str(px.dtype), recon_min=lo, recon_max=hi,
               recon_hist={int(k) - 4096: int(rh[k]) for k in nz},
               recon_slice_mean=smean, recon_slice_frac_gt_m500=sfrac)
    tmp = dst.with_suffix(".tmp")
    json.dump(out, open(tmp, "w"))
    tmp.rename(dst)
    return sid


if __name__ == "__main__":
    D.mkdir(parents=True, exist_ok=True)
    ids = [p[0] for p in pairing()]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as p:
        for s in p.imap_unordered(one, ids):
            print("done", s, flush=True)
