"""E00 stage 4: supervised targets on the CT grid for one subject.

usage: build_targets.py <subject_id|array_index> [fit_key]   (in a Slurm array the task id is the index)   (fit_key: full | translation_only_s1)

Each CT voxel is treated as a box in phantom space (alignment JSON transform). Targets are exact
box averages of the 0.25 mm phantom over that box (volume-weighted), not interpolation:
  density.npy  float16  mean mass density g/cm3
  cls.npy      uint8    majority name-level class (index into meta.json "classes")
  purity.npy   uint8    volume fraction of that class * 255   (255 = pure voxel, <255 = boundary)
  organ_pt.npy uint8    organ ID point-sampled at the voxel centre (native ID, for A15 / export)
  valid.npy    uint8    1 where the whole CT voxel lies inside the phantom grid
  ct.npy       int16    reconstruction as stored
Resumable: skipped if meta.json exists.
"""
import json
import os
import sys

import numpy as np
import pydicom
from numpy.lib.format import open_memmap
from scipy import sparse

from common import OUT, PH, SRC, pairing, phantom_file, xlsx_table

VR, VP = 0.6, 0.25  # assumed voxel sizes, mm (only the ratio is constrained by the fit)


def box_weights(n_recon, n_ph, A, sign, s):
    """Sparse [n_recon, n_ph] overlap fractions of each recon voxel's extent with phantom voxels,
    the coverage (row sum, 1 = fully inside), and the phantom index at the voxel centre."""
    e = (A + sign * s * VR * np.arange(n_recon + 1)) / VP  # voxel edges in phantom index units
    lo, hi = np.minimum(e[:-1], e[1:]), np.maximum(e[:-1], e[1:])
    rows, cols, vals = [], [], []
    for i in range(n_recon):
        for j in range(max(0, int(np.floor(lo[i]))), min(n_ph, int(np.ceil(hi[i])))):
            w = min(hi[i], j + 1) - max(lo[i], j)
            if w > 0:
                rows.append(i), cols.append(j), vals.append(w / (hi[i] - lo[i]))
    W = sparse.csr_matrix((vals, (rows, cols)), shape=(n_recon, n_ph), dtype=np.float32)
    centre = np.floor((lo + hi) / 2).astype(int)
    return W, np.asarray(W.sum(1)).ravel(), centre


def main(sid, fit_key):
    dst = OUT / "targets" / sid
    if (dst / "meta.json").exists():
        return print("exists", sid)
    dst.mkdir(parents=True, exist_ok=True)
    al = json.load(open(OUT / f"alignment/{sid}.json"))
    perm, flips = al["orientation"]["perm"], al["orientation"]["flips"]
    fit = al["fine"][fit_key]
    A, s = fit["A_mm"], fit.get("scale", [1, 1, 1])
    pf, pshape = phantom_file(sid)
    assert list(pshape) == al["phantom_shape_zyx"]
    vol = np.memmap(pf, dtype=np.uint8, mode="r").reshape(pshape)

    tab, _ = xlsx_table(sid)
    classes = sorted({v["name"] for v in tab.values()})
    cls_lut = np.full(256, 255, np.uint8)
    dens_lut = np.full(256, np.nan, np.float32)
    for k, v in tab.items():
        cls_lut[k], dens_lut[k] = classes.index(v["name"]), v["density"]
    mu_lut = np.pad(np.fromfile(PH / f"material_lac/{sid}/lin_att_120kV_121materials.bin", np.float32).reshape(121, 120)[:, 69], (0, 135))  # 70 keV, cm^-1

    d = SRC / f"recons/dcm_recons_copd/dcm_recon_{sid}"
    files = sorted(f for f in os.listdir(d) if f.endswith(".dcm"))
    rshape = (len(files), 768, 768)
    assert list(rshape) == al["recon"]["shape"]

    # weights per recon axis m (acting on phantom axis perm[m])
    Wm, cov, ctr = zip(*[box_weights(rshape[m], pshape[perm[m]], A[m], -1 if flips[m] else 1, s[m]) for m in range(3)])
    inv = [perm.index(a) for a in range(3)]  # phantom axis a <- recon axis inv[a]
    assert inv[0] == 0, "slice axis must map to phantom z (true for the cohort orientation)"
    Wz, Wy, Wx = Wm[inv[0]], Wm[inv[1]], Wm[inv[2]]  # rows indexed by recon idx along that recon axis
    inplane_ok = np.multiply.outer(cov[inv[1]] > 0.999, cov[inv[2]] > 0.999)  # [recon axis inv1, recon axis inv2]
    to_recon = (lambda a: a) if inv[1] == 1 else (lambda a: a.T)  # -> [row, col]

    out = {n: open_memmap(dst / f"{n}.npy", mode="w+", dtype=t, shape=rshape) for n, t in
           dict(ct="<i2", density="<f2", cls="u1", purity="u1", organ_pt="u1", valid="u1").items()}
    C = len(classes)
    acc = dict(n=np.zeros(C), ct=np.zeros(C), ct2=np.zeros(C), rho=np.zeros(C), mu=np.zeros(C))  # pure voxels only
    sx = sy = sxx = syy = sxy = n_valid = 0.0  # CT vs ideal mu70, all valid voxels
    for i, fn in enumerate(files):
        ct = pydicom.dcmread(d / fn).pixel_array
        out["ct"][i] = ct
        if cov[0][i] <= 0.999:
            out["cls"][i] = 255
            continue
        row = Wz.getrow(i)
        slab = np.asarray(vol[row.indices.min():row.indices.max() + 1])
        wz = np.zeros(slab.shape[0], np.float32)
        wz[row.indices - row.indices.min()] = row.data
        proj = lambda img: to_recon(np.asarray(Wy @ (Wx @ img.T).T))  # [ny,nx] -> [row,col]
        zavg = lambda lut: np.tensordot(wz, lut[slab], axes=1)
        valid = to_recon(inplane_ok)
        rho, mu = proj(zavg(dens_lut)), proj(zavg(mu_lut))
        cl = cls_lut[slab]
        best, arg = np.zeros((768, 768), np.float32), np.full((768, 768), 255, np.uint8)
        for c in np.unique(cl):
            f = proj(np.tensordot(wz, (cl == c).astype(np.float32), axes=1))
            m = f > best
            best[m], arg[m] = f[m], c
        pt = to_recon(np.asarray(vol[ctr[0][i]])[np.ix_(np.clip(ctr[inv[1]], 0, pshape[1] - 1), np.clip(ctr[inv[2]], 0, pshape[2] - 1))])
        out["density"][i] = np.where(valid, rho, 0)
        out["cls"][i] = np.where(valid, arg, 255)
        out["purity"][i] = np.where(valid, np.round(np.clip(best, 0, 1) * 255), 0)
        out["organ_pt"][i] = np.where(valid, pt, 255)
        out["valid"][i] = valid
        pure = valid & (best > 0.999)
        x, y = mu[valid].astype(np.float64), ct[valid].astype(np.float64)
        n_valid += x.size; sx += x.sum(); sy += y.sum(); sxx += (x * x).sum(); syy += (y * y).sum(); sxy += (x * y).sum()
        for c in np.unique(arg[pure]):
            m = pure & (arg == c)
            v = ct[m].astype(np.float64)
            acc["n"][c] += v.size; acc["ct"][c] += v.sum(); acc["ct2"][c] += (v * v).sum()
            acc["rho"][c] += rho[m].sum(); acc["mu"][c] += mu[m].sum()
    for a in out.values():
        a.flush()
    vx, vy, cxy = sxx / n_valid - (sx / n_valid) ** 2, syy / n_valid - (sy / n_valid) ** 2, sxy / n_valid - sx * sy / n_valid ** 2
    n = np.maximum(acc["n"], 1)
    meta = dict(
        id=sid, fit_key=fit_key, A_mm=A, scale=s, perm=perm, flips=flips, recon_shape=rshape, phantom_shape_zyx=pshape,
        assumed_voxel_mm=dict(recon=VR, phantom=VP), classes=classes, alignment_r=fit["r"],
        valid_fraction=float(n_valid / np.prod(rshape)),
        valid_slices=[int(np.flatnonzero(cov[0] > 0.999)[0]), int(np.flatnonzero(cov[0] > 0.999)[-1])],
        ct_vs_mu70=dict(n=int(n_valid), slope_HU_per_cm1=cxy / vx, intercept_HU=sy / n_valid - cxy / vx * sx / n_valid,
                        r=cxy / np.sqrt(vx * vy)),
        pure_voxel_stats={classes[c]: dict(n=int(acc["n"][c]), ct_mean=acc["ct"][c] / n[c],
                                           ct_sd=float(np.sqrt(max(acc["ct2"][c] / n[c] - (acc["ct"][c] / n[c]) ** 2, 0))),
                                           density_mean=acc["rho"][c] / n[c], mu70_mean=acc["mu"][c] / n[c])
                          for c in range(C) if acc["n"][c] > 0})
    json.dump(meta, open(dst / "meta.tmp", "w"), indent=1)
    os.rename(dst / "meta.tmp", dst / "meta.json")
    print("done", sid, "valid", meta["valid_fraction"], meta["ct_vs_mu70"])


if __name__ == "__main__":
    a = os.environ.get("SLURM_ARRAY_TASK_ID") or sys.argv[1]
    sid = pairing()[int(a)][0] if len(a) < 4 else a
    main(sid, sys.argv[2] if len(sys.argv) > 2 else "full")
