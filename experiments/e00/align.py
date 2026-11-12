"""E00 CT<->phantom geometric alignment for one subject (or several): python align.py <sid> [<sid> ...]

Model (per recon axis m = 0 slice, 1 row, 2 col; recon voxel centre y_m = 0.6*(idx+0.5) mm under the
0.6 mm assumption; phantom voxel centre x = 0.25*(idx+0.5) mm under the 0.25 mm assumption):
    x[perm[m]] = A_m + sign_m * s_m * y_m
so s_m = 1 means the voxel-size ratio is exactly 2.4; implied recon spacing along m = 0.6*s_m mm if the
phantom is truly 0.25 mm. Phantom image = 70 keV LAC of each label (lin_att column 69, cm^-1).

Stages: (a) masked NCC over all integer shifts for 48 signed permutations at 3 mm blocks, plus a coarse
scale scan (slice axis and in-plane) of the best candidate, then the 48-way search repeated at that scale;
(b) Nelder-Mead on 0.5 mm phantom / 1.8 mm recon blocks: translation-only at s=1, translation-only at
s_z=0.5476/0.6, then translation + 3 scales; (c) sensitivity, coverage boxes, labels-only anatomy checks,
3-plane overlay PNG. Output OUT/alignment/<sid>.json + <sid>.png (skipped if the JSON exists).
NOTE: correlation of the very pair being registered is not independent validation of the transform.
"""
import itertools
import json
import os
import sys
import time

import numpy as np
import pydicom
import scipy.fft as sfft
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize

from common import OUT, PH, SRC, phantom_file, xlsx_table

D = OUT / "alignment"
VR, VP = 0.6, 0.25           # assumed recon / phantom voxel sizes (mm); only their ratio is fitted
SZ_ALT = 0.5476 / 0.6        # slice-axis scale if recon slices were 0.5476 mm
WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
ORIENTS = [(p, f) for p in itertools.permutations(range(3)) for f in itertools.product((0, 1), repeat=3)]


# ---------------------------------------------------------------- coarse search
def orient(P, perm, flips):
    """Phantom array re-ordered so its axis m is phantom axis perm[m], reversed where flips[m]."""
    Q = np.transpose(P, perm)
    return np.ascontiguousarray(Q[tuple(slice(None, None, -1) if f else slice(None) for f in flips)])


class NCC:
    """Pearson r of A (moving) vs B over the overlap of their boxes, for every integer shift."""

    def __init__(self, B):
        self.B = B - B.mean()
        self.cache = {}

    def __call__(self, A, min_frac=0.6):
        B = self.B
        shp = tuple(sfft.next_fast_len(a + b - 1, real=True) for a, b in zip(A.shape, B.shape))
        if (A.shape, shp) not in self.cache:
            f = lambda x: sfft.rfftn(x, shp, workers=WORKERS)
            oa = np.ones(A.shape)
            self.cache[(A.shape, shp)] = (f(np.ones(B.shape)), f(B), f(B * B), f(oa[::-1, ::-1, ::-1]))
        F1b, Fb, Fbb, F1a = self.cache[(A.shape, shp)]
        A = (A - A.mean())[::-1, ::-1, ::-1]
        Fa, Faa = sfft.rfftn(A, shp, workers=WORKERS), sfft.rfftn(A * A, shp, workers=WORKERS)
        full = tuple(slice(0, a + b - 1) for a, b in zip(A.shape, B.shape))
        g = lambda x: sfft.irfftn(x, shp, workers=WORKERS)[full]
        n, sa, saa = g(F1a * F1b), g(Fa * F1b), g(Faa * F1b)
        sb, sbb, sab = g(F1a * Fb), g(F1a * Fbb), g(Fa * Fb)
        n = np.rint(n)
        with np.errstate(all="ignore"):
            r = (sab - sa * sb / n) / np.sqrt(np.clip(saa - sa**2 / n, 0, None) * np.clip(sbb - sb**2 / n, 0, None))
        r[(n < min_frac * min(A.size, B.size)) | ~np.isfinite(r)] = -2
        i = np.unravel_index(np.argmax(r), r.shape)
        # shift: A[j] overlays B[j + shift]
        return float(r[i]), [int(a - (s - 1)) for a, s in zip(i, A.shape)], float(n[i] / min(A.size, B.size))


def search48(P3, R3, ncc=None):
    ncc = ncc or NCC(R3)
    out = []
    for perm, flips in ORIENTS:
        r, sh, ov = ncc(orient(P3, perm, flips))
        out.append(dict(perm=list(perm), flips=list(flips), r=r, shift=sh, overlap=ov))
    return sorted(out, key=lambda d: -d["r"])


def rescale(R, s):
    """Resample recon block grid so one block spans the same phantom distance after scaling by s (per axis)."""
    ax = [((np.arange(int(n * si)) + 0.5) / si - 0.5) for n, si in zip(R.shape, s)]
    return map_coordinates(R, np.meshgrid(*ax, indexing="ij"), order=1, mode="nearest")


def coarse_A(cand, P3shape, s, block=3.0):
    """Translation A (mm) of the model from a 3 mm-block shift (derivation in module docstring)."""
    A = []
    for m in range(3):
        nb, sh = P3shape[cand["perm"][m]], cand["shift"][m]
        A.append(block * (nb + sh) if cand["flips"][m] else -block * sh)
    return A


# ---------------------------------------------------------------- data
def load_recon(sid):
    d = SRC / f"recons/dcm_recons_copd/dcm_recon_{sid}"
    files = sorted((f for f in os.listdir(d) if f.endswith(".dcm")), key=lambda f: int(f[6:-4]))
    rec = np.stack([pydicom.dcmread(d / f).pixel_array for f in files])
    assert rec.dtype == np.int16 and rec.ndim == 3, rec.shape
    return rec


def blocks(a, k):
    s = [n // k * k for n in a.shape]
    a = a[: s[0], : s[1], : s[2]]
    return a.reshape(s[0] // k, k, s[1] // k, k, s[2] // k, k).mean(axis=(1, 3, 5), dtype=np.float32)


def load_phantom(sid):
    """Stream labels in z-chunks -> mu70 at 0.5 mm (2^3 means), 3 mm (12^3 means), labels at 0.5 mm (stride 2)."""
    pf, shp = phantom_file(sid)
    tab, _ = xlsx_table(sid)
    lac = np.fromfile(PH / f"material_lac/{sid}/lin_att_120kV_121materials.bin", np.float32).reshape(121, 120)
    lut = np.full(256, np.nan, np.float32)
    for k in tab:
        lut[k] = lac[k, 69]  # column = keV-1 (pinned on 10313)
    nz, ny, nx = shp
    P05 = np.empty((nz // 2, ny // 2, nx // 2), np.float32)
    L2 = np.empty((nz // 2, ny // 2, nx // 2), np.uint8)
    P3 = np.empty((nz // 12, ny // 12, nx // 12), np.float32)
    hist = np.zeros(256, np.int64)
    mm = np.memmap(pf, np.uint8, "r", shape=shp)
    for z0 in range(0, nz // 2 * 2, 12):
        c = np.asarray(mm[z0: min(z0 + 12, nz // 2 * 2)])
        hist += np.bincount(c.ravel(), minlength=256)
        v = lut[c]
        b = v[:, : ny // 2 * 2, : nx // 2 * 2].reshape(len(c) // 2, 2, ny // 2, 2, nx // 2, 2).mean(axis=(1, 3, 5))
        P05[z0 // 2: z0 // 2 + len(b)] = b
        L2[z0 // 2: z0 // 2 + len(b)] = c[::2, ::2, ::2][:, : ny // 2, : nx // 2]
        if len(c) == 12:
            P3[z0 // 12] = b[:, : ny // 12 * 6, : nx // 12 * 6].reshape(1, 6, ny // 12, 6, nx // 12, 6).mean(axis=(1, 3, 5))[0]
    del mm
    unknown = {int(k): int(hist[k]) for k in np.flatnonzero(hist) if k not in tab}
    assert not unknown, (sid, "labels absent from table", unknown)
    assert np.isfinite(P05).all()
    return P05, P3, L2, tab, shp


# ---------------------------------------------------------------- anatomy (labels only, native phantom axes)
def anatomy(L2, tab):
    L = L2[::2, ::2, ::2]  # 1 mm
    ids = lambda *names: [k for k, v in tab.items() if v["name"] in names]
    m = lambda *names: np.isin(L, ids(*names))
    cen = lambda msk: np.array([c.mean() for c in np.nonzero(msk)])  # 1 mm units
    lung, spine = m("lung", "lung_parenchyma", "emphysema"), m("spine_cortical", "spine_trabecular")
    stern, scap, aorta = m("sternum", "sternum_trabecular"), m("scapulas", "scapulas_trabecular"), m("aorta")
    c_sp, c_st, c_sc, c_lu = cen(spine), cen(stern), cen(scap), cen(lung)
    sd = np.array([c.std() for c in np.nonzero(spine)])
    si = int(np.argmax(sd))  # the spine is the long S-I column
    d_ap, d_si = c_st - c_sp, c_sc - c_lu
    rest = [a for a in range(3) if a != si]
    ap = rest[int(np.argmax(abs(d_ap[rest])))]
    lr = [a for a in rest if a != ap][0]
    lz = np.nonzero(lung)
    n_lo, n_hi = int((lz[lr] < c_sp[lr]).sum()), int((lz[lr] > c_sp[lr]).sum())
    right_is_low = n_lo > n_hi  # right lung larger
    # aorta vs spine in the lower third of the lung S-I range
    zr = np.flatnonzero(lung.any(axis=tuple(a for a in range(3) if a != si)))
    lo_s, hi_s = zr.min(), zr.max()
    third = (lo_s, lo_s + (hi_s - lo_s) // 3) if d_si[si] > 0 else (hi_s - (hi_s - lo_s) // 3, hi_s)
    sl = [slice(None)] * 3
    sl[si] = slice(third[0], third[1] + 1)
    az, sp = np.nonzero(aorta[tuple(sl)]), np.nonzero(spine[tuple(sl)])
    aorta_minus_spine_lr = float(az[lr].mean() - sp[lr].mean()) if len(az[0]) and len(sp[0]) else None
    letters = [None] * 3  # anatomical direction of increasing native phantom index
    letters[ap] = "A" if d_ap[ap] > 0 else "P"
    letters[si] = "S" if d_si[si] > 0 else "I"
    letters[lr] = "L" if right_is_low else "R"
    aorta_left_ok = None if aorta_minus_spine_lr is None else bool((aorta_minus_spine_lr > 0) == right_is_low)
    return dict(phantom_axes_increasing=letters, ap_axis=ap, si_axis=si, lr_axis=lr,
                sternum_minus_spine_mm=d_ap.round(1).tolist(), scapula_minus_lung_mm=d_si.round(1).tolist(),
                lung_vox_1mm_low_side=n_lo, lung_vox_1mm_high_side=n_hi, lung_ratio_larger_over_smaller=round(max(n_lo, n_hi) / max(1, min(n_lo, n_hi)), 3),
                lower_third_aorta_minus_spine_lr_mm=aorta_minus_spine_lr, aorta_left_consistent_with_right_lung_larger=aorta_left_ok,
                ap_dominance=float(abs(d_ap[ap]) / (abs(d_ap[lr]) + 1e-9)),
                spine_elongation=float(sd[si] / np.sort(sd)[-2]), spine_sd_mm=sd.round(1).tolist(),
                note="1 mm label grid; centroids in mm from phantom origin; letters assume normal situs")


OPP = dict(A="P", P="A", S="I", I="S", L="R", R="L")


# ---------------------------------------------------------------- main per subject
def run(sid):
    dst = D / f"{sid}.json"
    if dst.exists():
        print(sid, "exists, skip")
        return
    t0 = time.time()
    rec = load_recon(sid)
    R3 = blocks(rec.clip(-1000, 1500), 5)
    R18 = blocks(rec.clip(-1000, 1500), 3)
    nrec = rec.shape
    rstats = dict(shape=list(nrec), min=int(rec.min()), max=int(rec.max()), frac_eq_m1000=float((rec == -1000).mean()))
    P05, P3, L2, tab, pshp = load_phantom(sid)
    print(sid, "loaded", nrec, pshp, f"{time.time() - t0:.0f}s", flush=True)

    # (a) 48-way search at s=1, coarse scale scan of the winner, 48-way search again at that scale
    s1 = search48(P3, R3)
    best = s1[0]
    Pc = orient(P3, best["perm"], best["flips"])
    scan, s = [], np.ones(3)

    def at(sv):
        r, sh, ov = NCC(rescale(R3, sv))(Pc)
        scan.append(dict(s=[round(x, 4) for x in sv], r=r))
        return r
    for _ in range(2):
        s[0] = max(np.arange(0.85, 1.0751, 0.01), key=lambda q: at([q, s[1], s[2]]))
        s[1] = s[2] = max(np.arange(0.90, 1.1001, 0.01), key=lambda q: at([s[0], q, q]))
    ncc_s = NCC(rescale(R3, s))
    s2 = search48(P3, rescale(R3, s), ncc_s)
    best, runner = s2[0], s2[1]
    print(sid, "coarse", best["perm"], best["flips"], round(best["r"], 4), "runner", runner["r"], "s", s, f"{time.time() - t0:.0f}s", flush=True)

    # (b) fine fit; parametrise translation at recon-axis centre (C) to decorrelate it from scale
    perm, flips = best["perm"], best["flips"]
    sg = np.array([-1.0 if f else 1.0 for f in flips])
    yc = np.array([VR * n / 2 for n in nrec])
    A0 = np.array(coarse_A(best, P3.shape, s))
    C0 = A0 + sg * s * yc
    Y = np.meshgrid(*[VR * (3 * np.arange(n) + 1.5) for n in R18.shape], indexing="ij")
    keep = np.ones(R18.shape, bool)
    keep[:2] = keep[-2:] = False  # recon edge bands at first/last ~6 slices

    def pidx(p, Ys):
        C, sc = p[:3], p[3:]
        idx = [None] * 3
        for m in range(3):
            idx[perm[m]] = (C[m] + sg[m] * sc[m] * (Ys[m] - yc[m])) / 0.5 - 0.5
        return idx
    ci = pidx(np.r_[C0, s], Y)
    for a in range(3):
        keep &= (ci[a] > 10) & (ci[a] < P05.shape[a] - 11)
    pts = np.flatnonzero(keep.ravel())
    pts = pts[:: max(1, len(pts) // 1_500_000)]
    Yf = [Y[m].ravel()[pts] for m in range(3)]
    Rs = R18.ravel()[pts].astype(np.float64)

    def r_of(p):
        v = map_coordinates(P05, pidx(p, Yf), order=1, mode="nearest")
        return float(np.corrcoef(v, Rs)[0, 1])

    def fit_t(C, sc):
        sim = np.vstack([C] + [C + np.eye(3)[i] for i in range(3)])
        for _ in range(2):  # restart once: a single Nelder-Mead run stalled 0.4 mm short on 10313
            o = minimize(lambda q: -r_of(np.r_[q, sc]), C, method="Nelder-Mead",
                         options=dict(initial_simplex=sim, xatol=0.01, fatol=1e-8, maxiter=400))
            C = o.x
            sim = np.vstack([C] + [C + np.eye(3)[i] * 0.5 for i in range(3)])
        return np.r_[o.x, sc], -o.fun

    # start translation-only fits from the coarse C re-expressed for the fixed scale (keep x at recon centre)
    toA = lambda p: (p[:3] - sg * p[3:] * yc).tolist()
    p_t1, r_t1 = fit_t(C0, np.ones(3))
    p_tz, r_tz = fit_t(C0, np.array([SZ_ALT, 1, 1]))
    starts = [np.r_[C0, s], p_t1, p_tz]
    p_full, r_full, ends = None, -2, []
    for p0 in starts:
        p0 = fit_t(p0[:3], p0[3:])[0]
        sim = np.vstack([p0] + [p0 + np.r_[np.eye(3)[i] * 0.5, 0, 0, 0] for i in range(3)]
                        + [p0 + np.r_[0, 0, 0, np.eye(3)[i] * 0.004] for i in range(3)])
        o = minimize(lambda q: -r_of(q), p0, method="Nelder-Mead",
                     options=dict(initial_simplex=sim, xatol=0.005, fatol=1e-8, maxiter=1500))
        ends.append(dict(start_scale=p0[3:].tolist(), A_mm=toA(o.x), scale=o.x[3:].tolist(), r=-o.fun))
        if -o.fun > r_full:
            p_full, r_full = o.x, -o.fun
    sens = {}
    for m in range(3):
        for d in (-1.0, 1.0):
            q = p_full.copy(); q[m] += d; sens[f"shift_axis{m}_{d:+.0f}mm"] = r_of(q) - r_full
        for d in (-0.01, 0.01):
            q = p_full.copy(); q[3 + m] *= 1 + d; sens[f"scale_axis{m}_{d:+.2f}"] = r_of(q) - r_full
    print(sid, "fine r_t1", r_t1, "r_tz", r_tz, "full", r_full, p_full[3:], f"{time.time() - t0:.0f}s", flush=True)

    # (c) coverage under the full fit
    A, sc = np.array(toA(p_full)), p_full[3:]
    cov = {}
    for m in range(3):
        a = perm[m]
        Lp = VP * pshp[a]
        y_ends = (np.array([0.0, Lp]) - A[m]) / (sg[m] * sc[m])          # recon mm where phantom box ends
        ridx = np.sort(y_ends / VR - 0.5)
        r_lo, r_hi = max(0, int(np.ceil(ridx[0]))), min(nrec[m] - 1, int(np.floor(ridx[1])))
        x_ends = A[m] + sg[m] * sc[m] * np.array([0.0, VR * nrec[m]])
        pi = np.sort(x_ends / VP - 0.5)
        p_lo, p_hi = max(0, int(np.ceil(pi[0]))), min(pshp[a] - 1, int(np.floor(pi[1])))
        cov[m] = dict(recon_axis=m, phantom_axis=a, recon_idx_in_phantom=[r_lo, r_hi], recon_n=nrec[m],
                      recon_frac_in_phantom=max(0, r_hi - r_lo + 1) / nrec[m],
                      phantom_idx_in_recon=[p_lo, p_hi], phantom_n=pshp[a],
                      phantom_frac_in_recon=max(0, p_hi - p_lo + 1) / pshp[a])
    an = anatomy(L2, tab)
    recon_letters = "".join(an["phantom_axes_increasing"][perm[m]] if sg[m] > 0 else OPP[an["phantom_axes_increasing"][perm[m]]]
                            for m in range(3))
    res = dict(
        id=sid, recon=rstats, phantom_shape_zyx=list(pshp),
        model="x_phantom[perm[m]] = A_m + sign_m*s_m*y_m; y_m = 0.6*(recon_idx+0.5) mm, x = 0.25*(phantom_idx+0.5) mm; sign=-1 where flips",
        phantom_image="70 keV LAC (lin_att col 69), cm^-1; recon HU clipped [-1000,1500]",
        coarse=dict(block_mm=3, search_scale1_top5=s1[:5], scale_scan=scan, coarse_scale=s.round(4).tolist(),
                    search_at_scale_top5=s2[:5], rank_of_s1_best_at_scale=next(i for i, d in enumerate(s2) if d["perm"] == s1[0]["perm"] and d["flips"] == s1[0]["flips"]),
                    worst_r=s2[-1]["r"]),
        orientation=dict(perm=perm, flips=flips, r3mm=best["r"]),
        runner_up=dict(perm=runner["perm"], flips=runner["flips"], r3mm=runner["r"], margin=best["r"] - runner["r"]),
        fine=dict(n_points=len(pts), translation_only_s1=dict(A_mm=toA(p_t1), r=r_t1),
                  translation_only_sz_0p5476=dict(A_mm=toA(p_tz), scale=[SZ_ALT, 1, 1], r=r_tz),
                  full=dict(A_mm=toA(p_full), scale=p_full[3:].tolist(), r=r_full), full_fit_from_each_start=ends,
                  implied_recon_spacing_mm_if_phantom_0p25=(VR * p_full[3:]).tolist(),
                  sensitivity_delta_r=sens),
        coverage=list(cov.values()),
        anatomy=dict(an, recon_axes_increasing=recon_letters),
        runtime_s=round(time.time() - t0),
    )
    overlay(sid, res, rec, L2, tab, p_full, perm, sg, yc)
    tmp = dst.with_suffix(".tmp")
    json.dump(res, open(tmp, "w"), indent=1)
    tmp.rename(dst)
    print(sid, "done", recon_letters, f"{time.time() - t0:.0f}s", flush=True)


def overlay(sid, res, rec, L2, tab, p, perm, sg, yc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nrec = res["recon"]["shape"]
    cov = res["coverage"]
    ctr = [(c["recon_idx_in_phantom"][0] + c["recon_idx_in_phantom"][1]) // 2 for c in cov]
    groups = dict(body=[k for k, v in tab.items() if v["name"] != "air"],
                  lung=[k for k, v in tab.items() if v["name"] in ("lung", "lung_parenchyma", "emphysema")],
                  bone=[k for k, v in tab.items() if any(w in v["name"] for w in ("spine", "rib", "stern", "scap"))])
    colors = dict(body="yellow", lung="cyan", bone="magenta")
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.4))
    for m, ax in enumerate(axs):
        # CT plane at index ctr[m] along recon axis m
        img = np.take(rec, ctr[m], axis=m)
        other = [a for a in range(3) if a != m]
        grids = np.meshgrid(*[np.arange(nrec[a]) for a in other], indexing="ij")
        Ys = [None] * 3
        Ys[m] = np.full(grids[0].shape, VR * (ctr[m] + 0.5))
        for g, a in zip(grids, other):
            Ys[a] = VR * (g + 0.5)
        idx = [None] * 3
        for k in range(3):
            x = p[k] + sg[k] * p[3 + k] * (Ys[k] - yc[k])
            idx[perm[k]] = np.rint((x / VP - 0.5) / 2).astype(int)  # 0.5 mm stride-2 label grid
        inside = np.all([(idx[a] >= 0) & (idx[a] < L2.shape[a]) for a in range(3)], axis=0)
        lab = np.full(grids[0].shape, 255, np.uint8)
        lab[inside] = L2[tuple(i[inside] for i in idx)]
        ax.imshow(img, cmap="gray", vmin=-1000, vmax=400, aspect=p[3 + other[0]] / p[3 + other[1]] if m else 1)
        for name, ids in groups.items():
            ax.contour(np.isin(lab, ids).astype(float), [0.5], colors=colors[name], linewidths=0.6)
        ax.contour(inside.astype(float), [0.5], colors="red", linewidths=0.8, linestyles="dashed")
        ax.set_title(f"recon axis {m} = {ctr[m]}  (rows: axis {other[0]}, cols: axis {other[1]})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    f = res["fine"]["full"]
    fig.suptitle(f"{sid}  perm {res['orientation']['perm']} flips {res['orientation']['flips']}  r={f['r']:.4f}  "
                 f"s={np.round(f['scale'], 4).tolist()}  axes {res['anatomy']['recon_axes_increasing']}  "
                 f"(yellow body, cyan lung, magenta bone, red dashed phantom box)", fontsize=9)
    fig.tight_layout()
    fig.savefig(D / f"{sid}.png", dpi=80)
    plt.close(fig)


if __name__ == "__main__":
    D.mkdir(parents=True, exist_ok=True)
    ids = sys.argv[1:]
    if os.environ.get("SLURM_ARRAY_TASK_ID") and not ids:
        from common import pairing
        ids = [pairing()[int(os.environ["SLURM_ARRAY_TASK_ID"])][0]]
    for sid in ids:
        run(sid)
