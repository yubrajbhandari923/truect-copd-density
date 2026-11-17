"""Real-patient round trip: predicted material/density map -> drishya variant-3 CT simulator (LEAP) -> simulated CT,
compared with the real patient CT (ground truth). Exploratory; owner request 2026-11-17.

Per series (inputs from infer.py, model u5_s11 by default, 0.6 mm grid in the training orientation):
  mu(r, E) = rho_pred(r) * (mu/rho)_m(E), m = DukeSim MC material of the predicted class (truect MC_OF_CLASS), mass
  attenuation from the dataset's own MC-GPU material files (1/total mean free path / nominal density).
  Two-basis decomposition (muscle + densest bone, 20-120 keV) as drishya variant 2 -> LEAP helical forward projection
  -> drishya variant 3 physics (120 kV spectrum + water BH, detector blur, smooth residual kernel, Poisson noise)
  -> fixed G_RECON LEAP FBP. Also the ideal 70 keV HU of the predicted map (no acquisition physics).
Geometry: TrueCT 10313 scan (dcm_008) helical trajectory (view spacing, pitch), extended/shortened symmetrically to the
patient's z length; tube current constant = median of 10313's mAs vector (real patients have no TCM vector here).
Volume: in-plane centre-padded/cropped to 768 (0.6 mm) and padded like drishya (128 px, 64 slices of air).
Compare (central slices, body mask, LEAP valid FOV r <= 215 mm): median HU per TotalSegmentator region, body MAE,
lung MLD / Perc15 / LAA-950, high-pass noise SD in eroded back muscle.
usage: simct.py <dataset_dir> <ts_dir> <infer_dir> <out_dir> <rank> <world> [model]
"""
import json, sys, time, zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
import scipy.ndimage as ndi
from nibabel.processing import resample_from_to, resample_to_output

import leap_recon as R
from physics import E, FIT, V3, Spectral, intensities, noisy, to_g

TRUECT = Path("/scratch/railabs/yb107/Datasets/TrueCT")
MC = Path("/scratch/railabs/yb107/Datasets/TrueCT/phantoms/COPD/material_mcgpu")
MC_FILES = {1: "air", 2: "body", 3: "sternum", 4: "muscle", 5: "lung", 6: "scapulas", 7: "spine_cortical", 8: "skin", 9: "aorta"}
MC_OF_CLASS = [1, 9, 2, 5, 4, 4, 5, 5, 4, 6, 6, 6, 6, 8, 7, 7, 3, 3, 9]  # truect/data.py
TRAIN_AX = ("I", "P", "L")
PAD_XY, PAD_Z, N = 128, 64, 768


def mass_att(name):
    """(mu/rho)(E) in cm^2/g at E = 1..120 keV from an MC-GPU material file."""
    lines = open(MC / f"{name}.mcgpu").read().splitlines()
    rho = float(lines[lines.index(next(l for l in lines if "NOMINAL DENSITY" in l)) + 1].strip("# "))
    n = int(lines[lines.index(next(l for l in lines if "NUMBER OF DATA VALUES" in l)) + 1].strip("# "))
    i0 = next(i for i, l in enumerate(lines) if not l.startswith("#"))
    tab = np.array([l.split()[:5] for l in lines[i0:i0 + n]], float)
    return np.interp(E * 1000, tab[:, 0], 1.0 / tab[:, 4]) / rho, rho


def materials():
    mr = {m: mass_att(f)[0] for m, f in MC_FILES.items()}  # cm^2/g
    rho = {m: mass_att(f)[1] for m, f in MC_FILES.items()}
    bone = max((7, 3, 6), key=lambda m: mr[m][69] * rho[m])
    B = np.stack([mr[4] * rho[4], mr[bone] * rho[bone]], 1) / 10.0  # mm^-1 at nominal density (drishya convention)
    coef = {m: np.linalg.lstsq(B[FIT], mr[m][FIT] / 10.0, rcond=None)[0] for m in mr}  # per g/cm^3
    err = max(float((np.abs(B[FIT] @ coef[m] - mr[m][FIT] / 10) / (mr[m][FIT] / 10)).max()) for m in mr)
    return B, coef, mr, dict(bone_basis=MC_FILES[bone], max_rel_fit_err=err, nominal_rho=rho)


def _hdr(p):
    d = pydicom.dcmread(p, stop_before_pixels=True)
    f = lambda t: np.frombuffer(bytes(d[t].value), "<f4")[0]  # noqa: E731
    return int(d.InstanceNumber), float(f((0x7031, 0x1001))), float(f((0x7031, 0x1002)))


def ref_geometry(cache):
    if cache.exists():
        return dict(np.load(cache))
    sdir = TRUECT / "projections/dcmproj_copd/dcm_008"
    with ProcessPoolExecutor(16) as ex:
        h = sorted(ex.map(_hdr, sorted(map(str, sdir.glob("proj_*.dcm"))), chunksize=64))
    ang, z = np.unwrap(np.array([r[1] for r in h])), np.array([r[2] for r in h])
    mas = np.fromfile(next(sdir.glob("mAs_vector_*.bin")), "<f4").astype(np.float64)
    nz = len(list((TRUECT / "recons/dcm_recons_copd/dcm_recon_10313").glob("*.dcm")))
    g = dict(ang=ang, z=z, mas=mas, nz=np.array(nz))
    np.savez(cache, **g)
    return g


def trajectory(ref, nz):
    ang = ref["ang"]; da = float(np.median(np.diff(ang))); pitch = float(np.polyfit(ang, ref["z"], 1)[0])
    n = len(ang) + int(round((nz - int(ref["nz"])) * 0.6 / abs(pitch * da)))
    a = ang[len(ang) // 2] + da * (np.arange(n) - (n - 1) / 2)  # symmetric about the reference mid angle
    zf = np.polyfit(ang, ref["z"], 1)
    return a, np.polyval(zf, a), np.full(n, float(np.median(ref["mas"])))


def fit768(n):
    """(src, dst) slices mapping an axis of length n onto a centred 768 axis (pad or crop)."""
    if n <= N:
        o = (N - n) // 2; return slice(0, n), slice(o, o + n)
    o = (n - N) // 2; return slice(o, o + N), slice(0, N)


def project(vol, ang, z):
    c_ = dict(R.CONV, nxy=vol.shape[1])
    L, _ = R.make_leap(len(ang), ang, z, vol.shape[0], c_)
    L.set_volume(vol.shape[2], vol.shape[1], vol.shape[0], 0.6, 0.6, c_["off_x_mm"], c_["off_y_mm"], 0.0)
    p = L.allocateProjections(); L.project(p, np.ascontiguousarray(vol))
    return np.asarray(p)


def to_train(img):
    return img.as_reoriented(nib.orientations.ornt_transform(nib.io_orientation(img.affine), nib.orientations.axcodes2ornt(TRAIN_AX)))


def ts_groups(names):
    g = lambda f: [i for i, n in names.items() if f(n)]  # noqa: E731
    return {"lung": g(lambda n: n.startswith("lung_")), "muscle": g(lambda n: n.startswith("autochthon_")),
            "aorta": g(lambda n: n == "aorta"), "heart": g(lambda n: n == "heart"), "spine": g(lambda n: n.startswith("vertebrae_")),
            "liver": g(lambda n: n == "liver"), "ribs": g(lambda n: n.startswith("rib_"))}


def lung_qib(h):
    return dict(mld=float(h.mean()), perc15=float(np.percentile(h, 15)), laa950=float((h < -950).mean() * 100)) if h.size else None


def hp_sd(h, m):
    return float((h - ndi.gaussian_filter(h, 2.0))[m].std()) if m.sum() > 500 else None


def run(series, ds, ts_dir, inf, out, model, B, coef, mr, ref, names):
    tm = {}; t00 = time.perf_counter()
    pd_img = nib.load(inf / "pred" / f"{series}__{model}__density.nii.gz")
    rho = np.asarray(pd_img.dataobj, np.float32)
    cls = np.asarray(nib.load(inf / "pred" / f"{series}__{model}__class.nii.gz").dataobj).astype(np.int64)
    ct06 = resample_to_output(nib.as_closest_canonical(nib.load(ds / "images" / f"{series}.nii.gz")), voxel_sizes=0.6, order=1, cval=-1024)
    seg = {k: to_train(resample_from_to(nib.load(ts_dir / series / f"{k}.nii.gz"), ct06, order=0)) for k in ("total", "body")}
    gt = np.asarray(to_train(ct06).dataobj, np.float32)
    assert gt.shape == rho.shape, (gt.shape, rho.shape)
    lab = np.asarray(seg["total"].dataobj).astype(np.int16); body = np.asarray(seg["body"].dataobj) > 0
    gt = np.where(body, np.clip(gt, -1024, 3071), -1000.0)
    Z, H, W = rho.shape
    mat = np.array(MC_OF_CLASS)[cls]
    (sy, dy), (sx, dx) = fit768(H), fit768(W)
    c = np.zeros((2, Z, N, N), np.float32)
    for k in range(2):
        lut = np.zeros(10, np.float32)
        for m in coef:
            lut[m] = coef[m][k]
        c[k][:, dy, dx] = (rho * lut[mat])[:, sy, sx]
    mu70 = np.zeros((Z, N, N), np.float32); lut70 = np.zeros(10, np.float32)
    for m in mr:
        lut70[m] = mr[m][69] / 10.0
    mu70[:, dy, dx] = (rho * lut70[mat])[:, sy, sx]
    vols = [np.pad(v[::-1], ((PAD_Z, PAD_Z), (PAD_XY, PAD_XY), (PAD_XY, PAD_XY))) for v in c]  # LEAP k order (drishya)
    del c
    ang, zz, mas = trajectory(ref, Z)
    t0 = time.perf_counter(); A1 = project(vols[0], ang, zz); A2 = project(vols[1], ang, zz); tm["forward_s"] = time.perf_counter() - t0
    del vols
    t0 = time.perf_counter()
    sp_ = Spectral(B, V3["al"], (float(A1.min()), float(A1.max())), (float(A2.min()), float(A2.max())))
    p = sp_(A1, A2, raw=True); del A1, A2
    ib, kk = intensities(p, V3["sb"], V3["ss"]); del p
    g0 = to_g(ib + V3["a"] * kk, sp_.poly); tm["physics_s"] = time.perf_counter() - t0
    t0 = time.perf_counter(); mu0 = R.fbp(g0, ang, zz, Z); tm["fbp_s"] = time.perf_counter() - t0; del g0
    t0 = time.perf_counter(); gn = noisy(ib, kk, V3["a"], V3["n0"], mas, np.random.default_rng(zlib.crc32(series.encode())), sp_.poly)
    del ib, kk
    mun = R.fbp(gn, ang, zz, Z); tm["noise_fbp_s"] = time.perf_counter() - t0; del gn

    def back(v768, fill=-1000.0):
        o = np.full((Z, H, W), fill, np.float32); o[:, sy, sx] = v768[:, dy, dx]; return o
    sims = {"ideal70": back(1000 * (mu70 / R.MU_WATER - 1)), "sim_noiseless": back(R.to_hu(mu0)), "sim_noisy": back(R.to_hu(mun))}
    del mu0, mun, mu70
    fov = np.zeros((H, W), bool); fov[sy, sx] = R.fov_mask()[dy, dx]
    roi = body & fov[None]; roi[:PAD_Z] = roi[Z - PAD_Z:] = False
    for k, v in sims.items():
        nib.save(nib.Nifti1Image(np.round(v).astype(np.int16), pd_img.affine), out / "vol" / f"{series}__{model}__{k}.nii.gz")
    G = ts_groups(names)
    res = dict(series=series, model=model, shape=[Z, H, W], n_views=len(ang), roi_voxels=int(roi.sum()), timing=tm)
    sm = lambda v: ndi.gaussian_filter(v, 1.0)  # noqa: E731  (light smoothing for bias maps, as drishya uses for bias)
    gts = sm(gt)
    for k, v in sims.items():
        r = res[k] = {}
        vs = sm(v)
        r["body_mae_hu"] = float(np.abs(v - gt)[roi].mean())
        r["body_mae_hu_smoothed"] = float(np.abs(vs - gts)[roi].mean())
        for reg, ids in G.items():
            m = np.isin(lab, ids) & roi
            if m.sum() > 500:
                r[f"{reg}_median_hu"] = float(np.median(v[m])); r[f"{reg}_bias_hu"] = float(np.median(vs[m] - gts[m]))
        r["lung_qib"] = lung_qib(v[np.isin(lab, G["lung"]) & roi])
        mus = ndi.binary_erosion(np.isin(lab, G["muscle"]) & roi, iterations=2)
        r["noise_sd_hp"] = hp_sd(v, mus)
    res["gt"] = {f"{reg}_median_hu": float(np.median(gt[np.isin(lab, ids) & roi])) for reg, ids in G.items() if (np.isin(lab, ids) & roi).sum() > 500}
    res["gt"]["lung_qib"] = lung_qib(gt[np.isin(lab, G["lung"]) & roi])
    res["gt"]["noise_sd_hp"] = hp_sd(gt, ndi.binary_erosion(np.isin(lab, G["muscle"]) & roi, iterations=2))
    figure(series, gt, sims, body, out / "fig" / f"{series}.png")
    res["timing"]["total_s"] = time.perf_counter() - t00
    print(series, json.dumps({k: (v if not isinstance(v, dict) else {a: (round(b, 1) if isinstance(b, float) else b) for a, b in v.items()})
                              for k, v in res.items() if k in ("gt", "sim_noisy", "timing")}), flush=True)
    return res


def figure(series, gt, sims, body, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z, x = gt.shape[0] // 2, gt.shape[2] // 2
    cols = [("real CT (truth)", gt), ("ideal 70 keV of prediction", sims["ideal70"]), ("sim noiseless", sims["sim_noiseless"]),
            ("sim noisy", sims["sim_noisy"]), ("sim noiseless - real", None)]
    fig, ax = plt.subplots(2, 5, figsize=(25, 10))
    for j, (t, v) in enumerate(cols):
        for i, (sl, name) in enumerate([(np.s_[z], "axial"), (np.s_[:, :, x], "sagittal")]):
            a = ax[i, j]
            if v is None:
                d = np.where(body, sims["sim_noiseless"] - gt, 0)[sl]
                im = a.imshow(ndi.gaussian_filter(d, 1.5), cmap="RdBu_r", vmin=-200, vmax=200); plt.colorbar(im, ax=a, fraction=0.04)
            else:
                a.imshow(v[sl], cmap="gray", vmin=-1000, vmax=400)
            a.set_title(f"{t} ({name})", fontsize=10); a.axis("off")
    fig.suptitle(series); fig.tight_layout(); fig.savefig(path, dpi=55); plt.close(fig)


if __name__ == "__main__":
    ds, ts_dir, inf, out = map(Path, sys.argv[1:5])
    rank, world = int(sys.argv[5]), int(sys.argv[6])
    model = sys.argv[7] if len(sys.argv) > 7 else "u5_s11"
    for d in ("vol", "fig", "metrics"):
        (out / d).mkdir(parents=True, exist_ok=True)
    B, coef, mr, info = materials()
    if rank == 0:
        json.dump(dict(info, coef={MC_FILES[m]: v.tolist() for m, v in coef.items()}, v3_params=V3), open(out / "materials.json", "w"), indent=1)
    print("materials", info, flush=True)
    assert info["max_rel_fit_err"] < 0.02, info
    ref = ref_geometry(out / f"geom_dcm_008_rank{rank}.npz")
    names = {int(k): v for k, v in json.load(open(ts_dir / "class_map_total.json")).items()}
    series = sorted(p.name[:-7] for p in (ds / "images").glob("*.nii.gz"))[rank::world]
    for s in series:
        f = out / "metrics" / f"{s}__{model}.json"
        if f.exists():
            continue
        json.dump(run(s, ds, ts_dir, inf, out, model, B, coef, mr, ref, names), open(f, "w"), indent=1)
    print("DONE", flush=True)
