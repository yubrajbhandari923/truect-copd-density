"""Test inference of the frozen-so-far E02 models on real (DUHS) chest CT; plausibility metrics, no ground truth.

Usage: python infer.py <shared_dir> <ts_dir> <out_dir> <rank> <world> [model,model,...]   (one GPU per process; cases split rank::world)

Per series: RAS -> resample to 0.6 mm isotropic (the TrueCT recon grid, itself an assumption) -> set voxels outside
the TotalSegmentator body mask to -1000 HU (no patient table in training) -> clip to [-1024, 3071] -> reorient to the
training array convention (z head->feet, y anterior->posterior, x patient right->left; established from phantom
labels: scapula superior, sternum anterior, descending aorta on the patient's left) -> pad in-plane to >= 768 with
-1000 -> each model's own full-volume inference. Orientation is checked per case with TotalSegmentator labels.
Outputs per series and model: density (g/cm3) and class NIfTI on the 0.6 mm grid; metrics JSON; overlay PNG.
"""
import json, sys, time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_from_to, resample_to_output

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e02"))
import train as t2d  # noqa: E402
import train3d as t3d  # noqa: E402
from truect.data import CLASSES, WORK  # noqa: E402

MODELS = ["u5_s11", "u5a_ref75k", "u4f_s11", "u3_s11"]
TRAIN_AX = ("I", "P", "L")  # training array (z, y, x)
BONE = [CLASSES.index(c) for c in ["ribs_cortical", "ribs_trabecular", "scapulas", "scapulas_trabecular",
                                   "spine_cortical", "spine_trabecular", "sternum", "sternum_trabecular"]]
LUNG = [CLASSES.index(c) for c in ["emphysema", "fissure", "lung", "lung_parenchyma"]]
OURS = {"aorta": [CLASSES.index("aorta")], "esophagus": [CLASSES.index("esophagus")],
        "spine": [CLASSES.index("spine_cortical"), CLASSES.index("spine_trabecular")],
        "ribs": [CLASSES.index("ribs_cortical"), CLASSES.index("ribs_trabecular")],
        "sternum": [CLASSES.index("sternum"), CLASSES.index("sternum_trabecular")],
        "scapula": [CLASSES.index("scapulas"), CLASSES.index("scapulas_trabecular")], "lung": LUNG}


def ts_groups(names):
    g = lambda f: [i for i, n in names.items() if f(n)]
    return {"aorta": g(lambda n: n == "aorta"), "esophagus": g(lambda n: n == "esophagus"),
            "spine": g(lambda n: n.startswith("vertebrae_")), "ribs": g(lambda n: n.startswith("rib_")),
            "sternum": g(lambda n: n == "sternum"), "scapula": g(lambda n: n.startswith("scapula_")),
            "lung": g(lambda n: n.startswith("lung_")), "heart": g(lambda n: n == "heart"),
            "muscle": g(lambda n: n.startswith("autochthon_"))}


def to_train(img):
    return img.as_reoriented(nib.orientations.ornt_transform(nib.io_orientation(img.affine),
                                                             nib.orientations.axcodes2ornt(TRAIN_AX)))


def load_models(dev, names=MODELS):
    out = {}
    for name in names:
        s = torch.load(WORK / "checkpoints/e02" / name / "last.pt", map_location="cpu", weights_only=False)
        cfg = s["cfg"]
        is3d = "filters" in cfg
        m = (t3d.build(cfg) if is3d else t2d.build(cfg)).to(dev).eval()
        m.load_state_dict(s["model"])
        out[name] = (m, cfg, is3d, s["step"])
    return out


def run_case(series, shared, ts_dir, out, models, norm, dev):
    img = nib.load(shared / "images" / f"{series}.nii.gz")
    ct06 = resample_to_output(nib.as_closest_canonical(img), voxel_sizes=0.6, order=1, cval=-1024)
    seg = {k: resample_from_to(nib.load(ts_dir / series / f"{k}.nii.gz"), ct06, order=0) for k in ("total", "body")}
    ctT, segT = to_train(ct06), {k: to_train(v) for k, v in seg.items()}
    ct = np.asarray(ctT.dataobj, np.float32)
    lab = np.asarray(segT["total"].dataobj).astype(np.int16)
    body = np.asarray(segT["body"].dataobj) > 0
    names = json.load(open(ts_dir / "class_map_total.json"))
    names = {int(k): v for k, v in names.items()}
    G = ts_groups(names)
    m_of = lambda k: np.isin(lab, G[k])
    # orientation check on this case: sternum anterior of spine (y), descending aorta on the patient's left (x)
    sp, st, ao = m_of("spine"), m_of("sternum"), m_of("aorta")
    chk = {}
    if sp.any() and st.any():
        chk["sternum_anterior"] = bool(np.nonzero(st)[1].mean() < np.nonzero(sp)[1].mean())
    if sp.any() and ao.any():
        zz = np.nonzero(ao)[0]; zm = (zz > np.percentile(zz, 30)) & (zz < np.percentile(zz, 70))
        a = np.array(np.nonzero(ao))[:, zm]; spy = np.nonzero(sp)[1].mean()
        a = a[:, a[1] > spy - 100]
        chk["desc_aorta_left"] = bool(a.shape[1] and a[2].mean() > np.nonzero(sp)[2].mean())
    ct = np.where(body, np.clip(ct, -1024, 3071), -1000.0).astype(np.float32)
    Z, H, W = ct.shape
    up = lambda n: max(768, -(-n // 32) * 32) - n  # 2D U-Nets need in-plane size divisible by 2^depth
    ph, pw = up(H), up(W)
    ctp = np.pad(ct, ((0, 0), (ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)), constant_values=-1000)
    crop = (slice(None), slice(ph // 2, ph // 2 + H), slice(pw // 2, pw // 2 + W))
    res = dict(series=series, shape_06mm=[Z, H, W], orientation_check=chk,
               body_fraction=float(body.mean()), hu_lung_median=float(np.median(ct[m_of("lung")])) if m_of("lung").any() else None)
    lung = m_of("lung")
    laa950 = float((ct[lung] < -950).mean()) if lung.any() else None
    res["laa950"] = laa950
    for name, (m, cfg, is3d, step) in models.items():
        torch.cuda.synchronize(); t0 = time.time()
        if is3d:
            pd, pc = t3d.predict_volume(m, ctp, cfg, norm, (0, Z - 1))
        else:
            pd, pc = t2d.predict_volume(m, ctp, cfg, norm, (0, Z - 1))
        torch.cuda.synchronize()
        r = res[name] = dict(step=step, sec=time.time() - t0)
        pd = pd[crop].astype(np.float32) if pd is not None else None
        pc = pc[crop] if pc is not None else None
        if pd is not None:
            pd[~body] = 0.0
            nib.save(nib.Nifti1Image(pd, ctT.affine), out / "pred" / f"{series}__{name}__density.nii.gz")
            for k in ("lung", "muscle", "spine", "aorta", "heart"):
                mk = m_of(k)
                if mk.any():
                    r[f"density_median_{k}"] = float(np.median(pd[mk]))
                    r[f"hu_median_{k}"] = float(np.median(ct[mk]))
            hl = ct[lung][::7]; dl = pd[lung][::7]
            r["lung_density_vs_hu_linear_r"] = float(np.corrcoef(hl, dl)[0, 1]) if lung.any() else None
            r["lung_density_minus_1+HU/1000_median"] = float(np.median(dl - (1 + hl / 1000))) if lung.any() else None
        if pc is not None:
            pc[~body] = 0
            nib.save(nib.Nifti1Image(pc, ctT.affine), out / "pred" / f"{series}__{name}__class.nii.gz")
            r["dice_vs_totalseg"] = {}
            for k, ids in OURS.items():
                a, b = np.isin(pc, ids), m_of(k)
                if b.any():
                    r["dice_vs_totalseg"][k] = float(2 * (a & b).sum() / (a.sum() + b.sum()))
            if lung.any():
                r["emphysema_fraction_of_ts_lung"] = float((pc[lung] == CLASSES.index("emphysema")).mean())
            for k in ("aorta", "heart"):
                mk = m_of(k)
                if mk.any():
                    r[f"bone_fraction_in_ts_{k}"] = float(np.isin(pc[mk], BONE).mean())
        if name == next(iter(models)):
            overlay(series, ct, pd, pc, lab, out / "overlays" / f"{series}.png")
        print(series, name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    return res


def overlay(series, ct, pd, pc, lab, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = ct.shape[0] // 2; x = ct.shape[2] // 2
    ims = [("CT", ct[z], dict(cmap="gray", vmin=-1000, vmax=400)), ("pred density", pd[z], dict(cmap="magma", vmin=0, vmax=1.5)),
           ("pred class", pc[z], dict(cmap="tab20", vmin=0, vmax=19, interpolation="nearest")),
           ("TotalSeg", lab[z] % 20, dict(cmap="tab20", vmin=0, vmax=19, interpolation="nearest")),
           ("CT sagittal", ct[:, :, x], dict(cmap="gray", vmin=-1000, vmax=400)),
           ("density sagittal", pd[:, :, x], dict(cmap="magma", vmin=0, vmax=1.5))]
    fig, ax = plt.subplots(1, 6, figsize=(30, 5.5))
    for a, (t, im, kw) in zip(ax, ims):
        a.imshow(im, **kw); a.set_title(f"{series[-28:]} {t}", fontsize=9); a.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=60); plt.close(fig)


if __name__ == "__main__":
    shared, ts_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    rank, world = int(sys.argv[4]), int(sys.argv[5])
    (out / "pred").mkdir(parents=True, exist_ok=True); (out / "overlays").mkdir(exist_ok=True)
    dev = torch.device("cuda")
    norm = json.load(open(t2d.NORM_PATH))
    models = load_models(dev, sys.argv[6].split(",") if len(sys.argv) > 6 else MODELS)
    series = sorted(p.name[:-7] for p in (shared / "images").glob("*.nii.gz"))[rank::world]
    for s in series:
        f = out / "metrics" / f"{s}.json"
        if f.exists():
            continue
        f.parent.mkdir(exist_ok=True)
        r = run_case(s, shared, ts_dir, out, models, norm, dev)
        json.dump(r, open(f, "w"), indent=1)
