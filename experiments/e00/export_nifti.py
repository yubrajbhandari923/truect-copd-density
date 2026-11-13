"""Export one TrueCT COPD subject to NIfTI, CT and phantom in one shared RAS world frame (mm).

usage: export_nifti.py <subject_id|array_index> [dest_root]   (in a Slurm array the task id is the index)

Writes <dest>/<id>/:
  ct.nii.gz              int16   reconstruction as stored (HU-like, min -1000), CT grid
  labels.nii.gz          uint8   phantom organ IDs, native 0.25 mm grid
  labels_ctgrid.nii.gz   uint8   organ ID point-sampled at each CT voxel centre; 255 = outside phantom
  density_ctgrid.nii.gz  float32 mass density g/cm3 box-averaged over each CT voxel; 0 outside phantom
  meta.json              transform, checks, provenance
Needs the alignment JSON and the build_targets.py output for the subject. Resumable (skips if meta.json exists).

World frame = phantom frame. Phantom array [z,y,x] increases S, L, P (cohort orientation; the descending-aorta
check below is recorded per subject). Phantom voxel j has centre 0.25*(j+0.5) mm; CT voxel i on recon axis m has
centre A_m + sign_m*s_m*0.6*(i+0.5) mm on phantom axis perm[m] (alignment fit "full").
Absolute voxel sizes 0.25 / 0.6 mm are ASSUMED (only their ratio is measured); orientation is in RAS.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

from common import OUT, PH, SRC, pairing, phantom_file

VR, VP = 0.6, 0.25
P = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)  # phantom (z:S, y:L, x:P) mm -> RAS mm


def affines(al):
    perm, flips = al["orientation"]["perm"], al["orientation"]["flips"]
    A, s = al["fine"]["full"]["A_mm"], al["fine"]["full"]["scale"]
    ph = np.eye(4)
    ph[:3, :3], ph[:3, 3] = P * VP, P @ np.full(3, VP / 2)
    G, g = np.zeros((3, 3)), np.zeros(3)
    for m in range(3):
        d = (-1 if flips[m] else 1) * s[m] * VR
        G[perm[m], m], g[perm[m]] = d, A[m] + d / 2
    ct = np.eye(4)
    ct[:3, :3], ct[:3, 3] = P @ G, P @ g
    return ct, ph


def desc_aorta_offset_mm(vol):
    """Median L-ward offset (mm, phantom +y) of descending aorta from the spine, 1 mm sampling.
    Descending aorta = aorta voxels within 40 mm anterior of the spine centroid. Expected > 0 (aorta left of spine)."""
    v = np.asarray(vol[::4, ::4, ::4])
    offs = []
    for sl in v:
        sp = np.argwhere(np.isin(sl, (9, 39, 40, 41, 42, 43)))
        ao = np.argwhere(sl == 25)
        if len(sp) < 20 or len(ao) < 5:
            continue
        cy, cx = sp.mean(0)
        d = ao[ao[:, 1] > cx - 40][:, 0]  # x increases posterior; 40 voxels = 40 mm at 1 mm sampling
        if len(d) >= 5:
            offs.append(np.median(d) - cy)
    return (float(np.median(offs)), len(offs)) if offs else (None, 0)


def save(arr, aff, path, desc):
    img = nib.as_closest_canonical(nib.Nifti1Image(arr, aff))
    img.header.set_xyzt_units("mm")
    img.header["descrip"] = desc[:79].encode()
    img.set_qform(img.affine, 1)
    img.set_sform(img.affine, 1)
    tmp = path.with_name(path.name.replace(".nii.gz", ".tmp.nii.gz"))
    nib.save(img, tmp)
    os.rename(tmp, path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return img, h.hexdigest()


def main(sid, dest):
    dst = dest / sid
    if (dst / "meta.json").exists():
        return print("exists", sid)
    dst.mkdir(parents=True, exist_ok=True)
    al = json.load(open(OUT / f"alignment/{sid}.json"))
    tg = OUT / "targets" / sid
    tmeta = json.load(open(tg / "meta.json"))
    assert tmeta["fit_key"] == "full"
    pf, pshape = phantom_file(sid)
    vol = np.fromfile(pf, np.uint8).reshape(pshape)
    ct_aff, ph_aff = affines(al)

    ct = np.load(tg / "ct.npy", mmap_mode="r")
    lab_ct = np.load(tg / "organ_pt.npy", mmap_mode="r")
    dens = np.load(tg / "density.npy", mmap_mode="r")
    sums = {}
    img_l, sums["labels.nii.gz"] = save(vol, ph_aff, dst / "labels.nii.gz", f"TrueCT COPD {sid} phantom organ ID 0.25mm(assumed)")
    img_c, sums["ct.nii.gz"] = save(np.asarray(ct), ct_aff, dst / "ct.nii.gz", f"TrueCT COPD {sid} recon as stored 0.6mm(assumed)")
    img_lc, sums["labels_ctgrid.nii.gz"] = save(np.asarray(lab_ct), ct_aff, dst / "labels_ctgrid.nii.gz", f"TrueCT COPD {sid} organ ID at CT voxel centre, 255 outside")
    _, sums["density_ctgrid.nii.gz"] = save(np.asarray(dens, np.float32), ct_aff, dst / "density_ctgrid.nii.gz", f"TrueCT COPD {sid} density g/cm3 CT-voxel mean")

    # self-check through the SAVED (canonical) affines: CT voxel centre -> world -> nearest native phantom voxel
    L, LC = np.asarray(img_l.dataobj), np.asarray(img_lc.dataobj)
    rng = np.random.default_rng(0)
    idx = np.argwhere(LC != 255)
    idx = idx[rng.choice(len(idx), min(20000, len(idx)), replace=False)]
    w = img_lc.affine @ np.c_[idx, np.ones(len(idx))].T
    j = np.rint((np.linalg.inv(img_l.affine) @ w)[:3]).astype(int).T
    ok = np.all((j >= 0) & (j < L.shape), 1)
    agree = float((L[tuple(j[ok].T)] == LC[tuple(idx[ok].T)]).mean())
    assert agree > 0.999, (sid, "affine self-check", agree)
    off, n_sl = desc_aorta_offset_mm(vol)

    meta = dict(
        id=sid, files=sums, source=dict(phantom=str(pf), recon_dir=str(SRC / f"recons/dcm_recons_copd/dcm_recon_{sid}"),
                                        material_table=str(PH / f"material_tables/challenge_materials_copd_smeared_{sid}.xlsx")),
        ct_shape_canonical=list(img_c.shape), labels_shape_canonical=list(img_l.shape),
        ct_affine_ras=img_c.affine.round(6).tolist(), labels_affine_ras=img_l.affine.round(6).tolist(),
        ct_spacing_mm=[float(x) for x in img_c.header.get_zooms()],
        assumed_voxel_mm=dict(ct=VR, phantom=VP), alignment=dict(fit="full", A_mm=al["fine"]["full"]["A_mm"],
            scale=al["fine"]["full"]["scale"], r=al["fine"]["full"]["r"], perm=al["orientation"]["perm"], flips=al["orientation"]["flips"]),
        self_check_ctgrid_vs_native_label_agreement=agree, self_check_n=int(ok.sum()),
        descending_aorta_left_of_spine_mm=off, aorta_check_slices=n_sl,
        ct_valid_fraction=tmeta["valid_fraction"], ct_vs_mu70=tmeta["ct_vs_mu70"],
        code_snapshot=os.environ.get("SLURM_SUBMIT_DIR"), slurm_job=os.environ.get("SLURM_JOB_ID"))
    json.dump(meta, open(dst / "meta.tmp", "w"), indent=1)
    os.rename(dst / "meta.tmp", dst / "meta.json")
    print("done", sid, "agree", agree, "aorta_L_mm", off, img_c.header.get_zooms())


if __name__ == "__main__":
    a = os.environ.get("SLURM_ARRAY_TASK_ID") or sys.argv[1]
    sid = pairing()[int(a)][0] if len(a) < 4 else a
    main(sid, Path(sys.argv[2]) if len(sys.argv) > 2 else SRC / "derived/copd_nifti")
