"""Collect every pipeline volume for a few showcase cases into one folder per case (for slides / 3D renders).

Usage: python prep.py <out_dir> truect <sid>...      (TrueCT validation subjects; runs u5a_wide here, GPU)
       python prep.py <out_dir> realct <series>...   (DUHS real CT; copies existing inference/TotalSeg/simct outputs)
TotalSegmentator for TrueCT cases is run afterwards by showcase.sbatch (separate env).
Predictions and GT class maps are written on the training grid (IPL); their affines put them in the same world frame
as ct.nii.gz, so any viewer / nibabel overlays them directly.
"""
import json, shutil, sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infer import load_models, to_train, t3d, t2d  # noqa: E402
from truect.data import load, split, WORK  # noqa: E402

MODEL = "u5a_wide"
TRUECT = Path("/scratch/railabs/yb107/Datasets/TrueCT/derived/copd_nifti")
DUHS = Path("/scratch/railabs/yb107/Datasets/DUHS_CT_CHEST_thin")
R = WORK / "results/realct"


def truect_case(sid, out, model):
    assert sid in split()["val"], f"{sid}: showcase uses validation subjects only (test is blinded until E04)"
    d = out / f"TrueCT_{sid}"; d.mkdir(parents=True, exist_ok=True)
    for f in ("ct.nii.gz", "density_ctgrid.nii.gz", "labels_ctgrid.nii.gz"):
        shutil.copy(TRUECT / sid / f, d / ("gt_" + f if f != "ct.nii.gz" else f))
    T = to_train(nib.load(TRUECT / sid / "ct.nii.gz"))
    t, _ = load(sid, ("ct", "cls"))
    assert np.array_equal(np.asarray(T.dataobj[::50]), t["ct"][::50]), sid  # NIfTI <-> training array mapping
    nib.save(nib.Nifti1Image(np.asarray(t["cls"]), T.affine), d / "gt_class.nii.gz")
    m, cfg, _, step = model
    pd, pc = t3d.predict_volume(m, np.asarray(t["ct"], np.float32), cfg, json.load(open(t2d.NORM_PATH)), (0, t["ct"].shape[0] - 1))
    nib.save(nib.Nifti1Image(pd.astype(np.float32), T.affine), d / f"pred_{MODEL}_density.nii.gz")
    nib.save(nib.Nifti1Image(pc, T.affine), d / f"pred_{MODEL}_class.nii.gz")
    print(sid, "done step", step, flush=True)


def realct_case(s, out):
    d = out / f"RealCT_{s.split('_Study')[0]}"; d.mkdir(parents=True, exist_ok=True)
    cp = {DUHS / "images" / f"{s}.nii.gz": "ct.nii.gz",
          R / "totalseg" / s / "total.nii.gz": "totalseg_total.nii.gz", R / "totalseg" / s / "body.nii.gz": "totalseg_body.nii.gz",
          R / f"inference_{MODEL}/pred/{s}__{MODEL}__density.nii.gz": f"pred_{MODEL}_density.nii.gz",
          R / f"inference_{MODEL}/pred/{s}__{MODEL}__class.nii.gz": f"pred_{MODEL}_class.nii.gz"}
    for k in ("ideal70", "sim_noiseless", "sim_noisy"):
        cp[R / f"simct/{MODEL}/vol/{s}__{MODEL}__{k}.nii.gz"] = f"simct_{k}.nii.gz"
    cp[R / f"simct/{MODEL}/fig/{s}__{MODEL}.png"] = "simct_slices.png"
    for a, b in cp.items():
        shutil.copy(a, d / b)
    (d / "series.txt").write_text(s + "\n")


if __name__ == "__main__":
    out, kind = Path(sys.argv[1]), sys.argv[2]
    if kind == "truect":
        import torch
        model = load_models(torch.device("cuda"), [MODEL])[MODEL]
        for sid in sys.argv[3:]:
            truect_case(sid, out, model)
    else:
        for s in sys.argv[3:]:
            realct_case(s, out)
    shutil.copy(R / "totalseg/class_map_total.json", out / "totalseg_class_map.json")
