"""Visual check of an exported subject using only the NIfTI files (their affines), no alignment code.

usage: check_nifti.py <subject_dir> <out.png>
Axial / coronal / sagittal CT through the volume centre, with native-label outlines (lung, bone, body) resampled
by nearest neighbour through the two affines. Rows displayed in radiological convention: patient right on image left.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

d, out = Path(sys.argv[1]), sys.argv[2]
ct, lab = nib.load(d / "ct.nii.gz"), nib.load(d / "labels.nii.gz")
C, L = np.asarray(ct.dataobj), np.asarray(lab.dataobj)
to_lab = np.linalg.inv(lab.affine) @ ct.affine
groups = {"body": [1, 4, 24], "lung": [6, 60, 120, *range(26, 39)], "bone": [3, 8, 9, *range(39, 60)]}
colors = {"body": "yellow", "lung": "cyan", "bone": "magenta"}
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
c = np.array(C.shape) // 2
for k, (name, axis) in enumerate([("axial", 2), ("coronal", 1), ("sagittal", 0)]):
    grid = [np.arange(n) for n in C.shape]
    grid[axis] = np.array([c[axis]])
    ijk = np.stack(np.meshgrid(*grid, indexing="ij"), -1).reshape(-1, 3)
    j = np.rint((to_lab[:3, :3] @ ijk.T).T + to_lab[:3, 3]).astype(int)
    ok = np.all((j >= 0) & (j < L.shape), 1)
    lv = np.full(len(j), 255, np.uint8)
    lv[ok] = L[tuple(j[ok].T)]
    shp = [len(g) for g in grid]
    img, lv = np.take(C, c[axis], axis=axis), lv.reshape(shp).squeeze(axis)
    # canonical RAS+: i->Right, j->Anterior, k->Superior. Flip both so image left = patient right
    # (sagittal: image left = anterior) and top = anterior (axial) / superior (coronal, sagittal).
    img, lv = img.T[::-1, ::-1], lv.T[::-1, ::-1]
    ax[k].imshow(img, cmap="gray", vmin=-1000, vmax=400)  # CT spacing ~0.6 mm isotropic
    for g, ids in groups.items():
        ax[k].contour(np.isin(lv, ids), levels=[0.5], colors=colors[g], linewidths=0.6)
    ax[k].set_title(f"{name} (from NIfTI affines only)")
fig.suptitle(f"{d.name}: image left = patient right (sagittal: anterior); top = anterior (axial) / superior")
plt.tight_layout()
plt.savefig(out, dpi=80)
print("wrote", out)
