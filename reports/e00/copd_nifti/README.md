# TrueCT COPD — NIfTI export (CT + phantom segmentation, co-registered)

Derived from the original files in this dataset (`phantoms/COPD`, `recons/dcm_recons_copd`, `pairing/`).
Nothing in the original tree was modified. Made 2026-11-12 in the `truect-copd-density` project
(git branch `experiment/e00-qualification`; exact code in `code/`, versions in `code/VERSION.txt`).
COPD set only (67 subjects); the liver and lung-lesion sets are **not** covered.

## Files

```
<id>/ct.nii.gz              int16    reconstruction values exactly as stored in the DICOMs (HU-like, min -1000)
<id>/labels.nii.gz          uint8    phantom organ IDs, native phantom grid (0.25 mm)
<id>/labels_ctgrid.nii.gz   uint8    organ ID at each CT voxel centre (nearest phantom voxel); 255 = outside phantom
<id>/density_ctgrid.nii.gz  float32  mass density g/cm3, exact volume average of the phantom over each CT voxel;
                                     0 = outside phantom (use labels_ctgrid == 255 to mask)
<id>/meta.json              transform, checks, source paths, SHA-256 of each NIfTI, Slurm job
subjects.tsv                one row per subject: shapes, spacing, all check results, notes
labels.tsv                  organ ID -> name, Monte-Carlo material ID, density range across subjects
code/                       scripts that produced everything here
```

All images are stored in canonical RAS+ orientation (nibabel `as_closest_canonical`). `ct.nii.gz` and
`labels.nii.gz` share one world frame, so they overlay directly in 3D Slicer / ITK-SNAP / nibabel with no
resampling step. The `*_ctgrid` files are on exactly the CT grid (same shape and affine as `ct.nii.gz`).

```python
import nibabel as nib, numpy as np
ct  = nib.load("10313/ct.nii.gz")            # (768, 768, n_slices) after canonicalisation, ~0.6 mm
lab = nib.load("10313/labels_ctgrid.nii.gz") # same grid; 255 = unknown
```

## How the geometry was obtained (read this before trusting the mm values)

The reconstruction DICOMs are MATLAB secondary captures with **no geometry tags** (no PixelSpacing,
SliceThickness, ImagePosition/Orientation, Rescale). Geometry therefore comes from registering each CT to
its phantom: 3 axis permutations x 8 flips scored, then translation + per-axis scale fitted by maximising the
correlation between the CT and the phantom rendered at 70 keV.

- **Measured:** orientation, translation, and the CT/phantom voxel-size **ratio** (2.4, per-axis scale
  0.998–1.004). Same axis mapping for all 67 subjects.
- **Assumed, not verified:** absolute voxel sizes 0.25 mm (phantom) and 0.6 mm (CT). If they are wrong, all
  mm values scale by one common factor; the CT<->label correspondence is unaffected.
- **Left/right inferred from anatomy:** no file states it. The descending aorta lies on the side this export
  calls patient-left in every subject (3.4–35.9 mm left of the spine, median 13.8 mm, all 67 subjects), as in normal anatomy.

## Verified (by computation on all 67, unless stated)

- Pairing: 67 subjects; phantom, recon, material-table and LAC folder ID sets identical; one recon per subject.
- Phantom `.bin`: uint8, C-order `[z, y, x]` = the three filename numbers reversed (confirmed by the
  registration working for all 67).
- Recon: 768 x 768 int16, 448/512/576/640 slices (= 64 x rotations), minimum exactly -1000 in every subject.
- Registration: correlation CT vs ideal 70 keV image 0.964–0.990 (median 0.984). 0.5476 mm slice spacing
  (suggested by the 64-row detector) fits worse than 0.6 mm in every subject.
- Overlays inspected **by eye** (all aligned): 10313, 11989, 14023, 14402, 16783, 18314, 18762, 24080
  (column `overlay_inspected_by_eye` in `subjects.tsv`). The other 59 rest on the numerical checks.
- NIfTI geometry self-check, every subject: CT voxel centre -> world -> nearest native phantom voxel gives the
  same label as `labels_ctgrid` for 99.95–100% (median 100%) of 20,000 sampled voxels.
- Material tables: the same 53 organ IDs with the same names and MC IDs in all 67 tables (a global ontology).
  47 IDs have voxels in every subject; IDs 26 and 106–110 never occur as voxels. Elemental compositions are
  identical across subjects; **density is subject-specific** (one scale factor per tissue group, about ±2% soft
  tissue/bone, 4–6% lung). Emphysema (ID 60) density varies 0.023–0.083 g/cm3.
- Emphysema burden: 0%–77% of lung voxels (median 5%).
- CT vs ideal 70 keV attenuation of the labels: correlation 0.946–0.985 (median 0.969), slope 4458–4953
  HU per cm^-1 (ideal water-referenced value ~5185), so the stored values are HU-like but not an exact 70 keV HU
  map. Mean CT of pure voxels: muscle -49 to +44 HU (median -1), emphysema -950 to -774, air -988 to -951.

## Known issues and unknowns

- **18762:** the body extends past the 768-pixel field of view; CT values near the cut edge may be biased.
  **11989:** body nearly touches the edge.
- Who reconstructed the images, with what kernel, is unknown.
- The challenge documentation (`reference/TrueCT-Documentation.pdf`) says the ground truth is 70 keV; the
  published challenge paper says 64 keV. 70 keV was used here.
- The dataset README calls the phantoms XCAT; the challenge paper says they were built from 67 COPDGene patient CTs.
- Challenge terms ask users to contact the organisers before using the data outside the challenge.
- `alignment_report_flags` in `subjects.tsv` come from an automatic heuristic (lung-size / aorta-side guesses);
  they do not indicate misregistration (all 6 low-correlation subjects were inspected by eye and are aligned).

## Re-running

Environment: Python 3.11, numpy, scipy, pydicom, openpyxl, nibabel, matplotlib. Order:
`align.py` (per subject) -> `build_targets.py` -> `export_nifti.py` -> `dataset_docs.py`;
`check_nifti.py <subject_dir> out.png` renders an overlay from the NIfTI files alone.
Paths at the top of `code/common.py` point at this dataset and at the project workspace.
