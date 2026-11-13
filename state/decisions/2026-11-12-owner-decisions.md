# Owner decisions, 2026-11-12

- **No DukeSim runs.** Use only the already-constructed TrueCT data; do not run DukeSim or
  the lab batch tool. The plan's optional resimulation check is out of scope. Reading the
  installed DukeSim source to mirror file formats remains fine.
  Impact: `per_subject_dukesim_export_validity` is assessed by a format validator only.
- **Recon provenance unknown.** The owner does not know who produced the reconstructions,
  the kernel, or the voxel size. Geometry rests on per-subject registration to the phantom.

Open owner questions live in `questions.md` at the repository root.

## Reusable NIfTI dataset in the source tree (2026-11-12)
Owner asked for a proper NIfTI dataset of CT + phantom segmentations plus findings and scripts saved in the
original dataset directory, mentioning only cases confirmed by analysis or by eye. Written under
`/scratch/railabs/yb107/Datasets/TrueCT/derived/copd_nifti/` (code: `experiments/e00/export_nifti.py`,
`check_nifti.py`, `dataset_docs.py` on `experiment/e00-qualification`). Source files are not modified.

## Primary material metric = 9 DukeSim Monte-Carlo materials (lead, 2026-11-13)
`truect.metrics` now reports `material_macro_dice` over the 9 MC material IDs (primary) besides the 19 anatomical
name classes (secondary). Reason: the plan's endpoint is canonical / DukeSim-relevant material; several name
classes (e.g. rib cortical vs trabecular) share one material. Noticed while reviewing E01 B4 (macro Dice 0.15 on
names); decided before any U-Net result and applied to all methods. Commit on main after 012d657.
