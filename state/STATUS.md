# TrueCT COPD density research — current status

**Plan:** `plans/plan-001.yaml`. **Updated:** 2026-11-13.

## Current phase

**E02 — U-Net core runs training.** E00 done (split frozen), E01 done.

## Established (evidence in `reports/e00/`, details in `experiments/e00.md`)

- 67/67 subjects paired; recon DICOMs carry no geometry → geometry from registration to phantom.
- Alignment fitted for all 67 (same orientation everywhere, r 0.964–0.990); 8 overlays checked by eye, all aligned.
  No exclusions.
- CT-grid training targets built for all 67 (`$W/results/e00/targets/`, 162 GB): density, majority class,
  purity, point label, valid mask. CT vs ideal 70 keV r 0.946–0.985.
- Reusable NIfTI dataset written to `/scratch/railabs/yb107/Datasets/TrueCT/derived/copd_nifti/` (owner-
  authorized; README, subjects.tsv, labels.tsv, code). Copies of the docs in `reports/e00/copd_nifti/`.
- Bug fixed: organ_pt was 0 instead of 255 outside the phantom (7ace4e9; training masks unaffected).

## Next action

1. ~~Freeze split~~ done: `configs/splits_copd.json` (46/10/11, stratified, seed 20261112, 10313 in train, no exclusions). Test IDs are not to be looked at until E04.
2. ~~FOV check~~ done: recon is a disc; eval/training region limited to r < 375 px (2d6bcc4).
3. E01 done (B3 density MAE 0.053, B4 material Dice 0.345). E02 seed-11 U3: 0.023 / 0.934. E02 2D runs done (reports/e02/validation_compare.json): fov loss costs in-body accuracy (material Dice -0.024, 10/10) -> keep body loss; 2.5D context and 3D help most on thin structures; U5 3D final (job 205034): density MAE 0.0207, material Dice 0.958, 19-class 0.839, best so far (10/10 density vs 2D). Nothing running. Next: 3D ablations, single seed (owner).
   HPC style: one job reserving many cores/GPUs, parallel inside (owner preference).

## Experiments

| Study | Current state | Journal |
|---|---|---|
| E00 data qualification | Targets + NIfTI done; split frozen | [e00](experiments/e00.md) |
| E01 calibration | Done (validation) | [e01](experiments/e01.md) |
| E02 U-Net | Core runs training | [e02](experiments/e02.md) |
| E03 conditional diffusion | Optional; undecided | [e03](experiments/e03.md) |
| E04 held-out evaluation | Not started | [e04](experiments/e04.md) |

## Open owner questions

`questions.md` (Q1–Q9); defaults are being followed. 10313 is development-exposed: never in validation/test.
