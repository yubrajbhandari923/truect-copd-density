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
3. E01 done (B3 density MAE 0.053, B4 material Dice 0.345). E02 seed-11 U3: 0.023 / 0.934. E02 2D runs done (reports/e02/validation_compare.json): fov loss costs in-body accuracy (material Dice -0.024, 10/10) -> keep body loss; 2.5D context and 3D help most on thin structures; U5 3D final (job 205034): density MAE 0.0207, material Dice 0.958, 19-class 0.839, best so far (10/10 density vs 2D). U5 ablations: capacity matters; Dice loss, w_cls 0.3 and 160^3 patches do not help the primary metrics; Batch 3 (2026-11-17): u5a_wide (filters x1.5) is the new best, 0.0201 / 0.961 / 0.845 at 75k (10/10 better than ref on density and material); lr 3e-4 and residual blocks do not help. A05/A11/A13-att cancelled by owner (material Dice > 0.95; focus on methods). E03 D1 pilot trained (80k) but sampling fails: MAE ~0.3 vs U5 0.02 (eps-prediction + linear schedule, high-t x0 not learned; diag 206091); next D1b v-prediction + zero-terminal-SNR. Real-CT plausibility test done (54 DUHS series, [realct](experiments/realct.md)): orientation correct 53/53; 3D models transfer to contrast CT (aorta Dice vs TotalSeg 0.78, soft-tissue densities ~1.0-1.05), 2D models do not (call 24-29% of aorta bone); lung density ~0.075 above 1+HU/1000, cause unknown. Next: D1b, A07 lookup-density eval, freeze E02 models (u5a_wide candidate).
   HPC style: one job reserving many cores/GPUs, parallel inside (owner preference).

## Experiments

| Study | Current state | Journal |
|---|---|---|
| E00 data qualification | Targets + NIfTI done; split frozen | [e00](experiments/e00.md) |
| E01 calibration | Done (validation) | [e01](experiments/e01.md) |
| E02 U-Net | Core runs training | [e02](experiments/e02.md) |
| E03 conditional diffusion | D1 pilot fails (MAE ~0.3); D1b planned | [e03](experiments/e03.md) |
| E04 held-out evaluation | Not started | [e04](experiments/e04.md) |

## Open owner questions

`questions.md` (Q1–Q9); defaults are being followed. 10313 is development-exposed: never in validation/test.
