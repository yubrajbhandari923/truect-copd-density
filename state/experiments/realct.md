# Real patient CT plausibility test (exploratory, outside plan-001)

Owner request 2026-11-16: run the trained models on real chest CT to see how they behave. No ground truth exists, so
this is a plausibility / domain-shift check, not an accuracy measurement. Not part of any confirmatory claim.

## Setup
- Data: DUHS VLM CT_CHEST (ld258, read-only) -> 54 thin original axial series, 53 patients, copied to
  `/scratch/railabs/yb107/Datasets/DUHS_CT_CHEST_thin` (sha256-verified, checks.json: no problems; README there).
  Siemens + GE, 0.63-0.94 mm in-plane, 0.6-0.625 mm slices, mostly contrast-enhanced (training is non-contrast).
- Preprocessing (infer.py): RAS -> 0.6 mm isotropic (ASSUMED TrueCT recon grid) -> TotalSegmentator 2.18 body mask,
  outside = -1000 HU (table removal) -> clip [-1024, 3071] -> training orientation (I,P,L) -> pad in-plane to >=768
  and a multiple of 32.
- Models: u5_s11, u5a_ref75k (3D), u4f_s11, u3_s11 (2D). u5a_wide (now best on VAL) not yet included.
- Code: branch experiment/realct-inference, snapshot bf924f7 (+ summarize.py 3d78274). Jobs 206032 (copy; TS failed:
  torchvision/torch CUDA mismatch, fixed with torchvision 0.29.1+cu126), 206049 (TS ok; rank 1 crashed on 2D size
  mismatch for >768 px slices, fixed by padding to multiple of 32), 206080 (rest). Outputs:
  `results/realct/{totalseg,inference}` in the workspace. Summary: `reports/realct/summary.json`, `per_case.csv`.

## Findings (54 cases, medians)
- Orientation convention verified on real anatomy: sternum anterior 53/53, descending aorta left 53/53 (one 165 mm
  FOV case had neither structure).
- 3D transfers, 2D does not, on contrast CT: aorta Dice vs TotalSeg 0.77-0.79 (3D) vs 0.07-0.11 (2D); 2D labels
  24-29% of the TS aorta as bone (iodinated blood ~200 HU looks like trabecular bone without 3D context); 3D 0.2-0.4%.
- Densities plausible: muscle 1.00-1.01, aorta 1.04-1.05, spine 0.98-0.99 g/cm3 (3D).
- Lung: density tracks HU (r 0.85) but sits ~0.075 g/cm3 above 1+HU/1000 (3D). Cause unknown: learned TrueCT mapping,
  partial-volume handling, or scanner/kernel shift. Not resolvable without ground truth.
- Emphysema label 2.7-3.4% of TS lung (3D) while LAA-950 median is ~0.4%: models call more emphysema than the
  threshold; direction of error unknown.
- Low agreement on sternum/scapula/esophagus (0.26-0.36): partly label-definition differences with TotalSeg; not
  investigated.

## Limits
No density ground truth; 0.6 mm grid and HU calibration across vendors assumed; contrast is out of distribution.
Overlays checked visually for one case only (RealCT-P05: anatomy and consolidation density plausible).

## u5a_wide (job 206110, code/realct-8558f51) — reports/realct/u5a_wide/
Orientation 53/53; aorta Dice vs TS 0.76, bone in TS aorta 0.5%; muscle 1.02, aorta 1.05, spine 0.98 g/cm3;
lung offset vs 1+HU/1000 0.049 (smaller than earlier 3D, 0.075); emphysema 5.7% of TS lung (more than earlier 3D).

## Simulated CT from u5a_wide predictions (job 206111, 54 cases) — reports/realct/simct_u5a_wide/
Drishya E01 best variants (v3_noiseless, v3; experiments/realct/simct, copied from drishya, not edited) applied to
mu = rho_pred x (mu/rho)_class (TrueCT 10313 per-subject LAC), 10313 helical geometry, constant reference mAs.
Median over cases (HU; real / ideal70 / sim noiseless / sim noisy):
muscle 37 / 10 / -23 / -28; liver 82 / 40 / 16 / 14; aorta 191 / 44 / 17 / 12; heart 150 / 40 / 12 / 9;
spine 275 / 148 / 98 / 99; lung -816 / -754 / -742 / -741; Perc15 -870 / -801 / -788 / -809; LAA-950 0.3 / 0.1 / 0.0 / 0.2 %;
body MAE vs real 81 / 90 / 99 HU; high-pass noise SD in muscle 26 / 15 / 11 / 50.
Reading: three separable gaps. (1) Model: predicted density gives soft tissue ~25-30 HU low and lung ~60 HU high
even at ideal 70 keV (consistent with the lung density offset above). (2) No iodine: aorta/heart contrast missing
(~150 HU), by construction. (3) Simulator: ~30 HU darker than ideal 70 keV in soft tissue, bone ~50 HU lower
(effective-energy / beam-hardening-correction difference; drishya's own E01 bias on TrueCT is +5..+28 HU, so the
offset here is partly the ideal70 reference, not verified). Noisy sim noise ~2x real: reference mAs and recon kernel
differ from the DUHS scans (patient mAs not used). Anatomy/geometry visually correct (example_fig.png, 2 cases viewed).

## Next
Look at more overlays (worst aorta / lung-offset cases); non-contrast screening data
(SHARED DLCS LungRads) would be a closer domain match.
