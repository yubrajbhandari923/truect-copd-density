# Showcase package: CT -> predicted phantom (u5a_wide), with 3D renders

Location: `$W/results/realct/showcase/` (W = /scratch/railabs/yb107/research-output/truect-copd-density).
Made by `prep.py` + `render.py` (this folder), job `showcase.sbatch`; model u5a_wide (E02, step 75000).

## Cases
| Folder | What it shows |
|---|---|
| TrueCT_10379 | TrueCT VAL, near-normal lung (0.4% emphysema): prediction matches phantom incl. a small RLL emphysema focus |
| TrueCT_24803 | TrueCT VAL, moderate (14%): emphysema pattern reproduced; best density slide (oblique view) |
| TrueCT_21400 | TrueCT VAL, severe (61%) |
| RealCT_RealCT-P45 | DUHS contrast CT, normal lung (LAA-950 0.2%); model calls some peri-mediastinal emphysema |
| RealCT_RealCT-P25 | DUHS contrast CT, most emphysematous (LAA-950 14%); model over-calls (33% of lung), contrast heart -> "bone" |
| RealCT_RealCT-P30 | DUHS contrast CT, typical case |

TrueCT cases are **validation** subjects (never trained on). Test subjects stay blinded until E04.

## Files per case
- `ct.nii.gz` input CT (TrueCT: dataset export, RAS; real CT: DUHS series as stored)
- `pred_u5a_wide_density.nii.gz` (g/cm3), `pred_u5a_wide_class.nii.gz` (19 classes, `truect.data.CLASSES`; real CT zeroed outside body, TrueCT unmasked)
- `totalseg_total.nii.gz`, `totalseg_body.nii.gz` (TotalSegmentator; class names in `../totalseg_class_map.json`)
- TrueCT only: `gt_density_ctgrid.nii.gz`, `gt_labels_ctgrid.nii.gz` (native organ IDs), `gt_class.nii.gz` (19 classes, 255 = unknown)
- Real CT only: `simct_{ideal70,sim_noiseless,sim_noisy}.nii.gz` (drishya-v3 simulated CT from the prediction), `simct_slices.png`
All volumes carry affines in one world frame: they overlay directly in 3D Slicer / ITK-SNAP.

## Renders (`renders/`)
- `figure_{anterior,left,oblique}_{dark,light}.png`: composed comparison (surfaces top, density volume renders bottom, legend, colour bar)
- `panels/<source>_<view>.png`: transparent, cropped to a common box per view (same scale across sources) for slide layouts
- `nifti/`: the exact volumes the renders are made from (canonical RAS, overlay on `ct.nii.gz`):
  `{pred_class,gt_class,totalseg}_groups.nii.gz` = surface label maps (1 Lung, 2 Bone, 3 Aorta, 4 Pulmonary vessels,
  5 Esophagus, 6 Emphysema; prediction masked to the TotalSeg body), `{ct,pred_density,gt_density}_density.nii.gz` =
  volume-rendered densities in g/cm3 (body-masked; CT as 1+HU/1000), `legend.json` = colours, colour map and opacity
  transfer function, so 3D Slicer / ITK-SNAP can reproduce the same look.
Surfaces = marching cubes per structure group (approach and colours from yb107 MedImgResearch visualizer.py), Taubin-smoothed.
Densities = GPU volume rendering; transfer function: emphysema (<0.1 g/cm3) orange, lung parenchyma blue, soft tissue transparent, bone white.
CT panel = 1+HU/1000 through the same function (display approximation; bone threshold raised so contrast blood is not bone).

## Caveats to keep on the slides
- Horizontal bands in predicted density = slab seams of the full-volume inference (96-slice slabs); a pipeline artefact, under investigation.
- Real CT is contrast-enhanced (out of training distribution); TotalSeg has no emphysema class and only pulmonary veins.
- Thin "emphysema" bits below the lungs in predictions = bowel/stomach gas labelled as emphysema.
