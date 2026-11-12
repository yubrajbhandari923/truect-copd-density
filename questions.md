# Questions for the project owner

Answer inline under each question whenever convenient. Nothing here blocks current work:
each open question lists the default I am proceeding with until you say otherwise.
Last updated: 2026-11-12.

## Open

### Q1. Stratify the train/validation/test split on emphysema burden?
Emphysema voxels are 0%–77% of the lung across subjects (median 5%). The plan specifies only
a seeded 46/10/11 split (seed 20261112, 10313 forced into training). With 11 test subjects a
plain shuffle could leave the test set with almost no severe cases.
**Default:** stratify on emphysema fraction, same seed and sizes, 10313 in training.
**Answer:**

### Q2. Drop the optional effective-atomic-number (Z_eff) analysis?
All 53 organ IDs map to 8 fixed compositions that are identical in every subject, so Z_eff is
a pure function of the label and carries no information beyond material classification.
**Default:** do not run it as a separate target; state the reason in the report.
**Answer:**

### Q3. Is validator-only acceptable for the "DukeSim export validity" primary endpoint?
With no DukeSim runs (see D1), `per_subject_dukesim_export_validity` can only mean: the
exported package passes format and consistency checks derived from the source files and
DukeSim's material-generation script. It cannot show that a simulation would succeed.
**Default:** report it as "format-valid", explicitly not "simulation-verified".
**Answer:**

### Q4. Which mono-energetic reference: 64 keV or 70 keV?
The local challenge documentation PDF says ground truth is 70 keV HU; the published challenge
paper says 64 keV. This affects only a sanity baseline (ideal HU computed from labels and
tables), not the density or material targets.
**Default:** use 70 keV (matches the local documentation and the prior 10313 study), note the
discrepancy.
**Answer:**

### Q5. Ablations with no supporting literature: keep all of them?
The literature review found nothing that justifies A05 (L1/Huber/gradient losses), A11
(training-set size), A18 (metadata conditioning) or A19 (seed stability), and judged several
cited references not to support the ablation they are attached to (details in
`docs/report/chapters/literature.qmd`).
**Default:** run them as planned and label the rationale as empirical, not literature-backed.
**Answer:**

### Q6. A16 (robustness to reconstruction variant) cannot run. Agreed to mark it not applicable?
There is exactly one reconstruction per subject, so the trigger for this conditional ablation
is not met. A12 (synthetic noise/blur perturbation) still covers robustness.
**Default:** mark A16 "trigger not met".
**Answer:**

### Q7. Corrections to wording in the plan and dataset README. Should I propose a plan revision?
- The COPD phantoms were built from 67 COPDGene patient CTs, not from the XCAT library
  (challenge paper). The plan and README call them XCAT.
- The plan's `status` is still `proposed`.
- `state/STATUS.md` points to `../plan.yaml`; the file is `plans/plan-001.yaml`.
- `settings.yaml` mentions `scripts/research_hpc.py`, which does not exist.
**Default:** fix the two pointer/tooling items myself; leave the plan text untouched and
record the corrections in the report.
**Answer:**

### Q8. Anyone I could ask about the reconstructions?
You do not know who produced `recons/dcm_recons_copd/` (see D2). If a lab contact might
(the DICOMs are MATLAB secondary captures dated 2022-02-08, 768x768), a name would let you
check later. Not needed for the work to proceed.
**Answer:**

## Decided

### D1. No DukeSim runs (2026-11-12)
Use only the already-constructed data. Do not run DukeSim or the lab batch tool at all, so
the plan's optional resimulation check is out of scope. The installed DukeSim source may
still be read to mirror file formats.

### D2. Recon provenance is unknown (2026-11-12)
Who reconstructed the images, the kernel and the voxel size are not known. Voxel size and
orientation rest on the per-subject registration to the phantom; the kernel stays unknown
and is listed as a limitation.
