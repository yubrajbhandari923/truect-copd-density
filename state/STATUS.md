# TrueCT COPD density research — current status

**Status recorded:** 2026-11-12. This is a clean start; no live HPC inspection
or scientific result is claimed here.

## Current phase

**E00 — source inventory, orientation and physical target qualification.**

## Next action

1. Connect to DRCC and read `/scratch/railabs/yb107/Datasets/TrueCT/README.md`
   and the contents of `pairing/`, `recons/`, `phantoms/`, `reference/` read-only.
2. Check/create `/scratch/railabs/yb107/research-output/truect-copd-density/`
   and bootstrap the project-local `uv` virtual environment there.
3. Inspect reference scripts in `/scratch/railabs/yb107/tmp/truect_case_10313_slides/`
   read-only, then start a small independent geometry audit in an E00 branch.
4. Determine real source inventory and CT-to-phantom pairing without assuming
   a previously reported subject count or completed QC.

## Experiments

| Study | Current state | Journal |
|---|---|---|
| E00 data qualification | Not yet run in this workspace | [e00](experiments/e00.md) |
| E01 calibration | Not started | [e01](experiments/e01.md) |
| E02 U-Net | Not started | [e02](experiments/e02.md) |
| E03 conditional diffusion | Optional; undecided | [e03](experiments/e03.md) |
| E04 held-out evaluation | Not started | [e04](experiments/e04.md) |

## Verified facts vs questions

- **User-provided source tree:** `README.md`, `pairing/`, `phantoms/`,
  `projections/`, `recons/`, `reference/`.
- **Not yet independently verified:** which files map to COPD subjects, exact
  count of usable cases, recon intensity units, phantom geometry, mass-density
  tables, device metadata, Slurm limits, Python/GPU package compatibility.
- **Research caution:** 10313 is available as a development reference and must
  not be treated as a blinded validation/test subject.

## Pointers

- [Scientific plan](../plan.yaml)
- [HPC and paths](../settings.yaml)
- [Run records](../reports/runs/)
- [Decisions](decisions/)
