# Owner decisions, 2026-11-12

- **No DukeSim runs.** Use only the already-constructed TrueCT data; do not run DukeSim or
  the lab batch tool. The plan's optional resimulation check is out of scope. Reading the
  installed DukeSim source to mirror file formats remains fine.
  Impact: `per_subject_dukesim_export_validity` is assessed by a format validator only.
- **Recon provenance unknown.** The owner does not know who produced the reconstructions,
  the kernel, or the voxel size. Geometry rests on per-subject registration to the phantom.

Open owner questions live in `questions.md` at the repository root.
