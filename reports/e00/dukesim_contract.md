# E00: DukeSim input contract for TrueCT COPD phantoms

Plan: `plans/plan-001.yaml`, experiment `e00_pairs_and_dukesim_targets`
(`scientific_principles.duke_sim_contract`, `source_data.inspect_not_assume`).
Date: 2026-11-12. Read-only investigation on `plp-drcc-login2`.

Each claim is tagged **OBSERVED** (seen in a file or computed from one; path given), **INFERRED**
(follows from observations but was not executed or stated anywhere) or **UNKNOWN**.

Analysis scripts (small reads only) and their stdout are reproducible from
`/scratch/railabs/yb107/research-output/truect-copd-density/tmp/dukesim_contract/`
(`xsub.py` and `xsub2.py` cover all 67 subjects; `lac.py` handles the LAC layout; `lab.py` reads 3 axial slices from each of 3 phantoms).
Abbreviations: `DS` = `/home/dmf46/DukeSim_v1.2.21`, `TC` = `/scratch/railabs/yb107/Datasets/TrueCT`,
`GB` = `/scratch/railabs/g-dukesimbatch`.

---

## Headline

1. **A real DukeSim install exists and is usable by the lab:** DukeSim v1.2.21 at `DS`, wrapped by the
   lab's Slurm batch tool `GB` (GEMSTONe). Its material-generation source (`DS/scripts/generate_materials/full_materials.py`)
   shows that the **xlsx material table is the master input**. DukeSim *derives* the LAC `.bin`, the
   conversion table `.in` and the `.mcgpu` files from it. The TrueCT `material_lac/<id>/` files match
   that generator's output format exactly.
2. **Minimal DukeSim phantom = uint8 label `.bin` + a phantom-parameter block (nx, ny, nz, sx, sy, sz in mm,
   NumberOfMaterials) + one material spreadsheet** (Name, one column per atomic number, Density, Organ ID,
   optional MC ID). The other route is to give the pre-built LAC `.bin` + `.in` + `.mcgpu` directory.
3. **Across the 67 COPD subjects the label set, names, MC IDs and elemental compositions are identical. Only
   densities vary.** They vary as **16 independent per-subject scale factors**, each applied to a whole group of IDs.
   Mass-attenuation rows are identical across subjects (max relative difference 2.3e-7), so the LAC file adds
   no information beyond the density column.
4. A canonical (median) catalog loses ≤ ~2% density for most groups, about 4-6% for the lung groups, and
   **up to 58% for emphysema (ID 60)**.

---

## Q1. File types: content, units, layout

### 1a. Phantom label `.bin` (`TC/phantoms/COPD/phantom_<id>_<a>_<b>_<c>_8bit_textured_copd_eroded_flipped.bin`)
- OBSERVED: 67 files. For all 67, file size = a·b·c bytes exactly (`xsub.py`: "size mismatches []").
- OBSERVED: values are uint8 organ IDs. In 3 axial slices from each of 3 phantoms the observed labels were
  {0,1,3,4,6,8,9,13,24,25,27–60,68,70,120}, all of them Organ IDs in the subject's table. The volume edges are all 0 (air).
  Prior full-volume count for 10313: 47 of 53 table IDs occur, and 26 and 106–110 are absent
  (`/scratch/railabs/yb107/tmp/truect_case_10313_slides/README.md`).
- OBSERVED (prior work; neighbour-agreement test in `/home/yb107/ctsynth_truect/README.md`): C-order array
  `[z,y,x] = (c,b,a)`, so x (=a) varies fastest. INFERRED: this matches DukeSim's `XCAT_nx=a, XCAT_ny=b, XCAT_nz=c`,
  with x fastest (DukeSim's downsampler and MC-GPU take nx, ny, nz in that order, `DS/scripts/dukesim.py:68`).
- OBSERVED: the file has no header and no geometry. The voxel size is *not* in the COPD filenames. The reference phantom name
  `challenge_reference_860_860_320_0.25_mm_10mat.bin` and the TrueCT PDF imply 0.25 mm.
  **UNKNOWN** for COPD (0.25 mm is assumed; the 10313 registration only pins the CT/phantom scale *ratio* at 2.40).

### 1b. Material spreadsheet `material_tables/challenge_materials_copd_smeared_<id>.xlsx`
- OBSERVED: one sheet with 53 data rows and 18 columns. The header is identical in all 67 files:
  `Name | 1 6 7 8 11 12 15 16 17 19 20 26 53 | Density | No. of elements with non-zero mass fractions | Organ ID | MC ID`.
  The integer headers are atomic numbers (H C N O Na Mg P S Cl K Ca Fe I). Their values are **mass fractions**
  (the TrueCT PDF says "percent mass", but the values are fractions, e.g. muscle O = 0.71). Density is in g/cm³.
- OBSERVED: the "No. of elements" column equals the count of nonzero fractions in every row of all 67 files.
- OBSERVED quirks:
  - Fraction sums ≠ 1 for lung, lung_parenchyma and emphysema (0.968) and for skin (0.9622). body sums to 0.999.
  - Air puts 0.012817 in the **Z=17 (Cl)** column. Real air has ~1.28% argon (Z=18), so this is presumably Ar mislabelled as Cl.
    The DukeSim example `GB/usr/example/Water_mat.xlsx` has the same row.
  - DukeSim's xcatattentab divides by the fraction sum (`DS/scripts/generate_materials/source_xcatattentab/xcatattentab.c:192`:
    `dMu=dDensity*dSumCS/dSumMass`), so the non-unit sums are effectively renormalised in the LAC.
    `full_materials.py:check_composition_sum` only prints a warning when a sum is outside [0.999, 1.001].

### 1c. Conversion table `material_lac/<id>/conversion_table_120kV_121materials.in`
- OBSERVED: text file with two blocks.
  - Block 1 has the header `# ORGAN_ID\tORGAN_NAME\tMC_MATERIAL\tDENSITY [g/cm3]`, then one row per Organ ID
    (53 rows), e.g. `1\t\tbody\t2\t\t0.92000771276497`.
  - Block 2 is headed `#[Monte Carlo material IDs, 2009-09-24]` and lists 9 rows of `MC_ID  name`
    (1 air, 2 body, 3 sternum, 4 muscle, 5 lung, 6 scapulas, 7 spine_cortical, 8 skin, 9 aorta).
- OBSERVED: the format is byte-for-byte that of `full_materials.py:make_conv_table`, which prints densities with Python `str(float)`.
  The table matches its xlsx exactly (name, MC ID, density) for all 53 IDs × 67 subjects (0 mismatches).
- OBSERVED (`DS/scripts/dukesim_params.py:614-650, 654-685`): DukeSim uses this table **only for MC-GPU**.
  - It replaces the voxel header with nx, ny, nz and voxel size in **cm** (0.1·sx).
  - It loads `<mcgpu_materials_directory>/<name>.mcgpu` for each name in block 2. A missing file is an error.

### 1d. LAC `material_lac/<id>/lin_att_120kV_121materials.bin`
- OBSERVED: 58,080 bytes, i.e. float32 little-endian with shape **[121 organ IDs, 120 energies]**, row-major
  (`full_materials.py:convert_lin_att` writes material by material, kV values each).
- OBSERVED: units are cm⁻¹. Column j corresponds to E = j+1 keV, i.e. 1..120 keV.
  - The generator call is `xcatattentab.x 1 1 {kV}` (start 1, step 1, max = kV).
  - The air row divided by 0.001205 gives 0.1871 cm²/g at column 59, which is NIST air at 60 keV (0.1875).
  - The 10313 study found a median relative error of 0 against density × composition × xraylib, with column = keV−1.
- OBSERVED: rows for Organ IDs not in the table (e.g. 2, 5, 7, 10–12, …) are copies of the air row in all 67 files.
  This is the generator's gap-filling behaviour: `convert_lin_att` writes `row[i][1]`, the first material, which must be air.
  The row count is max Organ ID + 1 = 121 = `NumberOfMaterials`.
  DukeSim checks the size as `4·N_Energies·N_Materials` (`dukesim_params.py:1316`).
- OBSERVED: values are cross-sections from xraylib `CS_Total` (`xcatattentab.c` header), i.e. total
  attenuation including coherent scattering.

### 1e. `.mcgpu` files (`TC/phantoms/mcgpu_materials/*.mcgpu`, `COPD/material_mcgpu/`)
- OBSERVED: these are PENELOPE-2006-derived MC-GPU material files. Header: material name, `NOMINAL DENSITY (g/cm^3)`,
  `NUMBER OF DATA VALUES 25000`. The body has columns Energy (eV) | Rayleigh / Compton / Photoelectric / Total mean free
  path (cm) | Rayleigh max cumulative F². The energy grid starts at 500 eV, consistent with `generate_materials.sh`
  (`500,{kV}5000`, 25000 values).
- OBSERVED: there are 17 files, and their md5s are **identical** across `phantoms/mcgpu_materials`, `COPD/`, `LungLesions/` and
  `LiverLesions/material_mcgpu/`. This is one shared catalogue (air, aorta, body, cartilage, kidney, liver, liver_lesion, lung,
  lung_lesion, muscle, pancreas, scapulas, skin, small_intestine, spine_cortical, spleen, sternum).
  The nominal densities are rounded values (lung 0.30, body 0.92, sternum 1.18, spine_cortical 1.38, …), *not* any
  subject's xlsx values.
- INFERRED: MC-GPU takes per-voxel density from the conversion table and scales the nominal-density mean free paths.
  So the subject-specific density enters scatter through the `.in`, and composition enters through the shared `.mcgpu`.
  Not verified in the MC-GPU binary.

### 1f. Which files are direct simulator inputs?
| file | DukeSim route A (`material_spreadsheet`) | route B (pre-built) | used by |
|---|---|---|---|
| label `.bin` | required (`XCAT_file`) | required | RTcat primary + MC-GPU scatter |
| xlsx | **required: master** | not used | generator → all three below |
| LAC `.bin` | generated into `material_module/linear_attenuation.bin` | `linear_attenuation_file` | RTcat (primary, polychromatic ray tracing) |
| `.in` | generated `conversion_table.in` | `material_conversion_table` | MC-GPU only |
| `.mcgpu` dir | generated, one file per unique MC ID | `mcgpu_materials_directory` | MC-GPU only |

OBSERVED: `DS/install/DukeSim_v1.2_documentation.pdf`, "Materials" section and Table 2 note: *"If material_spreadsheet is
defined, then linear_attenuation_file, material_conversion_table, and mcgpu_materials_directory do not have to be
defined."* `GB` always uses route A (`GB/stages/parameters/run.py` injects `material_spreadsheet`).
UNKNOWN: which route and which DukeSim version produced the 2022 challenge projections.

---

## Q2. Organ ID → (density, composition, MC ID, LAC row)

- OBSERVED: the label value **is** the Organ ID and indexes the LAC row directly (row k = Organ ID k, 0..120).
  Labels are not contiguous. 53 IDs are used, and the rest of the rows are air.
- OBSERVED hierarchy (identical in all 67 subjects): 53 Organ IDs → 19 names → 9 MC IDs → **8 distinct compositions**.
  - sternum/scapulas/ribs share one composition but have MC IDs 3 and 6.
  - All lung-type IDs (6, 26–38, 60, 106–110) share the lung composition and MC 5.
  - muscle/esophagus/fissure share MC 4. aorta/vessel share MC 9. Air is IDs 0 and 70.
- OBSERVED: **the LAC row is fully determined by density × composition.** For every set of IDs with the same composition,
  LAC_row/density is identical to < 1e-4 relative error, in all 67 files (0 violations). The mass attenuation
  per composition is identical across subjects (max relative difference 2.3e-7).
- OBSERVED (`full_materials.py:make_MC_IDs`): if the xlsx has an `MC ID` column, DukeSim uses it, keyed by **Name**.
  Otherwise it assigns one MC ID per unique composition. DukeSim writes one `.mcgpu` file per MC ID, named after the first
  row's Name and using that row's density as nominal.
  INFERRED: a predicted table that changes a Name→MC ID mapping inconsistently (one Name, two MC IDs)
  would be silently collapsed.
- Density-ladder structure (OBSERVED, all 67 subjects; ratios within a subject are exact to 1e-14):
  - Lung parenchyma 26–38 = base × [1, 1.5, 2.1, 2.7, 3.3, 3.9, 4.5, 5.3, 6.1, 6.9, 7.7, 8.5, 9.2].
  - Spine trabecular 39–43 = base × [1, 1.07, 1.14, 1.21, 1.28].
  - Ribs (45–49) and scapula (50–54) trabecular = base × [1, 1.069, 1.138, 1.207, 1.276].
  - Sternum trabecular 55–59 = base × [1, 1.071, 1.143, 1.214, 1.274].
  - INFERRED: these are "texture" density bins within one tissue.

---

## Q3. Variation across the 67 COPD subjects

OBSERVED (`xsub.py`, `xsub2.py`):
- **Same across all 67:** the xlsx header, the set of 53 Organ IDs (1 distinct set), the Name per ID, the MC ID per ID, the composition per ID
  (1 composition per Name), the conversion-table MC block (1 distinct), and the 17 `.mcgpu` files.
- **The `.in` matches the xlsx** for all 67 subjects (0 mismatches in ID set, name, MC ID or density).
- **The LAC files are all different** (67 distinct md5s), but only through density: mass-attenuation rows are identical.
- **Density varies per subject as 16 group scale factors.** Within a group, every ID has the same ratio to its cross-subject
  median. Ratio-to-median per subject:

| group (Organ IDs) | ratio range | SD | median density example (g/cm³) |
|---|---|---|---|
| air 0, 70 | 1.0000 (constant 0.001205) | 0 | 0.001205 |
| body 1 | 0.981–1.014 | 0.010 | 0.9225 |
| sternum 3 + trabecular 55–59 | 0.988–1.010 | 0.007 | 1.1806 |
| muscle 4 | 0.988–1.015 | 0.008 | 1.0494 |
| lung 6 | 0.956–1.061 | 0.031 | 0.2975 |
| scapulas 8 + trabecular 50–54 | 0.989–1.010 | 0.006 | 0.9606 |
| spine cortical 9 + trabecular 39–43 | 0.989–1.012 | 0.007 | 1.3792 |
| esophagus 13 | 0.990–1.018 | 0.009 | 1.0456 |
| skin 24 | 0.985–1.013 | 0.008 | 1.0907 |
| aorta 25 | 0.985–1.012 | 0.007 | 1.0608 |
| lung parenchyma ladder 26–38 | 0.960–1.029 | 0.019 | 0.1006 (ID 26) … 0.9254 (ID 38) |
| ribs cortical 44 + trabecular 45–49 | 0.990–1.012 | 0.006 | 0.9594 |
| **emphysema 60** | **0.430–1.585** | **0.333** | 0.0526 (range 0.0226–0.0834) |
| vessel 68 | 0.987–1.014 | 0.008 | 1.0595 |
| lung 106–110 (equal densities) | 0.959–1.064 | 0.032 | 0.2966 |
| fissure 120 | 0.984–1.012 | 0.009 | 1.0524 |

- OBSERVED: lung 6 and lung 106–110 are scaled independently of each other (Pearson r = −0.16 across subjects).
- INFERRED: "smeared" means these per-subject random density perturbations of a nominal XCAT table. The
  `.mcgpu` nominal densities (lung 0.30, body 0.92, …) look like the unperturbed nominal values.
- **Implication for a canonical catalog:** composition, MC ID and the label ontology lose nothing. Density loses
  ≤ 2% for soft tissue and bone, ~4% (parenchyma) and ~4–6% (lung 6/106–110) for lung, and **−57% / +58% for emphysema**.
  Within a subject the whole parenchyma ladder is fixed by one scalar, so a per-subject estimate of that scalar from CT is plausible.
  Whether the emphysema density can be identified from CT is an open scientific question.
  The plan's rule ("never copy held-out subject properties") is enforceable here: predict the label map, then
  either use train-median densities or estimate the 16 scale factors from the image, and record which.

---

## Q4. "smeared", "textured", "eroded", "flipped"

- OBSERVED: no file, script or document found defines these terms. Searched: TrueCT PDF, `TC/README.md`, the 10313 README and scripts,
  `/home/yb107/ctsynth_truect`, `/home/yb107/ctsynth_xcat` and the DukeSim scripts.
- INFERRED:
  - *smeared* (only on material tables): the per-subject density scale factors from Q3.
  - *textured*: the lung-parenchyma and trabecular density-bin IDs (26–38, 39–59) that give voxel-level texture.
  - *flipped*: an axis reversal relative to the native XCAT output. The 10313 study found the final array is (z,y,x) = (S,L,P).
  - *eroded*: UNKNOWN. Plausibly a morphological erosion of an organ/lung mask (e.g. to make room for skin or fissures).
  - "8bit" = uint8.
  - Liver files use "ribs_fixed" and "cropped", which confirms the suffixes describe post-processing steps.
- UNKNOWN: the exact operations. Ask the TrueCT/CVIT authors (Abadi/Segars) if it matters.

---

## Q5. DukeSim availability and what a run needs

OBSERVED:
- **Install:** `DS = /home/dmf46/DukeSim_v1.2.21`. This is the lab's admin install; it is visible to compute nodes.
  Contents: `DukeSim.py`, `scripts/{dukesim.py, dukesim_params.py, RTcat, mcgpu_CT/duke_mcgpu*, generate_materials/, mcr-toolkit*}`
  and the docs `install/DukeSim_v1.2_documentation.pdf`. There is also an `rtcat.license`.
- **Batch wrapper:** `GB = /scratch/railabs/g-dukesimbatch`.
  - Its `README_USER_GUIDE.md` says to create `usr/<netid>/` and run `./run.sh usr/<netid>/params.txt [jobid] [--test_mode]`.
    It submits one Slurm job per phantom and writes outputs under `GB/output/<jobid>/` (a shared, lab-wide location).
  - Scanners `w12`, `w20` and `duke-1` are available. **TrueCT was simulated on Duke-1** (TrueCT PDF, Table 1).
    `/scratch/railabs/dukesim_data/scanners/duke-1/duke-1.txt` gives: 120 kV, `NumberOfEnergies: 120`, 900×64 detector pixels of 1 mm, SID 1050,
    SIsoD 575, 1000 views/rotation, collimation 35.05 mm, EID, cylinder detector.
- **Inputs per phantom** (DukeSim phantom parameters, documentation Table 2 and `dukesim_params.py:281-303`):
  - `XCAT_file`: uint8 `.bin`, size must equal nx·ny·nz (`check_args`).
  - `XCAT_nx/ny/nz` (int) and `XCAT_sx/sy/sz` (**mm**).
  - `NumberOfMaterials`: GB sets it to max(Organ ID)+1 = 121. The check `max(label) > N_Materials+1` → error.
  - `msecs_per_phantom`: any value for a static phantom.
  - Either `material_spreadsheet` (xlsx) **or** `linear_attenuation_file` + `material_conversion_table` + `mcgpu_materials_directory`.
  - The GB metadata xlsx columns are `paramID, XCAT_file, material_spreadsheet, XCAT_nx, XCAT_ny, XCAT_nz, XCAT_sx, XCAT_sy, XCAT_sz,
    NumberOfMaterials, msecs_per_phantom` (`GB/usr/example/example_metadata.xlsx`).
- **Run-level parameters** come from a template, e.g. `GB/usr/_TEMPLATES/dukesim_parameters.txt`: mA, pitch, N_Rotations, BHC, noise,
  MC-GPU histories, `mcgpu_vox_size` and recon settings.
  - MC-GPU requires isotropic voxels if it downsamples (`dukesim_params.py:414`).
  - The scale factor is `int(mcgpu_vox_size / XCAT_sx)`.
- INFERRED: we *can* run DukeSim on a predicted phantom through GB. It needs GPU nodes and the lab's GEMSTONe environment, and it writes
  into `GB/output` (shared space, outside our workspace) unless overridden. Before any run, check the write location and confirm with the lead and the lab.
  None was run here.
- UNKNOWN: the exact 2022 acquisition parameters per case (mA/TCM profile, rotations, histories, DukeSim version). The
  `mAs_vector_<N>.bin` files in `projections/` give the TCM. Re-simulating a source phantom and comparing it with the supplied
  projections would be the real end-to-end acceptance test.

### Proposed export schema (route A, matching the source package)
For each predicted subject `<id>`:
1. `phantom_<id>_<nx>_<ny>_<nz>_pred.bin`: uint8, no header, C-order `[z,y,x]`, x fastest, same nx/ny/nz/orientation as the
   source phantom grid. Voxel size 0.25 mm isotropic (assumed; recorded explicitly).
2. `materials_<id>.xlsx`: one sheet in exactly the TrueCT column order:
   `Name | 1 6 7 8 11 12 15 16 17 19 20 26 53 | Density | No. of elements with non-zero mass fractions | Organ ID | MC ID`.
   Rows are sorted by Organ ID, and row 1 is air (Organ ID 0).
3. `metadata.xlsx` (GB format) or `phantom_params_<id>.txt` with `XCAT_file, XCAT_nx/ny/nz, XCAT_sx/sy/sz (mm), NumberOfMaterials=max ID+1,
   msecs_per_phantom`.
4. Optional route-B artefacts for byte-level comparison with the source package:
   - `conversion_table_120kV_<N>materials.in` in `make_conv_table` format.
   - `lin_att_120kV_<N>materials.bin` as float32 [N, 120]. Regenerate it with mass-attenuation rows from the source LAC: per composition,
     the source LAC row divided by the source density. This reproduces DukeSim's xraylib values without running xcatattentab.
   - The shared 17-file `.mcgpu` set, referenced and not copied.
5. A provenance JSON: model run, catalog source (train-median vs image-estimated scale factors) and the subject split.

### Mock-validator invariants (until a DukeSim run is authorised)
Phantom:
- V1 dtype uint8, file size == nx·ny·nz, no header.
- V2 nx, ny, nz and the voxel size match the declared metadata. Isotropic voxels if MC-GPU downsampling is used.
- V3 every label value ∈ the xlsx Organ ID set. `max(label) ≤ NumberOfMaterials−1` (stricter than DukeSim's own `≤ N+1` check).
- V4 orientation/axis order identical to the source phantom: a round-trip of the source labels through the exporter is byte-identical.

Material table:
- V5 xlsx: one sheet, exact header as above, Organ IDs unique and ascending, the first row is air with ID 0, and names have no spaces.
- V6 fractions ≥ 0. The `No. of elements` column equals the count of nonzero fractions. The fraction sum is within the source tolerance:
  either reproduce the source sums exactly (0.968 / 0.9622 are legitimate) or flag sums outside [0.999, 1.001] as DukeSim does.
- V7 density > 0. Air density = 0.001205. The density of each ID is within the train range for its group, or flagged.
- V8 consistency: one Name has one MC ID and one composition. One MC ID has one composition. Every MC ID name has a `<name>.mcgpu`
  in the shared catalogue.
- V9 ontology closure: IDs, names, MC IDs and compositions ⊆ the 53-ID TrueCT COPD catalogue (all of these are constant across subjects).
- V10 (if within-subject structure is kept) the ladder ratios are as in Q2 (lung parenchyma, 4 trabecular ladders) and lung 106–110 densities are equal.

Derived files (route B):
- V11 `.in` matches the xlsx exactly (ID set, name, MC ID, density repr), in `make_conv_table` format.
- V12 LAC size = 4·120·NumberOfMaterials. `LAC[k]/ρ_k` equals the catalogue mass attenuation for k's composition (rel. < 1e-5).
  Rows not in the table equal the air row.

Sanity checks:
- V13 recompute the source package for every training subject (xlsx → `.in` + LAC) and require byte-identity with
  `TC/.../material_lac/<id>/`. This validates the validator itself.
- V14 a monoenergetic forward check: `phantom → μ(70 keV)` HU via the LAC column 69. This is the challenge ground-truth definition
  ("mono-energetic representation of the phantoms at the 70 keV HU", TrueCT PDF p.6). Compare it with the recon after resampling.

---

## Q6. Effective atomic number

- OBSERVED: element Z values are the xlsx column headers. Mass fractions exist for all 53 IDs.
  - Atomic weights are in `DS/scripts/generate_materials/penelope_material/atomicweight.dat`.
  - Element mass-attenuation tables are in `/home/yb107/ctsynth_truect/data/element_mass_atten_xraylib.json`.
  - Per-composition μ/ρ(E) for 1–120 keV can be read straight from any LAC file.
- Options:
  1. Power law (Mayneord/Spiers): `Z_eff = (Σ_i a_i Z_i^m)^{1/m}`, m ≈ 2.94.
     Here `a_i = (w_i Z_i/A_i) / Σ_j (w_j Z_j/A_j)` is the electron fraction. It needs w (present), Z (headers) and A (available).
  2. Energy- or spectrum-dependent Z_eff, from matching the composition's photoelectric/Compton ratio, or from interpolating μ/ρ(E) between
     elements. The available xraylib tables and LAC μ/ρ support this.
  3. Mean electron density ratio ρ_e = ρ·Σ w_i Z_i/A_i as a companion quantity. It is defined only by the table.
- Caveats (OBSERVED):
  - The sums of w are < 1 for lung (0.968) and skin (0.9622). Renormalise, as xcatattentab does.
  - Air's 1.28% is in the Cl (Z=17) column; it is presumably Ar (Z=18). The choice changes Z_eff(air) only slightly; document it.
- INFERRED, key point: there are only **8 distinct compositions** and they are constant across subjects. So Z_eff is a deterministic
  function of the material class, takes 8 values, and carries **no information beyond the label map**. It is
  defensible as a secondary *derived* quantity, but it is not an independent regression target.

---

## Surprises / risks
- The LAC, `.in` and `.mcgpu` files are DukeSim *outputs* of the xlsx. Exporting a correct xlsx plus labels is sufficient for a DukeSim run.
- Emphysema density (ID 60) varies 3.7× across subjects (0.0226–0.0834 g/cm³). A canonical catalog is badly wrong there.
- Lung 6 and lung 106–110 densities are independently perturbed (r = −0.16).
- Unused LAC rows are air. A mislabelled voxel with an unused ID would therefore simulate as air, not error out: DukeSim's checks are lax.
- `.mcgpu` nominal densities differ from every subject's table. This is harmless if MC-GPU rescales by voxel density (INFERRED).
- GB writes to a shared lab output tree. A real DukeSim acceptance run needs the lead's decision on where it writes and on GPU use.
