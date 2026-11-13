"""Write the tables that document the NIfTI export, from computed results only.

usage: dataset_docs.py [dest_root]
Writes <dest>/subjects.tsv and <dest>/labels.tsv. Reads export meta.json, target meta.json, alignment
validation and the E00 summaries. Fails if any of the 67 subjects is missing an export.
"""
import csv
import json
import sys
from pathlib import Path

from common import OUT, SRC, pairing

VISUAL = {"10313", "14402", "16783", "24080", "11989", "18314", "14023", "18762"}  # overlays inspected by eye
NOTES = {"18762": "body extends past the 768-px field of view (axial); CT near the cut edge may be biased",
         "11989": "body nearly touches the field-of-view edge"}

dest = Path(sys.argv[1]) if len(sys.argv) > 1 else SRC / "derived/copd_nifti"
subs = {s["id"]: s for s in json.load(open(OUT / "summary/subjects_67.json"))}
val = json.load(open(OUT / "alignment/alignment_validation.json"))["subjects"]
rows = []
for sid, series in pairing():
    e = json.load(open(dest / sid / "meta.json"))
    t = json.load(open(OUT / "targets" / sid / "meta.json"))
    s = subs[sid]
    rows.append(dict(
        id=sid, projection_series=f"dcm_{int(series):03d}", phantom_file=Path(s["phantom_file"]).name,
        phantom_shape_zyx="x".join(map(str, s["phantom_shape_zyx"])), ct_shape="x".join(map(str, t["recon_shape"])),
        ct_spacing_mm_ras=" ".join(f"{z:.4f}" for z in e["ct_spacing_mm"]),
        align_r=f"{e['alignment']['r']:.4f}", nifti_affine_selfcheck=f"{e['self_check_ctgrid_vs_native_label_agreement']:.5f}",
        desc_aorta_left_of_spine_mm=f"{e['descending_aorta_left_of_spine_mm']:.1f}",
        ct_valid_fraction=f"{t['valid_fraction']:.3f}",
        ct_vs_ideal_mu70_r=f"{t['ct_vs_mu70']['r']:.4f}", ct_vs_ideal_mu70_slope_HU_per_cm1=f"{t['ct_vs_mu70']['slope_HU_per_cm1']:.0f}",
        emphysema_frac_of_lung=f"{s['emphysema_voxel_frac_of_lung']:.4f}", emphysema_density_g_cm3=f"{s['emphysema_density']:.4f}",
        ct_max=s["recon_max"], overlay_inspected_by_eye="yes" if sid in VISUAL else "no",
        alignment_report_flags=";".join(val[sid].get("flags", [])), notes=NOTES.get(sid, "")))
with open(dest / "subjects.tsv", "w", newline="") as f:
    w = csv.DictWriter(f, rows[0].keys(), delimiter="\t")
    w.writeheader()
    w.writerows(rows)

onto = json.load(open(OUT / "summary/ontology.json"))["organ_ids"]
with open(dest / "labels.tsv", "w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["organ_id", "name", "mc_id", "subjects_with_voxels", "density_min", "density_median", "density_max"])
    for k in sorted(onto, key=int):
        o = onto[k]
        w.writerow([k, "/".join(o["names"]), "/".join(map(str, o["mc_ids"])), o["n_subjects_present"],
                    f"{o['density']['min']:.4f}", f"{o['density']['median']:.4f}", f"{o['density']['max']:.4f}"])
print("wrote", dest / "subjects.tsv", dest / "labels.tsv", len(rows), "subjects")
