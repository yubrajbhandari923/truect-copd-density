"""E00 stage 3: condense inventory.json + volume_stats/*.json into small committed evidence.

Writes OUT/summary/{subjects_67.json, ontology.json, cohort_facts.json}. No volumes are read.
"""
import collections as C
import json

import numpy as np

from common import EXPOSED, OUT

inv = json.load(open(OUT / "inventory.json"))
S = inv["subjects"]
V = {s["id"]: json.load(open(OUT / f"volume_stats/{s['id']}.json")) for s in S}
(OUT / "summary").mkdir(exist_ok=True)

# ---- ontology: is one organ ID the same thing in every subject?
ids = sorted({int(k) for s in S for k in s["table"]})
onto = {}
for k in ids:
    rows = [s["table"][str(k)] for s in S if str(k) in s["table"]]
    rho = np.array([r["density"] for r in rows])
    fr = np.array([[r["fractions"][z] for z in sorted(r["fractions"], key=int)] for r in rows])
    vox = np.array([V[s["id"]]["label_hist"].get(str(k), 0) for s in S])
    onto[k] = dict(
        names=sorted({r["name"] for r in rows}), mc_ids=sorted({r["mc_id"] for r in rows}),
        n_tables=len(rows), n_subjects_present=int((vox > 0).sum()), voxels_total=int(vox.sum()),
        density=dict(min=float(rho.min()), median=float(np.median(rho)), max=float(rho.max()),
                     cv=float(rho.std() / rho.mean())),
        fraction_max_abs_spread=float((fr.max(0) - fr.min(0)).max()),
        fractions_median={z: float(v) for z, v in zip(sorted(rows[0]["fractions"], key=int), np.median(fr, 0))})
labels_not_in_table = {s["id"]: sorted(set(map(int, V[s["id"]]["label_hist"])) - set(map(int, s["table"]))) for s in S}
json.dump(dict(organ_ids=onto, labels_present_but_not_in_table={k: v for k, v in labels_not_in_table.items() if v},
               per_subject_density={s["id"]: {k: v["density"] for k, v in s["table"].items()} for s in S}),
          open(OUT / "summary/ontology.json", "w"), indent=1)

# ---- duplicates / derivatives: identical bytes, identical histograms, identical tables
def dup(key):
    g = C.defaultdict(list)
    for s in S:
        g[key(s)].append(s["id"])
    return [v for v in g.values() if len(v) > 1]

hist_vec = lambda s: tuple(sorted(V[s["id"]]["label_hist"].items()))
dups = dict(phantom_sha256=dup(lambda s: V[s["id"]]["phantom_sha256"]), label_histogram=dup(hist_vec),
            phantom_shape=dup(lambda s: tuple(s["phantom"]["shape_zyx"])),
            density_table=dup(lambda s: tuple(sorted((k, v["density"]) for k, v in s["table"].items()))),
            lac_sha256=dup(lambda s: s["lac"]["sha256"]))

# ---- per subject
subs = []
for s in S:
    v = V[s["id"]]
    rh = {int(k): n for k, n in v["recon_hist"].items()}
    n = sum(rh.values())
    flags = list(s["problems"])
    if labels_not_in_table[s["id"]]:
        flags.append(f"labels without table row: {labels_not_in_table[s['id']]}")
    if v["recon_shape"][0] != s["recon"]["n_slices"]:
        flags.append("slice count mismatch")
    subs.append(dict(
        id=s["id"], projection_series=s["series"], development_exposed=s["id"] in EXPOSED,
        phantom_file=s["phantom"]["file"], phantom_shape_zyx=s["phantom"]["shape_zyx"],
        phantom_sha256=v["phantom_sha256"], n_labels_present=len(v["label_hist"]),
        recon_shape=v["recon_shape"], recon_dtype=v["recon_dtype"], recon_min=v["recon_min"], recon_max=v["recon_max"],
        recon_frac_at_min=rh[v["recon_min"]] / n, recon_frac_eq_m1000=rh.get(-1000, 0) / n,
        n_views=s["projections"]["n_views"], table_sha256=s["table_sha256"], lac_sha256=s["lac"]["sha256"],
        emphysema_density=s["table"]["60"]["density"],
        emphysema_voxel_frac_of_lung=v["label_hist"].get("60", 0) / max(1, sum(
            n_ for k, n_ in v["label_hist"].items() if int(k) in {6, 60, *range(26, 39), *range(106, 111)})),
        file_level_status="OK" if not flags else "FLAGGED", flags=flags))
json.dump(subs, open(OUT / "summary/subjects_67.json", "w"), indent=1)

facts = dict(
    n_subjects=len(S), n_ok=sum(x["file_level_status"] == "OK" for x in subs),
    id_set_differences=inv["id_sets_equal"], unmapped_series=inv["series_unmapped"],
    duplicates=dups,
    recon_slices=C.Counter(x["recon_shape"][0] for x in subs), recon_inplane=C.Counter(str(x["recon_shape"][1:]) for x in subs),
    recon_min=C.Counter(x["recon_min"] for x in subs),
    recon_max_range=[min(x["recon_max"] for x in subs), max(x["recon_max"] for x in subs)],
    recon_geometry_tags_present=sorted(k for k in ("PixelSpacing", "SliceThickness", "ImagePositionPatient",
                                                   "RescaleSlope", "RescaleIntercept", "ImageOrientationPatient")
                                       if any(k in s["recon"]["header_first"] for s in S)),
    recon_modality=C.Counter(s["recon"]["header_first"].get("Modality") for s in S),
    n_labels_present=C.Counter(x["n_labels_present"] for x in subs),
    xlsx_vs_conv_mismatch={s["id"]: s["xlsx_vs_conv_mismatch_ids"] for s in S if s["xlsx_vs_conv_mismatch_ids"]},
    fraction_sum_range=[min(s["fraction_sum_range"][0] for s in S), max(s["fraction_sum_range"][1] for s in S)],
)
json.dump(facts, open(OUT / "summary/cohort_facts.json", "w"), indent=1)
print(json.dumps(facts, indent=1))
for k, o in onto.items():
    print(k, o["names"], o["mc_ids"], "present", o["n_subjects_present"], "vox %.3g" % o["voxels_total"],
          "rho med %.4f cv %.4f" % (o["density"]["median"], o["density"]["cv"]), "fracspread %.3g" % o["fraction_max_abs_spread"])
