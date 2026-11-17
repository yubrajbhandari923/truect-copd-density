"""Copy the thin-slice original axial chest series from the DUHS VLM CT_CHEST folder into the lab shared-data area,
verify the copies and record a manifest. Never writes to the source folder.

Selection (one rule, applied to every JSON sidecar): 3-D volume with > 50 slices, slice spacing <= 1.25 mm,
ImageType contains ORIGINAL, SeriesDescription without "MIP".
Usage: python copy_check.py <src_dir> <dst_dir>
"""
import csv, hashlib, json, shutil, sys, time
from pathlib import Path

import nibabel as nib
import numpy as np

SRC, DST = Path(sys.argv[1]), Path(sys.argv[2])
FIELDS = ["Manufacturer", "ManufacturersModelName", "ConvolutionKernel", "SeriesDescription", "BodyPart",
          "PatientPosition", "XRayTubeCurrent", "ExposureTime"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def selected(j, img):
    s, z = img.shape, img.header.get_zooms()
    return (len(s) == 3 and s[2] > 50 and z[2] <= 1.25 and "ORIGINAL" in j.get("ImageType", [])
            and "MIP" not in j.get("SeriesDescription", "").upper())


rows, problems = [], []
(DST / "images").mkdir(parents=True, exist_ok=True)
for js in sorted(SRC.glob("*.json")):
    j, src_nii = json.load(open(js)), js.with_suffix("").with_suffix(".nii.gz")
    if not src_nii.exists():
        src_nii = js.parent / (js.stem + ".nii.gz")
    img = nib.load(src_nii)
    if not selected(j, img):
        continue
    out = {}
    for p in (js, src_nii):
        d = DST / "images" / p.name
        if not d.exists() or d.stat().st_size != p.stat().st_size:
            shutil.copy2(p, d)
        out[p.suffix] = (sha(p), sha(d))
    v = np.asanyarray(nib.load(DST / "images" / src_nii.name).dataobj)  # full decode = gzip integrity check
    A, z = img.affine, np.array(img.header.get_zooms()[:3], float)
    cols = A[:3, :3] / z
    oblique_deg = float(np.degrees(np.arccos(np.clip(np.abs(cols).max(0), -1, 1))).max())
    r = dict(series=js.stem, patient=js.stem.split("_Study")[0], **{k: j.get(k) for k in FIELDS},
             shape="x".join(map(str, img.shape)), spacing_mm="x".join(f"{x:.3f}" for x in z),
             z_extent_mm=round(img.shape[2] * z[2], 1), axcodes="".join(nib.aff2axcodes(A)), oblique_deg=round(oblique_deg, 2),
             dtype=str(img.get_data_dtype()), slope=float(img.dataobj.slope), inter=float(img.dataobj.inter),
             hu_min=float(v.min()), hu_p50=float(np.median(v[::4, ::4, ::4])), hu_max=float(v.max()),
             sha256_nii=out[".gz"][1], sha256_json=out[".json"][1], copy_ok=all(a == b for a, b in out.values()))
    for msg, bad in [("copy checksum mismatch", not r["copy_ok"]), ("oblique > 1 deg", oblique_deg > 1),
                     ("HU min > -900 (not HU?)", r["hu_min"] > -900), ("HU max < 500", r["hu_max"] < 500),
                     ("non-unit/zero slope", r["slope"] not in (1.0,) and not np.isnan(r["slope"]))]:
        if bad:
            problems.append(f"{js.stem}: {msg}")
    rows.append(r)
    print(r["series"], r["shape"], r["spacing_mm"], r["axcodes"], r["hu_min"], r["hu_max"], r["copy_ok"], flush=True)

with open(DST / "manifest.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
json.dump(dict(source=str(SRC), n_series=len(rows), n_patients=len({r["patient"] for r in rows}), problems=problems,
               date=time.strftime("%Y-%m-%d"), selection=__doc__.split("Selection")[1].split("Usage")[0].strip()),
          open(DST / "checks.json", "w"), indent=1)
print(len(rows), "series;", len(problems), "problems:", *problems, sep="\n")
