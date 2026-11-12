"""E00 export self-test on real source packages. Usage: selftest_export.py <code_sha> sid [sid...]  (run via run.sbatch)"""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

import dukesim_export as d
from common import OUT, PH, phantom_file, xlsx_table

sha, sids = sys.argv[1], sys.argv[2:]
R = OUT / "export_validation"
rep = dict(code_commit=sha, job_id=os.environ.get("SLURM_JOB_ID"), subjects={}, negatives={})


def view(sid, root):
    lab, (z, y, x) = phantom_file(sid)
    v = root / sid
    v.mkdir(parents=True, exist_ok=True)
    for src in (lab, PH / f"material_tables/challenge_materials_copd_smeared_{sid}.xlsx",
                *(PH / f"material_lac/{sid}").glob("*")):
        if not (v / src.name).exists():
            (v / src.name).symlink_to(src)
    (v / "geometry.json").write_text(json.dumps(dict(label_file=lab.name, nx=x, ny=y, nz=z, sx_mm=0.25, sy_mm=0.25, sz_mm=0.25,
                                                      NumberOfMaterials=121, dtype="uint8", header_bytes=0)))
    return v


for sid in sids:
    v = view(sid, R / "src_view")
    val = d.validate_package(v)
    tab, _ = xlsx_table(sid)
    w = R / "regen" / sid
    shutil.rmtree(w, ignore_errors=True)
    d.write_tables(w, tab, sid)
    src_in = (PH / f"material_lac/{sid}/conversion_table_120kV_121materials.in").read_bytes()
    new_in = (w / "conversion_table_120kV_121materials.in").read_bytes()
    a = np.fromfile(PH / f"material_lac/{sid}/lin_att_120kV_121materials.bin", "<f4")
    b = np.fromfile(w / "lin_att_120kV_121materials.bin", "<f4")
    rel = np.abs(b.astype(np.float64) / a - 1)
    rt = dict(in_byte_identical=src_in == new_in, in_len=(len(src_in), len(new_in)),
              lac_shape_equal=a.shape == b.shape, lac_byte_identical=a.tobytes() == b.tobytes(),
              lac_max_abs=float(np.abs(a - b).max()), lac_max_rel=float(rel.max()),
              lac_n_elements_differing=int((a != b).sum()), lac_n=int(a.size))
    rep["subjects"][sid] = dict(validation=val, roundtrip=rt)
    print(sid, {k: x["status"] for k, x in val.items()}, rt, flush=True)

# V4: full write_package on one subject; label bytes must equal the source label file
sid = sids[0]
lab, shp = phantom_file(sid)
w = R / "regen_pkg" / sid
shutil.rmtree(w, ignore_errors=True)
lp = d.write_package(w, np.memmap(lab, np.uint8, "r", shape=shp), 0.25, xlsx_table(sid)[0], sid=sid, suffix="regen")
same = True
with open(lab, "rb") as f1, open(lp, "rb") as f2:
    while (b1 := f1.read(1 << 26)):
        if b1 != f2.read(len(b1)):
            same = False
            break
rep["V4_label_roundtrip"] = dict(subject=sid, byte_identical=same, written_pkg_validation={k: x["status"] for k, x in d.validate_package(w).items()})
print(rep["V4_label_roundtrip"], flush=True)
lp.unlink()  # our own 2-3 GB regenerated copy

# negatives on a copy of the real small files (label symlinked)
n = R / "neg"
shutil.rmtree(n, ignore_errors=True)
shutil.copytree(R / "src_view" / sid, n / "base", symlinks=True)
for k in ("A_label_not_in_table", "B_wrong_size", "C_density_nonpositive", "D_lac_truncated", "E_lac_float64"):
    shutil.copytree(n / "base", n / k, symlinks=True)
def rewrite(k, name, fn):
    f = n / k / name
    data = f.read_bytes() if f.is_symlink() else None
    f.unlink()
    return fn
tab = {int(o): dict(r) for o, r in xlsx_table(sid)[0].items()}
def put_xlsx(k, t):
    (n / k / f"challenge_materials_copd_smeared_{sid}.xlsx").unlink()
    d.write_xlsx(n / k / "materials_neg.xlsx", t)
    c = n / k / "conversion_table_120kV_121materials.in"; c.unlink(); c.write_bytes(d.conv_text(t).encode())
t = {o: dict(r) for o, r in tab.items()}; del t[60]; put_xlsx("A_label_not_in_table", t)
g = n / "B_wrong_size/geometry.json"; j = json.loads(g.read_text()); j["nx"] += 1; g.write_text(json.dumps(j))
t = {o: dict(r) for o, r in tab.items()}; t[4]["density"] = 0.0; put_xlsx("C_density_nonpositive", t)
f = n / "D_lac_truncated/lin_att_120kV_121materials.bin"; raw = f.read_bytes(); f.unlink(); f.write_bytes(raw[:-480])
f = n / "E_lac_float64/lin_att_120kV_121materials.bin"; raw = np.fromfile(f, "<f4"); f.unlink(); raw.astype("<f8").tofile(f)
expect = dict(A_label_not_in_table="V3", B_wrong_size="V1", C_density_nonpositive="V7", D_lac_truncated="V12", E_lac_float64="V12")
for k, v in expect.items():
    r = d.validate_package(n / k)
    rep["negatives"][k] = dict(expected_fail=v, statuses={a: b["status"] for a, b in r.items()}, hit=r[v]["status"] == "FAIL", reason=r[v]["reason"])
    print(k, rep["negatives"][k]["hit"], r[v]["reason"][:150], flush=True)
(R / "selftest.json").write_text(json.dumps(rep, indent=1))
