"""E00: DukeSim-compatible phantom package writer + validator (route A/B artefacts, no DukeSim run).

Formats mirror DukeSim v1.2.21 scripts/generate_materials/full_materials.py and the TrueCT COPD package.
Spec and evidence: reports/e00/dukesim_contract.md (on main). Table layout: common.xlsx_table.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import openpyxl

HERE = Path(__file__).parent
CAT = HERE / "dukesim_catalogue.json"
ZS = ["1", "6", "7", "8", "11", "12", "15", "16", "17", "19", "20", "26", "53"]
HDR = ["Name"] + ZS + ["Density", "No. of elements with non-zero mass fractions", "Organ ID", "MC ID"]
N_E = 120
AIR_DENSITY = 0.001205
CHUNK = 1 << 28


def catalogue():
    return json.loads(CAT.read_text())


def comp_key(fr):
    return tuple(round(float(fr.get(z, 0) or 0), 9) for z in ZS)


def build_catalogue(sid):
    """One-off: derive the fixed catalogue (ids, compositions, mass attenuation=row/density) from a source subject."""
    from common import PH, conv_table, xlsx_table
    tab, _ = xlsx_table(sid)
    lac = np.fromfile(PH / f"material_lac/{sid}/lin_att_120kV_121materials.bin", "<f4").reshape(121, N_E).astype(np.float64)
    comps, ids = {}, {}
    for oid, r in tab.items():
        k = comp_key(r["fractions"])
        c = comps.setdefault(k, dict(fractions=[r["fractions"][z] for z in ZS], rows=[]))
        c["rows"].append(lac[oid] / r["density"])
        ids[oid] = dict(name=r["name"], mc_id=r["mc_id"], density=r["density"])
    names = {}
    keys = list(comps)
    for oid, r in tab.items():
        ids[oid]["comp"] = f"c{keys.index(comp_key(r['fractions']))}"
    out = dict(zs=ZS, source_subject=sid, ids={str(k): v for k, v in ids.items()},
               comps={f"c{i}": dict(fractions=comps[k]["fractions"], mu_rho=np.mean(comps[k]["rows"], 0).tolist())
                      for i, k in enumerate(keys)},
               mc_names={str(m): n for m, n in sorted({v["mc_id"]: v["name"] for v in reversed(list(ids.values()))}.items())},
               mcgpu_files=sorted(p.stem for p in (PH.parent / "mcgpu_materials").glob("*.mcgpu")) if (PH.parent / "mcgpu_materials").exists() else [])
    CAT.write_text(json.dumps(out, indent=1))
    return out


def catalogue_table(scale=1.0):
    """A full 53-ID table from the catalogue (nominal = source-subject densities x scale); for tests / predicted tables."""
    cat = catalogue()
    return {int(k): dict(name=v["name"], mc_id=v["mc_id"], density=(v["density"] if int(k) in (0, 70) else v["density"] * scale),
                         fractions=dict(zip(ZS, cat["comps"][v["comp"]]["fractions"]))) for k, v in cat["ids"].items()}


# ------------------------------------------------------------------ writers
def conv_text(table):
    """full_materials.make_conv_table format (densities via str(float), rows in Organ-ID order)."""
    s = "# ORGAN_ID\tORGAN_NAME\tMC_MATERIAL\tDENSITY [g/cm3]\n"
    first = {}
    for oid in sorted(table):
        r = table[oid]
        s += f"{oid}\t\t{r['name']}\t{r['mc_id']}\t\t{r['density']!s}\n"
        first.setdefault(r["mc_id"], r["name"])
    s += "\n#[Monte Carlo material IDs, 2009-09-24]\n"
    return s + "".join(f"{m}\t{n}\n" for m, n in first.items())


def lac_array(table, cat=None):
    """float32 [max_id+1, 120] cm^-1: density x per-composition mass attenuation; unused IDs = air row."""
    cat = cat or catalogue()
    ckeys = {comp_key(dict(zip(ZS, c["fractions"]))): c["mu_rho"] for c in cat["comps"].values()}
    lac = np.zeros((max(table) + 1, N_E), np.float32)
    lac[:] = (table[0]["density"] * np.array(ckeys[comp_key(table[0]["fractions"])])).astype(np.float32)
    for oid, r in table.items():
        k = comp_key(r["fractions"])
        if k not in ckeys:
            raise ValueError(f"organ {oid} ({r['name']}): composition not in catalogue; cannot derive LAC")
        lac[oid] = (r["density"] * np.array(ckeys[k])).astype(np.float32)
    return lac


def write_xlsx(path, table):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append([int(h) if h.isdigit() else h for h in HDR])
    for oid in sorted(table):
        r = table[oid]
        fr = [r["fractions"].get(z, 0) for z in ZS]
        ws.append([r["name"]] + [0 if v == 0 else v for v in fr] +
                  [r["density"], sum(v != 0 for v in fr), oid, r["mc_id"]])
    wb.save(path)


def write_tables(out_dir, table, sid="pred", n_mat=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_xlsx(out / f"materials_{sid}.xlsx", table)
    (out / "conversion_table_120kV_121materials.in").write_bytes(conv_text(table).encode())
    lac = lac_array(table)
    assert lac.shape[0] == (n_mat or lac.shape[0])
    lac.astype("<f4").tofile(out / "lin_att_120kV_121materials.bin")


def write_package(out_dir, labels, voxel_mm, table, sid="pred", suffix="pred"):
    """labels: uint8 [z,y,x] array or memmap (written in chunks). Returns the label file path."""
    labels = np.asarray(labels) if not isinstance(labels, np.memmap) else labels
    if labels.dtype != np.uint8 or labels.ndim != 3:
        raise ValueError("labels must be a uint8 [z,y,x] volume")
    nz, ny, nx = labels.shape
    if int(labels.max()) >= 121 or not set(np.unique(labels).tolist()) <= set(table):
        raise ValueError("labels contain IDs outside the material table / NumberOfMaterials")
    out = Path(out_dir)
    write_tables(out, table, sid, n_mat=121)
    lp = out / f"phantom_{sid}_{nx}_{ny}_{nz}_{suffix}.bin"
    step = max(1, CHUNK // (ny * nx))
    with open(lp, "wb") as f:
        for z in range(0, nz, step):
            f.write(np.ascontiguousarray(labels[z:z + step]).tobytes())
    (out / "geometry.json").write_text(json.dumps(dict(
        label_file=lp.name, nx=nx, ny=ny, nz=nz, sx_mm=voxel_mm, sy_mm=voxel_mm, sz_mm=voxel_mm,
        NumberOfMaterials=max(table) + 1,
        array_order="C-order [z,y,x]; x fastest; same grid/orientation as source phantom", dtype="uint8", header_bytes=0,
        voxel_size_note="0.25 mm is ASSUMED for TrueCT COPD (not in source files)"), indent=1))
    return lp


# ------------------------------------------------------------------ validator
def _find(d, pat):
    h = sorted(d.glob(pat))
    return h[0] if len(h) == 1 else None


def validate_package(pkg):
    """Return {V1..V14: {status, reason}}; statuses PASS/FAIL/SKIP. Streams the label volume."""
    from common import xlsx_table_path
    d = Path(pkg)
    res = {}
    cat = catalogue()

    def put(v, ok, why):
        res[v] = dict(status="SKIP" if ok is None else "PASS" if ok else "FAIL", reason=why)

    lab, xl, inn, lacf, gj = (_find(d, p) for p in ("phantom_*.bin", "*.xlsx", "conversion_table_*.in", "lin_att_*.bin", "geometry.json"))
    geo = json.loads(gj.read_text()) if gj else None
    dims = (geo["nx"], geo["ny"], geo["nz"]) if geo else None
    nmat = geo["NumberOfMaterials"] if geo else None

    # V1 / V2
    if not (lab and geo):
        put("V1", False, f"missing label file or geometry.json (label={lab}, geometry={gj})")
        put("V2", False, "missing label file or geometry.json")
    else:
        sz = lab.stat().st_size
        put("V1", sz == int(np.prod(dims)) and geo.get("dtype") == "uint8",
            f"size {sz} vs nx*ny*nz {int(np.prod(dims))}; geometry dtype {geo.get('dtype')} (uint8 -> 1 byte/voxel, no header)")
        m = re.match(r"phantom_.*?_(\d+)_(\d+)_(\d+)_", lab.name)
        fd = tuple(map(int, m.groups())) if m else None
        vox = (geo["sx_mm"], geo["sy_mm"], geo["sz_mm"])
        ok = (fd is None or fd == dims) and len(set(vox)) == 1 and vox[0] > 0
        put("V2", ok, f"geometry dims {dims}, filename dims {fd}, voxel mm {vox} (isotropic={len(set(vox)) == 1}); "
                      "voxel size is unverifiable against source (assumed 0.25 mm)")

    # tables
    tab = None
    try:
        tab, hdr = xlsx_table_path(xl) if xl else (None, None)
    except AssertionError as e:
        put("V5", False, f"xlsx parse: {e}")
    if tab is not None:
        wb = openpyxl.load_workbook(xl)
        ids = list(tab)
        probs = []
        if wb.sheetnames != ["Sheet1"] and len(wb.sheetnames) != 1:
            probs.append(f"sheets {wb.sheetnames}")
        if hdr != HDR:
            probs.append("header mismatch")
        if ids != sorted(set(ids)):
            probs.append("Organ IDs not unique ascending")
        if not ids or ids[0] != 0 or tab[0]["name"] != "air":
            probs.append("first row is not air/ID 0")
        if any(" " in r["name"] for r in tab.values()):
            probs.append("names with spaces")
        put("V5", not probs, "; ".join(probs) or f"{len(ids)} rows, header exact, ascending unique IDs, air first, no spaces in names")
        # V6
        probs, off = [], []
        nnz = {int(r[-2]): r[-3] for r in wb.active.iter_rows(min_row=2, values_only=True) if r[0] is not None}
        for oid, r in tab.items():
            f = list(r["fractions"].values())
            if min(f) < 0:
                probs.append(f"{oid}: negative fraction")
            if sum(v != 0 for v in f) != nnz[oid]:
                probs.append(f"{oid}: n-nonzero")
            if not 0.999 <= sum(f) <= 1.001:
                off.append(f"{r['name']}={sum(f):.4f}")
        put("V6", not probs, "; ".join(probs) or f"fractions >= 0; sums outside [0.999,1.001] (source-known, flagged): {sorted(set(off))}")
        # V7
        bad = [o for o, r in tab.items() if not r["density"] > 0]
        air = tab[0]["density"] if 0 in tab else None
        put("V7", not bad and air == AIR_DENSITY, f"nonpositive density IDs {bad}; air density {air} (expect {AIR_DENSITY}); "
            "train-range check not done (no train ranges supplied)")
        # V8
        n2m, m2c, probs = {}, {}, []
        for oid, r in tab.items():
            ck = comp_key(r["fractions"])
            if n2m.setdefault(r["name"], r["mc_id"]) != r["mc_id"]:
                probs.append(f"name {r['name']} has 2 MC IDs")
            if m2c.setdefault(r["mc_id"], ck) != ck:
                probs.append(f"MC ID {r['mc_id']} has 2 compositions")
        # name->composition: a name must have one composition
        n2c = {}
        for r in tab.values():
            if n2c.setdefault(r["name"], comp_key(r["fractions"])) != comp_key(r["fractions"]):
                probs.append(f"name {r['name']} has 2 compositions")
        first = {}
        for r in tab.values():
            first.setdefault(r["mc_id"], r["name"])
        miss = [n for n in first.values() if cat["mcgpu_files"] and n not in cat["mcgpu_files"]]
        put("V8", not probs and not miss, "; ".join(probs) or f"name->MC->composition consistent; MC block names without .mcgpu in the shared 17-file catalogue: {miss}")
        # V9
        probs = []
        for oid, r in tab.items():
            c = cat["ids"].get(str(oid))
            if not c:
                probs.append(f"ID {oid} not in catalogue")
            elif (c["name"], c["mc_id"]) != (r["name"], r["mc_id"]) or \
                    comp_key(dict(zip(ZS, cat["comps"][c["comp"]]["fractions"]))) != comp_key(r["fractions"]):
                probs.append(f"ID {oid} name/MC/composition differs from catalogue")
        put("V9", not probs, "; ".join(probs[:8]) or f"{len(tab)} IDs match the 53-ID COPD catalogue (name, MC ID, composition)")
        # V10
        # ladder ratios taken from the catalogue (source subject), exact to ~1e-14 within a subject per the contract
        lad = {n: (range(a, b + 1), [cat["ids"][str(i)]["density"] / cat["ids"][str(a)]["density"] for i in range(a, b + 1)])
               for n, (a, b) in {"parenchyma 26-38": (26, 38), "spine 39-43": (39, 43), "ribs 45-49": (45, 49),
                                 "scapula 50-54": (50, 54), "sternum 55-59": (55, 59)}.items()}
        probs = []
        for n, (rg, rat) in lad.items():
            if all(i in tab for i in rg):
                base = tab[rg[0]]["density"]
                bad = [i for i, q in zip(rg, rat) if abs(tab[i]["density"] / base / q - 1) > 1e-9]
                probs += [f"{n}: ID {i} off ladder" for i in bad]
        if all(i in tab for i in range(106, 111)) and len({tab[i]["density"] for i in range(106, 111)}) != 1:
            probs.append("lung 106-110 densities unequal")
        put("V10", not probs, "; ".join(probs) or "ladder ratios (rel 1e-9 vs catalogue ratios) and lung 106-110 equality hold")
    else:
        for v in ("V5", "V6", "V7", "V8", "V9", "V10"):
            res.setdefault(v, dict(status="FAIL", reason="no/ambiguous xlsx"))

    # V3 (streams labels)
    if lab and tab is not None and nmat:
        cnt = np.zeros(256, np.int64)
        mm = np.memmap(lab, np.uint8, "r")
        for i in range(0, mm.size, CHUNK):
            cnt += np.bincount(mm[i:i + CHUNK], minlength=256)
        present = np.flatnonzero(cnt)
        notin = [int(v) for v in present if int(v) not in tab]
        put("V3", not notin and int(present.max()) <= nmat - 1,
            f"{len(present)} distinct labels, max {int(present.max())}, NumberOfMaterials {nmat}; labels not in table: {notin}")
    else:
        put("V3", False, "needs label file, xlsx and geometry")
    put("V4", None, "needs a source reference volume; done by the exporter round-trip self-test, not by validate_package")

    # V11 / V12
    if tab is not None and inn:
        txt = inn.read_bytes()
        put("V11", txt == conv_text(tab).encode(), "byte comparison of .in against make_conv_table text regenerated from the xlsx")
    else:
        put("V11", False, "missing .in or xlsx")
    if tab is not None and lacf:
        raw = lacf.stat().st_size
        want = 4 * N_E * (nmat or 0)
        if raw != want or not nmat:
            put("V12", False, f"LAC size {raw} != 4*120*NumberOfMaterials {want}")
        else:
            lac = np.fromfile(lacf, "<f4").reshape(nmat, N_E).astype(np.float64)
            probs, worst = [], 0.0
            for oid, r in tab.items():
                c = cat["comps"].get(cat["ids"].get(str(oid), {}).get("comp"))
                if c and oid < nmat and r["density"] > 0:
                    e = np.max(np.abs(lac[oid] / r["density"] / np.array(c["mu_rho"]) - 1))
                    worst = max(worst, e)
                    if e > 1e-5:
                        probs.append(f"ID {oid} rel {e:.2e}")
            unused = [k for k in range(nmat) if k not in tab]
            if unused and not np.array_equal(lac[unused], np.broadcast_to(lac[0], (len(unused), N_E))):
                probs.append("unused rows differ from air row")
            put("V12", not probs, "; ".join(probs[:8]) or f"size ok; max rel |LAC/rho / catalogue mu_rho - 1| = {worst:.2e} (<1e-5); {len(unused)} unused rows == air")
    else:
        put("V12", False, "missing LAC or xlsx")
    put("V13", None, "needs the source package for the same subject; done by the self-test round trip")
    put("V14", None, "needs a recon at 70 keV to compare against (mono HU definition); not checkable standalone")
    return dict(sorted(res.items(), key=lambda kv: int(kv[0][1:])))


if __name__ == "__main__":
    out = validate_package(sys.argv[1])
    print(json.dumps(out, indent=1))
    sys.exit(any(v["status"] == "FAIL" for v in out.values()))
