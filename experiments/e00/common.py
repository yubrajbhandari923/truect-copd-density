"""E00 shared paths and source parsers. Source tree is READ-ONLY."""
import hashlib
import re
from pathlib import Path

import numpy as np
import openpyxl

SRC = Path("/scratch/railabs/yb107/Datasets/TrueCT")
WORK = Path("/scratch/railabs/yb107/research-output/truect-copd-density")
OUT = WORK / "results/e00"
PH = SRC / "phantoms/COPD"
EXPOSED = {"10313"}  # development-exposed, never blinded test


def pairing():
    """[(subject_id, projection_series_idx)] from the CRLF index map, order preserved."""
    txt = (SRC / "pairing/copd_index_map.txt").read_text().replace("\r", "")
    return [tuple(l.split(",")) for l in txt.split("\n") if l.strip()]


def phantom_file(sid):
    """(path, array shape (z,y,x)). Filename gives (a,b,c); array is C-order [c,b,a]."""
    hits = sorted(PH.glob(f"phantom_{sid}_*.bin"))
    assert len(hits) == 1, (sid, hits)
    a, b, c = map(int, re.match(rf"phantom_{sid}_(\d+)_(\d+)_(\d+)_", hits[0].name).groups())
    return hits[0], (c, b, a)


def xlsx_table(sid):
    """{organ_id: {name, density, mc_id, fractions{Z: mass %}}} plus the raw header."""
    return xlsx_table_path(PH / f"material_tables/challenge_materials_copd_smeared_{sid}.xlsx")


def xlsx_table_path(p):
    rows = [r for r in openpyxl.load_workbook(p, data_only=True).active.iter_rows(values_only=True)
            if r and r[0] is not None]
    hdr = list(rows[0])
    i_d, i_o = hdr.index("Density"), hdr.index("Organ ID")
    i_m = hdr.index("MC ID") if "MC ID" in hdr else None
    tab = {}
    for r in rows[1:]:
        oid = int(r[i_o])
        assert oid not in tab, (sid, "duplicate organ id", oid)
        tab[oid] = dict(name=str(r[0]), density=float(r[i_d]),
                        mc_id=None if i_m is None else int(r[i_m]),
                        fractions={str(hdr[j]): float(r[j] or 0) for j in range(1, i_d)})
    return tab, [str(h) for h in hdr]


def conv_table(sid):
    """{organ_id: (name, mc_id, density)} from conversion_table_120kV_121materials.in."""
    out = {}
    for line in (PH / f"material_lac/{sid}/conversion_table_120kV_121materials.in").read_text().splitlines():
        if line.startswith("#[Monte"):
            break
        f = line.split()
        if len(f) == 4 and f[0].isdigit():
            out[int(f[0])] = (f[1], int(f[2]), float(f[3]))
    return out


def density_lut(sid):
    lut = np.full(256, np.nan, np.float32)
    for k, v in xlsx_table(sid)[0].items():
        lut[k] = v["density"]
    return lut


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()
