"""E00 stage 1: account for every COPD subject and its source files (metadata only, no volumes).

Writes OUT/inventory.json. Nothing is assumed to be 67: counts are reported as found.
"""
import json
import os
import sys

import numpy as np
import pydicom

from common import OUT, PH, SRC, conv_table, pairing, phantom_file, sha256, xlsx_table


def dcm_header(path):
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    return {e.keyword or str(e.tag): str(e.value)[:200] for e in ds if e.VR != "SQ"} | {
        "TransferSyntaxUID": str(ds.file_meta.get("TransferSyntaxUID", ""))}


def subject(sid, series):
    r = dict(id=sid, series=series, problems=[])
    try:
        pf, shape = phantom_file(sid)
        r["phantom"] = dict(file=pf.name, shape_zyx=shape, bytes=pf.stat().st_size,
                            size_matches_dims=pf.stat().st_size == int(np.prod(shape)))
        if not r["phantom"]["size_matches_dims"]:
            r["problems"].append("phantom size != product of filename dims")
    except Exception as e:
        r["problems"].append(f"phantom: {e!r}")
    try:
        tab, hdr = xlsx_table(sid)
        conv = conv_table(sid)
        r["xlsx_header"] = hdr
        r["table"] = tab
        r["table_sha256"] = sha256(PH / f"material_tables/challenge_materials_copd_smeared_{sid}.xlsx")
        bad = [k for k in set(tab) | set(conv) if k not in tab or k not in conv
               or abs(tab[k]["density"] - conv[k][2]) > 1e-6 * max(1, conv[k][2])
               or tab[k]["name"] != conv[k][0] or tab[k]["mc_id"] not in (None, conv[k][1])]
        r["xlsx_vs_conv_mismatch_ids"] = sorted(bad)
        fsum = {k: sum(v["fractions"].values()) for k, v in tab.items()}
        r["fraction_sum_range"] = [min(fsum.values()), max(fsum.values())]
        lac = PH / f"material_lac/{sid}/lin_att_120kV_121materials.bin"
        r["lac"] = dict(bytes=lac.stat().st_size, sha256=sha256(lac))
    except Exception as e:
        r["problems"].append(f"tables: {e!r}")
    try:
        d = SRC / f"recons/dcm_recons_copd/dcm_recon_{sid}"
        files = sorted(f for f in os.listdir(d) if f.endswith(".dcm"))
        sizes = {os.path.getsize(d / f) for f in files}
        r["recon"] = dict(n_slices=len(files), n_other=len(os.listdir(d)) - len(files),
                          first=files[0], last=files[-1], file_sizes=sorted(sizes),
                          header_first=dcm_header(d / files[0]),
                          header_last=dcm_header(d / files[-1]))
    except Exception as e:
        r["problems"].append(f"recon: {e!r}")
    try:
        d = SRC / f"projections/dcmproj_copd/dcm_{series}"
        names = os.listdir(d)
        proj = sorted(f for f in names if f.startswith("proj_"))
        mas = [f for f in names if f.startswith("mAs_vector")]
        r["projections"] = dict(n_views=len(proj), mas_files=mas,
                                mas_bytes=[os.path.getsize(d / f) for f in mas],
                                header_first=dcm_header(d / proj[0]))
    except Exception as e:
        r["problems"].append(f"projections: {e!r}")
    return r


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = pairing()
    ids = [p[0] for p in pairs]
    on_disk = dict(
        phantom_ids=sorted(f.name.split("_")[1] for f in PH.glob("phantom_*.bin")),
        recon_ids=sorted(d.replace("dcm_recon_", "") for d in os.listdir(SRC / "recons/dcm_recons_copd")),
        table_ids=sorted(f.stem.rsplit("_", 1)[1] for f in (PH / "material_tables").glob("*.xlsx")),
        lac_ids=sorted(os.listdir(PH / "material_lac")),
        series=sorted(os.listdir(SRC / "projections/dcmproj_copd")),
        material_mcgpu=sorted(os.listdir(PH / "material_mcgpu")),
        copd_dir_other=sorted(f for f in os.listdir(PH) if not f.startswith("phantom_")),
    )
    subs = []
    for i, (sid, ser) in enumerate(pairs):
        subs.append(subject(sid, ser))
        print(i, sid, subs[-1]["problems"], flush=True)
    out = dict(
        n_pairing_rows=len(pairs), n_unique_ids=len(set(ids)), n_unique_series=len({p[1] for p in pairs}),
        id_sets_equal={k: sorted(set(ids) ^ set(v)) for k, v in on_disk.items() if k.endswith("_ids")},
        series_unmapped=sorted(set(on_disk["series"]) - {f"dcm_{p[1]}" for p in pairs}),
        on_disk=on_disk, subjects=subs)
    json.dump(out, open(OUT / "inventory.json", "w"), indent=1)
    print("wrote", OUT / "inventory.json", "subjects with problems:",
          [s["id"] for s in subs if s["problems"]], file=sys.stderr)
