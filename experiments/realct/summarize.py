"""Summarise real-CT inference metrics: per-model medians and IQR over cases, per-case CSV.
Usage: python summarize.py <metrics_dir> <out_dir>
"""
import csv, json, sys
from pathlib import Path

import numpy as np

M, OUT = Path(sys.argv[1]), Path(sys.argv[2])
R = [json.load(open(f)) for f in sorted(M.glob("*.json"))]
MODELS = [k for k in R[0] if isinstance(R[0][k], dict) and "step" in R[0][k]]
rows = []
for r in R:
    for m in MODELS:
        x = r[m]
        rows.append(dict(series=r["series"], model=m, step=x["step"], laa950=r["laa950"], shape_06mm="x".join(map(str, r["shape_06mm"])),
                         orientation_ok=all(r["orientation_check"].values()) if r["orientation_check"] else None,
                         **{k: v for k, v in x.items() if isinstance(v, float) and k != "sec"},
                         **{f"dice_ts_{k}": v for k, v in x.get("dice_vs_totalseg", {}).items()}))
keys = sorted({k for r in rows for k in r} - {"series", "model", "step", "shape_06mm", "orientation_ok"})
q = lambda v: [round(float(np.percentile(v, p)), 4) for p in (25, 50, 75)] if v else None
summary = dict(n_cases=len(R),
               orientation={k: f"{sum(r['orientation_check'].get(k, False) for r in R)}/{sum(k in r['orientation_check'] for r in R)}"
                            for k in ("sternum_anterior", "desc_aorta_left")},
               per_model_p25_p50_p75={m: {k: q([r[k] for r in rows if r["model"] == m and r.get(k) is not None]) for k in keys} for m in MODELS})
OUT.mkdir(parents=True, exist_ok=True)
json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
with open(OUT / "per_case.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["series", "model", "step", "shape_06mm", "orientation_ok"] + keys); w.writeheader(); w.writerows(rows)
print(json.dumps(summary["orientation"]))
for m in MODELS:
    s = summary["per_model_p25_p50_p75"][m]
    print(m, {k: s[k][1] for k in keys if s[k] and (k.startswith(("dice_ts", "density_median", "bone_frac", "emph", "lung_")))})
