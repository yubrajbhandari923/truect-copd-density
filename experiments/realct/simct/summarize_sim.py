"""Aggregate simct metrics: per arm (real CT, ideal 70 keV of prediction, sim noiseless, sim noisy) the median over cases
of each region's median HU, bias vs real CT, body MAE, lung QIBs and noise. usage: summarize_sim.py <metrics_dir> <out_dir>
"""
import csv, json, sys
from pathlib import Path

import numpy as np

M, OUT = Path(sys.argv[1]), Path(sys.argv[2])
R = [json.load(open(f)) for f in sorted(M.glob("*.json"))]
ARMS = ["gt", "ideal70", "sim_noiseless", "sim_noisy"]
rows = []
for r in R:
    for a in ARMS:
        x = r[a]
        row = dict(series=r["series"], model=r["model"], arm=a, **{k: v for k, v in x.items() if isinstance(v, float)})
        row |= {f"lung_{k}": v for k, v in (x.get("lung_qib") or {}).items()}
        rows.append(row)
keys = sorted({k for r in rows for k in r} - {"series", "model", "arm"})
med = {a: {k: [round(float(np.percentile(v, p)), 1) for p in (25, 50, 75)] for k in keys
           if (v := [r[k] for r in rows if r["arm"] == a and r.get(k) is not None])} for a in ARMS}
OUT.mkdir(parents=True, exist_ok=True)
json.dump(dict(n_cases=len(R), model=R[0]["model"], p25_p50_p75=med), open(OUT / "simct_summary.json", "w"), indent=1)
with open(OUT / "simct_per_case.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["series", "model", "arm"] + keys); w.writeheader(); w.writerows(rows)
show = [k for k in keys if k.endswith("median_hu") or k.startswith(("body_mae_hu", "lung_mld", "lung_perc15", "lung_laa950", "noise"))]
print(f"{len(R)} cases; median over cases (IQR in json)")
print("metric".ljust(26), *[a.rjust(14) for a in ARMS])
for k in show:
    print(k.ljust(26), *[(str(med[a][k][1]) if k in med[a] else "-").rjust(14) for a in ARMS])
