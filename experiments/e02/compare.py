"""E02 validation comparison (10 val subjects): per-run medians, seed spread, paired subject-level differences with
95% bootstrap CIs (10k resamples of subjects). Inputs: results/e02/<run>/val_<step>.json and reports/e01/validation.json.
Usage: python compare.py <results_e02_dir> <e01_validation.json> <out.json>
"""
import json, sys
from pathlib import Path

import numpy as np

R, E01, OUT = Path(sys.argv[1]), json.load(open(sys.argv[2])), Path(sys.argv[3])
KEYS = ["density_mae", "density_mae_pure", "density_mae_boundary", "material_macro_dice", "macro_dice"]
CLS = ["emphysema", "lung_parenchyma", "fissure", "vessel", "spine_cortical", "ribs_cortical"]


def runs():
    out = {}
    for d in sorted(R.iterdir()):
        v = sorted(d.glob("val_*.json"), key=lambda p: int(p.stem[4:]))
        if v and not d.name.startswith(("smoke", "superseded")):
            out[d.name] = (int(v[-1].stem[4:]), json.load(open(v[-1])))
    for b, var in E01["chosen_variant"].items():
        out[f"E01_{b}"] = (0, {s: m[var] for s, m in E01["per_subject"].items()})
    return out


def flat(m):
    f = {k: m[k] for k in KEYS if k in m}
    f |= {f"mae_{c}": m["density_mae_by_class"][c] for c in CLS if c in m.get("density_mae_by_class", {})}
    f |= {f"dice_{c}": m["dice_by_class"][c] for c in CLS if c in m.get("dice_by_class", {})}
    return f


def paired(a, b, key, rng):
    s = sorted(set(a) & set(b))
    d = np.array([flat(a[x])[key] - flat(b[x])[key] for x in s])
    bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    return dict(n=len(d), mean_diff=float(d.mean()), ci95=np.percentile(bs, [2.5, 97.5]).tolist(),
                n_a_better=int((d < 0).sum() if "mae" in key else (d > 0).sum()))


if __name__ == "__main__":
    rr = runs()
    table = {}
    for name, (step, res) in rr.items():
        F = [flat(m) for m in res.values()]
        table[name] = dict(step=step, n_subjects=len(F), **{k: float(np.median([f[k] for f in F if k in f]))
                                                            for k in sorted(set().union(*F))})
    groups = {}
    for name in rr:
        if "_s" in name and not name.startswith("E01"):
            groups.setdefault(name.rsplit("_s", 1)[0], []).append(name)
    seeds = {g: {k: [float(np.mean([table[n][k] for n in ns])), float(np.std([table[n][k] for n in ns], ddof=1))]
                 for k in KEYS if all(k in table[n] for n in ns)}
             for g, ns in groups.items() if len(ns) > 1}
    rng = np.random.default_rng(0)
    C = [("u3_s11", "u3f_s11", "density_mae"), ("u3_s11", "u3f_s11", "material_macro_dice"),
         ("u2_s11", "u2f_s11", "material_macro_dice"), ("u1_s11", "u1f_s11", "density_mae"),
         ("u3f_s11", "u1f_s11", "density_mae"), ("u3f_s11", "u2f_s11", "material_macro_dice"),
         ("u4f_s11", "u3f_s11", "density_mae"), ("u4f_s11", "u3f_s11", "material_macro_dice"),
         ("u3_s11", "E01_B3", "density_mae"), ("u3_s11", "E01_B4", "material_macro_dice"),
         ("u5_s11", "u3_s11", "density_mae"), ("u5_s11", "u3_s11", "material_macro_dice")]
    C += [("u4f_s11", "u3f_s11", f"mae_{c}") for c in CLS] + [("u5_s11", "u3_s11", f"dice_{c}") for c in CLS]
    pairs = {f"{a} - {b}: {k}": paired(rr[a][1], rr[b][1], k, rng) for a, b, k in C if a in rr and b in rr}
    json.dump(dict(medians=table, seed_mean_sd_of_medians=seeds, paired=pairs), open(OUT, "w"), indent=1)
    cols = ["step"] + KEYS
    print("run".ljust(12), *[c[:12].rjust(12) for c in cols])
    for n, t in table.items():
        print(n.ljust(12), *[(f"{t[c]:.4f}" if isinstance(t.get(c), float) else str(t.get(c, "-"))).rjust(12) for c in cols])
    print("\nseed mean ± sd of medians"); [print(g, {k: f"{m:.4f}±{s:.4f}" for k, (m, s) in v.items()}) for g, v in seeds.items()]
    print("\npaired (a - b), mean over subjects, 95% bootstrap CI, #subjects where a is better")
    for k, v in pairs.items():
        print(f"{k:55s} {v['mean_diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] {v['n_a_better']}/{v['n']}")
