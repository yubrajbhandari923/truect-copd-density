"""Deterministic subject split, stratified on emphysema burden (default pending questions.md Q1).

usage: make_split.py subjects_67.json out.json [excluded_id ...]
Subjects sorted by emphysema fraction are cut into blocks of 6; within each block a seeded
shuffle assigns members so the overall sizes are 46/10/11 for N=67. Exposed subjects go to train.
"""
import json
import random
import sys

SEED, TARGET = 20261112, dict(train=46, val=10, test=11)


def split(subjects, excluded=()):
    S = [s for s in subjects if s["id"] not in excluded]
    exposed = [s["id"] for s in S if s["development_exposed"]]
    pool = sorted((s for s in S if not s["development_exposed"]),
                  key=lambda s: (s["emphysema_voxel_frac_of_lung"], s["id"]))
    n = len(S)
    n_test = round(n * TARGET["test"] / 67)
    n_val = round(n * TARGET["val"] / 67)
    rng = random.Random(SEED)
    out = dict(train=list(exposed), val=[], test=[])
    # one test and one val pick per stratum, strata of equal size along the sorted burden axis
    k = len(pool) / n_test
    strata = [pool[round(i * k):round((i + 1) * k)] for i in range(n_test)]
    val_strata = set(rng.sample(range(n_test), n_val))
    for i, st in enumerate(strata):
        ids = [s["id"] for s in st]
        rng.shuffle(ids)
        out["test"].append(ids.pop())
        if i in val_strata:
            out["val"].append(ids.pop())
        out["train"] += ids
    return {k_: sorted(v) for k_, v in out.items()}


if __name__ == "__main__":
    subs = json.load(open(sys.argv[1]))
    sp = split(subs, set(sys.argv[3:]))
    ids = [i for v in sp.values() for i in v]
    assert len(ids) == len(set(ids)) == len(subs) - len(sys.argv[3:])
    assert not {s["id"] for s in subs if s["development_exposed"]} & set(sp["val"] + sp["test"])
    frac = {s["id"]: s["emphysema_voxel_frac_of_lung"] for s in subs}
    summ = {k: dict(n=len(v), emph_frac_min=min(frac[i] for i in v), emph_frac_median=sorted(frac[i] for i in v)[len(v) // 2],
                    emph_frac_max=max(frac[i] for i in v)) for k, v in sp.items()}
    json.dump(dict(seed=SEED, stratified_on="emphysema_voxel_frac_of_lung", excluded=sys.argv[3:], balance=summ, **sp),
              open(sys.argv[2], "w"), indent=1)
    print(json.dumps(summ, indent=1))
