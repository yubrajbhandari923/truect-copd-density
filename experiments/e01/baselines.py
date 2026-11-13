"""E01 intensity-only baselines (B0-B4): fit on TRAIN voxel samples, select filter on VALIDATION, freeze LUTs.

Usage: python baselines.py OUT_DIR [--smoke]   (one Slurm job; parallel over subjects with multiprocessing)
Every baseline is a 1-D lookup on (optionally Gaussian-smoothed) CT, so it is frozen as a small table:
  B0  reference: rho = max(1 + HU/1000, 0.0012); not fitted (CT HU only approximately calibrated, see README)
  B1  isotonic (monotone) fit of per-bin median density vs raw CT
  B2  MLP (2x32 ReLU, squared loss) density from raw CT; tabulated exactly on every integer HU
  B3  B1 recipe on Gaussian-smoothed CT, sigma picked on validation (median subject density MAE)
  B4  per-bin majority class vs CT (raw or smoothed), sigma picked on validation (median subject macro Dice)
"""
import json, os, socket, sys, time
from multiprocessing import Pool

import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.isotonic import IsotonicRegression
from sklearn.neural_network import MLPRegressor

from truect.data import CLASSES, eval_mask, load, split
from truect.metrics import subject_metrics

SEED = 20261113
LO, HI, W = -1024, 3071, 4          # CT clip range and LUT bin width (HU)
CENTERS = LO + W * (np.arange((HI - LO + 1) // W) + 0.5)
SIGMAS = [0.5, 1.0, 1.5, 2.0, 3.0]  # Gaussian sigma in voxels (0.6 mm isotropic), 3-D
SMOKE = "--smoke" in sys.argv
N_PER_SUBJECT = 20_000 if SMOKE else 300_000
C = len(CLASSES)


def ct_and_smoothed(t):
    ct = np.clip(np.asarray(t["ct"], np.float32), LO, HI)
    yield 0.0, ct
    for s in SIGMAS:
        yield s, gaussian_filter(ct, s, mode="nearest")


def sample_train(sid):
    t, _ = load(sid)
    idx = np.flatnonzero(eval_mask(t))
    # shortcut: sampling with replacement (cheap on ~2e8 voxels); duplicates are negligible at 3e5 draws
    idx = np.sort(np.random.default_rng([SEED, int(sid)]).integers(0, len(idx), N_PER_SUBJECT) if len(idx) else idx)
    out = dict(rho=np.asarray(t["density"]).ravel()[idx].astype(np.float32),
               cls=np.asarray(t["cls"]).ravel()[idx])
    for s, x in ct_and_smoothed(t):
        out[s] = x.ravel()[idx]
    return out


def bin_of(x):
    return np.clip(((x - LO) // W).astype(np.int64), 0, len(CENTERS) - 1)


def fit_density_lut(x, rho):
    b = bin_of(x)
    o = np.lexsort((rho, b)); b, rho = b[o], rho[o]
    ub, start, n = np.unique(b, return_index=True, return_counts=True)
    med = rho[start + (n - 1) // 2]
    iso = IsotonicRegression(increasing=True).fit(CENTERS[ub], med, sample_weight=n)
    lut = np.interp(CENTERS, CENTERS[ub], iso.predict(CENTERS[ub]))
    assert (np.diff(lut) >= 0).all()
    return lut


def fit_class_lut(x, cls):
    counts = np.bincount(bin_of(x) * C + cls, minlength=len(CENTERS) * C).reshape(-1, C)
    full = np.flatnonzero(counts.sum(1))
    nearest = full[np.abs(np.arange(len(CENTERS))[:, None] - full[None]).argmin(1)]  # empty bins: nearest filled bin
    return counts[nearest].argmax(1).astype(np.uint8)


def fit_mlp(x, rho):
    sel = np.random.default_rng(SEED).choice(len(x), min(len(x), 1_000_000), replace=False)
    m = MLPRegressor(hidden_layer_sizes=(32, 32), early_stopping=True, random_state=SEED, max_iter=200).fit(x[sel, None] / 1000, rho[sel])
    hu = np.arange(LO, HI + 1, dtype=np.float32)
    return m, m.predict(hu[:, None] / 1000).astype(np.float32)


def apply_density(lut, x):
    return np.interp(x, CENTERS, lut).astype(np.float32)


def evaluate_val(args):
    sid, fit = args
    t, _ = load(sid)
    res = {}
    for s, x in ct_and_smoothed(t):
        tag = "raw" if s == 0 else f"s{s}"
        m = subject_metrics(t, apply_density(fit["B1"][s], x), fit["B4"][s][bin_of(x)])
        res[f"B1_{tag}"] = {k: v for k, v in m.items() if not k.startswith(("dice", "macro"))}
        res[f"B4_{tag}"] = {k: v for k, v in m.items() if not k.startswith("density")}
        if s == 0:
            res["B0"] = subject_metrics(t, np.maximum(1 + x / 1000, 0.0012).astype(np.float32))
            res["B2"] = subject_metrics(t, fit["B2"][x.astype(np.int64) - LO])
    return sid, res


def summary(per_subject, method):
    out = {}
    for k in ["density_mae", "density_rmse", "density_bias", "density_mae_pure", "density_mae_boundary", "macro_dice"]:
        v = [per_subject[s][method][k] for s in per_subject if k in per_subject[s][method]]
        if v:
            out[k] = dict(median=float(np.median(v)), min=float(np.min(v)), max=float(np.max(v)), mean=float(np.mean(v)))
    for k in ["density_mae_by_class", "dice_by_class"]:
        if k in next(iter(per_subject.values()))[method]:
            out[k + "_median"] = {c: float(np.median([d[method][k][c] for d in per_subject.values() if c in d[method][k]]))
                                  for c in CLASSES if any(c in d[method][k] for d in per_subject.values())}
    return out


def main():
    out_dir = sys.argv[1]; os.makedirs(out_dir, exist_ok=True)
    sp = split(); train, val = sp["train"], sp["val"]
    if SMOKE:
        train, val = train[:2], val[:1]
    nproc = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    t0 = time.time()
    with Pool(min(nproc, len(train))) as p:
        samples = p.map(sample_train, train)
    S = {k: np.concatenate([d[k] for d in samples]) for k in samples[0]}
    print(f"sampled {len(S['rho'])} train voxels in {time.time() - t0:.0f}s", flush=True)

    fit = dict(B1={}, B4={})
    for s in [0.0] + SIGMAS:
        fit["B1"][s] = fit_density_lut(S[s], S["rho"])
        fit["B4"][s] = fit_class_lut(S[s], S["cls"])
    mlp, fit["B2"] = fit_mlp(S[0.0], S["rho"])
    print(f"fitted in {time.time() - t0:.0f}s; MLP iters {mlp.n_iter_}", flush=True)

    with Pool(min(nproc, len(val))) as p:
        per_subject = dict(p.map(evaluate_val, [(sid, fit) for sid in val]))
    print(f"validated in {time.time() - t0:.0f}s", flush=True)

    tags = ["raw"] + [f"s{s}" for s in SIGMAS]
    med = lambda m, k: np.median([per_subject[s][m][k] for s in val])
    b3 = min(tags[1:], key=lambda g: med(f"B1_{g}", "density_mae"))
    b4 = min(tags, key=lambda g: -med(f"B4_{g}", "macro_dice"))
    sig = lambda g: 0.0 if g == "raw" else float(g[1:])
    chosen = {"B0": "B0", "B1": "B1_raw", "B2": "B2", "B3": f"B1_{b3}", "B4": f"B4_{b4}"}

    prov = dict(job=os.environ.get("SLURM_JOB_ID"), host=socket.gethostname(), code=os.environ.get("SNAPSHOT"),
                seed=SEED, n_per_subject=N_PER_SUBJECT, n_train_subjects=len(train), val_subjects=val,
                runtime_s=round(time.time() - t0), smoke=SMOKE)
    lut_grid = dict(lo=LO, hi=HI, bin_width=W, bin_centers_note="center_i = lo + bin_width*(i+0.5)",
                    ct_preprocessing="clip CT to [lo, hi] as float32; smoothing = scipy gaussian_filter(sigma voxels, mode=nearest), 3-D")
    frozen = dict(provenance=prov, grid=lut_grid,
                  B0=dict(formula="max(1 + HU/1000, 0.0012)"),
                  B1=dict(sigma=0.0, density_lut=fit["B1"][0.0].tolist(), apply="np.interp(ct, centers, lut)"),
                  B2=dict(density_lut_1hu=fit["B2"].tolist(), apply="lut[int(clip(ct)) - lo]",
                          mlp=dict(input="ct/1000", coefs=[c.tolist() for c in mlp.coefs_],
                                   intercepts=[c.tolist() for c in mlp.intercepts_], activation="relu")),
                  B3=dict(sigma=sig(b3), density_lut=fit["B1"][sig(b3)].tolist(), apply="np.interp(smooth(ct), centers, lut)"),
                  B4=dict(sigma=sig(b4), class_lut=fit["B4"][sig(b4)].tolist(), classes=CLASSES,
                          apply="lut[clip((smooth(ct) - lo) // bin_width, 0, nbins-1)]"))
    baselines = dict(provenance=prov, grid=lut_grid, sigma_grid_voxels=SIGMAS, voxel_mm=0.6, chosen_variant=chosen,
                     selection="B3: sigma minimising median val density MAE; B4: sigma (incl. raw) maximising median val macro Dice",
                     methods=__doc__, frozen_params="reports/e01/frozen_comparators.json")
    validation = dict(provenance=prov, chosen_variant=chosen,
                      summary={k: summary(per_subject, v) for k, v in chosen.items()},
                      all_variants_summary={m: summary(per_subject, m) for m in next(iter(per_subject.values()))},
                      per_subject=per_subject)
    for name, obj in [("baselines", baselines), ("validation", validation), ("frozen_comparators", frozen)]:
        json.dump(obj, open(f"{out_dir}/{name}.json", "w"), indent=1)
    print(json.dumps({k: {m: v[m]["median"] for m in ("density_mae", "macro_dice") if m in v}
                      for k, v in validation["summary"].items()}, indent=1))


if __name__ == "__main__":
    main()
