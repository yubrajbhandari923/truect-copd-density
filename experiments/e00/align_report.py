"""Aggregate OUT/alignment/<sid>.json -> OUT/alignment/alignment_validation.json + contact_sheet.png.

Flags are fixed rules written before looking at the cohort results (not tuned to make subjects pass):
orientation/axis letters != majority, full-fit r < 0.95 or a robust outlier (median - 3*1.4826*MAD),
3 mm runner-up margin < 0.05, any |scale-1| > 0.003, fine optimum not a local maximum under +-1 mm,
orientation rank changed by the coarse scale scan, anatomy checks inconsistent/weak, coverage fractions
robust outliers. Registration r is self-consistency of the same pair, not independent validation.
"""
import json
from collections import Counter

import numpy as np

from common import pairing
from align import D


def robust_lo(v, x):
    v = np.asarray(v, float)
    med, mad = np.median(v), 1.4826 * np.median(abs(v - np.median(v)))
    return mad > 0 and abs(x - med) > 3 * mad


def main():
    ids = [s for s, _ in pairing()]
    J = {s: json.load(open(D / f"{s}.json")) for s in ids if (D / f"{s}.json").exists()}
    missing = [s for s in ids if s not in J]
    key = lambda j: f"perm{j['orientation']['perm']}flips{j['orientation']['flips']}"
    maj = Counter(key(j) for j in J.values()).most_common(1)[0][0]
    maj_ax = Counter(j["anatomy"]["recon_axes_increasing"] for j in J.values()).most_common(1)[0][0]
    rs = [j["fine"]["full"]["r"] for j in J.values()]
    cf = lambda j, m, k: j["coverage"][m][k]
    rows = {}
    for s, j in J.items():
        f, an = j["fine"], j["anatomy"]
        sc = np.array(f["full"]["scale"])
        flags = []
        if key(j) != maj: flags.append("orientation_not_majority")
        if an["recon_axes_increasing"] != maj_ax: flags.append("anatomical_axes_not_majority")
        if f["full"]["r"] < 0.95: flags.append("r_below_0.95")
        if robust_lo(rs, f["full"]["r"]) and f["full"]["r"] < np.median(rs): flags.append("r_low_outlier")
        if j["runner_up"]["margin"] < 0.05: flags.append("runner_up_margin_below_0.05")
        if np.any(abs(sc - 1) > 0.003): flags.append("scale_dev_gt_0.3pct")
        if any(v > 0 for k, v in f["sensitivity_delta_r"].items() if k.startswith("shift")): flags.append("not_local_max_1mm")
        if j["coarse"]["rank_of_s1_best_at_scale"] != 0: flags.append("orientation_changed_by_scale_scan")
        if an["aorta_left_consistent_with_right_lung_larger"] is not True: flags.append("aorta_vs_lung_side_inconsistent")
        if an["lung_ratio_larger_over_smaller"] < 1.03: flags.append("lung_LR_ratio_weak_lt_1.03")
        if an["ap_dominance"] < 2: flags.append("ap_axis_ambiguous")
        if an["spine_elongation"] < 2: flags.append("si_axis_ambiguous")
        for m in range(3):
            for k in ("recon_frac_in_phantom", "phantom_frac_in_recon"):
                if robust_lo([cf(x, m, k) for x in J.values()], cf(j, m, k)): flags.append(f"coverage_outlier_axis{m}_{k}")
        rows[s] = dict(
            orientation=dict(perm=j["orientation"]["perm"], flips=j["orientation"]["flips"]),
            recon_axes_increasing=an["recon_axes_increasing"], phantom_axes_increasing=an["phantom_axes_increasing"],
            r_full=f["full"]["r"], r_translation_only_s1=f["translation_only_s1"]["r"],
            r_translation_only_sz_0p5476=f["translation_only_sz_0p5476"]["r"],
            r3mm=j["orientation"]["r3mm"], runner_up=j["runner_up"],
            A_mm=f["full"]["A_mm"], A_mm_translation_only_s1=f["translation_only_s1"]["A_mm"], scale=f["full"]["scale"],
            implied_recon_spacing_mm=f["implied_recon_spacing_mm_if_phantom_0p25"],
            coarse_scale=j["coarse"]["coarse_scale"], sensitivity_delta_r=f["sensitivity_delta_r"],
            coverage=j["coverage"], recon_shape=j["recon"]["shape"], phantom_shape_zyx=j["phantom_shape_zyx"],
            anatomy=an, flags=flags)
    arr = lambda f: np.array([f(r) for r in rows.values()])
    S, A = arr(lambda r: r["scale"]), arr(lambda r: r["A_mm"])
    summ = dict(
        n=len(rows), missing=missing, majority_orientation=maj,
        orientation_counts=Counter(key(j) for j in J.values()),
        recon_axes_counts=Counter(r["recon_axes_increasing"] for r in rows.values()),
        r_full_min_med_max=np.percentile(arr(lambda r: r["r_full"]), [0, 50, 100]).tolist(),
        margin_min_med_max=np.percentile(arr(lambda r: r["runner_up"]["margin"]), [0, 50, 100]).tolist(),
        scale_min_per_axis=S.min(0).tolist(), scale_median_per_axis=np.median(S, 0).tolist(), scale_max_per_axis=S.max(0).tolist(),
        A_mm_min=A.min(0).tolist(), A_mm_max=A.max(0).tolist(), A_mm_std=A.std(0).tolist(),
        n_flagged=sum(bool(r["flags"]) for r in rows.values()),
        flag_counts=Counter(f for r in rows.values() for f in r["flags"]),
        caution="r is intensity self-consistency of the registered pair (phantom 70 keV LAC vs recon HU), "
                "not independent validation; absolute L/R and S/I come from label anatomy assuming normal situs.")
    json.dump(dict(summary=summ, subjects=rows), open(D / "alignment_validation.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))
    contact_sheet(list(rows))


def contact_sheet(ids, ncol=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ims = [plt.imread(D / f"{s}.png")[::2, ::2] for s in ids if (D / f"{s}.png").exists()]
    h, w = max(i.shape[0] for i in ims), max(i.shape[1] for i in ims)
    nrow = -(-len(ims) // ncol)
    sheet = np.ones((nrow * h, ncol * w, 3), np.float32)
    for k, im in enumerate(ims):
        r, c = divmod(k, ncol)
        sheet[r * h: r * h + im.shape[0], c * w: c * w + im.shape[1]] = im[..., :3]
    plt.imsave(D / "contact_sheet.png", sheet)


if __name__ == "__main__":
    main()
