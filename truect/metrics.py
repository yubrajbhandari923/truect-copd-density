"""Per-subject metrics on the shared evaluation region (truect.data.eval_mask). Used by every comparator.

subject_metrics(t, pred_density=None, pred_cls=None) walks slices so a whole volume never sits in float64.
Density: MAE, RMSE, bias (pred - true) in g/cm3, overall and split pure/boundary (purity == 255) and per true class.
Material: per-class Dice and macro Dice over classes present in the truth region.
"""
import numpy as np

from truect.data import CLASSES, eval_mask

C = len(CLASSES)


def subject_metrics(t, pred_density=None, pred_cls=None):
    n_slices = t["valid"].shape[0]
    z = lambda: np.zeros(C)
    acc = dict(n=z(), ae=z(), se=z(), err=z(), n_pure=0, ae_pure=0.0, n_edge=0, ae_edge=0.0,
               tp=z(), n_true=z(), n_pred=z())
    for i in range(n_slices):
        m = eval_mask(t, i)
        if not m.any():
            continue
        cls = np.asarray(t["cls"][i])[m]
        assert (cls < C).all(), "class 255 inside the evaluation region"
        acc["n"] += np.bincount(cls, minlength=C)
        if pred_density is not None:
            e = np.asarray(pred_density[i], np.float64)[m] - np.asarray(t["density"][i], np.float64)[m]
            acc["ae"] += np.bincount(cls, np.abs(e), C)
            acc["se"] += np.bincount(cls, e * e, C)
            acc["err"] += np.bincount(cls, e, C)
            pure = np.asarray(t["purity"][i])[m] == 255
            acc["n_pure"] += int(pure.sum()); acc["ae_pure"] += float(np.abs(e[pure]).sum())
            acc["n_edge"] += int((~pure).sum()); acc["ae_edge"] += float(np.abs(e[~pure]).sum())
        if pred_cls is not None:
            p = np.asarray(pred_cls[i])[m]
            acc["tp"] += np.bincount(cls[p == cls], minlength=C)
            acc["n_true"] += np.bincount(cls, minlength=C)
            acc["n_pred"] += np.bincount(p[p < C], minlength=C)
    N = acc["n"].sum()
    out = dict(n_voxels=int(N))
    if pred_density is not None:
        out.update(density_mae=acc["ae"].sum() / N, density_rmse=float(np.sqrt(acc["se"].sum() / N)),
                   density_bias=acc["err"].sum() / N,
                   density_mae_pure=acc["ae_pure"] / max(acc["n_pure"], 1),
                   density_mae_boundary=acc["ae_edge"] / max(acc["n_edge"], 1),
                   density_mae_by_class={CLASSES[c]: acc["ae"][c] / acc["n"][c] for c in range(C) if acc["n"][c]})
    if pred_cls is not None:
        present = acc["n_true"] > 0
        dice = 2 * acc["tp"] / np.maximum(acc["n_true"] + acc["n_pred"], 1)
        out.update(dice_by_class={CLASSES[c]: float(dice[c]) for c in range(C) if present[c]},
                   macro_dice=float(dice[present].mean()))
    return {k: (float(v) if isinstance(v, np.floating) else v) for k, v in out.items()}
