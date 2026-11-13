import numpy as np

from truect.metrics import CLASSES, subject_metrics


def fake(seed=0):
    r = np.random.default_rng(seed)
    t = dict(valid=np.ones((2, 4, 4), np.uint8), organ_pt=np.ones((2, 4, 4), np.uint8),
             cls=r.integers(0, 3, (2, 4, 4)).astype(np.uint8), purity=np.full((2, 4, 4), 255, np.uint8),
             density=r.random((2, 4, 4)).astype(np.float16))
    t["organ_pt"][0, 0, 0] = 0      # outside-body air: excluded
    t["valid"][1, 3, 3] = 0         # outside phantom: excluded
    t["cls"][1, 3, 3] = 255         # would assert if the mask leaked it
    return t


def test_perfect_prediction():
    t = fake()
    m = subject_metrics(t, t["density"], t["cls"])
    assert m["n_voxels"] == 30
    assert m["density_mae"] == 0 and m["density_rmse"] == 0 and m["macro_dice"] == 1


def test_known_errors():
    t = fake()
    m = subject_metrics(t, np.asarray(t["density"], np.float32) + 0.1, np.zeros_like(t["cls"]))
    assert abs(m["density_mae"] - 0.1) < 1e-3 and abs(m["density_bias"] - 0.1) < 1e-3
    mask = (t["valid"] == 1) & (t["organ_pt"] != 0)
    n0 = (t["cls"][mask] == 0).sum()
    assert abs(m["dice_by_class"][CLASSES[0]] - 2 * n0 / (n0 + mask.sum())) < 1e-9
    assert set(m["dice_by_class"]) == {CLASSES[c] for c in np.unique(t["cls"][mask])}
