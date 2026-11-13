import numpy as np

from truect.data import fov
from truect.metrics import CLASSES, subject_metrics

S = 16


def fake(seed=0):
    r = np.random.default_rng(seed)
    t = dict(valid=np.ones((2, S, S), np.uint8), organ_pt=np.ones((2, S, S), np.uint8),
             cls=r.integers(0, 3, (2, S, S)).astype(np.uint8), purity=np.full((2, S, S), 255, np.uint8),
             density=r.random((2, S, S)).astype(np.float16))
    t["organ_pt"][0, 8, 8] = 0      # outside-body air: excluded
    t["valid"][1, 7, 7] = 0         # outside phantom: excluded
    t["cls"][1, 7, 7] = 255         # would assert if the mask leaked it
    t["cls"][:, 0, 0] = 255         # corner is outside the field of view: would assert if leaked
    return t


def test_fov_is_a_disc():
    f = fov((S, S))
    assert f[8, 8] and f[0, 8] and not f[0, 0] and not f[15, 15]
    big = fov((768, 768))
    assert big[383, 10] and not big[383, 2] and not big[0, 0]  # radius 375 px about centre 383.5


def test_perfect_prediction():
    t = fake()
    m = subject_metrics(t, t["density"], t["cls"])
    assert m["n_voxels"] == 2 * fov((S, S)).sum() - 2
    assert m["density_mae"] == 0 and m["density_rmse"] == 0 and m["macro_dice"] == 1


def test_known_errors():
    t = fake()
    m = subject_metrics(t, np.asarray(t["density"], np.float32) + 0.1, np.zeros_like(t["cls"]))
    assert abs(m["density_mae"] - 0.1) < 1e-3 and abs(m["density_bias"] - 0.1) < 1e-3
    mask = (t["valid"] == 1) & (t["organ_pt"] != 0) & fov((S, S))
    n0 = (t["cls"][mask] == 0).sum()
    assert abs(m["dice_by_class"][CLASSES[0]] - 2 * n0 / (n0 + mask.sum())) < 1e-9
    assert set(m["dice_by_class"]) == {CLASSES[c] for c in np.unique(t["cls"][mask])}


def test_material_dice_merges_same_material():
    t = fake()
    # swap ribs_cortical <-> ribs_trabecular everywhere: wrong anatomy, same DukeSim material
    a, b = CLASSES.index("ribs_cortical"), CLASSES.index("ribs_trabecular")
    t["cls"][:] = np.where(t["cls"] == 0, a, np.where(t["cls"] == 1, b, t["cls"]))
    pred = np.where(t["cls"] == a, b, np.where(t["cls"] == b, a, t["cls"]))
    m = subject_metrics(t, None, pred)
    assert m["material_macro_dice"] == 1.0 and m["macro_dice"] < 1.0
