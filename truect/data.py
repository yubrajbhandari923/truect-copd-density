"""Access to the E00 CT-grid targets and the frozen split. Shared by E01-E04.

Arrays per subject (memmaps, all [slice, row, col] on the CT grid, see experiments/e00/build_targets.py):
  ct int16, density float16 g/cm3, cls uint8 (index into CLASSES, 255 = unknown), purity uint8 (255 = pure),
  organ_pt uint8 (native organ ID, 255 = outside phantom), valid uint8.
Test subjects are refused unless allow_test=True (only E04, after methods are frozen).
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORK = Path("/scratch/railabs/yb107/research-output/truect-copd-density")
TARGETS = WORK / "results/e00/targets"
CLASSES = ['air', 'aorta', 'body', 'emphysema', 'esophagus', 'fissure', 'lung', 'lung_parenchyma', 'muscle',
           'ribs_cortical', 'ribs_trabecular', 'scapulas', 'scapulas_trabecular', 'skin', 'spine_cortical',
           'spine_trabecular', 'sternum', 'sternum_trabecular', 'vessel']  # identical in every subject's meta.json
ARRAYS = ("ct", "density", "cls", "purity", "organ_pt", "valid")


def split():
    return json.load(open(ROOT / "configs/splits_copd.json"))


def load(sid, names=ARRAYS, allow_test=False, root=TARGETS):
    if not allow_test and sid in split()["test"]:
        raise PermissionError(f"{sid} is a held-out test subject")
    meta = json.load(open(root / sid / "meta.json"))
    assert meta["classes"] == CLASSES, sid
    return {n: np.load(root / sid / f"{n}.npy", mmap_mode="r") for n in names}, meta


FOV_FRACTION = 375 / 384  # recon is a disc of radius ~384 px (CT == -1000 beyond 386 px); keep a ~5 mm margin


def fov(shape):
    """CT-derived reconstruction field of view for an in-plane shape (rows, cols): True inside the disc."""
    n, m = shape[-2:]
    yy, xx = np.mgrid[:n, :m]
    return np.hypot(yy - (n - 1) / 2, xx - (m - 1) / 2) < FOV_FRACTION * min(n, m) / 2


def eval_mask(t, i=slice(None)):
    """Evaluation region: CT voxel fully inside the phantom, not outside-body air (organ ID 0), and inside the
    reconstruction field of view. Identical for every method, never a model input."""
    v = np.asarray(t["valid"][i])
    return (v == 1) & (np.asarray(t["organ_pt"][i]) != 0) & fov(v.shape)
