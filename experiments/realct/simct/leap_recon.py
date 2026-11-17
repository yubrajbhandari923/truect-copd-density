# Copied verbatim from drishya main 02d129f (drishya/leap_recon.py) for the real-CT simulation check; do not edit here.
"""LEAP helical FBP of TrueCT projections on the supplied-recon grid — the reconstruction approved at G_RECON (2026-11-14).

Settings and their evidence: reports/e00/geometry_and_preprocessing.md. All conventions were set on the reference
cylinder only. Valid in-plane FOV r <= ~215 mm (bright rim beyond ~225 mm, cause unresolved).
LEAP (LLNL, commit 0c8846f) is imported lazily from $LEAP_SRC so this module imports without it.
Note: LEAP's helical FBP backward pass is not a correct gradient (CTsynth pilot) — do not backpropagate through fbp().
"""
import os
import sys

import numpy as np

SOD, SDD, PIX = 575.0, 1050.0, 1.0  # mm; curved detector, 1 mm pixels (DICOM-CT-PD tags)
N_ROWS, N_COLS = 64, 900
MU_WATER = 0.0192733  # /mm, polyfit on reference cylinder body (E00 job 205278)
CONV = dict(rot_sign=-1, row_flip=True, ch_flip=False, col_shift=-1.0,
            off_x_mm=-0.2925746180057162, off_y_mm=-0.2977597886646287,  # reference-only shift calibration, E00 job 205395 (full precision)
            nxy=768, vox_mm=0.6, dz_mm=0.6)
VALID_FOV_RADIUS_MM = 215.0


def _leap():
    if "LEAP_SRC" in os.environ and os.environ["LEAP_SRC"] not in sys.path:
        sys.path.insert(0, os.environ["LEAP_SRC"])
    from leapctype import tomographicModels
    return tomographicModels


def make_leap(n_views, ang_rad, z_mm, nz, conv=CONV):
    """LEAP model for the helical scan. ang_rad: unwrapped tube angles; z_mm: table z per view. Returns (model, pitch mm/rad)."""
    c = conv
    L = _leap()()
    pitch = float(np.polyfit(ang_rad, z_mm, 1)[0])
    L.set_conebeam(n_views, N_ROWS, N_COLS, PIX, PIX, 0.5 * (N_ROWS - 1), 0.5 * (N_COLS - 1) + c["col_shift"],
                   np.degrees(c["rot_sign"] * ang_rad).astype(np.float32), SOD, SDD)
    L.set_curvedDetector()
    L.set_helicalPitch(c["rot_sign"] * pitch)
    L.set_volume(c["nxy"], c["nxy"], nz, c["vox_mm"], c["dz_mm"], c["off_x_mm"], c["off_y_mm"], 0.0)
    return L, pitch


def prep_g(g, conv=CONV):
    """(view, row, ch) line integrals from ctpd.read_series -> LEAP detector orientation."""
    if conv["row_flip"]:
        g = g[:, ::-1, :]
    if conv["ch_flip"]:
        g = g[:, :, ::-1]
    return np.ascontiguousarray(g, dtype=np.float32)


def fbp(g, ang_rad, z_mm, nz, conv=CONV):
    """Helical FBP. g must already be prep_g'd; it may be modified in place. Returns mu [slice, y, x] (/mm) in supplied
    DICOM slice order, volume centred on mid-scan table z."""
    L, _ = make_leap(len(g), ang_rad, z_mm, nz, conv)
    f = L.allocateVolume()
    L.FBP(g, f)
    return np.ascontiguousarray(np.asarray(f)[::-1])


def to_hu(mu, mu_water=MU_WATER):
    """HU, clipped at -1000 like the supplied TrueCT recon."""
    return np.maximum(1000.0 * (mu / mu_water - 1.0), -1000.0)


def fov_mask(shape_yx=(768, 768), vox_mm=0.6, r_mm=VALID_FOV_RADIUS_MM):
    """In-plane mask of the valid reconstruction region (image-centre radius)."""
    ny, nx = shape_yx
    yy, xx = np.mgrid[:ny, :nx]
    return np.hypot(yy - (ny - 1) / 2, xx - (nx - 1) / 2) * vox_mm <= r_mm
