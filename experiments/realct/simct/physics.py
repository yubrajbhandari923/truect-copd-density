"""Drishya E01 variant-3 forward physics, copied verbatim (function bodies) from drishya branch experiment/e01-forward
11ce89c: water_mu_mm, spectrum_weights, Spectral (forward_spectral.py); scatter_kernel, intensities, to_g, noisy
(forward_v3.py). Only imports/paths changed. Calibrated parameters (drishya results/e01/v3/e01_v3.json, fit on TrueCT
10313): Al 13 mm, scatter-like kernel sigma 50 px amplitude 0.0272, detector blur 0.4 px, N0 411.6 photons per mAs unit.
"""
import json
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import spekpy
from scipy.signal import fftconvolve
from scipy.special import logsumexp

import leap_recon as R

HERE = Path(__file__).resolve().parent
E = np.arange(1, 121, dtype=np.float64)  # keV
FIT = E >= 20
V3 = dict(al=13.0, ss=50.0, a=0.027159092108356556, sb=0.4, n0=411.57948611418146)  # drishya e01_v3.json params


def water_mu_mm():
    d = json.loads((HERE / "water_xraylib.json").read_text())
    return np.exp(np.interp(np.log(E), np.log(d["energies_keV"]), np.log(d["water_cm2_g"]))) / 10.0


def spectrum_weights(al_mm):
    s = spekpy.Spek(kvp=120, th=12, dk=1.0)
    s.filter("Al", al_mm)
    e, f = s.get_spectrum()
    w = np.interp(E, e, f, left=0, right=0) * E  # energy-integrating detector
    return w / w.sum()


class Spectral:
    def __init__(self, B, al_mm, a1rng, a2rng, n1=2048, n2=512):
        self.w, mw = spectrum_weights(al_mm), water_mu_mm()
        L = np.linspace(0, 800, 801)
        pw = -np.log(np.exp(-np.outer(L, mw)) @ self.w)
        V = np.stack([pw ** k for k in (4, 3, 2, 1)], 1)  # 4th-order, no constant: zero path -> zero
        self.poly = np.append(np.linalg.lstsq(V, R.MU_WATER * L, rcond=None)[0], 0.0)
        self.water_fit_max_err = float(np.abs(np.polyval(self.poly, pw) - R.MU_WATER * L).max())
        pad = lambda r: (r[0] - 0.02 * (r[1] - r[0]) - 1e-3, r[1] + 0.02 * (r[1] - r[0]) + 1e-3)  # noqa: E731
        self.a1 = np.linspace(*pad(a1rng), n1); self.a2 = np.linspace(*pad(a2rng), n2)
        x = self.a1[:, None, None] * B[None, None, :, 0] + self.a2[None, :, None] * B[None, None, :, 1]
        p_poly = -logsumexp(-x, b=self.w[None, None, :], axis=-1)
        self.lut = np.polyval(self.poly, p_poly).astype(np.float32)
        self.lut_raw = p_poly.astype(np.float32)

    def __call__(self, A1, A2, chunk=250, raw=False):
        out, lut = np.empty_like(A1), (self.lut_raw if raw else self.lut)
        s1, s2 = (len(self.a1) - 1) / (self.a1[-1] - self.a1[0]), (len(self.a2) - 1) / (self.a2[-1] - self.a2[0])
        for v0 in range(0, len(A1), chunk):
            a, b = A1[v0:v0 + chunk], A2[v0:v0 + chunk]
            out[v0:v0 + chunk] = ndi.map_coordinates(lut, [(a - self.a1[0]) * s1, (b - self.a2[0]) * s2], order=1, mode="nearest")
        return out


def scatter_kernel(ss):
    xr, xc = np.arange(-63, 64), np.arange(-min(int(4 * ss), 899), min(int(4 * ss), 899) + 1)
    k = np.outer(np.exp(-xr ** 2 / (2 * ss ** 2)), np.exp(-xc ** 2 / (2 * ss ** 2)))
    return (k / (2 * np.pi * ss ** 2)).astype(np.float32)


def intensities(p, sb, ss, chunk=500):
    ib, kk = np.empty_like(p), np.empty_like(p)
    ker = scatter_kernel(ss)[None]
    for v in range(0, len(p), chunk):
        ip = np.exp(-p[v:v + chunk])
        ib[v:v + chunk] = ndi.gaussian_filter(ip, (0, sb, sb), mode="nearest") if sb > 0 else ip
        kk[v:v + chunk] = fftconvolve(ib[v:v + chunk] * p[v:v + chunk], ker, mode="same", axes=(1, 2))
    return ib, kk


def to_g(i, poly):
    return np.polyval(poly, -np.log(np.maximum(i, 1e-12))).astype(np.float32)


def noisy(ib, kk, a, n0, mas, rng, poly, chunk=500):
    out = np.empty_like(ib)
    for v in range(0, len(ib), chunk):
        lam = (n0 * mas[v:v + chunk])[:, None, None] * (ib[v:v + chunk] + a * kk[v:v + chunk])
        n = np.maximum(rng.poisson(lam), 1).astype(np.float32)
        out[v:v + chunk] = to_g(n / (n0 * mas[v:v + chunk])[:, None, None], poly)
    return out
