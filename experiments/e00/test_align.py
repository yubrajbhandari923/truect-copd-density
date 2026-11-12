"""Synthetic check: the 48-way masked-NCC search recovers a known permutation, flips and shift."""
import numpy as np
from scipy.ndimage import gaussian_filter

from align import coarse_A, orient, search48


def test_search_recovers_known_orientation():
    rng = np.random.default_rng(0)
    P = gaussian_filter(rng.normal(size=(18, 22, 26)), 1.5)  # anisotropic box so permutations are distinct
    perm, flips, off = (2, 0, 1), (1, 0, 1), (5, 3, 7)
    Q = orient(P, perm, flips)
    R = rng.normal(scale=0.01, size=(36, 30, 34))
    R[off[0]:off[0] + Q.shape[0], off[1]:off[1] + Q.shape[1], off[2]:off[2] + Q.shape[2]] += Q
    res = search48(P, R)
    best = res[0]
    assert (tuple(best["perm"]), tuple(best["flips"])) == (perm, flips)
    assert best["shift"] == list(off) and best["r"] > 0.99 and res[1]["r"] < 0.9
    # model x[perm[m]] = A_m + sign_m*y_m (block=1): recon block centre k+0.5 maps to phantom centre b+0.5
    A = coarse_A(best, P.shape, np.ones(3), block=1.0)
    for m in range(3):
        k = off[m] + 2  # recon block inside the embedded copy
        j = k - off[m]
        b = P.shape[perm[m]] - 1 - j if flips[m] else j
        x = A[m] + (-1 if flips[m] else 1) * (k + 0.5)
        assert np.isclose(x, b + 0.5)
