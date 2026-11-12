import numpy as np

from build_targets import box_weights


def test_box_weights_conserve_mass_and_flag_coverage():
    # 10 recon voxels of 0.6 mm over a 40-voxel 0.25 mm phantom axis, reversed, offset 1.3 mm
    W, cov, ctr = box_weights(10, 40, A=7.3, sign=-1, s=1.0)
    x = np.random.default_rng(0).random(40).astype(np.float32)
    inside = cov > 0.999
    assert inside.sum() >= 8 and np.allclose(cov[inside], 1, atol=1e-5)
    # box average of a constant is the constant; total mass over fully covered voxels is conserved
    assert np.allclose((W @ np.ones(40, np.float32))[inside], 1, atol=1e-5)
    e = (7.3 - 0.6 * np.arange(11)) / 0.25
    for i in np.flatnonzero(inside):
        lo, hi = sorted((e[i], e[i + 1]))
        fine = np.linspace(lo, hi, 20001)[:-1] + (hi - lo) / 40000
        assert abs((W @ x)[i] - x[np.floor(fine).astype(int)].mean()) < 1e-3
        assert ctr[i] == int(np.floor((lo + hi) / 2))
    # a voxel hanging off the end is not fully covered
    W2, cov2, _ = box_weights(10, 40, A=-0.3, sign=1, s=1.0)
    assert cov2[0] < 0.999 and cov2[1] > 0.999
