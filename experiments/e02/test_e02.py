import numpy as np
import torch

from train import build, ct_stack, losses


def cfg(**k):
    return dict(heads="both", context=0, base=8, depth=3, w_cls=1.0) | k


def test_heads_and_context():
    for heads, keys in [("density", {"density"}), ("cls", {"cls"}), ("both", {"density", "cls"})]:
        for ctx in (0, 2):
            out = build(cfg(heads=heads, context=ctx))(torch.zeros(2, 2 * ctx + 1, 64, 64))
            assert set(out) == keys
            assert "density" not in out or out["density"].shape == (2, 64, 64)
            assert "cls" not in out or out["cls"].shape == (2, 19, 64, 64)


def test_ct_stack_clamps():
    ct = np.arange(5)[:, None, None] * np.ones((1, 2, 2))
    assert ct_stack(ct, 0, 1)[:, 0, 0].tolist() == [0, 0, 1]
    assert ct_stack(ct, np.array([3, 4]), 1)[:, :, 0, 0].tolist() == [[2, 3, 4], [3, 4, 4]]


def test_loss_ignores_outside_mask():
    d = torch.rand(1, 4, 4); m = torch.zeros(1, 4, 4, dtype=torch.bool); m[0, :2] = True
    cl = torch.where(m, 3, 255)
    logits = torch.full((1, 19, 4, 4), -10.0); logits[:, 3] = 10
    out = {"density": torch.where(m, d, d + 100), "cls": logits}  # huge error only outside the mask
    L = losses(out, d, cl, m, cfg())
    assert L["l1"] < 1e-6 and L["ce"] < 1e-6


def test_crop3d_pads_outside():
    from train3d import crop
    a = np.arange(4 * 5 * 6).reshape(4, 5, 6)
    out = crop(a, np.array([-1, 3, 2]), 3, -7)
    assert out.shape == (3, 3, 3)
    assert (out[0] == -7).all() and (out[:, 2] == -7).all()  # z=-1 and row 5 are outside
    assert out[1, 0, 0] == a[0, 3, 2] and out[2, 1, 2] == a[1, 4, 4]
