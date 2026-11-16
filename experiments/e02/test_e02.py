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


def test_dice_term_ignores_outside_mask():
    from train3d import loss3d
    m = torch.zeros(1, 4, 4, 4, dtype=torch.bool); m[0, :2] = True
    cl = torch.where(m, 3, 255); d = torch.rand(1, 4, 4, 4)
    logits = torch.full((1, 19, 4, 4, 4), -10.0); logits[:, 3] = 10
    logits[:, :, 2:] = torch.randn(1, 19, 2, 4, 4)  # garbage only outside the mask
    L = loss3d({"density": d, "cls": logits}, d, cl, m, dict(w_cls=1.0, dice=1.0))
    assert L["dice"] < 0.01 and L["ce"] < 1e-6  # absent classes add ~smooth/(smooth+eps) each


def test_grad_and_huber_terms():
    from train3d import loss3d
    m = torch.zeros(1, 4, 4, 4, dtype=torch.bool); m[0, 1:3, 1:3, 1:3] = True
    d = torch.rand(1, 4, 4, 4); cl = torch.where(m, 3, 255); logits = torch.zeros(1, 19, 4, 4, 4)
    L = loss3d({"density": d + 0.3, "cls": logits}, d, cl, m, dict(w_cls=0.0, w_grad=1.0, huber=0.05))
    assert L["grad"] < 1e-6  # a constant offset has no gradient error; mask-edge jumps are excluded
    assert abs(L["huber"] - (0.3 - 0.025)) < 1e-5 and abs(L["loss"] - L["huber"]) < 1e-6


def test_train_subsets_nested():
    from train3d import train_subjects
    s = [train_subjects(dict(n_train=n)) for n in (10, 20, 40)] + [train_subjects({})]
    assert [len(x) for x in s] == [10, 20, 40, 46] and all(a == b[:len(a)] for a, b in zip(s, s[1:]))
