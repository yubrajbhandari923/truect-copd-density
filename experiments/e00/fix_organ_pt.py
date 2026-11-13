"""One-off repair of targets built before commit fixing build_targets.py: organ_pt was left 0 (air) instead of
255 (unknown) on CT slices not fully inside the phantom. Sets organ_pt = 255 wherever valid == 0. Idempotent.

usage: fix_organ_pt.py <subject_id> [...]
"""
import sys

import numpy as np

from common import OUT

for sid in sys.argv[1:]:
    d = OUT / "targets" / sid
    op = np.load(d / "organ_pt.npy", mmap_mode="r+")
    v = np.load(d / "valid.npy", mmap_mode="r")
    n = 0
    for i in range(op.shape[0]):
        bad = (v[i] == 0) & (op[i] != 255)
        n += int(bad.sum())
        op[i][bad] = 255
    op.flush()
    print("fixed", sid, n)
