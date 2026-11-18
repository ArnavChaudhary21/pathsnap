"""Loading of the cached candidate tables (built by scripts/03_build_dataset.py)."""
from __future__ import annotations

import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")


def load_cached(partition: str, tid: str) -> dict:
    z = np.load(os.path.join(CACHE, partition, f"{tid}.npz"))
    return {k: z[k] for k in z.files}


def stack(partition: str, tids, obs_stride: int = 1):
    """Concatenate trips. obs_stride>1 keeps every k-th observation (all its
    candidates) - consecutive 1 Hz fixes are nearly duplicates, so striding
    removes redundancy without dropping any trip."""
    X, y, grp, obs_l, lab_l = [], [], [], [], []
    for gi, tid in enumerate(tids):
        d = load_cached(partition, tid)
        keep = (d["obs"] % obs_stride == 0) & (d["obs_label"][d["obs"]] >= 0)
        X.append(d["X"][keep]); y.append(d["y"][keep]); grp.append(np.full(keep.sum(), gi))
        obs_l.append(d["obs"][keep] + gi * 10_000_000)
    return (np.concatenate(X), np.concatenate(y), np.concatenate(grp), np.concatenate(obs_l))
