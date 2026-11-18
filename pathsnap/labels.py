"""Observation-level label construction from the chronological reference route.

The ``.route`` file lists the arcs the vehicle traversed in order; it has far
fewer rows than the ``.track`` file and carries no timestamps. To obtain a
reference arc per GPS observation we solve a monotone alignment problem:

    choose j(0) <= j(1) <= ... <= j(n-1)   (indices into the route)
    minimising  sum_i  min(d(p_i, route[j(i)]), CAP)

where d is the projected point-to-segment distance in metres. The
non-decreasing constraint encodes the fact that both sequences are
chronological; route arcs may be skipped (short arcs passed between two
fixes). CAP bounds the influence of gross GPS outliers.

This is solved exactly with dynamic programming in O(n*m) time and memory
(prefix minima of the previous row). Points whose aligned arc is still far
away (> ``uncertain_m``) are marked -1 = "uncertain" and excluded from
training and from metric denominators (reported separately).
"""
from __future__ import annotations

import numpy as np

from .geo import point_segment
from .graph import RoadGraph


def align_track_to_route(pxy: np.ndarray, route: np.ndarray, g: RoadGraph,
                         cap: float = 100.0, uncertain_m: float = 50.0):
    n, m = len(pxy), len(route)
    ax, ay, bx, by = g.ax[route], g.ay[route], g.bx[route], g.by[route]
    back = np.empty((n, m), dtype=np.int32)
    prev = None
    for i in range(n):
        d, _ = point_segment(pxy[i, 0], pxy[i, 1], ax, ay, bx, by)
        d = np.minimum(d, cap)
        if prev is None:
            cur = d
            back[i] = np.arange(m)
        else:
            # prefix minimum of prev and its argmin
            pm = np.minimum.accumulate(prev)
            idx = np.arange(m)
            is_new = np.concatenate([[True], prev[1:] < pm[:-1]])
            arg = np.maximum.accumulate(np.where(is_new, idx, 0))
            cur = pm + d
            back[i] = arg
        prev = cur
    j = np.empty(n, dtype=np.int64)
    j[-1] = int(np.argmin(prev))
    for i in range(n - 1, 0, -1):
        j[i - 1] = back[i, j[i]]
    arcs = route[j]
    dist, t = point_segment(pxy[:, 0], pxy[:, 1], g.ax[arcs], g.ay[arcs], g.bx[arcs], g.by[arcs])
    label = np.where(dist <= uncertain_m, arcs, -1)
    return {"route_idx": j, "arc": arcs, "label": label, "dist": dist, "t": t}
