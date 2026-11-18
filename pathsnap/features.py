"""Candidate generation and candidate-level features.

Every GPS observation i gets a candidate set C_i of directed arcs near it.
Each (i, arc) pair becomes one row with the features below; the supervised
target (built elsewhere, from labels.py) is "this arc is the reference arc
for observation i".

Only the track, nodes and arcs are used here - never the reference route -
so the same code runs unchanged at inference time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geo import LocalProjection, angle_diff, point_segment
from .graph import RoadGraph
from .io import TripInputs

FEATURES = [
    "dist", "dist_rel", "dist_rank", "n_cand", "log_n_cand_20m",
    "head_diff", "head_diff_rel", "cos_head", "speed", "turn_rate",
    "win_dist", "win_dist_rel", "win_dist_far", "win_head_diff",
    "t_along", "off_end", "log_arc_len", "out_deg_to", "in_deg_from",
    "dt_prev", "dt_next", "radius_level",
]


@dataclass
class CandidateConfig:
    radius: float = 50.0       # metres, first search radius
    max_cand: int = 10         # keep the k nearest arcs
    expand: tuple = (2.0, 4.0)  # radius multipliers if nothing is found
    head_win: int = 2          # +-obs used to estimate GPS heading
    ctx_win: int = 5           # +-obs used for the window-distance feature


@dataclass
class PreparedTrip:
    ti: TripInputs
    proj: LocalProjection
    pxy: np.ndarray
    t: np.ndarray
    g: RoadGraph


def prepare_trip(ti: TripInputs) -> PreparedTrip:
    proj = LocalProjection.for_track(ti.track[:, :2])
    return PreparedTrip(ti, proj, proj.to_xy(ti.track[:, :2]), ti.track[:, 2].copy(),
                        RoadGraph(proj.to_xy(ti.nodes), ti.arcs))


def _shift(a, k):
    """a[i+k] with edge clamping."""
    n = len(a)
    idx = np.clip(np.arange(n) + k, 0, n - 1)
    return a[idx]


def motion(pxy: np.ndarray, t: np.ndarray, w: int):
    fwd, bwd = _shift(pxy, w), _shift(pxy, -w)
    d = fwd - bwd
    heading = np.arctan2(d[:, 1], d[:, 0])
    span = np.maximum(_shift(t, w) - _shift(t, -w), 1e-6)
    speed = np.hypot(d[:, 0], d[:, 1]) / span
    # turn rate: change between backward and forward half-headings
    h1 = np.arctan2(*(pxy - bwd)[:, ::-1].T)
    h2 = np.arctan2(*(fwd - pxy)[:, ::-1].T)
    turn = angle_diff(h1, h2)
    return heading, speed, turn


def build_candidates(pt: PreparedTrip, cfg: CandidateConfig = CandidateConfig()):
    """Returns dict with obs (row->obs index), arc, X (rows x features), t_along."""
    g, pxy, t = pt.g, pt.pxy, pt.t
    n = len(pxy)
    obs_l, arc_l, d_l, t_l, lvl_l, n20_l = [], [], [], [], [], []
    for i in range(n):
        lvl = 0
        arcs, d, tt = g.arcs_near(pxy[i, 0], pxy[i, 1], cfg.radius)
        for k, mult in enumerate(cfg.expand, start=1):
            if len(arcs):
                break
            arcs, d, tt = g.arcs_near(pxy[i, 0], pxy[i, 1], cfg.radius * mult)
            lvl = k
        if not len(arcs):
            continue
        n20 = int((d <= 20).sum())
        arcs, d, tt = arcs[:cfg.max_cand], d[:cfg.max_cand], tt[:cfg.max_cand]
        obs_l.append(np.full(len(arcs), i))
        arc_l.append(arcs); d_l.append(d); t_l.append(tt)
        lvl_l.append(np.full(len(arcs), lvl)); n20_l.append(np.full(len(arcs), n20))
    if not obs_l:
        return {"obs": np.empty(0, int), "arc": np.empty(0, int), "X": np.empty((0, len(FEATURES))),
                "t_along": np.empty(0)}
    obs = np.concatenate(obs_l); arc = np.concatenate(arc_l)
    dist = np.concatenate(d_l); tal = np.concatenate(t_l)
    lvl = np.concatenate(lvl_l); n20 = np.concatenate(n20_l)

    # per-observation group helpers (rows are grouped and sorted by distance)
    starts = np.flatnonzero(np.r_[True, obs[1:] != obs[:-1]])
    counts = np.diff(np.r_[starts, len(obs)])
    grp = np.repeat(np.arange(len(starts)), counts)
    rank = np.arange(len(obs)) - starts[grp]

    def rel(v):
        return v - np.minimum.reduceat(v, starts)[grp]

    heading, speed, turn = motion(pxy, t, cfg.head_win)
    bearing = g.bearing[arc]
    hd = angle_diff(heading[obs], bearing)
    hd = np.where(g.length[arc] > 1e-6, hd, np.pi / 2)

    # context window: how well does this arc explain the neighbouring fixes
    ax, ay, bx, by = g.ax[arc], g.ay[arc], g.bx[arc], g.by[arc]
    wd = []
    for o in range(-cfg.ctx_win, cfg.ctx_win + 1):
        q = np.clip(obs + o, 0, n - 1)
        wd.append(point_segment(pxy[q, 0], pxy[q, 1], ax, ay, bx, by)[0])
    wd = np.stack(wd)
    win_dist = wd.mean(0)
    win_far = np.median(wd, 0)  # robust variant
    # heading agreement over a longer window (+-ctx_win)
    lh = np.arctan2(*(_shift(pxy, cfg.ctx_win) - _shift(pxy, -cfg.ctx_win))[:, ::-1].T)
    win_hd = np.where(g.length[arc] > 1e-6, angle_diff(lh[obs], bearing), np.pi / 2)

    # off_end: distance of the unclamped projection beyond the segment ends (m)
    L = np.maximum(g.length[arc], 1e-6)
    traw = ((pxy[obs, 0] - ax) * (bx - ax) + (pxy[obs, 1] - ay) * (by - ay)) / (L * L)
    off_end = np.maximum(0, np.maximum(-traw, traw - 1)) * L

    dtv = np.diff(t)
    dt_prev = np.r_[0.0, dtv][obs]
    dt_next = np.r_[dtv, 0.0][obs]

    X = np.column_stack([
        dist, rel(dist), rank, counts[grp], np.log1p(n20),
        hd, rel(hd), np.cos(hd), speed[obs], turn[obs],
        win_dist, rel(win_dist), win_far, win_hd,
        tal, off_end, np.log1p(g.length[arc]), g.out_deg[g.arcs[arc, 1]], g.in_deg[g.arcs[arc, 0]],
        np.log1p(dt_prev), np.log1p(dt_next), lvl,
    ]).astype(np.float32)
    return {"obs": obs, "arc": arc, "X": X, "t_along": tal}
