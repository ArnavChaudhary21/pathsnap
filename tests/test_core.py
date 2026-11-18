import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.features import PreparedTrip
from pathsnap.graph import RoadGraph
from pathsnap.hmm import HMMParams, viterbi_core, viterbi_decode
from pathsnap.labels import align_track_to_route


def test_viterbi_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(50):
        sizes = rng.integers(1, 4, size=rng.integers(1, 6))
        E = [rng.normal(size=k) for k in sizes]
        T = [rng.normal(size=(sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]
        best, best_path = -np.inf, None
        for path in itertools.product(*[range(k) for k in sizes]):
            sc = E[0][path[0]] + sum(T[i][path[i], path[i + 1]] + E[i + 1][path[i + 1]] for i in range(len(T)))
            if sc > best:
                best, best_path = sc, list(path)
        assert viterbi_core(E, T) == best_path


def _two_parallel_roads():
    # road A: y=0 eastbound (arcs 0,1,2); road B: y=30 eastbound (arcs 3,4,5), not connected to A
    xy = np.array([[0, 0], [100, 0], [200, 0], [300, 0], [0, 30], [100, 30], [200, 30], [300, 30]], float)
    arcs = np.array([[0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [6, 7]])
    return RoadGraph(xy, arcs)


def test_network_distance_and_one_way():
    g = _two_parallel_roads()
    assert abs(g.route_distance(0, 0.5, 1, 0.5, 1000) - 100) < 1e-9
    assert abs(g.route_distance(0, 0.2, 0, 0.7, 1000) - 50) < 1e-9
    assert g.route_distance(1, 0.5, 0, 0.5, 1000) == np.inf      # against one-way direction
    assert g.route_distance(0, 0.5, 4, 0.5, 1000) == np.inf      # other component


def test_hmm_fixes_isolated_wrong_emission():
    g = _two_parallel_roads()
    xs = np.arange(10, 300, 10.0)
    pxy = np.column_stack([xs, np.full(len(xs), 14.0)])            # ambiguous: between both roads
    pt = PreparedTrip(ti=None, proj=None, pxy=pxy, t=np.arange(len(xs), dtype=float), g=g)
    obs, arc, ta, prob = [], [], [], []
    for i, x in enumerate(xs):
        a_road, b_road = int(x // 100), 3 + int(x // 100)
        pa = 0.3 if i == 12 else 0.7                                 # one misleading step
        for a, p in ((a_road, pa), (b_road, 1 - pa)):
            obs.append(i); arc.append(a); ta.append((x % 100) / 100); prob.append(p)
    pred, _ = viterbi_decode(pt, np.array(obs), np.array(arc), np.array(ta), np.array(prob), HMMParams())
    assert all(p < 3 for p in pred), pred                           # stays on road A throughout


def test_alignment_is_monotone_and_skips():
    g = _two_parallel_roads()
    route = np.array([0, 1, 2])
    pxy = np.array([[5, 3], [60, -2], [250, 4], [290, 1]], float)   # skips arc 1 entirely in the middle
    al = align_track_to_route(pxy, route, g)
    assert list(al["arc"]) == [0, 0, 2, 2]
    assert np.all(np.diff(al["route_idx"]) >= 0)
