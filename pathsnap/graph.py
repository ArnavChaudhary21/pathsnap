"""Road network in projected (metre) coordinates.

Provides
* segment geometry (length, bearing) per directed arc,
* a spatial index built on points densified along every arc, used for
  radius queries of candidate arcs,
* bounded single-source Dijkstra with a per-trip cache, used for the
  network distance between two candidate positions.
"""
from __future__ import annotations

import heapq

import numpy as np
from scipy.spatial import cKDTree

from .geo import point_segment


class RoadGraph:
    def __init__(self, node_xy: np.ndarray, arcs: np.ndarray, densify_step: float = 10.0):
        self.xy = node_xy
        self.arcs = arcs
        self.n_nodes = len(node_xy)
        self.n_arcs = len(arcs)
        a, b = node_xy[arcs[:, 0]], node_xy[arcs[:, 1]]
        self.ax, self.ay, self.bx, self.by = a[:, 0], a[:, 1], b[:, 0], b[:, 1]
        d = b - a
        self.length = np.hypot(d[:, 0], d[:, 1])
        self.bearing = np.arctan2(d[:, 1], d[:, 0])

        # out-adjacency (CSR) of arcs leaving each node
        order = np.argsort(arcs[:, 0], kind="stable")
        self.out_arcs = order
        self.out_ptr = np.zeros(self.n_nodes + 1, dtype=np.int64)
        np.add.at(self.out_ptr, arcs[:, 0] + 1, 1)
        self.out_ptr = np.cumsum(self.out_ptr)
        self.in_deg = np.bincount(arcs[:, 1], minlength=self.n_nodes)
        self.out_deg = np.diff(self.out_ptr)

        # densified sample points for spatial lookup
        self.step = densify_step
        nseg = np.maximum(1, np.ceil(self.length / densify_step).astype(int))
        arc_idx = np.repeat(np.arange(self.n_arcs), nseg + 1)
        offs = np.concatenate([np.arange(k + 1) / k for k in nseg])
        px = self.ax[arc_idx] + offs * (self.bx - self.ax)[arc_idx]
        py = self.ay[arc_idx] + offs * (self.by - self.ay)[arc_idx]
        self._sample_arc = arc_idx
        self._tree = cKDTree(np.column_stack([px, py]))
        self._sp_cache: dict[int, tuple[float, dict[int, float]]] = {}

    # ------------------------------------------------------------------ spatial
    def arcs_near(self, x: float, y: float, radius: float):
        """Arcs whose segment lies within ``radius`` metres of (x, y).

        Returns (arc_ids, distances, t) sorted by distance.
        """
        idx = self._tree.query_ball_point([x, y], radius + self.step / 2)
        if not idx:
            return np.empty(0, int), np.empty(0), np.empty(0)
        cand = np.unique(self._sample_arc[idx])
        dist, t = point_segment(x, y, self.ax[cand], self.ay[cand], self.bx[cand], self.by[cand])
        keep = dist <= radius
        cand, dist, t = cand[keep], dist[keep], t[keep]
        o = np.argsort(dist, kind="stable")
        return cand[o], dist[o], t[o]

    def dist_to_arcs(self, x, y, arc_ids):
        return point_segment(x, y, self.ax[arc_ids], self.ay[arc_ids], self.bx[arc_ids], self.by[arc_ids])

    # ------------------------------------------------------------------ network
    def shortest_from(self, src: int, cutoff: float) -> dict[int, float]:
        """Distances from node ``src`` to every node reachable within ``cutoff`` m."""
        hit = self._sp_cache.get(src)
        if hit is not None and hit[0] >= cutoff:
            return hit[1]
        dist = {src: 0.0}
        heap = [(0.0, src)]
        ptr, out, to, L = self.out_ptr, self.out_arcs, self.arcs[:, 1], self.length
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, np.inf):
                continue
            for k in range(ptr[u], ptr[u + 1]):
                arc = out[k]
                nd = d + L[arc]
                if nd > cutoff:
                    continue
                v = int(to[arc])
                if nd < dist.get(v, np.inf):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        self._sp_cache[src] = (cutoff, dist)
        return dist

    def route_distance(self, arc_a: int, t_a: float, arc_b: int, t_b: float, cutoff: float) -> float:
        """Driving distance from position t_a on arc_a to position t_b on arc_b.

        Returns ``inf`` if not reachable within ``cutoff`` metres.
        """
        La, Lb = self.length[arc_a], self.length[arc_b]
        if arc_a == arc_b and t_b >= t_a - 1e-9:
            return (t_b - t_a) * La
        rest = (1.0 - t_a) * La
        if rest > cutoff:
            return np.inf
        dmap = self.shortest_from(int(self.arcs[arc_a, 1]), cutoff - rest)
        mid = dmap.get(int(self.arcs[arc_b, 0]))
        if mid is None:
            return np.inf
        return rest + mid + t_b * Lb

    def clear_cache(self):
        self._sp_cache.clear()
