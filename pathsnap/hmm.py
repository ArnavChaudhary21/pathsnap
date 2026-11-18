"""HMM + Viterbi route reconstruction.

Hidden state at observation i: one candidate (arc, position t on the arc)
from the candidate set C_i produced by features.build_candidates.

Emission   log e_i(c) = w_e * log( p(c) / sum_{c' in C_i} p(c') )
           where p is the classifier's probability that c is the true arc.
Transition log a(c -> c') = -| d_net(c, c') - d_gps(i, j) | / beta_ij
           d_net: driving distance along the directed network from the
           projection of fix i on arc c to the projection of fix j on c'
           (bounded Dijkstra), d_gps: straight-line distance between fixes.
           beta_ij = beta * (1 + gap_growth * max(0, dt_ij - 1)) so that the
           model is more tolerant after GPS gaps (the vehicle may have
           turned, so network and straight distances legitimately differ).
           Small backward jitter on the same arc (<= backtrack_m) is treated
           as standing still instead of requiring a loop.
Invalid    If c' is not reachable from c within the search cutoff (one-way
           violation, disconnected component, broken network) the
           transition gets the finite log-penalty -break_penalty instead of
           -inf, so the decoder can still recover a route across network
           breaks rather than failing.
Missing    Observations without candidates are skipped; the chain links the
candidates previous and next observations that have candidates, with dt and
           d_gps computed across the gap.
Decoding is done in log space (no underflow on 19k-step sequences).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .metrics import group_starts


@dataclass
class HMMParams:
    beta: float = 5.0
    emission_weight: float = 1.0
    break_penalty: float = 30.0
    gap_growth: float = 0.1
    backtrack_m: float = 15.0
    cutoff_factor: float = 3.0
    cutoff_slack: float = 150.0
    vmax: float = 60.0          # m/s, bounds the search radius after gaps
    cutoff_cap: float = 8000.0

    def to_dict(self):
        return asdict(self)


def viterbi_decode(pt, obs, arc, t_along, prob, params: HMMParams = HMMParams()):
    """Returns (pred_arc[n_obs], pred_t[n_obs]) with -1 for obs without candidates."""
    g, pxy, ts = pt.g, pt.pxy, pt.t
    n_obs = len(pxy)
    pred = np.full(n_obs, -1, dtype=np.int64)
    pred_t = np.zeros(n_obs)
    if len(obs) == 0:
        return pred, pred_t
    st = group_starts(obs)
    en = np.r_[st[1:], len(obs)]
    steps = obs[st]
    pc = np.clip(prob, 1e-9, 1.0)

    def emis(s):
        p = pc[st[s]:en[s]]
        return params.emission_weight * np.log(p / p.sum())

    emissions = [emis(s) for s in range(len(st))]
    trans = []
    L = g.length
    for s in range(1, len(st)):
        i, j = steps[s - 1], steps[s]
        a0, a1 = st[s - 1], en[s - 1]
        b0, b1 = st[s], en[s]
        dgps = float(np.hypot(*(pxy[j] - pxy[i])))
        dt = float(ts[j] - ts[i])
        beta = params.beta * (1.0 + params.gap_growth * max(0.0, dt - 1.0))
        cutoff = min(params.cutoff_cap, max(params.cutoff_factor * dgps + params.cutoff_slack, params.vmax * dt))
        T = np.empty((a1 - a0, b1 - b0))
        for u in range(a0, a1):
            au, tu = arc[u], t_along[u]
            for v in range(b0, b1):
                av, tv = arc[v], t_along[v]
                if au == av and (tu - tv) * L[au] <= params.backtrack_m:
                    dn = max(0.0, (tv - tu) * L[au])
                else:
                    dn = g.route_distance(au, tu, av, tv, cutoff)
                T[u - a0, v - b0] = -params.break_penalty if not np.isfinite(dn) else -abs(dn - dgps) / beta
        trans.append(T)
    path = viterbi_core(emissions, trans)
    for s, k in enumerate(path):
        r = st[s] + k
        pred[steps[s]] = arc[r]
        pred_t[steps[s]] = t_along[r]
    return pred, pred_t


def viterbi_core(emissions, transitions):
    """Max-product (log-sum) Viterbi.

    delta_0(k)   = E_0(k)
    delta_s(k)   = max_j [ delta_{s-1}(j) + T_s(j, k) ] + E_s(k)
    psi_s(k)     = argmax_j [ ... ]
    path_{S-1}   = argmax_k delta_{S-1}(k);  path_{s-1} = psi_s(path_s)
    """
    score = np.asarray(emissions[0], float)
    backs = []
    for E, T in zip(emissions[1:], transitions):
        tot = score[:, None] + T
        bp = np.argmax(tot, axis=0)
        score = tot[bp, np.arange(tot.shape[1])] + E
        backs.append(bp)
    path = [int(np.argmax(score))]
    for bp in reversed(backs):
        path.append(int(bp[path[-1]]))
    return path[::-1]
