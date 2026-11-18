"""Evaluation metrics.

Conventions (used identically for every model and for the decoder):
* Observations whose reference label is -1 (uncertain / off-network) are
  excluded from every denominator.
* Observation-level Top-1 Accuracy: predicted arc == reference arc; an
  observation with no candidates or no prediction counts as wrong.
* Candidate-level Macro F1 / Precision / Recall: each candidate row is a
  binary example; the predicted positive is the top-scored candidate of its
  observation (the decision actually used by the system), so it does not
  depend on the calibration of a 0.5 threshold.
* Log Loss: binary cross-entropy of the row probabilities.
* Candidate Miss Rate: reference arc absent from the candidate set.
* Route Continuity: fraction of consecutive predicted pairs (i, i+1) that are
  drivable: same arc, or network distance <= 2 * GPS displacement + 50 m.
* Route F1 (length-weighted): overlap between the set of predicted arcs and
  the set of reference-route arcs, weighted by arc length.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score


def group_starts(obs):
    return np.flatnonzero(np.r_[True, obs[1:] != obs[:-1]])


def top1_rows(obs, score):
    """Boolean mask of the highest-scoring row within each observation."""
    st = group_starts(obs)
    cnt = np.diff(np.r_[st, len(obs)])
    grp = np.repeat(np.arange(len(st)), cnt)
    # stable argmax per group: sort by (group, -score)
    order = np.lexsort((-score, grp))
    first = np.r_[True, grp[order][1:] != grp[order][:-1]]
    mask = np.zeros(len(obs), bool)
    mask[order[first]] = True
    return mask


def per_obs_prediction(obs, arc, score, n_obs):
    pred = np.full(n_obs, -1, dtype=np.int64)
    m = top1_rows(obs, score)
    pred[obs[m]] = arc[m]
    return pred


def candidate_metrics(obs, y, prob, obs_label):
    """Row-level metrics restricted to rows of labelled observations."""
    keep = obs_label[obs] >= 0
    o, yy, p = obs[keep], y[keep], prob[keep]
    yhat = top1_rows(o, p).astype(int)
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return {
        "macro_f1": float(f1_score(yy, yhat, average="macro")),
        "precision_pos": float(precision_score(yy, yhat, zero_division=0)),
        "recall_pos": float(recall_score(yy, yhat, zero_division=0)),
        "log_loss": float(log_loss(yy, pc, labels=[0, 1])),
    }


def obs_metrics(pred, obs_label, obs=None, arc=None):
    lab = obs_label >= 0
    out = {"top1_acc": float(np.mean(pred[lab] == obs_label[lab])), "n_labelled_obs": int(lab.sum())}
    if obs is not None:
        present = np.zeros(len(obs_label), bool)
        hit = arc == obs_label[obs]
        present[obs[hit]] = True
        out["candidate_miss_rate"] = float(1 - present[lab].mean())
    return out


def route_continuity(pred, t_along_pred, pt, slack=50.0, factor=2.0):
    g, pxy = pt.g, pt.pxy
    ok = tot = 0
    for i in range(len(pred) - 1):
        a, b = pred[i], pred[i + 1]
        if a < 0 or b < 0:
            continue
        tot += 1
        if a == b:
            ok += 1
            continue
        gc = float(np.hypot(*(pxy[i + 1] - pxy[i])))
        lim = factor * gc + slack
        if g.route_distance(a, t_along_pred[i], b, t_along_pred[i + 1], lim) <= lim:
            ok += 1
    return ok / tot if tot else float("nan")


def route_overlap(pred, route, g):
    P = set(int(a) for a in pred if a >= 0)
    R = set(int(a) for a in route)
    L = g.length
    inter = sum(L[a] for a in P & R)
    prec = inter / max(sum(L[a] for a in P), 1e-9)
    rec = inter / max(sum(L[a] for a in R), 1e-9)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"route_precision": float(prec), "route_recall": float(rec), "route_f1": float(f1)}
