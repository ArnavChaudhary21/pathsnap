"""Shared evaluation helpers for classifiers on the student-testing partition."""
from __future__ import annotations

import time

import numpy as np

from .data import load_cached
from .features import prepare_trip
from .io import load_reference_route, load_trip_inputs, trip_dir
from .metrics import (candidate_metrics, obs_metrics, per_obs_prediction, route_continuity,
                      route_overlap, top1_rows)

_PT_CACHE: dict = {}


def prepared(partition, tid):
    key = (partition, tid)
    if key not in _PT_CACHE:
        _PT_CACHE[key] = prepare_trip(load_trip_inputs(trip_dir(partition, tid)))
    return _PT_CACHE[key]


def rows_top1(obs, y, prob):
    """Observation-level top-1 accuracy computed from candidate rows (labelled obs only)."""
    m = top1_rows(obs, prob)
    n_obs = len(np.unique(obs))
    return float(y[m].sum() / n_obs)


def eval_classifier_on_partition(model, partition, tids, with_route=True):
    """Classifier-only evaluation. Returns (summary, per_trip list, raw dict per trip)."""
    all_obs, all_y, all_p, all_lab = [], [], [], []
    per_trip, raw = [], {}
    infer_s = 0.0
    n_obs_total = 0
    off = 0
    for tid in tids:
        d = load_cached(partition, tid)
        t0 = time.perf_counter()
        p = model.predict_proba(d["X"])[:, 1]
        infer_s += time.perf_counter() - t0
        n_obs_total += int(d["n_obs"])
        pred = per_obs_prediction(d["obs"], d["arc"], p, int(d["n_obs"]))
        m = obs_metrics(pred, d["obs_label"], d["obs"], d["arc"])
        row = {"trip": tid, **m}
        if with_route:
            pt = prepared(partition, tid)
            tsel = np.zeros(len(pred))
            mask = top1_rows(d["obs"], p)
            tsel[d["obs"][mask]] = d["t_along"][mask]
            row["route_continuity"] = route_continuity(pred, tsel, pt)
            row.update(route_overlap(pred, load_reference_route(trip_dir(partition, tid)), pt.g))
        per_trip.append(row)
        raw[tid] = {"prob": p, "pred": pred}
        all_obs.append(d["obs"] + off); all_y.append(d["y"]); all_p.append(p)
        all_lab.append(d["obs_label"]); off += int(d["n_obs"])
    obs = np.concatenate(all_obs); y = np.concatenate(all_y); p = np.concatenate(all_p)
    lab = np.concatenate(all_lab)
    summ = candidate_metrics(obs, y, p, lab)
    pred_all = np.concatenate([raw[t]["pred"] for t in tids])
    summ.update(obs_metrics(pred_all, lab, obs, np.concatenate([load_cached(partition, t)["arc"] for t in tids])))
    if with_route:
        for k in ("route_continuity", "route_f1", "route_precision", "route_recall"):
            summ[k + "_mean"] = float(np.mean([r[k] for r in per_trip]))
    summ["inference_s_total"] = infer_s
    summ["inference_ms_per_1000_obs"] = 1000 * infer_s / n_obs_total * 1000
    return summ, per_trip, raw
