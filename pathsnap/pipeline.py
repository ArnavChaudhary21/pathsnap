"""End-to-end inference used by the API.

predict_trip()   uses ONLY .track/.nodes/.arcs (load_trip_inputs).
evaluate_trip()  is the only place that reads .route; it runs after the
                 prediction is final and never feeds anything back into it.
"""
from __future__ import annotations

import json
import os
import time

import joblib
import numpy as np

from .features import CandidateConfig, build_candidates, prepare_trip
from .hmm import HMMParams, viterbi_decode
from .io import load_reference_route, load_trip_inputs
from .labels import align_track_to_route
from .metrics import obs_metrics, per_obs_prediction, route_continuity, route_overlap, top1_rows


class Predictor:
    def __init__(self, artifact_dir: str):
        with open(os.path.join(artifact_dir, "manifest.json")) as f:
            self.manifest = json.load(f)
        self.model = joblib.load(os.path.join(artifact_dir, "classifier.joblib"))
        self.cand_cfg = CandidateConfig(**{k: tuple(v) if isinstance(v, list) else v
                                           for k, v in self.manifest["candidate_config"].items()})
        self.hmm = HMMParams(**self.manifest["hmm_params"])

    def predict_trip(self, folder: str) -> dict:
        timing = {}
        t0 = time.perf_counter()
        ti = load_trip_inputs(folder)
        pt = prepare_trip(ti)
        timing["load_project_s"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        c = build_candidates(pt, self.cand_cfg)
        timing["candidates_features_s"] = time.perf_counter() - t1
        t2 = time.perf_counter()
        prob = self.model.predict_proba(c["X"])[:, 1] if len(c["obs"]) else np.empty(0)
        timing["classifier_s"] = time.perf_counter() - t2
        t3 = time.perf_counter()
        pred, pred_t = viterbi_decode(pt, c["obs"], c["arc"], c["t_along"], prob, self.hmm)
        timing["viterbi_s"] = time.perf_counter() - t3
        n = len(pt.pxy)
        clf = per_obs_prediction(c["obs"], c["arc"], prob, n) if len(c["obs"]) else np.full(n, -1)
        clf_t = np.zeros(n)
        if len(c["obs"]):
            m = top1_rows(c["obs"], prob)
            clf_t[c["obs"][m]] = c["t_along"][m]
        timing["total_s"] = time.perf_counter() - t0
        return {"pt": pt, "pred": pred, "pred_t": pred_t, "clf": clf, "clf_t": clf_t,
                "n_candidates": int(len(c["obs"])), "timing": timing}


def evaluate_trip(folder: str, out: dict) -> dict | None:
    """Server-side scoring against the reference route (never returned raw)."""
    route = load_reference_route(folder)
    if route is None:
        return None
    pt = out["pt"]
    labels = align_track_to_route(pt.pxy, route, pt.g)["label"]
    res = {}
    for k, (p, t) in {"hmm": (out["pred"], out["pred_t"]), "classifier_only": (out["clf"], out["clf_t"])}.items():
        m = obs_metrics(p, labels)
        r = route_overlap(p, route, pt.g)
        res[k] = {"top1_accuracy": m["top1_acc"], "route_continuity": route_continuity(p, t, pt), **r}
    res["n_labelled_obs"] = int((labels >= 0).sum())
    res["n_uncertain_obs_excluded"] = int((labels < 0).sum())
    return res
