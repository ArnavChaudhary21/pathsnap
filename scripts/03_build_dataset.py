"""Build candidate/feature tables for train and student_test trips.

Writes cache/<partition>/<trip>.npz with
  obs, arc, X           candidate rows (inputs only, no ground truth)
  y                     1 if arc == reference arc of that observation
  obs_label             per-observation reference arc (-1 = uncertain/off-network)
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from joblib import Parallel, delayed

from pathsnap.features import CandidateConfig, build_candidates, prepare_trip
from pathsnap.io import list_trips, load_reference_route, load_trip_inputs, trip_dir
from pathsnap.labels import align_track_to_route

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")


def one(part, tid):
    t0 = time.time()
    f = trip_dir(part, tid)
    pt = prepare_trip(load_trip_inputs(f))
    c = build_candidates(pt, CandidateConfig())
    lab = align_track_to_route(pt.pxy, load_reference_route(f), pt.g)["label"]   # label side only
    y = (c["arc"] == lab[c["obs"]]).astype(np.int8)
    os.makedirs(os.path.join(CACHE, part), exist_ok=True)
    np.savez_compressed(os.path.join(CACHE, part, f"{tid}.npz"), obs=c["obs"], arc=c["arc"], X=c["X"],
                        t_along=c["t_along"], y=y, obs_label=lab, n_obs=len(pt.pxy))
    return part, tid, len(pt.pxy), len(c["obs"]), time.time() - t0


if __name__ == "__main__":
    jobs = [(p, t) for p in ("train", "student_test") for t in list_trips(p)]
    res = Parallel(n_jobs=10, verbose=0)(delayed(one)(p, t) for p, t in jobs)
    for r in res:
        print(*r[:4], f"{r[4]:.1f}s")
    print("total rows", sum(r[3] for r in res))
