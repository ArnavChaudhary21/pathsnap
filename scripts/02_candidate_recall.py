"""Candidate-generation design study on TRAIN trips only.

For a grid of search radii and max-candidate counts we measure
* candidate recall (reference arc present in the set) = 1 - Candidate Miss Rate
* mean candidate-set size (drives training cost and Viterbi cost, O(K^2)).
Labels with status 'uncertain' (aligned arc > 50 m away) are excluded.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from pathsnap.features import prepare_trip
from pathsnap.io import list_trips, load_reference_route, load_trip_inputs, trip_dir
from pathsnap.labels import align_track_to_route

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RADII = [20, 30, 50, 75, 100]
KS = [3, 5, 8, 10, 15, 20]

rows = []
for tid in list_trips("train"):
    f = trip_dir("train", tid)
    pt = prepare_trip(load_trip_inputs(f))
    lab = align_track_to_route(pt.pxy, load_reference_route(f), pt.g)["label"]
    for i in np.flatnonzero(lab >= 0)[::2]:   # every 2nd labelled obs is plenty for this study
        arcs, d, _ = pt.g.arcs_near(pt.pxy[i, 0], pt.pxy[i, 1], max(RADII))
        pos = np.flatnonzero(arcs == lab[i])
        rows.append((tid, d[pos[0]] if len(pos) else np.inf, pos[0] if len(pos) else 10**6,
                     *[int((d <= r).sum()) for r in RADII]))
    print(tid, flush=True)
df = pd.DataFrame(rows, columns=["trip", "label_dist", "label_rank", *[f"n_{r}" for r in RADII]])
out = []
for r in RADII:
    for k in KS:
        # with nearest-first truncation, the label survives iff within r and rank < k
        ok = (df.label_dist <= r) & (df.label_rank < k)
        out.append({"radius": r, "max_cand": k, "recall": ok.mean(),
                    "miss_rate": 1 - ok.mean(), "mean_set_size": np.minimum(df[f"n_{r}"], k).mean()})
res = pd.DataFrame(out)
res.to_csv(os.path.join(ROOT, "results", "candidate_recall_train.csv"), index=False)
print(res.pivot(index="radius", columns="max_cand", values="recall").round(4))
print(res.pivot(index="radius", columns="max_cand", values="mean_set_size").round(2))
