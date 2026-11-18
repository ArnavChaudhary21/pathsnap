"""Export the feature-engineered dataset (cache/*.npz) to CSV.

  python scripts/09_export_features_csv.py

Writes results/features/train_features.csv and student_test_features.csv.
One row = one (GPS observation, candidate road arc) pair:
  trip_id, obs, arc       identifiers
  <22 features>           pathsnap.features.FEATURES
  y                       target: 1 if arc is the reference arc of this observation
  obs_label               reference arc of the observation (-1 = uncertain, excluded in training)
Requires scripts/03_build_dataset.py to have been run first.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd

from pathsnap.data import load_cached
from pathsnap.features import FEATURES
from pathsnap.io import list_trips

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "features")
os.makedirs(OUT, exist_ok=True)

for part in ("train", "student_test"):
    frames = []
    for tid in list_trips(part):
        d = load_cached(part, tid)
        df = pd.DataFrame(d["X"], columns=FEATURES)
        df.insert(0, "trip_id", tid)
        df.insert(1, "obs", d["obs"])
        df.insert(2, "arc", d["arc"])
        df["y"] = d["y"]
        df["obs_label"] = d["obs_label"][d["obs"]]
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    path = os.path.join(OUT, f"{part}_features.csv")
    full.to_csv(path, index=False, float_format="%.6g")
    print(f"{part}: {len(full):,} rows, {full.trip_id.nunique()} trips, "
          f"positive rate {full.y.mean():.3f} -> {path}")
