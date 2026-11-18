"""Checkpoint 2: train and compare candidate classifiers.

  python scripts/04_train_models.py logreg gnb knn rf lgbm mlp

Protocol
* Training data: 80 train trips, every 3rd observation (all its candidates),
  observations with uncertain labels removed.
* 5-fold GroupKFold, groups = trip -> no trip is split across folds.
* Per fold we record validation AND training-fold scores (over/under-fitting).
* Final model: refit on all 80 trips, evaluated on all observations of the
  10 student-test trips (classifier-only, no HMM).
Results: results/experiments/<model>.json, artifacts/models/<model>.joblib
"""
import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib
import numpy as np
from sklearn.model_selection import GroupKFold

from pathsnap.data import stack
from pathsnap.evaluate import eval_classifier_on_partition, rows_top1
from pathsnap.io import list_trips
from pathsnap.metrics import candidate_metrics
from pathsnap.models import CONFIGS, LOSS, SEED, make_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "results", "experiments")
MOD = os.path.join(ROOT, "artifacts", "models")
os.makedirs(EXP, exist_ok=True)
os.makedirs(MOD, exist_ok=True)
STRIDE = 3
N_FOLDS = 5
TRAIN_EVAL_CAP = 150_000   # rows of the training fold re-scored for the train-vs-val gap


def fold_scores(model, X, y, obs):
    p = model.predict_proba(X)[:, 1]
    obs = np.unique(obs, return_inverse=True)[1]     # compact ids (order preserved)
    lab = np.zeros(obs.max() + 1, dtype=np.int64)   # all rows here are labelled observations
    m = candidate_metrics(obs, y, p, lab)
    m["top1_acc"] = rows_top1(obs, y, p)
    return m


def subsample_obs(obs, cap, rng):
    u = np.unique(obs)
    if len(obs) <= cap:
        return np.ones(len(obs), bool)
    take = rng.choice(u, size=int(len(u) * cap / len(obs)), replace=False)
    return np.isin(obs, take)


def run(name, params=None, tag=None):
    tag = tag or name
    train_t, test_t = list_trips("train"), list_trips("student_test")
    X, y, grp, obs = stack("train", train_t, STRIDE)
    rng = np.random.default_rng(SEED)
    folds = []
    for k, (tr, va) in enumerate(GroupKFold(n_splits=N_FOLDS).split(X, y, grp)):
        mdl = make_model(name, params)
        t0 = time.perf_counter()
        mdl.fit(X[tr], y[tr])
        fit_s = time.perf_counter() - t0
        va_m = fold_scores(mdl, X[va], y[va], obs[va])
        sub = tr[subsample_obs(obs[tr], TRAIN_EVAL_CAP, rng)]
        tr_m = fold_scores(mdl, X[sub], y[sub], obs[sub])
        folds.append({"fold": k, "fit_s": fit_s, "val": va_m, "train": tr_m,
                      "val_trips": sorted({train_t[g] for g in np.unique(grp[va])})})
        print(f"[{tag}] fold {k}: val F1={va_m['macro_f1']:.4f} top1={va_m['top1_acc']:.4f} "
              f"ll={va_m['log_loss']:.4f} | train F1={tr_m['macro_f1']:.4f} ({fit_s:.1f}s)", flush=True)

    final = make_model(name, params)
    t0 = time.perf_counter()
    final.fit(X, y)
    fit_all_s = time.perf_counter() - t0
    st, per_trip, _ = eval_classifier_on_partition(final, "student_test", test_t)
    print(f"[{tag}] student_test: {json.dumps({k: round(v, 4) for k, v in st.items()})}", flush=True)

    agg = lambda part, key: (float(np.mean([f[part][key] for f in folds])), float(np.std([f[part][key] for f in folds])))
    res = {
        "model": tag, "algorithm": name, "family": CONFIGS[name]["family"],
        "config": {**{k: v for k, v in CONFIGS[name].items() if k != "family"}, **(params or {})},
        "loss": LOSS[name], "seed": SEED, "obs_stride": STRIDE, "n_train_rows": int(len(y)),
        "positive_rate": float(y.mean()),
        "cv": {k: {"mean": agg("val", k)[0], "std": agg("val", k)[1]} for k in ("macro_f1", "top1_acc", "log_loss")},
        "cv_train": {k: {"mean": agg("train", k)[0], "std": agg("train", k)[1]} for k in ("macro_f1", "top1_acc", "log_loss")},
        "folds": folds, "fit_all_s": fit_all_s, "student_test": st, "student_test_per_trip": per_trip,
        "env": {"python": platform.python_version(), "machine": platform.processor()},
    }
    with open(os.path.join(EXP, f"{tag}.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    joblib.dump(final, os.path.join(MOD, f"{tag}.joblib"), compress=3)
    return res


if __name__ == "__main__":
    for n in sys.argv[1:]:
        run(n)
