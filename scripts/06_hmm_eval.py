"""Checkpoint 3b: HMM/Viterbi route reconstruction experiments.

  python scripts/06_hmm_eval.py lgbm_tuned [n_decoder_trials]

1. Out-of-fold (OOF) classifier probabilities for the 80 train trips (same
   5 GroupKFold folds) - emissions on train trips must come from a model
   that did not see those trips, otherwise decoder tuning is optimistic.
2. Decoder parameters (beta, emission weight, break penalty, gap growth)
   tuned with Optuna TPE on a fixed subset of train trips using OOF
   probabilities. Objective: observation-level Top-1 accuracy.
3. Student-test comparison: classifier-only vs HMM(default) vs HMM(tuned):
   Top-1, route continuity, length-weighted route F1, per trip.
4. Folium maps for every student-test trip, and export of the final
   inference artifacts to artifacts/final/.
"""
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

from pathsnap.data import load_cached, stack
from pathsnap.evaluate import prepared
from pathsnap.features import FEATURES, CandidateConfig
from pathsnap.hmm import HMMParams, viterbi_decode
from pathsnap.io import list_trips, load_reference_route, trip_dir
from pathsnap.metrics import obs_metrics, per_obs_prediction, route_continuity, route_overlap, top1_rows
from pathsnap.models import SEED, make_model
from pathsnap.viz import trip_map

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "hmm")
MAPS = os.path.join(ROOT, "results", "maps")
os.makedirs(OUT, exist_ok=True)
os.makedirs(MAPS, exist_ok=True)


def oof_probs(tag, name, params):
    path = os.path.join(ROOT, "cache", f"oof_{tag}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    tids = list_trips("train")
    X, y, grp, _ = stack("train", tids, 3)
    out = {}
    for tr, va in GroupKFold(n_splits=5).split(X, y, grp):
        m = make_model(name, params)
        m.fit(X[tr], y[tr])
        for gi in np.unique(grp[va]):
            d = load_cached("train", tids[gi])
            out[tids[gi]] = m.predict_proba(d["X"])[:, 1]
        print("oof fold done", flush=True)
    np.savez_compressed(path, **out)
    return out


def decode_eval(partition, tid, prob, params):
    d = load_cached(partition, tid)
    pt = prepared(partition, tid)
    pred, pt_t = viterbi_decode(pt, d["obs"], d["arc"], d["t_along"], prob, params)
    return pred, pt_t, d, pt


def evaluate_trip(partition, tid, prob, params):
    d = load_cached(partition, tid)
    pt = prepared(partition, tid)
    route = load_reference_route(trip_dir(partition, tid))
    clf = per_obs_prediction(d["obs"], d["arc"], prob, int(d["n_obs"]))
    m = top1_rows(d["obs"], prob)
    clf_t = np.zeros(int(d["n_obs"])); clf_t[d["obs"][m]] = d["t_along"][m]
    t0 = time.perf_counter()
    hmm, hmm_t = viterbi_decode(pt, d["obs"], d["arc"], d["t_along"], prob, params)
    dec_s = time.perf_counter() - t0
    row = {"trip": tid, "n_obs": int(d["n_obs"]), "n_labelled": int((d["obs_label"] >= 0).sum()),
           "decode_s": dec_s}
    for k, (p, t) in {"clf": (clf, clf_t), "hmm": (hmm, hmm_t)}.items():
        row[f"{k}_top1"] = obs_metrics(p, d["obs_label"])["top1_acc"]
        row[f"{k}_continuity"] = route_continuity(p, t, pt)
        ro = route_overlap(p, route, pt.g)
        row[f"{k}_route_f1"] = ro["route_f1"]
        row[f"{k}_route_precision"] = ro["route_precision"]
        row[f"{k}_route_recall"] = ro["route_recall"]
    lab = d["obs_label"] >= 0
    row["fixed_by_hmm"] = int(((clf != d["obs_label"]) & (hmm == d["obs_label"]) & lab).sum())
    row["broken_by_hmm"] = int(((clf == d["obs_label"]) & (hmm != d["obs_label"]) & lab).sum())
    return row, clf, hmm


def summarise(rows, prefix):
    df = pd.DataFrame(rows)
    w = df.n_labelled / df.n_labelled.sum()
    return {f"{prefix}_{k}": float(v) for k, v in {
        "clf_top1": (df.clf_top1 * w).sum(), "hmm_top1": (df.hmm_top1 * w).sum(),
        "clf_continuity": df.clf_continuity.mean(), "hmm_continuity": df.hmm_continuity.mean(),
        "clf_route_f1": df.clf_route_f1.mean(), "hmm_route_f1": df.hmm_route_f1.mean(),
        "fixed": df.fixed_by_hmm.sum(), "broken": df.broken_by_hmm.sum(), "decode_s": df.decode_s.sum()}.items()}


def main(tag, n_trials):
    meta = json.load(open(os.path.join(ROOT, "results", "experiments", f"{tag}.json")))
    name, params = meta["algorithm"], meta["config"]
    model = joblib.load(os.path.join(ROOT, "artifacts", "models", f"{tag}.joblib"))

    # ---- 1. OOF emissions for train trips
    oof = oof_probs(tag, name, params)

    # ---- 2. decoder tuning on train trips (OOF), fixed subset for cost
    tids = list_trips("train")
    sizes = {t: int(load_cached("train", t)["n_obs"]) for t in tids}
    tune_trips = [t for t in tids if sizes[t] <= 4000][::2][:25]
    obs_lab = {t: load_cached("train", t)["obs_label"] for t in tune_trips}

    def objective(trial):
        p = HMMParams(beta=trial.suggest_float("beta", 1.0, 60.0, log=True),
                      emission_weight=trial.suggest_float("emission_weight", 0.2, 3.0, log=True),
                      break_penalty=trial.suggest_float("break_penalty", 5.0, 150.0, log=True),
                      gap_growth=trial.suggest_float("gap_growth", 0.0, 0.5))
        hit = tot = 0
        for t in tune_trips:
            pred, *_ = decode_eval("train", t, oof[t], p)
            lab = obs_lab[t] >= 0
            hit += int((pred[lab] == obs_lab[t][lab]).sum()); tot += int(lab.sum())
        return hit / tot

    study = optuna.create_study(direction="maximize", study_name="hmm_decoder",
                                storage=f"sqlite:///{os.path.join(OUT, 'decoder_study.db')}", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8))
    if len(study.trials) == 0:
        study.enqueue_trial({k: getattr(HMMParams(), k) for k in ("beta", "emission_weight", "break_penalty", "gap_growth")})
    remaining = n_trials - len([t for t in study.trials if t.state.is_finished()])
    t0 = time.time()
    if remaining > 0:
        study.optimize(objective, n_trials=remaining,
                       callbacks=[lambda s, t: print(f"decoder trial {t.number} top1={t.value:.4f} {t.params}", flush=True)])
    study.trials_dataframe().to_csv(os.path.join(OUT, "decoder_trials.csv"), index=False)
    tuned = HMMParams(**study.best_params)
    default = HMMParams()
    clf_tune = []
    for t in tune_trips:
        d = load_cached("train", t)
        clf_tune.append(obs_metrics(per_obs_prediction(d["obs"], d["arc"], oof[t], int(d["n_obs"])), d["obs_label"]))
    tune_info = {"tune_trips": tune_trips, "n_trials": len(study.trials), "wall_s": time.time() - t0,
                 "default_params": default.to_dict(), "tuned_params": tuned.to_dict(),
                 "default_top1_oof": study.trials[0].value, "best_top1_oof": study.best_value,
                 "classifier_only_top1_oof": float(np.average([c["top1_acc"] for c in clf_tune],
                                                              weights=[c["n_labelled_obs"] for c in clf_tune]))}

    # ---- 3. student-test comparison (model refit on all train trips)
    res = {"model": tag, "decoder_tuning": tune_info}
    per = {}
    for label, p in (("default", default), ("tuned", tuned)):
        rows = []
        for t in list_trips("student_test"):
            d = load_cached("student_test", t)
            prob = model.predict_proba(d["X"])[:, 1]
            row, clf, hmm = evaluate_trip("student_test", t, prob, p)
            rows.append(row)
            pt = prepared("student_test", t)                 # maps for both decoder settings
            trip_map(pt.ti, pt.g, pt.proj, reference=load_reference_route(trip_dir("student_test", t)),
                     predicted=hmm, classifier_only=clf,
                     title=f"student_test/{t}  HMM({label}) top-1 {row['hmm_top1']:.3f}").save(
                os.path.join(MAPS, f"student_test_{t}_hmm_{label}.html"))
        per[label] = rows
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f"student_test_per_trip_{label}.csv"), index=False)
        res.update(summarise(rows, f"student_test_{label}"))
    # Deployed decoder = the one with the higher student-test HMM Top-1 (decoder
    # selection on student-test is allowed; the evaluator partition is never used).
    chosen = "tuned" if res["student_test_tuned_hmm_top1"] >= res["student_test_default_hmm_top1"] else "default"
    deployed = tuned if chosen == "tuned" else default
    res["deployed_decoder"] = chosen
    with open(os.path.join(OUT, "hmm_results.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(json.dumps(res, indent=2, default=float))

    # ---- 4. export inference artifacts
    fin = os.path.join(ROOT, "artifacts", "final")
    os.makedirs(fin, exist_ok=True)
    shutil.copy(os.path.join(ROOT, "artifacts", "models", f"{tag}.joblib"), os.path.join(fin, "classifier.joblib"))
    json.dump({"classifier": tag, "algorithm": name, "classifier_params": params, "features": FEATURES,
               "candidate_config": CandidateConfig().__dict__, "hmm_params": deployed.to_dict(),
               "hmm_params_source": chosen},
              open(os.path.join(fin, "manifest.json"), "w"), indent=2, default=list)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 30)
