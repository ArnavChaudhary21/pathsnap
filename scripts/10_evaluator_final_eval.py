"""Final held-out evaluation on the 10 evaluator-testing trips.

  python scripts/10_evaluator_final_eval.py [folder]

Run ONCE, after artifacts/final/ has been frozen by scripts/06_hmm_eval.py.
Nothing here feeds back into model, feature, hyperparameter, decoder or
threshold choices - the evaluator partition is used for reporting only, as
required by the hackathon guidelines.

It uses exactly the code path of the API (pathsnap.pipeline.Predictor +
evaluate_trip), so the numbers match what the evaluator sees in the UI.

Outputs (results/final_eval/):
  evaluator_test_per_trip.csv    per-trip Top-1, route continuity, route P/R/F1, timings
  evaluator_test_summary.json    observation-weighted Top-1 + mean route metrics
  maps/evaluator_<trip>.html     Folium maps: GPS, network, reference, classifier-only, HMM route
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd

from pathsnap.io import load_reference_route
from pathsnap.pipeline import Predictor, evaluate_trip
from pathsnap.viz import trip_map

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = (sys.argv[1] if len(sys.argv) > 1 else
            os.environ.get("PATHSNAP_EVAL_DIR",
                           os.path.join(os.path.dirname(ROOT), "evaluator_test", "evaluator_test")))
OUT = os.path.join(ROOT, "results", "final_eval")
os.makedirs(os.path.join(OUT, "maps"), exist_ok=True)

pred = Predictor(os.path.join(ROOT, "artifacts", "final"))
rows = []
for tid in sorted(d for d in os.listdir(EVAL_DIR) if os.path.isdir(os.path.join(EVAL_DIR, d))):
    folder = os.path.join(EVAL_DIR, tid)
    out = pred.predict_trip(folder)                 # inputs only (.track/.nodes/.arcs)
    m = evaluate_trip(folder, out)                  # reference route read after prediction
    row = {"trip": tid, "n_obs": len(out["pt"].pxy), "n_labelled": m["n_labelled_obs"],
           "n_uncertain_excluded": m["n_uncertain_obs_excluded"]}
    for k in ("classifier_only", "hmm"):
        for mk, v in m[k].items():
            row[f"{k}_{mk}"] = v
    row.update({f"time_{k}": v for k, v in out["timing"].items()})
    rows.append(row)
    pt = out["pt"]
    trip_map(pt.ti, pt.g, pt.proj, reference=load_reference_route(folder), predicted=out["pred"],
             classifier_only=out["clf"],
             title=f"evaluator_test/{tid}  HMM top-1 {m['hmm']['top1_accuracy']:.3f}").save(
        os.path.join(OUT, "maps", f"evaluator_{tid}.html"))
    print(f"{tid}: clf top1={m['classifier_only']['top1_accuracy']:.4f}  "
          f"hmm top1={m['hmm']['top1_accuracy']:.4f}  hmm route F1={m['hmm']['route_f1']:.4f}  "
          f"({out['timing']['total_s']:.1f}s)", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "evaluator_test_per_trip.csv"), index=False)
w = df.n_labelled / df.n_labelled.sum()
summary = {"partition": os.path.basename(os.path.normpath(EVAL_DIR)), "n_trips": len(df),
           "model": pred.manifest["classifier"], "hmm_params": pred.manifest["hmm_params"]}
for k in ("classifier_only", "hmm"):
    summary[f"{k}_top1_accuracy"] = float((df[f"{k}_top1_accuracy"] * w).sum())
    for mk in ("route_continuity", "route_precision", "route_recall", "route_f1"):
        summary[f"{k}_{mk}_mean"] = float(df[f"{k}_{mk}"].mean())
summary["mean_total_s_per_trip"] = float(df.time_total_s.mean())
with open(os.path.join(OUT, "evaluator_test_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
