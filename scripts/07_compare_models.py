"""Model comparison table + required plots, built only from results/experiments/*.json.

Outputs (results/comparison/):
  model_comparison.csv / .md       consolidated table (report section 3.1)
  cv_macro_f1.png                  CV Macro F1 per model (mean +- std over folds)
  student_test_top1_f1.png         student-test Top-1 accuracy and Macro F1
  log_loss.png                     CV and student-test log loss
  train_vs_val.png                 train-fold vs validation-fold Macro F1 (over/under-fitting)
  confusion_<model>.png            candidate-level confusion matrix on student test (top-1 decision)
  tuned_vs_untuned.csv             if <model>_tuned.json exists
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from pathsnap.data import load_cached
from pathsnap.io import list_trips
from pathsnap.metrics import top1_rows

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "results", "experiments")
OUT = os.path.join(ROOT, "results", "comparison")
os.makedirs(OUT, exist_ok=True)
ORDER = ["logreg", "gnb", "knn", "rf", "lgbm", "mlp", "lgbm_tuned", "rf_tuned", "mlp_tuned"]
NAMES = {"logreg": "Logistic Regression", "gnb": "Gaussian Naive Bayes", "knn": "KNN (distance-weighted)",
         "rf": "Random Forest", "lgbm": "LightGBM", "mlp": "MLP", "lgbm_tuned": "LightGBM (Bayes-tuned)",
         "rf_tuned": "Random Forest (Bayes-tuned)", "mlp_tuned": "MLP (Bayes-tuned)"}

res = {}
for f in glob.glob(os.path.join(EXP, "*.json")):
    r = json.load(open(f))
    res[r["model"]] = r
models = [m for m in ORDER if m in res]

rows = []
for m in models:
    r = res[m]
    st = r["student_test"]
    cfg = ", ".join(f"{k}={v}" for k, v in r["config"].items())
    rows.append({
        "Model": NAMES[m], "Algorithm family": r["family"], "Key configuration": cfg,
        "CV Macro F1": f"{r['cv']['macro_f1']['mean']:.4f} ± {r['cv']['macro_f1']['std']:.4f}",
        "CV Top-1": f"{r['cv']['top1_acc']['mean']:.4f}",
        "Top-1 Accuracy (student test)": f"{st['top1_acc']:.4f}",
        "Route continuity (student test)": f"{st['route_continuity_mean']:.4f}",
        "Student-Test Macro F1": f"{st['macro_f1']:.4f}",
        "Precision (pos)": f"{st['precision_pos']:.4f}", "Recall (pos)": f"{st['recall_pos']:.4f}",
        "Log Loss (student test)": f"{st['log_loss']:.4f}",
        "Route F1 (student test)": f"{st['route_f1_mean']:.4f}",
        "Inference (ms / 1000 obs)": f"{st['inference_ms_per_1000_obs']:.1f}",
        "Fit time all train (s)": f"{r['fit_all_s']:.1f}",
    })
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "model_comparison.csv"), index=False)
with open(os.path.join(OUT, "model_comparison.md"), "w", encoding="utf-8") as f:
    f.write(df.to_markdown(index=False))
print(df.drop(columns=["Key configuration"]).to_string(index=False))

lab = [NAMES[m].replace(" (", "\n(") for m in models]
x = np.arange(len(models))


def bar(ax, vals, err=None, title="", ylim=None, fmt="{:.4f}"):
    b = ax.bar(x, vals, yerr=err, capsize=4, color="#4c78a8")
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=8); ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    for xi, v in zip(x, vals):
        ax.text(xi, v, fmt.format(v), ha="center", va="bottom", fontsize=8)


fig, ax = plt.subplots(figsize=(11, 4.5))
bar(ax, [res[m]["cv"]["macro_f1"]["mean"] for m in models], [res[m]["cv"]["macro_f1"]["std"] for m in models],
    "5-fold GroupKFold CV Macro F1 (80 train trips), mean ± std", (0.85, 1.0))
fig.tight_layout(); fig.savefig(os.path.join(OUT, "cv_macro_f1.png"), dpi=130); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(15, 4.5))
bar(axs[0], [res[m]["student_test"]["top1_acc"] for m in models], title="Student test: observation Top-1 accuracy", ylim=(0.8, 1.0))
bar(axs[1], [res[m]["student_test"]["macro_f1"] for m in models], title="Student test: candidate Macro F1", ylim=(0.85, 1.0))
fig.tight_layout(); fig.savefig(os.path.join(OUT, "student_test_top1_f1.png"), dpi=130); plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 4.5))
w = 0.38
ax.bar(x - w / 2, [res[m]["cv"]["log_loss"]["mean"] for m in models], w, label="CV (mean)")
ax.bar(x + w / 2, [res[m]["student_test"]["log_loss"] for m in models], w, label="Student test")
ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=8); ax.set_yscale("log"); ax.legend()
ax.set_title("Binary log loss of candidate probabilities (lower is better, log scale)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "log_loss.png"), dpi=130); plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.bar(x - w / 2, [res[m]["cv_train"]["macro_f1"]["mean"] for m in models], w, label="training folds")
ax.bar(x + w / 2, [res[m]["cv"]["macro_f1"]["mean"] for m in models], w, label="validation folds")
ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=8); ax.set_ylim(0.85, 1.0); ax.legend()
ax.set_title("Train-fold vs validation-fold Macro F1 (gap = overfitting indicator)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "train_vs_val.png"), dpi=130); plt.close(fig)

# per-fold stability
fig, ax = plt.subplots(figsize=(11, 4.5))
for m in models:
    ax.plot(range(5), [f["val"]["macro_f1"] for f in res[m]["folds"]], marker="o", label=NAMES[m])
ax.set_xlabel("fold"); ax.set_ylabel("validation Macro F1"); ax.set_xticks(range(5)); ax.legend(fontsize=8)
ax.set_title("Per-fold validation Macro F1 (stability across trip groups)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "cv_folds.png"), dpi=130); plt.close(fig)

# confusion matrices on student test (top-1 decision rule), from saved models
tids = list_trips("student_test")
cache = [load_cached("student_test", t) for t in tids]
for m in models:
    p = os.path.join(ROOT, "artifacts", "models", f"{m}.joblib")
    if not os.path.exists(p):
        continue
    mdl = joblib.load(p)
    yt, yp = [], []
    for d in cache:
        keep = d["obs_label"][d["obs"]] >= 0
        prob = mdl.predict_proba(d["X"][keep])[:, 1]
        yp.append(top1_rows(d["obs"][keep], prob).astype(int)); yt.append(d["y"][keep])
    cm = confusion_matrix(np.concatenate(yt), np.concatenate(yp))
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ConfusionMatrixDisplay(cm, display_labels=["other arc", "reference arc"]).plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(NAMES[m], fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"confusion_{m}.png"), dpi=130); plt.close(fig)

for base in ("lgbm", "rf", "mlp"):
    if base in res and f"{base}_tuned" in res:
        u, t = res[base], res[f"{base}_tuned"]
        comp = pd.DataFrame({
            "metric": ["CV Macro F1", "Student-test Top-1", "Student-test Macro F1", "Student-test Log Loss",
                       "Inference ms/1000 obs", "Route continuity"],
            "untuned": [u["cv"]["macro_f1"]["mean"], u["student_test"]["top1_acc"], u["student_test"]["macro_f1"],
                        u["student_test"]["log_loss"], u["student_test"]["inference_ms_per_1000_obs"],
                        u["student_test"]["route_continuity_mean"]],
            "tuned": [t["cv"]["macro_f1"]["mean"], t["student_test"]["top1_acc"], t["student_test"]["macro_f1"],
                      t["student_test"]["log_loss"], t["student_test"]["inference_ms_per_1000_obs"],
                      t["student_test"]["route_continuity_mean"]]})
        comp.to_csv(os.path.join(OUT, f"tuned_vs_untuned_{base}.csv"), index=False)
        print(comp.to_string(index=False))
